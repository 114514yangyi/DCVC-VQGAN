import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import warnings

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from ..layers.layers import MaskedConv2d, subpel_conv3x3
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste
from compressai.models import CompressionModel
from compressai.ans import BufferedRansEncoder, RansDecoder


class DCVC_net(CompressionModel):
    def __init__(self, lmbda=1.0):
        super().__init__()
        out_channel_mv = 128
        out_channel_N = 64
        out_channel_M = 96

        self.lmbda = lmbda

        self.out_channel_mv = out_channel_mv
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.bitEstimator_z = EntropyBottleneck(out_channel_N)
        self.bitEstimator_z_mv = EntropyBottleneck(out_channel_N)


        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.gaussian_conditional_mv = GaussianConditional(None)

        self.mvEncoder = nn.Sequential(
            nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        )

        self.mvDecoder_part1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, 2, 3, stride=2, padding=1, output_padding=1),
        )

        self.mvDecoder_part2 = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )

        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N+3, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N*2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_M, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3, stride=1, padding=1)
        )

        self.mvpriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_mv, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.mvpriorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_N * 3 // 2, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N * 3 // 2, out_channel_mv*2, 3, stride=1, padding=1)
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d(
            out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1
        )

        self.auto_regressive_mv = MaskedConv2d(
            out_channel_mv, 2 * out_channel_mv, kernel_size=5, padding=2, stride=1
        )

        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(out_channel_mv * 12 // 3, out_channel_mv * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 10 // 3, out_channel_mv * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 8 // 3, out_channel_mv * 6 // 3, 1),
        )

        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.opticFlow = ME_Spynet()
        self.entropy_bottleneck = None  # Deprecated, but kept for compatibility

    def motioncompensation(self, ref, mv):
        ref_feature = self.feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        context = self.context_refine(prediction_init)

        return context
    
    def pixel_motioncompensation(self, ref, mv):
        warped = flow_warp(ref, mv)
        return warped

    def mv_refine(self, ref, mv):
        return self.mvDecoder_part2(torch.cat((mv, ref), 1)) + mv

    def calculate_mse(self, x, x_hat):
        """Calculate MSE in RGB domain"""
        mse_loss = F.mse_loss(x, x_hat, reduction='mean')
        return mse_loss

    def forward(self, referframe, input_image, stage=1):
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        z_mv = self.mvpriorEncoder(mvfeature)

        _, z_mv_likelihoods = self.bitEstimator_z_mv(z_mv)
        z_mv_medians = self.bitEstimator_z_mv._get_medians()
        compressed_z_mv = quantize_ste(z_mv - z_mv_medians)+ z_mv_medians

        params_mv = self.mvpriorDecoder(compressed_z_mv)

        quant_mv = self.gaussian_conditional_mv.quantize(
            mvfeature, "noise" if self.training else "dequantize"
        )

        ctx_params_mv = self.auto_regressive_mv(quant_mv)
        gaussian_params_mv = self.entropy_parameters_mv(
            torch.cat((params_mv, ctx_params_mv), dim=1)
        )
        scales_hat_mv, means_hat_mv = gaussian_params_mv.chunk(2, 1)
        _, mv_likelihoods = self.gaussian_conditional_mv(mvfeature, scales_hat_mv, means=means_hat_mv)

        quant_mv_upsample = self.mvDecoder_part1(quant_mv)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)
        pixel_rec = self.pixel_motioncompensation(referframe, quant_mv_upsample_refine)

        temporal_prior_params = self.temporalPriorEncoder(context)
        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        _, z_likelihoods = self.bitEstimator_z(z)
        z_medians = self.bitEstimator_z._get_medians()
        compressed_z = quantize_ste(z - z_medians) + z_medians
            
        params = self.priorDecoder(compressed_z)
        feature_renorm = feature

        compressed_y_renorm = self.gaussian_conditional.quantize(
            feature_renorm, "noise" if self.training else "dequantize"
        )

        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        _, y_likelihoods = self.gaussian_conditional(
            feature_renorm, scales_hat, means=means_hat
        )


        recon_image_feature = self.contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]

        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * pixel_num)
        bpp_mv_z = torch.log(z_mv_likelihoods).sum() / (-math.log(2) * pixel_num)
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * pixel_num)
        bpp_mv_y = torch.log(mv_likelihoods).sum() / (-math.log(2) * pixel_num)

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        loss = 0
        #loss calculation
        if stage == 1:
            #in stage 1, we calculate L_me = lambda*distortion(rec,inp) + bpp_mv_y + bpp_mv_z
            mse_loss = self.calculate_mse(pixel_rec, input_image)
            distortion = self.lmbda * mse_loss
            L_me = distortion + bpp_mv_y + bpp_mv_z
            bpp_train = bpp_mv_y + bpp_mv_z
            loss = L_me
        elif stage == 2:
            #in stage 2, we train other modules except mv generation module. at this time, we freeze the mv generation module and calculate L_rec = lambda*distortion(rec,inp)
            mse_loss = self.calculate_mse(recon_image, input_image)
            L_rec = self.lmbda * mse_loss
            bpp_train = torch.tensor(0)
            loss = L_rec
        elif stage == 3:
            #in stage 3, the mv generation module is still frozen, and we calculate L_con = lambda*distortion(rec,inp) + bpp_y + bpp_z
            mse_loss = self.calculate_mse(recon_image, input_image)
            distortion = self.lmbda * mse_loss
            L_con = distortion + bpp_y + bpp_z
            bpp_train = bpp_y + bpp_z
            loss = L_con
        elif stage == 4:
            #in stage 4, we train all modules and calculate L_all = lambda*distortion(rec,inp) + bpp
            mse_loss = self.calculate_mse(recon_image, input_image)
            distortion = self.lmbda * mse_loss
            L_all = distortion + bpp
            bpp_train = bpp
            loss = L_all

        return {"bpp_mv_y": bpp_mv_y, "bpp_mv_z": bpp_mv_z, "bpp_y": bpp_y, "bpp_z": bpp_z,
                "bpp": bpp, "recon_image": recon_image if stage !=1 else pixel_rec, "context": context, "loss": loss,
                "mse_loss": mse_loss, "bpp_train": bpp_train}

    def load_dict(self, pretrained_dict):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight
        self.load_state_dict(result_dict)

    @torch.no_grad()
    def compress(self, referframe, input_image):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )
        
        # Motion vector compression
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        z_mv = self.mvpriorEncoder(mvfeature)
        z_mv_strings = self.bitEstimator_z_mv.compress(z_mv)
        z_mv_hat = self.bitEstimator_z_mv.decompress(z_mv_strings, z_mv.size()[-2:])
        params_mv = self.mvpriorDecoder(z_mv_hat)
        
        mv_y_strings, quant_mv = self._compress_ar_mv(mvfeature, params_mv)

        # Motion compensation
        quant_mv_upsample = self.mvDecoder_part1(quant_mv)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)

        # Residual compression
        temporal_prior_params = self.temporalPriorEncoder(context)
        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        z_strings = self.bitEstimator_z.compress(z)
        z_hat = self.bitEstimator_z.decompress(z_strings, z.size()[-2:])
        params = self.priorDecoder(z_hat)
        
        y_strings, _ = self._compress_ar_res(feature, params, temporal_prior_params)
        
        return {
            "strings": [y_strings, z_strings, mv_y_strings, z_mv_strings],
            "shape": z.size()[-2:],
            "mv_shape": z_mv.size()[-2:],
        }

    @torch.no_grad()
    def decompress(self, referframe, strings):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).",
                stacklevel=2,
            )
        
        y_strings, z_strings, mv_y_strings, z_mv_strings = strings["strings"]
        z_shape = strings["shape"]
        mv_z_shape = strings["mv_shape"]
        
        # Motion vector decompression
        z_mv_hat = self.bitEstimator_z_mv.decompress([z_mv_strings], mv_z_shape)
        params_mv = self.mvpriorDecoder(z_mv_hat)
        quant_mv = self._decompress_ar_mv(mv_y_strings, params_mv)

        # Motion compensation
        quant_mv_upsample = self.mvDecoder_part1(quant_mv)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)

        # Residual decompression
        z_hat = self.bitEstimator_z.decompress([z_strings], z_shape)
        params = self.priorDecoder(z_hat)
        temporal_prior_params = self.temporalPriorEncoder(context)
        compressed_y_renorm = self._decompress_ar_res(y_strings, params, temporal_prior_params)

        # Final image reconstruction
        recon_image_feature = self.contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))
        
        return {"recon_image": recon_image.clamp_(0,1)}

    def _compress_ar_mv(self, y, params):
        cdf = self.gaussian_conditional_mv.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional_mv.cdf_length.tolist()
        offsets = self.gaussian_conditional_mv.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        context_prediction = self.auto_regressive_mv
        entropy_parameters = self.entropy_parameters_mv
        gaussian_conditional = self.gaussian_conditional_mv

        kernel_size = context_prediction.kernel_size[0]
        padding = (kernel_size - 1) // 2
        height, width = y.size(2), y.size(3)

        y_hat = F.pad(y, (padding, padding, padding, padding))
        masked_weight = context_prediction.weight * context_prediction.mask

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(y_crop, masked_weight, bias=context_prediction.bias)
                
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = gaussian_conditional.build_indexes(scales_hat)
                y_q = gaussian_conditional.quantize(y[:, :, h, w], "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        string = encoder.flush()
        
        return string, y_hat[:, :, padding:-padding, padding:-padding]

    def _compress_ar_res(self, y, params, temporal_params):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        context_prediction = self.auto_regressive
        entropy_parameters = self.entropy_parameters
        gaussian_conditional = self.gaussian_conditional

        kernel_size = context_prediction.kernel_size[0]
        padding = (kernel_size - 1) // 2
        height, width = y.size(2), y.size(3)

        y_hat = F.pad(y, (padding, padding, padding, padding))
        masked_weight = context_prediction.weight * context_prediction.mask
        
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(y_crop, masked_weight, bias=context_prediction.bias)

                p = params[:, :, h:h+1, w:w+1]
                temp_p = temporal_params[:, :, h:h+1, w:w+1]
                
                gaussian_params = entropy_parameters(torch.cat((temp_p, p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = gaussian_conditional.build_indexes(scales_hat)
                y_q = gaussian_conditional.quantize(y[:, :, h, w], "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())
        
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        string = encoder.flush()

        return string, y_hat[:, :, padding:-padding, padding:-padding]

    def _decompress_ar_mv(self, string, params):
        cdf = self.gaussian_conditional_mv.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional_mv.cdf_length.tolist()
        offsets = self.gaussian_conditional_mv.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(string)

        context_prediction = self.auto_regressive_mv
        entropy_parameters = self.entropy_parameters_mv
        gaussian_conditional = self.gaussian_conditional_mv

        kernel_size = context_prediction.kernel_size[0]
        padding = (kernel_size - 1) // 2
        height, width = params.size(2), params.size(3)
        
        y_hat = torch.zeros(
            (params.size(0), self.out_channel_mv, height + 2 * padding, width + 2 * padding),
            device=params.device,
        )
        
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(y_crop, context_prediction.weight, bias=context_prediction.bias)

                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1).to(params.device)
                rv = gaussian_conditional.dequantize(rv, means_hat)
                y_hat[:, :, h + padding, w + padding] = rv
        
        return F.pad(y_hat, (-padding, -padding, -padding, -padding))

    def _decompress_ar_res(self, string, params, temporal_params):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()
        
        decoder = RansDecoder()
        decoder.set_stream(string)

        context_prediction = self.auto_regressive
        entropy_parameters = self.entropy_parameters
        gaussian_conditional = self.gaussian_conditional
        
        kernel_size = context_prediction.kernel_size[0]
        padding = (kernel_size - 1) // 2
        height, width = params.size(2), params.size(3)
        
        y_hat = torch.zeros(
            (params.size(0), self.out_channel_M, height + 2 * padding, width + 2 * padding),
            device=params.device,
        )

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(y_crop, context_prediction.weight, bias=context_prediction.bias)
                
                p = params[:, :, h:h+1, w:w+1]
                temp_p = temporal_params[:, :, h:h+1, w:w+1]
                gaussian_params = entropy_parameters(torch.cat((temp_p, p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(indexes.squeeze().tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1).to(params.device)
                rv = gaussian_conditional.dequantize(rv, means_hat)
                y_hat[:, :, h + padding, w + padding] = rv

        return F.pad(y_hat, (-padding, -padding, -padding, -padding))
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the inherited method to use the standard PyTorch loader.
        This completely bypasses the problematic `remap_old_keys` function
        for DCVC_net instances, while allowing library models to use it.
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        super(CompressionModel, self).load_state_dict(new_state_dict, strict=strict)