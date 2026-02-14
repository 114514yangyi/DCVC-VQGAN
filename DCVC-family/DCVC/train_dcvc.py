import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import math
from PIL import Image
import torchvision.transforms as transforms
import time
import logging
import datetime
from timm.utils import unwrap_model
from torch_ema import ExponentialMovingAverage

# Add Accelerate imports
from accelerate import Accelerator
from accelerate.utils import set_seed

# Import models
from src.models.DCVC_net_compressai import DCVC_net

# Import CompressAI models and utilities
from compressai.zoo import models as compressai_models
from compressai.optimizers import net_aux_optimizer

# Import datasets
from dataset import VimeoGOPDataset, BVI_AOM_Dataset, HEVCB_Dataset, VideoGOPDataset, VideoValidationDataset

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Add deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from compressai.optimizers import net_aux_optimizer
def configure_optimizers(net, args, is_finetuning=False):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    learning_rate = args.learning_rate if not is_finetuning else args.finetune_lr
    conf = {
        "net": {"type": "Adam", "lr": learning_rate},
        "aux": {"type": "Adam", "lr": learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"]


def setup_device_for_mps_fallback(device, logger=None):
    """
    Setup device with MPS fallback support.
    If MPS device is detected, enable CPU fallback for unsupported operations.
    
    Args:
        device: torch.device object
        logger: Optional logger for logging messages
    
    Returns:
        device: torch.device object (may be changed to CPU if MPS has issues)
    """
    # Check if MPS is available and being used
    if device.type == 'mps':
        # Enable MPS fallback to CPU for unsupported operations
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        if logger:
            logger.warning("MPS device detected. Enabling CPU fallback for unsupported operations.")
            logger.warning("This may result in slower performance but ensures compatibility.")
        else:
            print("Warning: MPS device detected. Enabling CPU fallback for unsupported operations.")
            print("This may result in slower performance but ensures compatibility.")
    
    return device


def setup_logging(checkpoint_dir, accelerator):
    """Setup logging configuration with Accelerate support"""
    log_dir = os.path.join(checkpoint_dir, 'logs')
    
    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        # Configure logging for main process
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
    else:
        # Suppress most logging for non-main processes
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
    
    return logger


def compress_i_frame_with_padding(i_frame_model, frame_tensor, calculate_bpp=True):
    """
    Compress I-frame with proper padding and optionally calculate accurate BPP.

    Args:
        i_frame_model: CompressAI model for I-frame compression (expects RGB)
        frame_tensor: Input RGB frame tensor [B, C(=3), H, W] in [0,1]
        calculate_bpp: Whether to calculate accurate BPP (default: True)

    Returns:
        If calculate_bpp=True: (encoded_result, bpp) where encoded_result['x_hat']
        is RGB; If calculate_bpp=False: encoded_result
    """
    B, C, H, W = frame_tensor.shape

    # Calculate padding (kept consistent with original logic)
    padding_r, padding_b = H % 16, W % 16

    # Apply padding if needed
    if padding_r > 0 or padding_b > 0:
        rgb_padded = torch.nn.functional.pad(frame_tensor, (0, padding_b, 0, padding_r), mode='reflect')
    else:
        rgb_padded = frame_tensor

    # Compress in RGB space
    encoded = i_frame_model(rgb_padded)

    # Reconstructed RGB
    x_hat_rgb = encoded['x_hat']

    # Remove padding (crop) if it was applied
    if padding_r > 0 or padding_b > 0:
        x_hat_rgb = x_hat_rgb[:, :, :H, :W]

    encoded['x_hat'] = x_hat_rgb

    # Calculate BPP if requested
    if calculate_bpp:
        num_pixels = B * H * W
        total_bits = 0.0
        import math
        for likelihood in encoded['likelihoods'].values():
            bits = torch.log(likelihood).sum() / (-math.log(2))
            total_bits += bits
        bpp = total_bits.item() / num_pixels
        return encoded, bpp
    else:
        return encoded


def compress_p_frame_with_padding(model, ref_frame, current_frame, stage):
    """
    Compress P-frame with proper padding to multiples of 16.

    Args:
        model: DCVC model for P-frame compression
        ref_frame: Reference frame tensor [B, C(=3), H, W] in [0,1]
        current_frame: Current frame tensor [B, C(=3), H, W] in [0,1]
        stage: Training stage

    Returns:
        dict: Result dictionary with padded processing and cropped output
    """
    B, C, H, W = current_frame.shape

    # Calculate padding (same as I-frame logic)
    padding_r, padding_b = H % 16, W % 16

    # Apply padding if needed
    if padding_r > 0 or padding_b > 0:
        ref_padded = torch.nn.functional.pad(ref_frame, (0, padding_b, 0, padding_r), mode='reflect')
        current_padded = torch.nn.functional.pad(current_frame, (0, padding_b, 0, padding_r), mode='reflect')
    else:
        ref_padded = ref_frame
        current_padded = current_frame

    # Process P-frame with padding
    result = model(ref_padded, current_padded, stage=stage)

    # Remove padding from reconstructed frame if it was applied
    if padding_r > 0 or padding_b > 0:
        result["recon_image"] = result["recon_image"][:, :, :H, :W]

    return result


def calculate_rgb_metrics(rgb_ref, rgb_target):
    """
    Calculate MSE and PSNR between two RGB tensors.

    Args:
        rgb_ref: Reference RGB tensor (B, 3, H, W)
        rgb_target: Target RGB tensor (B, 3, H, W)

    Returns:
        dict: {'mse': float, 'psnr': float}
    """
    # Calculate MSE
    mse = F.mse_loss(rgb_ref, rgb_target, reduction='mean').item()

    # Calculate PSNR (assuming input range [0, 1])
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')

    return {
        'mse': mse,
        'psnr': psnr
    }


def evaluate_hevc_b(model, i_frame_model, hevc_b_dir, device, stage, max_frames=96, accelerator=None, use_video_dataset=False):
    """
    Evaluate DCVC model on HEVC-B dataset with reset mechanism using online I-frame compression.
    Now properly includes I-frame BPP in the overall evaluation metrics.
    Returns detailed per-frame statistics including I-frame and first 3 P-frames.
    
    Args:
        use_video_dataset: If True, use VideoValidationDataset instead of HEVCB_Dataset
    """
    model.eval()
    i_frame_model.eval()

    if use_video_dataset and hevc_b_dir:
        # Use video dataset for validation
        eval_dataset = VideoValidationDataset(
            video_dir=hevc_b_dir,
            transform=transforms.ToTensor(),
            max_frames=max_frames
        )
    else:
        # Use original HEVCB_Dataset (image sequences)
        eval_dataset = HEVCB_Dataset(root_dir=hevc_b_dir, max_frames=max_frames)

    if len(eval_dataset) == 0:
        if accelerator is None or accelerator.is_main_process:
            print(f"Warning: No HEVC-B sequences found in {hevc_b_dir}")
        return None

    results = {}

    if accelerator is None or accelerator.is_main_process:
        print(f"  Evaluating on {len(eval_dataset)} HEVC-B sequences")

    with torch.no_grad():
        for seq_idx in range(len(eval_dataset)):
            try:
                sequence_data = eval_dataset[seq_idx]
            except Exception as e:
                if accelerator is None or accelerator.is_main_process:
                    print(f"Error loading sequence {seq_idx} from evaluation dataset: {e}")
                raise  # 直接抛出异常，中断评估
            
            if sequence_data is None:
                if accelerator is None or accelerator.is_main_process:
                    print(f"Warning: Sequence {seq_idx} returned None, skipping...")
                continue

            frames = sequence_data['frames'].unsqueeze(0).to(device)
            seq_name = sequence_data['name']
            num_frames = sequence_data['num_frames']

            if accelerator is None or accelerator.is_main_process:
                print(f"\n=== Sequence: {seq_name} ({num_frames} frames) ===")

            total_bpp = 0
            total_mse = 0
            total_psnr = 0

            # Track P-frame statistics separately
            p_frame_bpp_sum = 0
            p_frame_mse_sum = 0
            p_frame_psnr_sum = 0
            p_frame_count = 0

            # Store first 3 P-frame details
            first_three_p_frames = []

            # Process I-frame with BPP calculation
            i_frame = frames[0, 0, ...]
            i_frame_batch = i_frame.unsqueeze(0)

            # Compress I-frame
            i_frame_result, i_frame_bpp = compress_i_frame_with_padding(
                i_frame_model, i_frame_batch, calculate_bpp=True
            )
            ref_frame = i_frame_result['x_hat']  # RGB

            i_frame_metric = calculate_rgb_metrics(ref_frame, i_frame_batch)
            i_frame_mse = i_frame_metric['mse']
            i_frame_psnr = i_frame_metric['psnr']

            if accelerator is None or accelerator.is_main_process:
                print(f"Frame 0 (I-frame): BPP={i_frame_bpp:.6f}, PSNR={i_frame_psnr:.2f} dB, MSE={i_frame_mse:.6f}")

            # Add I-frame metrics to totals
            total_bpp += i_frame_bpp
            total_mse += i_frame_mse
            total_psnr += i_frame_psnr

            # Process P-frames
            for frame_idx in range(1, num_frames):
                current_frame = frames[0, frame_idx, ...].unsqueeze(0)

                # Use padding for P-frame compression
                result = compress_p_frame_with_padding(model, ref_frame, current_frame, stage)

                p_frame_bpp = result["bpp_train"].item()
                p_frame_metric = calculate_rgb_metrics(result["recon_image"], current_frame)
                p_frame_psnr = p_frame_metric['psnr']
                p_frame_mse = p_frame_metric['mse']

                if accelerator is None or accelerator.is_main_process:
                    print(f"Frame {frame_idx} (P-frame): BPP={p_frame_bpp:.6f}, "
                          f"PSNR={p_frame_psnr:.2f} dB, MSE={p_frame_mse:.6f}")

                # Add to totals
                total_bpp += p_frame_bpp
                total_mse += p_frame_mse
                total_psnr += p_frame_psnr

                # Track P-frame statistics
                p_frame_bpp_sum += p_frame_bpp
                p_frame_mse_sum += p_frame_mse
                p_frame_psnr_sum += p_frame_psnr
                p_frame_count += 1

                # Store first 3 P-frame details
                if frame_idx <= 3:
                    first_three_p_frames.append({
                        'frame_idx': frame_idx,
                        'bpp': p_frame_bpp,
                        'psnr': p_frame_psnr,
                        'mse': p_frame_mse
                    })

                ref_frame = result["recon_image"]

            # Calculate averages (including I-frame in all metrics)
            avg_mse = total_mse / num_frames
            avg_psnr = total_psnr / num_frames
            avg_bpp = total_bpp / num_frames

            # Calculate P-frame averages
            avg_p_frame_bpp = p_frame_bpp_sum / p_frame_count if p_frame_count > 0 else 0
            avg_p_frame_mse = p_frame_mse_sum / p_frame_count if p_frame_count > 0 else 0
            avg_p_frame_psnr = p_frame_psnr_sum / p_frame_count if p_frame_count > 0 else 0

            results[seq_name] = {
                'avg_psnr': avg_psnr,
                'avg_bpp': avg_bpp,
                'avg_mse': avg_mse,
                'num_frames': num_frames,
                'i_frame_bpp': i_frame_bpp,
                'i_frame_psnr': i_frame_psnr,
                'i_frame_mse': i_frame_mse,
                'avg_p_frame_bpp': avg_p_frame_bpp,
                'avg_p_frame_psnr': avg_p_frame_psnr,
                'avg_p_frame_mse': avg_p_frame_mse,
                'p_frame_count': p_frame_count,
                'first_three_p_frames': first_three_p_frames
            }

    # Calculate overall averages
    if results:
        psnr_list = [results[seq]['avg_psnr'] for seq in results]
        bpp_list = [results[seq]['avg_bpp'] for seq in results]
        mse_list = [results[seq]['avg_mse'] for seq in results]

        # I-frame specific metrics
        i_frame_psnr_list = [results[seq]['i_frame_psnr'] for seq in results]
        i_frame_bpp_list = [results[seq]['i_frame_bpp'] for seq in results]
        i_frame_mse_list = [results[seq]['i_frame_mse'] for seq in results]

        # P-frame specific metrics
        p_frame_psnr_list = [results[seq]['avg_p_frame_psnr'] for seq in results if results[seq]['p_frame_count'] > 0]
        p_frame_bpp_list = [results[seq]['avg_p_frame_bpp'] for seq in results if results[seq]['p_frame_count'] > 0]
        p_frame_mse_list = [results[seq]['avg_p_frame_mse'] for seq in results if results[seq]['p_frame_count'] > 0]

        overall_results = {
            'avg_psnr': sum(psnr_list) / len(psnr_list),
            'avg_bpp': sum(bpp_list) / len(bpp_list),
            'avg_mse': sum(mse_list) / len(mse_list),
            'num_sequences': len(psnr_list),
            'avg_i_frame_psnr': sum(i_frame_psnr_list) / len(i_frame_psnr_list),
            'avg_i_frame_bpp': sum(i_frame_bpp_list) / len(i_frame_bpp_list),
            'avg_i_frame_mse': sum(i_frame_mse_list) / len(i_frame_mse_list),
            'avg_p_frame_psnr': sum(p_frame_psnr_list) / len(p_frame_psnr_list) if p_frame_psnr_list else 0,
            'avg_p_frame_bpp': sum(p_frame_bpp_list) / len(p_frame_bpp_list) if p_frame_bpp_list else 0,
            'avg_p_frame_mse': sum(p_frame_mse_list) / len(p_frame_mse_list) if p_frame_mse_list else 0,
            'detailed_results': results  # Store per-sequence detailed results
        }

        if accelerator is None or accelerator.is_main_process:
            print(f"\n=== OVERALL RESULTS ===")
            print(f"All Frames - Avg BPP={overall_results['avg_bpp']:.6f}, "
                  f"Avg PSNR={overall_results['avg_psnr']:.2f} dB, "
                  f"Avg MSE={overall_results['avg_mse']:.6f} "
                  f"({overall_results['num_sequences']} sequences)")
            print(f"\nI-frame avg: BPP={overall_results['avg_i_frame_bpp']:.6f}, "
                  f"PSNR={overall_results['avg_i_frame_psnr']:.2f} dB, "
                  f"MSE={overall_results['avg_i_frame_mse']:.6f}")
            print(f"P-frame avg: BPP={overall_results['avg_p_frame_bpp']:.6f}, "
                  f"PSNR={overall_results['avg_p_frame_psnr']:.2f} dB, "
                  f"MSE={overall_results['avg_p_frame_mse']:.6f}")

            # Print first 3 P-frame statistics (averaged across all sequences)
            print(f"\n=== FIRST 3 P-FRAMES DETAILED STATISTICS ===")
            for p_idx in range(1, 4):
                bpp_values = []
                psnr_values = []
                mse_values = []
                for seq_name, seq_results in results.items():
                    for p_frame_info in seq_results['first_three_p_frames']:
                        if p_frame_info['frame_idx'] == p_idx:
                            bpp_values.append(p_frame_info['bpp'])
                            psnr_values.append(p_frame_info['psnr'])
                            mse_values.append(p_frame_info['mse'])

                if bpp_values:
                    avg_bpp = sum(bpp_values) / len(bpp_values)
                    avg_psnr = sum(psnr_values) / len(psnr_values)
                    avg_mse = sum(mse_values) / len(mse_values)
                    print(f"P-frame {p_idx}: BPP={avg_bpp:.6f}, PSNR={avg_psnr:.2f} dB, MSE={avg_mse:.6f}")

        return overall_results

    return None


def train_one_epoch(model, i_frame_model, train_loader, optimizer, device, stage, epoch,
                   gradient_accumulation_steps=1, use_gop_optimization=False, 
                   grad_clip_max_norm=None, ema=None, accelerator=None, 
                   is_finetuning=False, phase_name="Training"):
    """
    Unified training function for both main training and finetuning.
    """
    model.train()
    i_frame_model.eval()
    
    # Get unwrapped model for accessing methods
    unwrapped_model = accelerator.unwrap_model(model) if accelerator else model

    # Update quantiles
    unwrapped_model.bitEstimator_z._update_quantiles()
    unwrapped_model.bitEstimator_z_mv._update_quantiles()

    # Control parameter freezing based on stage (only for main training, not finetuning)
    if not is_finetuning and stage in [2, 3]:
        for param in unwrapped_model.opticFlow.parameters():
            param.requires_grad = False
        for param in unwrapped_model.mvEncoder.parameters():
            param.requires_grad = False
        for param in unwrapped_model.mvDecoder_part1.parameters():
            param.requires_grad = False
        for param in unwrapped_model.mvDecoder_part2.parameters():
            param.requires_grad = False
        for param in unwrapped_model.mvpriorEncoder.parameters():
            param.requires_grad = False
        for param in unwrapped_model.mvpriorDecoder.parameters():
            param.requires_grad = False
        for param in unwrapped_model.auto_regressive_mv.parameters():
            param.requires_grad = False
        for param in unwrapped_model.entropy_parameters_mv.parameters():
            param.requires_grad = False
        for param in unwrapped_model.bitEstimator_z_mv.parameters():
            param.requires_grad = False
    elif not is_finetuning:
        for param in unwrapped_model.opticFlow.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvEncoder.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvDecoder_part1.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvDecoder_part2.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvpriorEncoder.parameters():
            param.requires_grad = True
        for param in unwrapped_model.mvpriorDecoder.parameters():
            param.requires_grad = True
        for param in unwrapped_model.auto_regressive_mv.parameters():
            param.requires_grad = True
        for param in unwrapped_model.entropy_parameters_mv.parameters():
            param.requires_grad = True
        for param in unwrapped_model.bitEstimator_z_mv.parameters():
            param.requires_grad = True
    
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
    progress_desc = f"{phase_name} Stage {stage} Epoch {epoch}" if not is_finetuning else f"{phase_name} Epoch {epoch}"
    progress_bar = tqdm(train_loader, desc=progress_desc, 
                       disable=(accelerator and not accelerator.is_main_process))
    
    for batch_idx, gop_batch in enumerate(progress_bar):
        gop_batch = gop_batch.to(device)
        batch_size, gop_size, _, _, _ = gop_batch.shape
        
        if use_gop_optimization:
            # GOP-level optimization (accumulate gradients across GOP)
            optimizer.zero_grad()
            batch_loss = 0
            batch_mse = 0
            batch_bpp = 0
            batch_psnr = 0
            
            reference_frames = None
            
            for frame_pos in range(gop_size):
                current_frames = gop_batch[:, frame_pos, :, :, :].to(device)
                
                if frame_pos == 0:  # First frame (I-frame) in each sequence
                    with torch.no_grad():
                        # Compress I-frame in RGB
                        i_frame_result, _ = compress_i_frame_with_padding(i_frame_model, current_frames)
                        reference_frames = i_frame_result['x_hat'].detach()  # RGB
                else:  # P-frames
                    result = model(reference_frames, current_frames, stage=stage)
                    
                    batch_loss += result["loss"]
                    batch_mse += result["mse_loss"].item() * batch_size
                    batch_bpp += result["bpp_train"].item() * batch_size
                    batch_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
                    
                    # Use reconstructed frame as reference for next P-frame
                    reference_frames = result["recon_image"]
            
            if gop_size > 1:
                batch_loss /= (gop_size - 1)  # Average over P-frames
                batch_loss /= gradient_accumulation_steps

                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    # Stop training
                    raise RuntimeError("Loss is NaN or Inf, stopping training")

                if accelerator:
                    accelerator.backward(batch_loss)
                else:
                    batch_loss.backward()
                
                if grad_clip_max_norm is not None:
                    if accelerator:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                
                optimizer.step()
                
                if ema is not None:
                    ema.update(unwrapped_model.parameters())
                
                total_loss += batch_loss.item() * (gop_size - 1)
                total_mse += batch_mse
                total_bpp += batch_bpp
                total_psnr += batch_psnr
                n_frames += batch_size * (gop_size - 1)

        else:
            # Per-frame optimization with GOP-level structure (detached references)
            reference_frames = None
            
            for frame_pos in range(gop_size):
                current_frames = gop_batch[:, frame_pos, :, :, :].to(device)
                
                if frame_pos == 0:  # First frame (I-frame) - compress and use as reference
                    with torch.no_grad():
                        # Compress I-frame in RGB
                        i_frame_result, _ = compress_i_frame_with_padding(i_frame_model, current_frames)
                        reference_frames = i_frame_result['x_hat'].detach()  # RGB
                else:  # P-frames - optimize individually but maintain GOP structure
                    optimizer.zero_grad()

                    # Process P-frame with current reference (both RGB)
                    result = model(reference_frames, current_frames, stage=stage)
                    loss = result["loss"]

                    if torch.isnan(loss) or torch.isinf(loss):
                        #stop training
                        raise RuntimeError("Loss is NaN or Inf, stopping training")

                    if accelerator:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    
                    if grad_clip_max_norm is not None:
                        if accelerator:
                            accelerator.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                    
                    optimizer.step()
                    
                    if ema is not None:
                        ema.update(unwrapped_model.parameters())
                    
                    # Collect statistics
                    total_loss += result["loss"].item()
                    total_mse += result["mse_loss"].item() * batch_size
                    total_bpp += result["bpp_train"].item() * batch_size
                    total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
                    n_frames += batch_size
                    
                    with torch.no_grad():
                        reference_frames = result["recon_image"].detach()
        
        # Update progress bar
        if n_frames > 0 and (accelerator is None or accelerator.is_main_process):
            postfix_dict = {
                'loss': f"{total_loss / n_frames:.4f}",
                'mse': f"{total_mse / n_frames:.6f}",
                'bpp': f"{total_bpp / n_frames:.4f}",
                'psnr': f"{total_psnr / n_frames:.2f}"
            }
            # Add prefix for finetuning to distinguish metrics
            if is_finetuning:
                postfix_dict = {f"ft_{k}": v for k, v in postfix_dict.items()}
            
            progress_bar.set_postfix(postfix_dict)
    
    # Calculate epoch statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        avg_psnr = total_psnr / n_frames
        avg_bpp = total_bpp / n_frames
    else:
        avg_loss = 0
        avg_mse = 0
        avg_psnr = 0
        avg_bpp = 0
    
    # Update quantiles at the end
    unwrapped_model.bitEstimator_z._update_quantiles()
    unwrapped_model.bitEstimator_z_mv._update_quantiles()

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "bpp": avg_bpp,
    }


def load_checkpoint_for_stage(checkpoint_dir, stage, quality_index, lambda_value, device, logger, accelerator=None):
    """Load checkpoint from previous stage or resume current stage."""
    stage_checkpoint_patterns = [
        f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage}_best.pth',
        f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage}_latest.pth',
        f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage}.pth'
    ]
    
    # Try to load current stage checkpoint first
    for pattern in stage_checkpoint_patterns:
        checkpoint_path = os.path.join(checkpoint_dir, pattern)
        if os.path.exists(checkpoint_path):
            if accelerator is None or accelerator.is_main_process:
                logger.info(f"Loading checkpoint from current stage: {checkpoint_path}")
            return torch.load(checkpoint_path, map_location=device), True
    
    # Try to load from previous stage
    if stage > 1:
        prev_stage_patterns = [
            f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage-1}_best.pth',
            f'model_dcvc_lambda_{lambda_value}_quality_{quality_index}_stage_{stage-1}.pth'
        ]
        
        for pattern in prev_stage_patterns:
            checkpoint_path = os.path.join(checkpoint_dir, pattern)
            if os.path.exists(checkpoint_path):
                if accelerator is None or accelerator.is_main_process:
                    logger.info(f"Loading checkpoint from previous stage: {checkpoint_path}")
                return torch.load(checkpoint_path, map_location=device), False
    
    if accelerator is None or accelerator.is_main_process:
        logger.info("No checkpoint found, starting from scratch")
    return None, False


def setup_training_config(args):
    """Setup training configuration based on stage."""
    is_finetuning = (args.stage == 5)
    
    # Determine parameters based on stage
    if is_finetuning:
        epochs = args.finetune_epochs
        batch_size = args.finetune_batch_size
        gop_size = args.finetune_gop_size
        phase_name = "BVI-AOM Finetune"
        use_gop_optimization = True  # Always use GOP optimization for BVI-AOM
        eval_stage = 4  # Use stage 4 for evaluation during finetuning
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        gop_size = args.gop_size
        phase_name = "Training"
        use_gop_optimization = (args.stage == 4 and args.stage4_gop_opt)
        eval_stage = args.stage
    
    return {
        'is_finetuning': is_finetuning,
        'epochs': epochs,
        'batch_size': batch_size,
        'gop_size': gop_size,
        'phase_name': phase_name,
        'use_gop_optimization': use_gop_optimization,
        'eval_stage': eval_stage
    }


def setup_dataset(args, config):
    """Setup dataset based on stage."""
    train_transform = transforms.ToTensor()  # Always use ToTensor for consistency
    
    if config['is_finetuning']:
        # BVI-AOM dataset for finetuning
        if not args.bvi_aom_dir:
            raise ValueError("For BVI-AOM finetuning (stage 5), --bvi_aom_dir must be provided")
        
        dataset = BVI_AOM_Dataset(
            root_dir=args.bvi_aom_dir,
            gop_size=config['gop_size'],
            crop_size=args.crop_size,
            max_sequences=args.finetune_max_sequences,
            transform=train_transform
        )
        
        if len(dataset) == 0:
            raise RuntimeError("No BVI-AOM sequences found")
            
    else:
        # Main training stages (1-4)
        if args.use_video_dataset:
            # Use MP4 video dataset
            if not args.train_video_dir:
                raise ValueError("For video dataset, --train_video_dir must be provided")
            
            dataset = VideoGOPDataset(
                video_dir=args.train_video_dir,
                gop_size=config['gop_size'],
                transform=train_transform,
                crop_size=args.crop_size
            )
            
            if len(dataset) == 0:
                raise RuntimeError(f"No video files found in {args.train_video_dir}")
        else:
            # Use Vimeo-90k image sequence dataset (traditional way)
            if not args.vimeo_dir or not args.septuplet_list:
                raise ValueError("For image dataset, --vimeo_dir and --septuplet_list must be provided")

            dataset = VimeoGOPDataset(
                root_dir=args.vimeo_dir,
                septuplet_list_path=args.septuplet_list,
                gop_size=config['gop_size'],
                transform=train_transform,
                crop_size=args.crop_size
            )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return dataset, data_loader


def main():
    parser = argparse.ArgumentParser(description='DCVC Training with Four Stages + BVI-AOM Finetuning')
    
    # Dataset arguments
    parser.add_argument('--vimeo_dir', type=str, help='Path to Vimeo-90k "sequences" directory')
    parser.add_argument('--septuplet_list', type=str, help='Path to "sep_trainlist.txt" for Vimeo-90k')
    parser.add_argument('--hevc_b_dir', type=str, help='Path to HEVC-B dataset directory for evaluation')
    parser.add_argument('--bvi_aom_dir', type=str, help='Path to BVI-AOM dataset for finetuning')
    
    # Video dataset arguments (new)
    parser.add_argument('--train_video_dir', type=str, default=None,
                        help='Path to training video directory (MP4 files)')
    parser.add_argument('--val_video_dir', type=str, default=None,
                        help='Path to validation video directory (MP4 files)')
    parser.add_argument('--use_video_dataset', action='store_true',
                        help='Use MP4 video dataset instead of image sequences')
    parser.add_argument('--i_frame_model_name', type=str, default='cheng2020-anchor', 
                        help='CompressAI I-frame model name (e.g., cheng2020-anchor, bmshj2018-factorized, etc.)')
    parser.add_argument('--i_frame_quality', type=int, default=6, 
                        help='Quality level for CompressAI pretrained model (1-8, higher = better quality)')
    parser.add_argument('--i_frame_pretrained', action='store_true', default=True,
                        help='Use CompressAI pretrained weights for I-frame model')
    
    
    # Training arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_dcvc', help='Directory to save checkpoints')
    parser.add_argument('--lambda_value', type=float, required=True, help='Lambda value for rate-distortion trade-off')
    parser.add_argument('--quality_index', type=int, required=True, help='Quality index (1-4) for the model name')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4, 5], 
                        help='Training stage (1-4 for main training, 5 for BVI-AOM finetuning)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for this stage')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--force_learning_rate', type=float, default=None, help='Force Learning rate')
    parser.add_argument('--crop_size', type=int, default=256, help='Random crop size for training patches')
    parser.add_argument('--gop_size', type=int, default=7, help='Number of frames in a sequence for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--stage4_gop_opt', action='store_true', help='Whether to use GOP optimization in stage 4')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Stage 5 (BVI-AOM finetuning) specific arguments
    parser.add_argument('--finetune_epochs', type=int, default=10, help='Number of epochs for BVI-AOM finetuning')
    parser.add_argument('--finetune_gop_size', type=int, default=32, help='GOP size for BVI-AOM finetuning')
    parser.add_argument('--finetune_batch_size', type=int, default=2, help='Batch size for BVI-AOM finetuning')
    parser.add_argument('--finetune_max_sequences', type=int, default=None, help='Max BVI-AOM sequences')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='Learning rate for BVI-AOM finetuning')
    
    # Resume and loading arguments
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--load_pretrained', type=str, default=None, help='Path to load pretrained weights only')
    
    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=1, help='HEVC-B evaluation frequency')
    parser.add_argument('--eval_max_frames', type=int, default=96, help='Maximum frames per HEVC-B sequence')
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation during training')
    
    # Learning rate scheduler
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='LR reduction factor')
    parser.add_argument('--scheduler_patience', type=int, default=3, help='Scheduler patience')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # EMA arguments
    parser.add_argument('--use_ema', action='store_true',default=True, help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--evaluate_both', action='store_true', default=True, help='Evaluate both normal and EMA weights')

    # Training options
    parser.add_argument('--grad_clip_max_norm', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    
    # Accelerate arguments
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'],
                       help='Whether to use mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    device = accelerator.device
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create checkpoint directory (only on main process)
    if accelerator.is_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.checkpoint_dir, accelerator)
    
    # Setup MPS fallback if needed (after logger is initialized, before any model operations)
    device = setup_device_for_mps_fallback(device, logger if accelerator.is_main_process else None)
    
    # Setup training configuration
    config = setup_training_config(args)
    
    # Stage descriptions
    stage_descriptions = {
        1: "Warm up MV generation part",
        2: "Train other modules",
        3: "Train with bit cost", 
        4: "End-to-end training",
        5: "BVI-AOM finetuning"
    }
    
    if accelerator.is_main_process:
        logger.info("=" * 80)
        logger.info(f"DCVC TRAINING - STAGE {args.stage}: {stage_descriptions[args.stage]}")
        logger.info("=" * 80)
        logger.info(f"Lambda value: {args.lambda_value}")
        logger.info(f"Quality index: {args.quality_index}")
        logger.info(f"Device: {device}")
        logger.info(f"Num processes: {accelerator.num_processes}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Epochs: {config['epochs']}")
        logger.info(f"Batch size: {config['batch_size']}")
        logger.info(f"GOP size: {config['gop_size']}")
        logger.info(f"GOP optimization: {config['use_gop_optimization']}")
        logger.info("=" * 80)

    # Load CompressAI I-frame model
    if accelerator.is_main_process:
        logger.info(f"Loading CompressAI I-frame model: {args.i_frame_model_name}")
        logger.info(f"Quality level: {args.i_frame_quality}")
        logger.info(f"Using pretrained weights: {args.i_frame_pretrained}")
    
    try:
        # Load CompressAI pretrained model
        i_frame_model = compressai_models[args.i_frame_model_name](
            quality=args.i_frame_quality,
            pretrained=args.i_frame_pretrained
        ).to(device)
        
        # Set to evaluation mode
        i_frame_model.eval()
        
        if accelerator.is_main_process:
            logger.info(f"Successfully loaded CompressAI model: {args.i_frame_model_name}")
            
    except KeyError:
        available_models = list(compressai_models.keys())
        error_msg = f"Model '{args.i_frame_model_name}' not found in CompressAI zoo. Available models: {available_models}"
        if accelerator.is_main_process:
            logger.error(error_msg)
        raise ValueError(error_msg)
        
    except Exception as e:
        error_msg = f"Error loading CompressAI model: {e}"
        if accelerator.is_main_process:
            logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Compile I-frame model if enabled
    if args.compile:
        i_frame_model = torch.compile(i_frame_model)
        if accelerator.is_main_process:
            logger.info("I-frame model compiled for faster inference")

    # Initialize DCVC model
    model = DCVC_net(lmbda=args.lambda_value).to(device)
    
    # Compile DCVC model if enabled
    if args.compile:
        model = torch.compile(model)
        if accelerator.is_main_process:
            logger.info("DCVC model compiled for faster training")

    # Check HEVC-B availability
    hevc_b_available = args.hevc_b_dir and os.path.exists(args.hevc_b_dir) and not args.skip_eval
    if hevc_b_available and accelerator.is_main_process:
        logger.info(f"HEVC-B evaluation enabled. Will evaluate every {args.eval_freq} epochs.")

    # Load checkpoint handling for stage 5 (finetuning)
    if config['is_finetuning']:
        # Check for explicit resume path
        if args.resume:
            if accelerator.is_main_process:
                logger.info(f"Resuming stage 5 training from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            unwrapped_model = model
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            unwrapped_model.load_state_dict(state_dict, strict=True)
            if accelerator.is_main_process:
                logger.info("Loaded checkpoint for resuming BVI-AOM finetuning")

        # Check for explicit load_pretrained path
        elif args.load_pretrained:
            if accelerator.is_main_process:
                logger.info(f"Loading pretrained weights for stage 5: {args.load_pretrained}")
            checkpoint = torch.load(args.load_pretrained, map_location=device)
            unwrapped_model = model
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            unwrapped_model.load_state_dict(state_dict, strict=True)
            if accelerator.is_main_process:
                logger.info("Loaded pretrained weights for BVI-AOM finetuning")

        # Neither resume nor load_pretrained provided - use autofind
        else:
            # Try to find existing stage 5 checkpoint first
            stage5_checkpoint, is_resuming_stage5 = load_checkpoint_for_stage(
                args.checkpoint_dir, 5, args.quality_index, args.lambda_value, device, logger, accelerator
            )

            if is_resuming_stage5 and stage5_checkpoint:
                # Found existing stage 5 checkpoint - resume from it
                unwrapped_model = model
                state_dict = stage5_checkpoint.get('model_state_dict', stage5_checkpoint)
                unwrapped_model.load_state_dict(state_dict, strict=True)
                if accelerator.is_main_process:
                    logger.info("Autofind: Resuming BVI-AOM finetuning from existing stage 5 checkpoint")
            else:
                # No stage 5 checkpoint found - try to load from stage 4
                stage4_checkpoint, _ = load_checkpoint_for_stage(
                    args.checkpoint_dir, 4, args.quality_index, args.lambda_value, device, logger, accelerator
                )

                if stage4_checkpoint:
                    unwrapped_model = model
                    state_dict = stage4_checkpoint.get('model_state_dict', stage4_checkpoint)
                    unwrapped_model.load_state_dict(state_dict, strict=True)
                    if accelerator.is_main_process:
                        logger.info("Autofind: Starting BVI-AOM finetuning from stage 4 checkpoint")
                else:
                    raise RuntimeError("Autofind failed: No stage 4 checkpoint found for BVI-AOM finetuning")

    # Setup dataset and data loader
    dataset, train_loader = setup_dataset(args, config)
    
    if accelerator.is_main_process:
        if config['is_finetuning']:
            dataset_name = "BVI-AOM"
        elif args.use_video_dataset:
            dataset_name = "Video (MP4)"
        else:
            dataset_name = "Vimeo-90k"
        logger.info(f"{dataset_name} dataset size: {len(dataset)} samples")
        logger.info(f"Training batches per epoch: {len(train_loader)}")

    # Initialize optimizer
    optimizer = configure_optimizers(model, args, is_finetuning=config['is_finetuning'])

    # Prepare with Accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    # Initialize EMA
    ema = None
    if args.use_ema:
        if accelerator.is_main_process:
            logger.info(f"Using EMA with decay rate: {args.ema_decay}")
        unwrapped_model = accelerator.unwrap_model(model)
        ema = ExponentialMovingAverage(unwrapped_model.parameters(), decay=args.ema_decay)
        ema.to(device)
    
    # Setup scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )
        if accelerator.is_main_process:
            logger.info("ReduceLROnPlateau scheduler initialized")

    # Initialize tracking variables
    start_epoch = 0
    best_loss = float('inf')
    best_loss_ema = float('inf')
    global_best_loss = float('inf')
    
    # Handle checkpoint restoration for stages 1-4
    if not config['is_finetuning']:
        # Check for explicit resume path
        if args.resume:
            if accelerator.is_main_process:
                logger.info(f"Resuming training from: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_loss_ema = checkpoint.get('best_loss_ema', float('inf'))
            global_best_loss = checkpoint.get('global_best_loss', float('inf'))

            if args.use_ema and 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])

            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Check for explicit load_pretrained path
        elif args.load_pretrained:
            if accelerator.is_main_process:
                logger.info(f"Loading pretrained weights: {args.load_pretrained}")
            checkpoint = torch.load(args.load_pretrained, map_location=device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(state_dict, strict=True)

        # Neither resume nor load_pretrained provided - start from scratch
        else:
            if accelerator.is_main_process:
                logger.info(f"Starting stage {args.stage} training from scratch")

    # Handle checkpoint restoration for stage 5 (finetuning)
    else:
        # Only restore training state if resuming (not for load_pretrained)
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_loss_ema = checkpoint.get('best_loss_ema', float('inf'))
            global_best_loss = checkpoint.get('global_best_loss', float('inf'))

            if args.use_ema and 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])

            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if accelerator.is_main_process:
                logger.info(f"Resumed from epoch {start_epoch}")

    if args.force_learning_rate is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.force_learning_rate
            if accelerator.is_main_process:
                logger.info(f"Forced learning rate to {args.force_learning_rate:.6f}")

        # Reset scheduler when forcing learning rate
        if scheduler is not None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=args.scheduler_factor,
                patience=args.scheduler_patience,
                min_lr=args.scheduler_min_lr,
            )
            if accelerator.is_main_process:
                logger.info("Scheduler reset due to forced learning rate")

    # ==================== UNIFIED TRAINING LOOP ====================
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        if accelerator.is_main_process:
            logger.info("")
            logger.info("=" * 100)
            logger.info(f"  {config['phase_name']} Stage {args.stage} - Epoch {epoch + 1}/{config['epochs']}")
            logger.info("=" * 100)
            logger.info(f"  Learning Rate: {current_lr:.6e}")
            logger.info("-" * 100)

        # Train one epoch
        train_stats = train_one_epoch(
            model, i_frame_model, train_loader, optimizer, device, 
            args.stage, epoch + 1, args.gradient_accumulation_steps, 
            config['use_gop_optimization'], args.grad_clip_max_norm, ema, accelerator,
            config['is_finetuning'], config['phase_name']
        )
        
        epoch_duration = time.time() - epoch_start_time

        if accelerator.is_main_process:
            logger.info("")
            logger.info("-" * 100)
            logger.info(f"  TRAINING RESULTS - Epoch {epoch + 1} (Duration: {epoch_duration:.2f}s)")
            logger.info("-" * 100)
            logger.info(f"  Train Loss:  {train_stats['loss']:.6f}")
            logger.info(f"  Train PSNR:  {train_stats['psnr']:.2f} dB")
            logger.info(f"  Train BPP:   {train_stats['bpp']:.6f}")
            logger.info(f"  Train MSE:   {train_stats['mse']:.6f}")

        # Evaluation
        hevc_b_results = None
        hevc_b_results_ema = None

        if (epoch + 1) % args.eval_freq == 0 and hevc_b_available:
            if accelerator.is_main_process:
                logger.info("")
                logger.info("=" * 100)
                logger.info(f"  HEVC-B EVALUATION - Epoch {epoch + 1}")
                logger.info("=" * 100)

            # HEVC-B evaluation
            hevc_b_results = evaluate_hevc_b(
                model, i_frame_model, args.hevc_b_dir, device,
                config['eval_stage'], args.eval_max_frames, accelerator=accelerator,
                use_video_dataset=args.use_video_dataset
            )

            # EMA evaluation on HEVC-B
            if args.use_ema and args.evaluate_both:
                unwrapped_model = accelerator.unwrap_model(model)
                ema.store(unwrapped_model.parameters())
                ema.copy_to(unwrapped_model.parameters())

                if accelerator.is_main_process:
                    logger.info("")
                    logger.info("-" * 100)
                    logger.info("  HEVC-B EVALUATION (EMA)")
                    logger.info("-" * 100)

                hevc_b_results_ema = evaluate_hevc_b(
                    model, i_frame_model, args.hevc_b_dir, device,
                    config['eval_stage'], args.eval_max_frames, accelerator=accelerator,
                    use_video_dataset=args.use_video_dataset
                )

                ema.restore(unwrapped_model.parameters())

            if hevc_b_results and accelerator.is_main_process:
                rd_loss = hevc_b_results['avg_bpp'] + args.lambda_value * hevc_b_results['avg_mse']
                logger.info("")
                logger.info("-" * 100)
                logger.info("  TEST RESULTS (Regular Model)")
                logger.info("-" * 100)
                logger.info(f"  Overall:")
                logger.info(f"    PSNR:     {hevc_b_results['avg_psnr']:.4f} dB")
                logger.info(f"    BPP:      {hevc_b_results['avg_bpp']:.6f}")
                logger.info(f"    MSE:      {hevc_b_results['avg_mse']:.6f}")
                logger.info(f"    RD Loss:  {rd_loss:.6f} (lambda={args.lambda_value})")
                logger.info(f"  Sequences: {hevc_b_results['num_sequences']}")
                logger.info("")
                logger.info(f"  I-frame:")
                logger.info(f"    PSNR:     {hevc_b_results['avg_i_frame_psnr']:.4f} dB")
                logger.info(f"    BPP:      {hevc_b_results['avg_i_frame_bpp']:.6f}")
                logger.info(f"    MSE:      {hevc_b_results['avg_i_frame_mse']:.6f}")
                logger.info("")
                logger.info(f"  P-frame (avg):")
                logger.info(f"    PSNR:     {hevc_b_results['avg_p_frame_psnr']:.4f} dB")
                logger.info(f"    BPP:      {hevc_b_results['avg_p_frame_bpp']:.6f}")
                logger.info(f"    MSE:      {hevc_b_results['avg_p_frame_mse']:.6f}")

                # Log first 3 P-frame details
                logger.info("")
                logger.info(f"  First 3 P-frames (detailed):")
                detailed_results = hevc_b_results.get('detailed_results', {})
                for p_idx in range(1, 4):
                    bpp_values = []
                    psnr_values = []
                    mse_values = []
                    for seq_name, seq_results in detailed_results.items():
                        for p_frame_info in seq_results['first_three_p_frames']:
                            if p_frame_info['frame_idx'] == p_idx:
                                bpp_values.append(p_frame_info['bpp'])
                                psnr_values.append(p_frame_info['psnr'])
                                mse_values.append(p_frame_info['mse'])

                    if bpp_values:
                        avg_bpp = sum(bpp_values) / len(bpp_values)
                        avg_psnr = sum(psnr_values) / len(psnr_values)
                        avg_mse = sum(mse_values) / len(mse_values)
                        logger.info(f"    P-frame {p_idx}: BPP={avg_bpp:.6f}, PSNR={avg_psnr:.2f} dB, MSE={avg_mse:.6f}")

                if hevc_b_results_ema:
                    rd_loss_ema = hevc_b_results_ema['avg_bpp'] + args.lambda_value * hevc_b_results_ema['avg_mse']
                    logger.info("")
                    logger.info("-" * 100)
                    logger.info("  TEST RESULTS (EMA Model)")
                    logger.info("-" * 100)
                    logger.info(f"  Overall:")
                    logger.info(f"    PSNR:     {hevc_b_results_ema['avg_psnr']:.4f} dB")
                    logger.info(f"    BPP:      {hevc_b_results_ema['avg_bpp']:.6f}")
                    logger.info(f"    MSE:      {hevc_b_results_ema['avg_mse']:.6f}")
                    logger.info(f"    RD Loss:  {rd_loss_ema:.6f} (lambda={args.lambda_value})")
                    logger.info(f"  Sequences: {hevc_b_results_ema['num_sequences']}")
                    logger.info("")
                    logger.info(f"  I-frame:")
                    logger.info(f"    PSNR:     {hevc_b_results_ema['avg_i_frame_psnr']:.4f} dB")
                    logger.info(f"    BPP:      {hevc_b_results_ema['avg_i_frame_bpp']:.6f}")
                    logger.info(f"    MSE:      {hevc_b_results_ema['avg_i_frame_mse']:.6f}")
                    logger.info("")
                    logger.info(f"  P-frame (avg):")
                    logger.info(f"    PSNR:     {hevc_b_results_ema['avg_p_frame_psnr']:.4f} dB")
                    logger.info(f"    BPP:      {hevc_b_results_ema['avg_p_frame_bpp']:.6f}")
                    logger.info(f"    MSE:      {hevc_b_results_ema['avg_p_frame_mse']:.6f}")

                    # Log first 3 P-frame details for EMA
                    logger.info("")
                    logger.info(f"  First 3 P-frames (detailed):")
                    detailed_results_ema = hevc_b_results_ema.get('detailed_results', {})
                    for p_idx in range(1, 4):
                        bpp_values = []
                        psnr_values = []
                        mse_values = []
                        for seq_name, seq_results in detailed_results_ema.items():
                            for p_frame_info in seq_results['first_three_p_frames']:
                                if p_frame_info['frame_idx'] == p_idx:
                                    bpp_values.append(p_frame_info['bpp'])
                                    psnr_values.append(p_frame_info['psnr'])
                                    mse_values.append(p_frame_info['mse'])

                        if bpp_values:
                            avg_bpp = sum(bpp_values) / len(bpp_values)
                            avg_psnr = sum(psnr_values) / len(psnr_values)
                            avg_mse = sum(mse_values) / len(mse_values)
                            logger.info(f"    P-frame {p_idx}: BPP={avg_bpp:.6f}, PSNR={avg_psnr:.2f} dB, MSE={avg_mse:.6f}")

        # Determine current test loss for best model tracking
        current_test_loss = train_stats['loss']  # Default to training loss
        current_test_loss_ema = None
        
        if hevc_b_results:
            current_test_loss = hevc_b_results['avg_bpp'] + args.lambda_value * hevc_b_results['avg_mse']
        if hevc_b_results_ema:
            current_test_loss_ema = hevc_b_results_ema['avg_bpp'] + args.lambda_value * hevc_b_results_ema['avg_mse']

        # Update scheduler
        if scheduler:
            scheduler.step(current_test_loss)

        # Check for best model
        is_best = current_test_loss < best_loss
        is_best_ema = False
        if is_best:
            best_loss = current_test_loss
        
        if args.use_ema and current_test_loss_ema is not None:
            if current_test_loss_ema < best_loss_ema:
                best_loss_ema = current_test_loss_ema
                is_best_ema = True

        # Check for global best
        current_best_loss = min(current_test_loss, current_test_loss_ema or float('inf'))
        is_global_best = current_best_loss < global_best_loss
        is_ema_global_best = (current_test_loss_ema is not None and 
                             current_test_loss_ema == current_best_loss and 
                             is_global_best)
        
        if is_global_best:
            global_best_loss = current_best_loss

        # Save checkpoints (only on main process)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_stats['loss'],
                'best_loss': best_loss,
                'global_best_loss': global_best_loss,
                'stage': args.stage,
                'quality_index': args.quality_index,
                'lambda_value': args.lambda_value,
                'is_finetuning': config['is_finetuning']
            }
            
            if args.use_ema:
                checkpoint_data['ema_state_dict'] = ema.state_dict()
                checkpoint_data['best_loss_ema'] = best_loss_ema
            
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            # Save latest checkpoint
            latest_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_latest.pth'
            )
            torch.save(checkpoint_data, latest_path)
            
            # Save best checkpoint
            if is_best or is_global_best:
                best_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best.pth'
                )
                torch.save(checkpoint_data, best_path)
                logger.info("")
                logger.info("=" * 100)
                logger.info(f"  *** NEW BEST MODEL (Regular) - Epoch {epoch + 1} ***")
                logger.info("=" * 100)
                logger.info(f"  Best Loss:  {best_loss:.6f}")
                if hevc_b_results:
                    logger.info(f"  PSNR:       {hevc_b_results['avg_psnr']:.4f} dB")
                    logger.info(f"  BPP:        {hevc_b_results['avg_bpp']:.6f}")
                logger.info(f"  Saved to:   {os.path.basename(best_path)}")
                logger.info("=" * 100)

            # Save best EMA model
            if args.use_ema and (is_best_ema or is_ema_global_best):
                ema.store(unwrapped_model.parameters())
                ema.copy_to(unwrapped_model.parameters())

                ema_checkpoint_data = checkpoint_data.copy()
                ema_checkpoint_data['model_state_dict'] = unwrapped_model.state_dict()

                ema_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best_ema.pth'
                )
                torch.save(ema_checkpoint_data, ema_path)

                ema.restore(unwrapped_model.parameters())
                logger.info("")
                logger.info("=" * 100)
                logger.info(f"  *** NEW BEST MODEL (EMA) - Epoch {epoch + 1} ***")
                logger.info("=" * 100)
                logger.info(f"  Best Loss:  {best_loss_ema:.6f}")
                if hevc_b_results_ema:
                    logger.info(f"  PSNR:       {hevc_b_results_ema['avg_psnr']:.4f} dB")
                    logger.info(f"  BPP:        {hevc_b_results_ema['avg_bpp']:.6f}")
                logger.info(f"  Saved to:   {os.path.basename(ema_path)}")
                logger.info("=" * 100)

            # Save global best
            if is_global_best:
                if is_ema_global_best:
                    ema.store(unwrapped_model.parameters())
                    ema.copy_to(unwrapped_model.parameters())

                global_checkpoint_data = checkpoint_data.copy()
                global_checkpoint_data['model_state_dict'] = unwrapped_model.state_dict()
                global_checkpoint_data['is_ema'] = is_ema_global_best

                global_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_global_best.pth'
                )
                torch.save(global_checkpoint_data, global_path)

                global_state_dict_path = os.path.join(
                    args.checkpoint_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_global_best_state_dict.pth'
                )
                torch.save(unwrapped_model.state_dict(), global_state_dict_path)

                if is_ema_global_best:
                    ema.restore(unwrapped_model.parameters())

                model_type = "EMA" if is_ema_global_best else "Regular"
                logger.info("")
                logger.info("*" * 100)
                logger.info(f"  *** NEW GLOBAL BEST MODEL ({model_type}) - Epoch {epoch + 1} ***")
                logger.info("*" * 100)
                logger.info(f"  Global Best Loss:  {global_best_loss:.6f}")
                if is_ema_global_best and hevc_b_results_ema:
                    logger.info(f"  PSNR:              {hevc_b_results_ema['avg_psnr']:.4f} dB")
                    logger.info(f"  BPP:               {hevc_b_results_ema['avg_bpp']:.6f}")
                elif hevc_b_results:
                    logger.info(f"  PSNR:              {hevc_b_results['avg_psnr']:.4f} dB")
                    logger.info(f"  BPP:               {hevc_b_results['avg_bpp']:.6f}")
                logger.info(f"  Saved to:          {os.path.basename(global_path)}")
                logger.info("*" * 100)

        # End of epoch separator
        if accelerator.is_main_process:
            logger.info("")
            logger.info("=" * 100)
            logger.info(f"  End of Epoch {epoch + 1}/{config['epochs']}")
            logger.info("=" * 100)
            logger.info("")

    # Save final model
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Save with stage-specific naming
        final_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_final.pth'
        )
        torch.save(unwrapped_model.state_dict(), final_path)
        logger.info(f"Final model for stage {args.stage} saved to {final_path}")
        
        # Save with standard naming based on stage
        if args.stage == 4:
            standard_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_quality_{args.quality_index}_psnr.pth'
            )
            torch.save(unwrapped_model.state_dict(), standard_path)
            logger.info(f"Final model with standard naming saved to {standard_path}")
        elif config['is_finetuning']:
            standard_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_quality_{args.quality_index}_psnr_finetuned.pth'
            )
            torch.save(unwrapped_model.state_dict(), standard_path)
            logger.info(f"Final finetuned model with standard naming saved to {standard_path}")

    accelerator.wait_for_everyone()

    # Clean up accelerator
    accelerator.end_training()

    if accelerator.is_main_process:
        phase_name = "BVI-AOM finetuning" if config['is_finetuning'] else f"Stage {args.stage} training"

        logger.info("")
        logger.info("=" * 100)
        logger.info(f"  TRAINING COMPLETED - {phase_name.upper()}")
        logger.info("=" * 100)
        logger.info(f"  Total Epochs:  {config['epochs']}")
        logger.info(f"  Stage:         {args.stage}")
        logger.info(f"  Lambda:        {args.lambda_value}")
        logger.info(f"  Quality Index: {args.quality_index}")
        logger.info("")
        logger.info("  BEST RESULTS:")
        logger.info("  " + "-" * 96)
        logger.info(f"  Best Loss (Regular):  {best_loss:.6f}")
        if args.use_ema:
            logger.info(f"  Best Loss (EMA):      {best_loss_ema:.6f}")
        logger.info(f"  Global Best Loss:     {global_best_loss:.6f}")
        logger.info("=" * 100)
        logger.info("")


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()
