#!/usr/bin/env python3
"""
Standalone HEVC-B Evaluation Script for DCVC Models

This script evaluates DCVC P-frame and CompressAI I-frame models on the HEVC-B dataset,
providing results in both estimation mode (training forward pass) and real compression mode.
DCVC is a single-rate model, so only one lambda/quality level is evaluated per run.

Usage:
    python evaluate_dcvc_hevc_b.py --p_frame_model_path <path> --hevc_b_dir <path> [options]
"""

import os
import argparse
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm
import logging
import time
import warnings

# Import model classes and utilities
from src.models.DCVC_net_compressai import DCVC_net

# Import CompressAI models
from compressai.zoo import models as compressai_models

# Import dataset class
from dataset import HEVCB_Dataset


def setup_logging(log_file=None):
    """Setup logging configuration"""
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    handlers = [logging.StreamHandler()]  # Always log to console
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    return logger


def load_model_checkpoint(model, checkpoint_path, device, logger):
    """
    Load model weights from checkpoint, handling various checkpoint formats.
    Returns the lambda value from checkpoint if available.
    """
    logger.info(f"Loading model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        logger.info("Loaded model_state_dict from checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        logger.info("Loaded state_dict from checkpoint")
    else:
        # Assume the entire checkpoint is the state dict
        state_dict = checkpoint
        logger.info("Loaded state dict directly from checkpoint")
    
    # Remove "module." prefix from keys if present (for models saved with DataParallel)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove "module." (7 characters)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict, strict=True)
    logger.info(f"Model loaded successfully with {len(new_state_dict)} parameters")
    
    # Extract lambda value from checkpoint if available
    checkpoint_lambda = None
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'stage' in checkpoint:
            logger.info(f"Checkpoint stage: {checkpoint['stage']}")
        if 'lambda_value' in checkpoint:
            checkpoint_lambda = checkpoint['lambda_value']
            logger.info(f"Checkpoint lambda: {checkpoint_lambda}")
        if 'quality_index' in checkpoint:
            logger.info(f"Checkpoint quality index: {checkpoint['quality_index']}")
    
    return checkpoint_lambda


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


def calculate_real_compression_bpp(p_frame_model, i_frame_model, ref_frame, current_frame, logger):
    """
    Calculate BPP using real compression/decompression.
    """
    try:
        # Suppress warnings for GPU inference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Compress P-frame
            compressed_data = p_frame_model.compress(ref_frame, current_frame)
            
            # Calculate actual bits
            total_bits = 0
            
            # Count bits from all strings
            if 'strings' in compressed_data:
                strings = compressed_data['strings']
                # strings = [y_strings, z_strings, mv_y_strings, z_mv_strings]
                for string_data in strings:
                    if isinstance(string_data, list):
                        for s in string_data:
                            total_bits += len(s) * 8
                    else:
                        total_bits += len(string_data) * 8
            
            # Calculate BPP
            B, C, H, W = current_frame.shape
            num_pixels = B * H * W
            bpp = total_bits / num_pixels
            
            # Decompress to get reconstructed frame
            recon_data = p_frame_model.decompress(ref_frame, compressed_data)
            recon_frame = recon_data['recon_image']
            
            return bpp, recon_frame
            
    except Exception as e:
        logger.warning(f"Real compression failed: {e}. Falling back to estimation.")
        return None, None


def calculate_rgb_metrics(rgb_ref, rgb_target):
    """
    Calculate MSE and PSNR between two RGB tensors.

    Args:
        rgb_ref: Reference RGB tensor (B, 3, H, W)
        rgb_target: Target RGB tensor (B, 3, H, W)

    Returns:
        dict: {
            'mse': float,
            'psnr': float
        }
    """
    # Calculate MSE
    mse = F.mse_loss(rgb_ref, rgb_target, reduction='mean').item()

    # Calculate PSNR (assuming input range [0, 1])
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')

    return {
        'mse': mse,
        'psnr': psnr
    }


def evaluate_hevc_b(p_frame_model, i_frame_model, hevc_b_dir, device, 
                   lambda_value, max_frames=96, intra_period=-1, 
                   use_real_compression=False, stage=4, logger=None):
    """
    Evaluate DCVC model on HEVC-B dataset.
    
    Args:
        p_frame_model: DCVC P-frame model
        i_frame_model: CompressAI I-frame model  
        hevc_b_dir: Path to HEVC-B dataset
        device: Device to run evaluation on
        lambda_value: Lambda value for rate-distortion trade-off
        max_frames: Maximum frames per sequence
        intra_period: Period for inserting I-frames (default: -1 for no periodic I-frames)
        use_real_compression: Whether to use real compression or estimation
        stage: DCVC training stage to use (default: 4)
        logger: Logger instance
        
    Returns:
        dict: Evaluation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    p_frame_model.eval()
    i_frame_model.eval()

    eval_dataset = HEVCB_Dataset(root_dir=hevc_b_dir, max_frames=max_frames)
    
    if len(eval_dataset) == 0:
        logger.warning(f"No HEVC-B sequences found in {hevc_b_dir}")
        return None
    
    compression_mode = "Real Compression" if use_real_compression else "Estimation"
    
    # Log evaluation configuration
    if intra_period > 0:
        logger.info(f"Evaluating on {len(eval_dataset)} HEVC-B sequences using {compression_mode}")
        logger.info(f"I-frames will be inserted at frames: 0, {intra_period}, {intra_period*2}, {intra_period*3}, ...")
    else:
        logger.info(f"Evaluating on {len(eval_dataset)} HEVC-B sequences using {compression_mode}")
        logger.info("Only frame 0 will be an I-frame, all others will be P-frames")
    
    logger.info(f"Lambda value: {lambda_value}")
    
    # Update model lambda
    if hasattr(p_frame_model, 'lmbda'):
        p_frame_model.lmbda = lambda_value
        logger.info(f"Updated model lambda to {lambda_value}")
    
    sequence_results = {}
    
    # Overall metrics across all sequences
    total_sequences = 0
    total_frames = 0
    total_bpp = 0
    total_mse = 0
    total_psnr = 0
    
    # Separate I-frame and P-frame tracking
    total_i_frame_bpp = 0
    total_i_frame_mse = 0
    total_i_frame_psnr = 0
    total_i_frame_count = 0
    
    total_p_frame_bpp = 0
    total_p_frame_bpp_breakdown = {
        'bpp_y': 0, 'bpp_z': 0, 'bpp_mv_y': 0, 'bpp_mv_z': 0
    }
    total_p_frame_count = 0
    
    with torch.no_grad():
        for seq_idx in range(len(eval_dataset)):
            sequence_data = eval_dataset[seq_idx]
            if sequence_data is None:
                continue
                
            frames = sequence_data['frames'].unsqueeze(0).to(device)  # Add batch dimension
            seq_name = sequence_data['name']
            num_frames = sequence_data['num_frames']
            
            logger.info(f"\n=== Sequence: {seq_name} ({num_frames} frames) ===")
            
            # Initialize metrics for this sequence
            seq_total_bpp = 0
            seq_total_mse = 0
            seq_total_psnr = 0
            
            # Initialize separate tracking for detailed metrics
            seq_p_frame_bpp = 0
            seq_p_frame_bpp_breakdown = {
                'bpp_y': 0, 'bpp_z': 0, 'bpp_mv_y': 0, 'bpp_mv_z': 0
            }
            seq_i_frame_bpp = 0
            seq_i_frame_mse = 0
            seq_i_frame_psnr = 0
            seq_p_frame_count = 0
            seq_i_frame_count = 0
            
            # Initialize reference frame
            ref_frame = None
            
            # Process all frames
            for frame_idx in range(num_frames):
                current_frame = frames[0, frame_idx, ...].unsqueeze(0)  # [1, C, H, W]
                
                # Determine if this frame should be an I-frame
                is_i_frame = (frame_idx == 0) or (intra_period > 0 and frame_idx % intra_period == 0)
                
                if is_i_frame:
                    # Process as I-frame
                    i_frame_result, i_frame_bpp = compress_i_frame_with_padding(
                        i_frame_model, current_frame, calculate_bpp=True
                    )
                    ref_frame = i_frame_result['x_hat']
                    
                    # Calculate I-frame metrics
                    i_frame_metric = calculate_rgb_metrics(ref_frame, current_frame)
                    i_frame_mse = i_frame_metric['mse']
                    i_frame_psnr = i_frame_metric['psnr']

                    # Print I-frame metrics
                    logger.info(f"Frame {frame_idx} (I-frame): BPP={i_frame_bpp:.6f}, PSNR={i_frame_psnr:.2f} dB")
                    
                    # Add to sequence totals
                    seq_total_bpp += i_frame_bpp
                    seq_total_mse += i_frame_mse
                    seq_total_psnr += i_frame_psnr
                    
                    # Track I-frame metrics separately
                    seq_i_frame_bpp += i_frame_bpp
                    seq_i_frame_mse += i_frame_mse
                    seq_i_frame_psnr += i_frame_psnr
                    seq_i_frame_count += 1
                    
                else:
                    # Process as P-frame
                    if use_real_compression:
                        # Use real compression
                        real_bpp, real_recon = calculate_real_compression_bpp(
                            p_frame_model, i_frame_model, ref_frame, current_frame, logger
                        )
                        
                        if real_bpp is not None and real_recon is not None:
                            p_frame_bpp = real_bpp
                            p_frame_recon = real_recon
                            
                            # For real compression, we don't have detailed breakdown
                            p_frame_bpp_breakdown = {
                                'bpp_y': 0, 'bpp_z': 0, 'bpp_mv_y': 0, 'bpp_mv_z': 0
                            }
                            
                            logger.info(f"Frame {frame_idx} (P-frame): BPP={p_frame_bpp:.6f} [Real Compression]")
                        else:
                            # Fall back to estimation if real compression fails
                            logger.warning(f"Frame {frame_idx}: Real compression failed, using estimation")
                            result = p_frame_model(ref_frame, current_frame, stage=stage)
                            p_frame_bpp = result["bpp_train"].item()
                            p_frame_recon = result["recon_image"]
                            p_frame_bpp_breakdown = {
                                'bpp_y': result["bpp_y"].item(),
                                'bpp_z': result["bpp_z"].item(), 
                                'bpp_mv_y': result["bpp_mv_y"].item(),
                                'bpp_mv_z': result["bpp_mv_z"].item()
                            }
                            
                            logger.info(f"Frame {frame_idx} (P-frame): BPP={p_frame_bpp:.6f} "
                                      f"(Y:{p_frame_bpp_breakdown['bpp_y']:.6f}, Z:{p_frame_bpp_breakdown['bpp_z']:.6f}, "
                                      f"MV_Y:{p_frame_bpp_breakdown['bpp_mv_y']:.6f}, MV_Z:{p_frame_bpp_breakdown['bpp_mv_z']:.6f}) [Estimation]")
                    else:
                        # Use estimation (forward pass)
                        result = p_frame_model(ref_frame, current_frame, stage=stage)
                        p_frame_bpp = result["bpp_train"].item()
                        p_frame_recon = result["recon_image"]
                        p_frame_bpp_breakdown = {
                            'bpp_y': result["bpp_y"].item(),
                            'bpp_z': result["bpp_z"].item(), 
                            'bpp_mv_y': result["bpp_mv_y"].item(),
                            'bpp_mv_z': result["bpp_mv_z"].item()
                        }
                        
                        logger.info(f"Frame {frame_idx} (P-frame): BPP={p_frame_bpp:.6f} "
                                  f"(Y:{p_frame_bpp_breakdown['bpp_y']:.6f}, Z:{p_frame_bpp_breakdown['bpp_z']:.6f}, "
                                  f"MV_Y:{p_frame_bpp_breakdown['bpp_mv_y']:.6f}, MV_Z:{p_frame_bpp_breakdown['bpp_mv_z']:.6f}) [Estimation]")
                    
                    # Calculate P-frame metrics
                    p_frame_metric = calculate_rgb_metrics(p_frame_recon, current_frame)
                    p_frame_psnr = p_frame_metric['psnr']
                    p_frame_mse = p_frame_metric['mse']
                    
                    logger.info(f"         PSNR: {p_frame_psnr:.2f} dB")
                    
                    # Add to sequence totals
                    seq_total_bpp += p_frame_bpp
                    seq_total_mse += p_frame_mse
                    seq_total_psnr += p_frame_psnr
                    
                    # Track P-frame detailed metrics
                    seq_p_frame_bpp += p_frame_bpp
                    for key in p_frame_bpp_breakdown:
                        seq_p_frame_bpp_breakdown[key] += p_frame_bpp_breakdown[key]
                    seq_p_frame_count += 1
                    
                    # Update reference for next frame
                    ref_frame = p_frame_recon
            
            # Calculate sequence averages
            seq_avg_mse = seq_total_mse / num_frames
            seq_avg_psnr = seq_total_psnr / num_frames
            seq_avg_bpp = seq_total_bpp / num_frames
            
            # Calculate separate I-frame and P-frame metrics for this sequence
            seq_avg_i_frame_bpp = seq_i_frame_bpp / seq_i_frame_count if seq_i_frame_count > 0 else 0
            seq_avg_i_frame_mse = seq_i_frame_mse / seq_i_frame_count if seq_i_frame_count > 0 else 0
            seq_avg_i_frame_psnr = seq_i_frame_psnr / seq_i_frame_count if seq_i_frame_count > 0 else 0
            
            seq_avg_p_frame_bpp = seq_p_frame_bpp / seq_p_frame_count if seq_p_frame_count > 0 else 0
            seq_avg_p_frame_bpp_breakdown = {}
            for key in seq_p_frame_bpp_breakdown:
                seq_avg_p_frame_bpp_breakdown[key] = seq_p_frame_bpp_breakdown[key] / seq_p_frame_count if seq_p_frame_count > 0 else 0
            
            # Calculate RD cost using lambda
            seq_rd_cost = lambda_value * seq_avg_mse + seq_avg_bpp
            
            # Print sequence summary
            logger.info(f"Sequence Summary: Avg BPP={seq_avg_bpp:.6f}, Avg PSNR={seq_avg_psnr:.2f} dB, RD Cost={seq_rd_cost:.6f}")
            logger.info(f"  I-frames: {seq_i_frame_count}, P-frames: {seq_p_frame_count}")
            logger.info(f"  I-frame: BPP={seq_avg_i_frame_bpp:.6f}, PSNR={seq_avg_i_frame_psnr:.2f} dB")
            if not use_real_compression or any(v > 0 for v in seq_avg_p_frame_bpp_breakdown.values()):
                logger.info(f"  P-frame: Total_BPP={seq_avg_p_frame_bpp:.6f} "
                          f"(Y:{seq_avg_p_frame_bpp_breakdown['bpp_y']:.6f}, Z:{seq_avg_p_frame_bpp_breakdown['bpp_z']:.6f}, "
                          f"MV_Y:{seq_avg_p_frame_bpp_breakdown['bpp_mv_y']:.6f}, MV_Z:{seq_avg_p_frame_bpp_breakdown['bpp_mv_z']:.6f})")
            
            # Store sequence results
            sequence_results[seq_name] = {
                'avg_psnr': seq_avg_psnr,
                'avg_bpp': seq_avg_bpp,
                'avg_mse': seq_avg_mse,
                'rd_cost': seq_rd_cost,
                'i_frame_bpp': seq_avg_i_frame_bpp,
                'i_frame_psnr': seq_avg_i_frame_psnr,
                'i_frame_mse': seq_avg_i_frame_mse,
                'p_frame_bpp': seq_avg_p_frame_bpp,
                'p_frame_bpp_breakdown': seq_avg_p_frame_bpp_breakdown,
                'i_frame_count': seq_i_frame_count,
                'p_frame_count': seq_p_frame_count,
                'num_frames': num_frames
            }
            
            # Add to overall totals
            total_sequences += 1
            total_frames += num_frames
            total_bpp += seq_total_bpp
            total_mse += seq_total_mse
            total_psnr += seq_total_psnr
            
            total_i_frame_bpp += seq_i_frame_bpp
            total_i_frame_mse += seq_i_frame_mse
            total_i_frame_psnr += seq_i_frame_psnr
            total_i_frame_count += seq_i_frame_count
            
            total_p_frame_bpp += seq_p_frame_bpp
            for key in seq_p_frame_bpp_breakdown:
                total_p_frame_bpp_breakdown[key] += seq_p_frame_bpp_breakdown[key]
            total_p_frame_count += seq_p_frame_count
    
    # Calculate overall averages
    overall_avg_mse = total_mse / total_frames
    overall_avg_psnr = total_psnr / total_frames
    overall_avg_bpp = total_bpp / total_frames
    overall_rd_cost = lambda_value * overall_avg_mse + overall_avg_bpp
    
    # Calculate separate I-frame and P-frame averages
    overall_avg_i_frame_bpp = total_i_frame_bpp / total_i_frame_count if total_i_frame_count > 0 else 0
    overall_avg_i_frame_mse = total_i_frame_mse / total_i_frame_count if total_i_frame_count > 0 else 0
    overall_avg_i_frame_psnr = total_i_frame_psnr / total_i_frame_count if total_i_frame_count > 0 else 0
    
    overall_avg_p_frame_bpp = total_p_frame_bpp / total_p_frame_count if total_p_frame_count > 0 else 0
    overall_avg_p_frame_bpp_breakdown = {}
    for key in total_p_frame_bpp_breakdown:
        overall_avg_p_frame_bpp_breakdown[key] = total_p_frame_bpp_breakdown[key] / total_p_frame_count if total_p_frame_count > 0 else 0
    
    # Create overall results
    overall_results = {
        'avg_psnr': overall_avg_psnr,
        'avg_bpp': overall_avg_bpp,
        'avg_mse': overall_avg_mse,
        'rd_cost': overall_rd_cost,
        'avg_i_frame_bpp': overall_avg_i_frame_bpp,
        'avg_i_frame_psnr': overall_avg_i_frame_psnr,
        'avg_i_frame_mse': overall_avg_i_frame_mse,
        'avg_p_frame_bpp': overall_avg_p_frame_bpp,
        'avg_p_frame_bpp_breakdown': overall_avg_p_frame_bpp_breakdown,
        'total_i_frame_count': total_i_frame_count,
        'total_p_frame_count': total_p_frame_count,
        'total_sequences': total_sequences,
        'total_frames': total_frames,
        'lambda_value': lambda_value,
        'compression_mode': compression_mode,
        'sequence_results': sequence_results
    }
    
    logger.info(f"\n=== OVERALL RESULTS ACROSS ALL SEQUENCES ({compression_mode}) ===")
    logger.info(f"Lambda {lambda_value}: Avg BPP={overall_avg_bpp:.6f}, Avg PSNR={overall_avg_psnr:.2f} dB, "
              f"RD Cost={overall_rd_cost:.6f}")
    logger.info(f"Total sequences: {total_sequences}, Total frames: {total_frames}")
    logger.info(f"Frame distribution: I-frames={total_i_frame_count}, P-frames={total_p_frame_count}")
    logger.info(f"I-frame avg: BPP={overall_avg_i_frame_bpp:.6f}, PSNR={overall_avg_i_frame_psnr:.2f} dB")
    if not use_real_compression or any(v > 0 for v in overall_avg_p_frame_bpp_breakdown.values()):
        breakdown = overall_avg_p_frame_bpp_breakdown
        logger.info(f"P-frame avg: BPP={overall_avg_p_frame_bpp:.6f} "
                  f"(Y:{breakdown['bpp_y']:.6f}, Z:{breakdown['bpp_z']:.6f}, "
                  f"MV_Y:{breakdown['bpp_mv_y']:.6f}, MV_Z:{breakdown['bpp_mv_z']:.6f})")
    
    return overall_results


def main():
    parser = argparse.ArgumentParser(description='HEVC-B Evaluation Script for DCVC Models')
    
    # Model paths
    parser.add_argument('--p_frame_model_path', type=str, required=True, 
                       help='Path to DCVC P-frame model checkpoint')
    parser.add_argument('--i_frame_model_name', type=str, default='cheng2020-anchor', 
                       help='CompressAI I-frame model name (e.g., cheng2020-anchor, bmshj2018-factorized, etc.)')
    parser.add_argument('--i_frame_quality', type=int, default=6, 
                       help='Quality level for CompressAI pretrained model (1-8, higher = better quality)')
    parser.add_argument('--i_frame_pretrained', action='store_true', default=True,
                       help='Use CompressAI pretrained weights for I-frame model')
    
    # Dataset arguments
    parser.add_argument('--hevc_b_dir', type=str, required=True, 
                       help='Path to HEVC-B dataset directory')
    parser.add_argument('--max_frames', type=int, default=96, 
                       help='Maximum frames to process per HEVC-B sequence')
    
    # Evaluation arguments
    parser.add_argument('--lambda_value', type=float, default=None, 
                       help='Lambda value for rate-distortion trade-off (if not provided, will try to extract from checkpoint)')
    parser.add_argument('--intra_period', type=int, default=-1, 
                       help='Period for inserting I-frames (default: -1 for no periodic I-frames, 0 also disables, positive values enable)')
    parser.add_argument('--stage', type=int, default=4, choices=[1, 2, 3, 4],
                       help='DCVC training stage to use for evaluation (default: 4)')
    
    # Compression mode
    parser.add_argument('--use_real_compression', action='store_true', 
                       help='Use real compression/decompression instead of estimation (forward pass)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Directory to save evaluation results and logs')
    parser.add_argument('--save_results', action='store_true', 
                       help='Save detailed results to JSON file')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda/cpu). If not specified, uses cuda if available')
    
    # Compilation arguments
    parser.add_argument('--disable_compilation', action='store_true', 
                       help='Disable torch.compile for models (useful for debugging)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Setup output directory and logging
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "dcvc_evaluation.log")
    else:
        log_file = None
    
    logger = setup_logging(log_file)
    
    # Log evaluation configuration
    logger.info("=" * 80)
    logger.info("DCVC HEVC-B EVALUATION CONFIGURATION")
    logger.info("=" * 80)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Device: {device}")
    
    # Log compression mode
    compression_mode = "Real Compression" if args.use_real_compression else "Estimation (Forward Pass)"
    logger.info(f"COMPRESSION MODE: {compression_mode}")
    
    # Log compilation status
    if args.disable_compilation:
        logger.info("MODEL COMPILATION: DISABLED")
    else:
        logger.info("MODEL COMPILATION: ENABLED (torch.compile)")
    
    # Log intra period configuration
    if args.intra_period > 0:
        logger.info(f"INTRA PERIOD: {args.intra_period} (I-frames at intervals of {args.intra_period} frames)")
    else:
        logger.info("INTRA PERIOD: DISABLED (only frame 0 will be I-frame)")
    
    logger.info("=" * 80)
    
    # Validate paths
    if not os.path.exists(args.hevc_b_dir):
        raise ValueError(f"HEVC-B directory does not exist: {args.hevc_b_dir}")
    if not os.path.exists(args.p_frame_model_path):
        raise ValueError(f"P-frame model checkpoint does not exist: {args.p_frame_model_path}")
    
    # Load CompressAI I-frame model
    logger.info(f"Loading CompressAI I-frame model: {args.i_frame_model_name}")
    logger.info(f"Quality level: {args.i_frame_quality}")
    logger.info(f"Using pretrained weights: {args.i_frame_pretrained}")
    
    try:
        i_frame_model = compressai_models[args.i_frame_model_name](
            quality=args.i_frame_quality,
            pretrained=args.i_frame_pretrained
        ).to(device)
        i_frame_model.eval()
        logger.info(f"Successfully loaded CompressAI model: {args.i_frame_model_name}")
    except KeyError:
        available_models = list(compressai_models.keys())
        error_msg = f"Model '{args.i_frame_model_name}' not found in CompressAI zoo. Available models: {available_models}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading CompressAI model: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Compile I-frame model for performance if compilation is enabled
    if not args.disable_compilation:
        i_frame_model = torch.compile(i_frame_model)
        logger.info("I-frame model compiled for faster inference")
    else:
        logger.info("I-frame model loaded (compilation disabled)")
    
    # Load DCVC P-frame model 
    # Initialize with temporary lambda, will be updated based on checkpoint or argument
    temp_lambda = args.lambda_value if args.lambda_value is not None else 0.1
    p_frame_model = DCVC_net(lmbda=temp_lambda).to(device)
    
    # Load checkpoint and extract lambda if available
    checkpoint_lambda = load_model_checkpoint(p_frame_model, args.p_frame_model_path, device, logger)
    p_frame_model.eval()
    
    # Determine final lambda value
    if args.lambda_value is not None:
        final_lambda = args.lambda_value
        logger.info(f"Using lambda from command line argument: {final_lambda}")
    elif checkpoint_lambda is not None:
        final_lambda = checkpoint_lambda
        logger.info(f"Using lambda from checkpoint: {final_lambda}")
    else:
        final_lambda = 0.1  # Default value
        logger.warning(f"No lambda value found in checkpoint or command line, using default: {final_lambda}")
    
    # Update model lambda
    p_frame_model.lmbda = final_lambda
    
    # Compile P-frame model for performance if compilation is enabled
    if not args.disable_compilation:
        p_frame_model = torch.compile(p_frame_model)
        logger.info("P-frame model compiled for faster inference")
    else:
        logger.info("P-frame model loaded (compilation disabled)")
    
    # Run evaluation
    logger.info("Starting DCVC HEVC-B evaluation...")
    eval_start_time = time.time()
    
    try:
        results = evaluate_hevc_b(
            p_frame_model=p_frame_model,
            i_frame_model=i_frame_model,
            hevc_b_dir=args.hevc_b_dir,
            device=device,
            lambda_value=final_lambda,
            max_frames=args.max_frames,
            intra_period=args.intra_period,
            use_real_compression=args.use_real_compression,
            stage=args.stage,
            logger=logger
        )
        
        eval_duration = time.time() - eval_start_time
        logger.info(f"Evaluation completed in {eval_duration:.2f} seconds")
        
        if results:
            # Save results if requested
            if args.save_results and args.output_dir:
                import json
                results_file = os.path.join(args.output_dir, "dcvc_hevc_b_results.json")
                
                # Convert to JSON-serializable format
                json_results = {}
                for k, v in results.items():
                    if isinstance(v, dict):
                        json_results[k] = {}
                        for kk, vv in v.items():
                            if isinstance(vv, dict):
                                json_results[k][kk] = {str(kkk): float(vvv) if isinstance(vvv, (int, float)) else vvv for kkk, vvv in vv.items()}
                            elif isinstance(vv, (int, float)):
                                json_results[k][kk] = float(vv)
                            else:
                                json_results[k][kk] = vv
                    elif isinstance(v, (int, float)):
                        json_results[k] = float(v)
                    else:
                        json_results[k] = v
                
                # Add configuration info to JSON
                json_results['_config'] = {
                    'lambda_value': final_lambda,
                    'intra_period': args.intra_period,
                    'max_frames': args.max_frames,
                    'stage': args.stage,
                    'use_real_compression': args.use_real_compression,
                    'compression_mode': compression_mode,
                    'i_frame_model': args.i_frame_model_name,
                    'i_frame_quality': args.i_frame_quality
                }
                
                with open(results_file, 'w') as f:
                    json.dump(json_results, f, indent=2)
                logger.info(f"Results saved to: {results_file}")
            
            # Print final summary
            logger.info("=" * 80)
            logger.info(f"FINAL DCVC EVALUATION SUMMARY ({compression_mode})")
            logger.info("=" * 80)
            logger.info(f"Lambda {final_lambda:6.3f}: BPP={results['avg_bpp']:.6f}, PSNR={results['avg_psnr']:.2f} dB, "
                      f"RD Cost={results['rd_cost']:.6f}")
            logger.info(f"             I-frames: {results['total_i_frame_count']}, P-frames: {results['total_p_frame_count']}")
            logger.info(f"             Total sequences: {results['total_sequences']}, Total frames: {results['total_frames']}")
            logger.info("=" * 80)
            
        else:
            logger.error("Evaluation failed or returned no results")
            return 1
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise
    
    logger.info("DCVC evaluation script completed successfully!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())