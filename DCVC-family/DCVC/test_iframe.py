#!/usr/bin/env python3
"""
Standalone HEVC-B I-Frame Evaluation Script

This script evaluates CompressAI I-frame models on the HEVC-B dataset,
providing results in both estimation mode (training forward pass) and real compression mode.
All frames are treated as independent I-frames.

Usage:
    python test_iframe.py --hevc_b_dir <path> [options]
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


def evaluate_hevc_b_iframes(i_frame_model, hevc_b_dir, device, 
                           max_frames=96,logger=None):
    """
    Evaluate CompressAI I-frame model on HEVC-B dataset.
    All frames are treated as independent I-frames.
    
    Args:
        i_frame_model: CompressAI I-frame model  
        hevc_b_dir: Path to HEVC-B dataset
        device: Device to run evaluation on
        max_frames: Maximum frames per sequence
        logger: Logger instance
        
    Returns:
        dict: Evaluation results
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    i_frame_model.eval()

    eval_dataset = HEVCB_Dataset(root_dir=hevc_b_dir, max_frames=max_frames)
    
    if len(eval_dataset) == 0:
        logger.warning(f"No HEVC-B sequences found in {hevc_b_dir}")
        return None

    compression_mode = "Estimation (Forward Pass)"

    # Log evaluation configuration
    logger.info(f"Evaluating on {len(eval_dataset)} HEVC-B sequences using {compression_mode}")
    logger.info("All frames will be processed as independent I-frames")
    
    sequence_results = {}
    
    # Overall metrics across all sequences
    total_sequences = 0
    total_frames = 0
    total_bpp = 0
    total_mse = 0
    total_psnr = 0
    
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
            
            # Process all frames as I-frames
            for frame_idx in range(num_frames):
                current_frame = frames[0, frame_idx, ...].unsqueeze(0)  # [1, C, H, W]
                
                # Use estimation (forward pass)
                i_frame_result, frame_bpp = compress_i_frame_with_padding(
                    i_frame_model, current_frame, calculate_bpp=True
                )
                frame_recon = i_frame_result['x_hat']
                logger.info(f"Frame {frame_idx} (I-frame): BPP={frame_bpp:.6f} [Estimation]")
                
                # Calculate I-frame metrics
                frame_metric = calculate_rgb_metrics(frame_recon, current_frame)
                frame_psnr = frame_metric['psnr']
                frame_mse = frame_metric['mse']
                
                logger.info(f"         PSNR: {frame_psnr:.2f} dB")
                
                # Add to sequence totals
                seq_total_bpp += frame_bpp
                seq_total_mse += frame_mse
                seq_total_psnr += frame_psnr
            
            # Calculate sequence averages
            seq_avg_mse = seq_total_mse / num_frames
            seq_avg_psnr = seq_total_psnr / num_frames
            seq_avg_bpp = seq_total_bpp / num_frames
            
            # Print sequence summary
            logger.info(f"Sequence Summary: Avg BPP={seq_avg_bpp:.6f}, Avg PSNR={seq_avg_psnr:.2f} dB")
            
            # Store sequence results
            sequence_results[seq_name] = {
                'avg_psnr': seq_avg_psnr,
                'avg_bpp': seq_avg_bpp,
                'avg_mse': seq_avg_mse,
                'num_frames': num_frames
            }
            
            # Add to overall totals
            total_sequences += 1
            total_frames += num_frames
            total_bpp += seq_total_bpp
            total_mse += seq_total_mse
            total_psnr += seq_total_psnr
    
    # Calculate overall averages
    overall_avg_mse = total_mse / total_frames
    overall_avg_psnr = total_psnr / total_frames
    overall_avg_bpp = total_bpp / total_frames
    
    # Create overall results
    overall_results = {
        'avg_psnr': overall_avg_psnr,
        'avg_bpp': overall_avg_bpp,
        'avg_mse': overall_avg_mse,
        'total_sequences': total_sequences,
        'total_frames': total_frames,
        'compression_mode': compression_mode,
        'sequence_results': sequence_results
    }
    
    logger.info(f"\n=== OVERALL RESULTS ACROSS ALL SEQUENCES ({compression_mode}) ===")
    logger.info(f"I-frame Only: Avg BPP={overall_avg_bpp:.6f}, Avg PSNR={overall_avg_psnr:.2f} dB")
    logger.info(f"Total sequences: {total_sequences}, Total frames: {total_frames}")
    logger.info("=" * 80)
    
    return overall_results


def main():
    parser = argparse.ArgumentParser(description='HEVC-B I-Frame Evaluation Script')
    # CUDA_VISIBLE_DEVICES=2 python test_iframe.py --hevc_b_dir /home/zhan5096/Project/dataset/HEVC-B/cropped --i_frame_quality 3 --i_frame_pretrained
    # I-frame model arguments
    parser.add_argument('--i_frame_model_name', type=str, default='cheng2020-anchor', 
                       help='CompressAI I-frame model name (e.g., IntraNoAR, bmshj2018-factorized, etc.)')
    parser.add_argument('--i_frame_quality', type=int, default=3, 
                       help='Quality level for CompressAI pretrained model (1-8, higher = better quality)')
    parser.add_argument('--i_frame_pretrained', action='store_true', default=True,
                       help='Use CompressAI pretrained weights for I-frame model')
    
    # Dataset arguments
    parser.add_argument('--hevc_b_dir', type=str, required=True, 
                       help='Path to HEVC-B dataset directory')
    parser.add_argument('--max_frames', type=int, default=96, 
                       help='Maximum frames to process per HEVC-B sequence')
    
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
        log_file = os.path.join(args.output_dir, "iframe_evaluation.log")
    else:
        log_file = None
    
    logger = setup_logging(log_file)
    
    # Log evaluation configuration
    logger.info("=" * 80)
    logger.info("I-FRAME HEVC-B EVALUATION CONFIGURATION")
    logger.info("=" * 80)
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info(f"Device: {device}")
    
    # Log compression mode
    compression_mode = "Estimation (Forward Pass)"
    logger.info(f"COMPRESSION MODE: {compression_mode}")
    
    # Log compilation status
    if args.disable_compilation:
        logger.info("MODEL COMPILATION: DISABLED")
    else:
        logger.info("MODEL COMPILATION: ENABLED (torch.compile)")
    
    logger.info("=" * 80)
    
    # Validate paths
    if not os.path.exists(args.hevc_b_dir):
        raise ValueError(f"HEVC-B directory does not exist: {args.hevc_b_dir}")
    
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
    
    # Run evaluation
    logger.info("Starting I-frame HEVC-B evaluation...")
    eval_start_time = time.time()
    
    try:
        results = evaluate_hevc_b_iframes(
            i_frame_model=i_frame_model,
            hevc_b_dir=args.hevc_b_dir,
            device=device,
            max_frames=args.max_frames,
            logger=logger
        )
        
        eval_duration = time.time() - eval_start_time
        logger.info(f"Evaluation completed in {eval_duration:.2f} seconds")
        
        if results:
            # Save results if requested
            if args.save_results and args.output_dir:
                import json
                results_file = os.path.join(args.output_dir, "iframe_hevc_b_results.json")
                
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
                    'max_frames': args.max_frames,
                    'compression_mode': compression_mode,
                    'i_frame_model': args.i_frame_model_name,
                    'i_frame_quality': args.i_frame_quality
                }
                
                with open(results_file, 'w') as f:
                    json.dump(json_results, f, indent=2)
                logger.info(f"Results saved to: {results_file}")
            
            # Print final summary
            logger.info("=" * 80)
            logger.info(f"FINAL I-FRAME EVALUATION SUMMARY ({compression_mode})")
            logger.info("=" * 80)
            logger.info(f"I-frame Only: BPP={results['avg_bpp']:.6f}, PSNR={results['avg_psnr']:.2f} dB")
            logger.info(f"             Total sequences: {results['total_sequences']}, Total frames: {results['total_frames']}")
            logger.info("=" * 80)
            
        else:
            logger.error("Evaluation failed or returned no results")
            return 1
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise
    
    logger.info("I-frame evaluation script completed successfully!")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
