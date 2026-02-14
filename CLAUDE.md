# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Disclaimer**: This documentation was generated with the assistance of Claude Code and may contain errors or inaccuracies. Please verify critical information before use.
>
> **Contributing**: OpenDCVCs is an open-source project for the research community. We welcome contributions! Feel free to use, modify, and contribute to this codebase.

## Overview

OpenDCVCs is a PyTorch implementation of the DCVC (Deep Contextual Video Compression) series of learned video compression models, built on top of CompressAI for entropy coding support. The project contains training-ready code for multiple DCVC variants (DCVC, DCVC-TCM, DCVC-HEM, DCVC-DC).

**Important**: This is a reimplementation based on CompressAI. The original estimation-based code is on the main branch.

## Repository Structure

```
DCVC-family/
├── DCVC/              # Base DCVC model
│   ├── src/
│   │   ├── models/    # DCVC_net_compressai.py (main P-frame model)
│   │   ├── layers/    # Custom layers (GDN, MaskedConv2d)
│   │   ├── ops/       # Operations (parametrizers, bound_ops)
│   │   └── utils/     # Utility functions
│   ├── train_dcvc.py  # Main training script
│   ├── test_video.py  # HEVC-B video evaluation script
│   ├── test_iframe.py # I-frame only evaluation script
│   ├── dataset.py     # Dataset classes (VimeoGOPDataset, HEVCB_Dataset, BVI_AOM_Dataset)
│   └── train.sh       # SLURM training script template
│
└── DCVC-TCM/          # DCVC with Temporal Context Modeling
    ├── src/
    │   ├── models/    # video_net_dmc_compressai.py (TCM variant)
    │   ├── entropy_models/  # Custom entropy models
    │   └── ...        # Similar structure to DCVC
    └── ...            # Similar scripts to DCVC
```

## Key Architecture

### Model Components

**DCVC Base Model** (DCVC-family/DCVC/src/models/DCVC_net_compressai.py):
- Inherits from `compressai.models.CompressionModel`
- **Motion Coding**: `mvEncoder` → `mvDecoder_part1/part2` with ME_Spynet for optical flow
- **Residual Coding**: `contextualEncoder` → `contextualDecoder_part1/part2`
- **Entropy Models**: `EntropyBottleneck` for hyperpriors (z, z_mv), `GaussianConditional` for main latents
- **Context Modeling**: Uses warped reference features and motion-compensated priors

**DCVC-TCM** (DCVC-family/DCVC-TCM/src/models/video_net_dmc_compressai.py):
- Extends base DCVC with multi-scale temporal context mining
- `FeatureExtractor` + `MultiScaleContextFusion` for hierarchical temporal modeling
- `ContextualEncoder`/`ContextualDecoder` with multi-scale feature fusion

### Training Stages

Training uses a progressive, stage-based approach:

1. **Stage 1**: Train motion estimation (optical flow) and motion coding modules
2. **Stage 2**: Freeze motion modules, train residual coding with fixed motion
3. **Stage 3**: Continue stage 2 training (motion modules remain frozen)
4. **Stage 4**: Unfreeze all modules, end-to-end joint optimization (default: no GOP optimization)
   - **Stage 4 + GOP Opt**: After completing stage 4, run again with `--stage4_gop_opt` for 7-frame error-aware GOP-level finetuning
5. **Stage 5**: Fine-tuning on BVI-AOM dataset with 32-frame sequences

**Stage progression**: Each stage loads checkpoints from the previous stage automatically. Use `--resume` to continue training within the same stage.

#### GOP Size Training Protocol (Stages 1-4)

**IMPORTANT**: Within each stage, follow this GOP progression protocol:

1. Start with `--gop_size 2`, train for several epochs
2. Reset learning rate to 1e-4 (use `--force_learning_rate 1e-4`), increase to `--gop_size 3`
3. Continue with `--gop_size 5` (reset LR again)
4. Finally train with `--gop_size 7` (reset LR again)
5. For Stage 4 GOP optimization, enable `--stage4_gop_opt` with `--gop_size 7` for error-aware finetuning

This progressive GOP size increase helps the model gradually learn longer temporal dependencies.

### Datasets

- **Vimeo90k**: Primary training dataset (7-frame septuplets)
  - Requires: `--vimeo_dir` and `--septuplet_list` paths
  - Randomly crops `--crop_size` patches (default: 256x256)
  - Selects random `--gop_size` frames (default: 7) from each septuplet

- **HEVC-B**: Evaluation dataset (5 sequences)
  - Requires: `--hevc_b_dir` pointing to RGB PNG sequences
  - Used for validation during training (controlled by `--eval_freq`)

- **BVI-AOM**: Fine-tuning dataset (stage 5)
  - Optional: `--finetune_max_sequences` to limit number of sequences

### I-Frame Compression

Uses CompressAI pre-trained models (not trained in this repo):
- Loaded via `compressai.zoo.models`
- Specified by `--i_frame_model` (e.g., 'bmshj2018-hyperprior')
- Quality controlled by `--i_frame_quality` (1-8)
- Optional: `--i_frame_pretrained` for custom checkpoint paths

## Training Commands

### Setup Environment

```bash
module load gcc
module load conda/2024.09
module load cuda/12.6.0
module load openmpi/5.0.5

conda activate dcvc
```

### Basic Training (DCVC)

```bash
# Stage 1: Train motion modules (follow GOP progression)
# Step 1a: Start with GOP 2
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --vimeo_dir /path/to/vimeo_septuplet/sequences \
  --septuplet_list /path/to/sep_trainlist.txt \
  --hevc_b_dir /path/to/HEVC-B/png_sequences \
  --checkpoint_dir /path/to/checkpoints \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2

# Step 1b: Increase to GOP 3, reset LR
accelerate launch DCVC-family/DCVC/train_dcvc.py \
  [same args] \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --resume /path/to/stage_1_gop2_checkpoint.pth

# Step 1c: Increase to GOP 5, reset LR
# (use --gop_size 5 --force_learning_rate 1e-4 --resume ...)

# Step 1d: Final training with GOP 7, reset LR
# (use --gop_size 7 --force_learning_rate 1e-4 --resume ...)

# Stage 2-3: Train residual coding (motion frozen)
# Repeat the same GOP progression (2→3→5→7) for each stage

# Stage 4: End-to-end optimization (no GOP opt)
accelerate launch DCVC-family/DCVC/train_dcvc.py \
  [same args with GOP progression 2→3→5→7] \
  --stage 4 \
  --use_scheduler \
  --scheduler_patience 3

# Stage 4 + GOP Opt: Error-aware GOP finetuning (after completing stage 4)
accelerate launch DCVC-family/DCVC/train_dcvc.py \
  [same args] \
  --stage 4 \
  --stage4_gop_opt \
  --gop_size 7 \
  --resume /path/to/stage_4_gop7_checkpoint.pth

# Stage 5: BVI-AOM finetuning with 32 frames
accelerate launch DCVC-family/DCVC/train_dcvc.py \
  [same args] \
  --stage 5 \
  --finetune_gop_size 32 \
  --finetune_epochs 10 \
  --resume /path/to/stage_4_final_checkpoint.pth
```

### Training DCVC-TCM

```bash
# Similar to DCVC, but requires I-frame pretrained model
accelerate launch DCVC-family/DCVC-TCM/train_dcvc.py \
  [same args as DCVC] \
  --i_frame_pretrained /path/to/intra_model.pth.tar \
  [other args]
```

### Important Training Arguments

- `--lambda_value`: Rate-distortion tradeoff (higher = better quality, lower compression)
- `--quality_index`: Integer quality level (1-8, used for checkpoint naming)
- `--gop_size`: Number of frames per GOP. **Follow progression: 2 → 3 → 5 → 7 within each stage**
- `--force_learning_rate`: Override/reset learning rate. **Required when changing GOP size** (typically use 1e-4)
- `--stage`: Training stage (1-5). See "Training Stages" section for details
- `--stage4_gop_opt`: Enable GOP-level optimization in stage 4 (error-aware finetuning)
- `--use_scheduler`: Enable ReduceLROnPlateau scheduler
- `--scheduler_patience`: Epochs to wait before reducing LR
- `--resume`: Continue training from checkpoint (same stage)
- `--load_pretrained`: Load weights but reset training state (useful for transfer)
- `--finetune_gop_size`: GOP size for stage 5 BVI-AOM finetuning (default: 32)

## Evaluation Commands

### Evaluate on HEVC-B

```bash
# Full video evaluation (I-frame + P-frames)
python DCVC-family/DCVC/test_video.py \
  --p_frame_model_path /path/to/checkpoint.pth \
  --hevc_b_dir /path/to/HEVC-B/sequences \
  --i_frame_model bmshj2018-hyperprior \
  --i_frame_quality 6 \
  --max_frames 96

# I-frame only evaluation
python DCVC-family/DCVC/test_iframe.py \
  --hevc_b_dir /path/to/HEVC-B/sequences \
  --i_frame_model bmshj2018-hyperprior \
  --quality 6
```

### Evaluation Modes

Both scripts support two modes:
1. **Estimation mode** (default): Training forward pass with likelihoods
2. **Real compression mode** (`--real_compression`): Actual entropy coding with CompressAI

Evaluation outputs per-sequence and average PSNR/MS-SSIM/BPP metrics.

## Checkpoint Management

Checkpoint naming convention:
```
model_dcvc_lambda_{lambda}_quality_{quality}_stage_{stage}_{latest|best}.pth
```

Checkpoints contain:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`, `stage`: Training progress
- `lambda_value`, `quality_index`: Configuration
- `best_metric`: Best evaluation score

**Auto-loading**: Training automatically loads from previous stage if current stage checkpoint doesn't exist.

## Code Patterns

### RGB vs YUV
- Models operate in **RGB space** (not YUV)
- Input frames: [B, 3, H, W] in range [0, 1]
- Padding handled internally (16-pixel alignment for I-frames)

### Stage Parameter
- Model forward pass requires `stage=` argument during training
- Stage controls which modules are active/frozen
- Use `stage=4` for inference/evaluation

### Accelerate Integration
- All training uses Hugging Face Accelerate for distributed training
- Checkpoints saved/loaded via `accelerator.save()`/`accelerator.load()`
- Main process checks via `accelerator.is_main_process`

## Common Issues

1. **Padding errors**: Ensure input dimensions are compatible with model downsampling (typically 16x for I-frames)

2. **Memory issues**: Reduce `--batch_size` or `--crop_size` for training, or `--max_frames` for evaluation

3. **Stage mismatch**: If loading a checkpoint fails, verify the stage number and checkpoint naming

4. **Learning rate**: Use `--force_learning_rate` if scheduler has reduced LR too much or you need manual control

5. **GOP size**: Training GOP size (`--gop_size`) can differ from evaluation (HEVC-B uses 32-frame GOPs). Remember to follow the GOP progression protocol (2→3→5→7) within each stage

6. **Forgetting to reset LR**: When increasing GOP size, always use `--force_learning_rate 1e-4` to reset the learning rate

## Development Workflow

1. **For each stage** (1-4), follow the GOP progression: 2 → 3 → 5 → 7
   - Train with `--gop_size 2` for several epochs
   - Resume with `--gop_size 3 --force_learning_rate 1e-4`
   - Resume with `--gop_size 5 --force_learning_rate 1e-4`
   - Resume with `--gop_size 7 --force_learning_rate 1e-4`

2. Progress through stages: 1 → 2 → 3 → 4 → (optional: 4 + GOP opt) → (optional: 5 for BVI-AOM)

3. After completing Stage 4 with GOP 7, optionally enable GOP optimization:
   - Run Stage 4 again with `--stage4_gop_opt --gop_size 7` for error-aware finetuning

4. Monitor training logs in `checkpoint_dir/logs/`

5. Evaluate on HEVC-B after each stage completion using `test_video.py`

6. Compare BPP vs PSNR/MS-SSIM against baseline codecs

7. Use `--skip_eval` to disable automatic evaluation during training

## Dependencies

Key packages (see requirements.txt):
- PyTorch 2.6+ with CUDA 12.6
- CompressAI (for entropy coding, pre-trained I-frame models)
- accelerate (distributed training)
- pytorch-msssim (MS-SSIM metric)
- timm (model utilities)
- torch-ema (exponential moving average, optional)
