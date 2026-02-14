# OpenDCVCs: A PyTorch Open Source Implementation and Performance Evaluation of the DCVC series Video Codecs

## Currently, we are reimplementing the project based on the CompressAI repository to support entropy coding, and will release it soon. The original estimation-based code is available on the main branch.

## Overview

**OpenDCVCs** is an open-source PyTorch implementation and benchmarking suite for the DCVC series of learned video compression models. Built on top of CompressAI, it provides comprehensive, training-ready code with real entropy coding support for advanced video compression codecs.

### Supported Algorithms

| Model      | Status | Description                                                                                                   |
|------------|--------|--------------------------------------------------------------------------------------------------------------|
| DCVC       | ✅ Ready | Feature-domain conditional coding with contextual entropy modeling                                            |
| DCVC-TCM   | ✅ Ready | Multi-scale temporal context mining and refilling for richer temporal modeling                                |
| DCVC-HEM   | 🚧 Coming | Hybrid spatial-temporal entropy modeling and multi-granularity quantization                                   |
| DCVC-DC    | 🚧 Coming | Hierarchical quality, offset diversity, and quadtree-based entropy coding for robust, diverse context mining  |

## Features

- ✅ Full training & evaluation pipelines for DCVC and DCVC-TCM
- ✅ Real entropy coding via CompressAI integration
- ✅ Multi-stage progressive training protocol (5 stages)
- ✅ Support for multiple datasets (Vimeo90k, HEVC-B, BVI-AOM)
- ✅ Accelerate-based distributed training
- ✅ Comprehensive evaluation tools with PSNR/MS-SSIM/BPP metrics

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Requires PyTorch 2.6+ with CUDA 12.6
# See requirements.txt for full dependency list
```

### Training DCVC

Training follows a **progressive GOP size protocol** within each stage:

```bash
# Stage 1: Motion coding with GOP progression (2→3→5→7)
# Step 1: Start with GOP 2
CUDA_VISIBLE_DEVICES=0 accelerate launch DCVC-family/DCVC/train_dcvc.py \
  --vimeo_dir /path/to/vimeo_septuplet/sequences \
  --septuplet_list /path/to/sep_trainlist.txt \
  --hevc_b_dir /path/to/HEVC-B/sequences \
  --checkpoint_dir ./checkpoints \
  --i_frame_quality 6 \
  --lambda_value 2048 \
  --quality_index 4 \
  --stage 1 \
  --epochs 10 \
  --batch_size 4 \
  --gop_size 2

# Step 2: Increase to GOP 3, reset LR
accelerate launch DCVC-family/DCVC/train_dcvc.py \
  [same args] \
  --gop_size 3 \
  --force_learning_rate 1e-4 \
  --resume ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_1_latest.pth

# Step 3: GOP 5 with LR reset
# (repeat with --gop_size 5 --force_learning_rate 1e-4 --resume ...)

# Step 4: GOP 7 with LR reset
# (repeat with --gop_size 7 --force_learning_rate 1e-4 --resume ...)

# Stages 2-4: Repeat GOP progression for each stage
# Stage 2-3: Residual coding (motion frozen)
# Stage 4: End-to-end optimization
# Stage 4 + GOP Opt: Error-aware finetuning with --stage4_gop_opt
# Stage 5: BVI-AOM finetuning with 32-frame GOPs
```

**Important**: Always reset learning rate to 1e-4 when changing GOP size!

### Training DCVC-TCM

```bash
# Same as DCVC, but add I-frame pretrained model
accelerate launch DCVC-family/DCVC-TCM/train_dcvc.py \
  [same args as DCVC] \
  --i_frame_pretrained /path/to/intra_model.pth.tar
```

### Evaluation

```bash
# Evaluate on HEVC-B dataset
python DCVC-family/DCVC/test_video.py \
  --p_frame_model_path ./checkpoints/model_dcvc_lambda_2048.0_quality_4_stage_4_best.pth \
  --hevc_b_dir /path/to/HEVC-B/sequences \
  --i_frame_model bmshj2018-hyperprior \
  --i_frame_quality 6 \
  --max_frames 96

# Optional: Real compression mode (slower, uses actual entropy coding)
python DCVC-family/DCVC/test_video.py [same args] --real_compression
```

## Training Protocol

### Multi-Stage Training

1. **Stage 1**: Train motion estimation and motion coding modules
2. **Stage 2**: Freeze motion modules, train residual coding
3. **Stage 3**: Continue residual coding training
4. **Stage 4**: Unfreeze all, end-to-end joint optimization
5. **Stage 4 + GOP Opt** (optional): Error-aware GOP-level finetuning with `--stage4_gop_opt`
6. **Stage 5** (optional): Fine-tune on BVI-AOM with 32-frame sequences

### GOP Size Progression (Critical!)

**Within each stage (1-4)**, follow this GOP size progression:

1. Start with `--gop_size 2`, train for several epochs
2. Increase to `--gop_size 3` with `--force_learning_rate 1e-4`
3. Continue with `--gop_size 5` (reset LR)
4. Finish with `--gop_size 7` (reset LR)

This progressive increase helps the model gradually learn longer temporal dependencies.

## Dataset Preparation

### Vimeo90k (Training)
- Download from [Vimeo90k website](http://toflow.csail.mit.edu/)
- Structure: `sequences/[video_id]/im1.png ... im7.png`
- Requires `sep_trainlist.txt` file listing training sequences

### HEVC-B (Evaluation)
- 5 sequences: BQTerrace, BasketballDrive, Cactus, Kimono, ParkScene
- Convert to RGB PNG sequences (96+ frames per sequence)
- Use 32-frame GOPs for evaluation

### BVI-AOM (Stage 5 Fine-tuning)
- Optional dataset for final fine-tuning
- 64-frame sequences, used with 32-frame GOPs

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--stage` | Training stage (1-5) |
| `--gop_size` | Frames per GOP. **Follow 2→3→5→7 progression!** |
| `--force_learning_rate` | Reset LR (required when changing GOP size, use 1e-4) |
| `--lambda_value` | Rate-distortion tradeoff (e.g., 256, 512, 1024, 2048) |
| `--quality_index` | Quality level 1-8 (for checkpoint naming) |
| `--i_frame_quality` | CompressAI I-frame model quality (1-8) |
| `--stage4_gop_opt` | Enable GOP optimization in stage 4 |
| `--use_scheduler` | Enable ReduceLROnPlateau scheduler |
| `--resume` | Continue training from checkpoint |
| `--finetune_gop_size` | GOP size for stage 5 (default: 32) |

## Project Structure

```
OpenDCVCs_compressai/
├── DCVC-family/
│   ├── DCVC/              # Base DCVC model
│   │   ├── src/
│   │   │   ├── models/    # DCVC_net_compressai.py
│   │   │   ├── layers/    # Custom layers (GDN, etc.)
│   │   │   ├── ops/       # Operations
│   │   │   └── utils/     # Utilities
│   │   ├── train_dcvc.py  # Training script
│   │   ├── test_video.py  # Video evaluation
│   │   ├── test_iframe.py # I-frame evaluation
│   │   └── dataset.py     # Dataset classes
│   │
│   └── DCVC-TCM/          # DCVC with Temporal Context Modeling
│       ├── src/
│       │   ├── models/    # video_net_dmc_compressai.py
│       │   └── ...
│       └── ...
│
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── CLAUDE.md             # Detailed guide for Claude Code
```

## Architecture Highlights

### DCVC
- Motion coding: ME_Spynet + MV encoder/decoder
- Residual coding: Contextual encoder/decoder with warped references
- Entropy models: EntropyBottleneck (hyperpriors) + GaussianConditional (main latents)
- RGB-space compression (not YUV)

### DCVC-TCM
- Multi-scale temporal context extraction via FeatureExtractor
- Hierarchical context fusion with MultiScaleContextFusion
- Enhanced contextual encoder/decoder with multi-scale features


## Contributing

OpenDCVCs is an open-source project built for the research community. We welcome contributions of all kinds! Feel free to use, modify, and contribute to this codebase.

Ways to contribute:
- Report bugs and issues
- Submit pull requests with improvements
- Add support for new DCVC variants (DCVC-HEM, DCVC-DC)
- Improve documentation
- Share trained models and benchmarks

## Disclaimer

This documentation was generated with the assistance of Claude Code and may contain errors or inaccuracies. Please verify critical information before use. If you find any issues, please open an issue or submit a pull request.



## Acknowledgments

This implementation builds upon:
- [CompressAI](https://github.com/InterDigitalInc/CompressAI) for entropy coding
- Original DCVC series papers and implementations
- Hugging Face Accelerate for distributed training

## Bib

```
@article{zhang2025opendcvcs,
  title={Opendcvcs: a pytorch open source implementation and performance evaluation of the dcvc series video codecs},
  author={Zhang, Yichi and Zhu, Fengqing},
  journal={arXiv preprint arXiv:2508.04491},
  year={2025}
}
```

