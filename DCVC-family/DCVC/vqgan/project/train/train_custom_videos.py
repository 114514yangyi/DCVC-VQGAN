import os.path
import sys

# 添加项目根目录到 Python 路径，以便可以直接运行脚本
# 注意：当前文件位于 <repo>/train/train_custom_videos.py，故只需回到上一级目录
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)



import argparse
from datetime import datetime
import torch
from torchvision.transforms import transforms
from train.trainVqVae import TrainVqVae
from models.model_adapter import create_model
from train.train_utils import get_model_size, NormalizeInverse, \
    analyze_model_architecture, build_id_name
from train.video_utils import video_pipe
from train.auto_normalize import compute_normalize_params
from log_utils.log_utils import LogManager
from train.video_utils import list_videos




PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = PYTORCH_CUDA_ALLOC_CONF
# os.environ["WANDB_MODE"] = "offline"


################################
# ========================
# train_custom_videos.py 文件详细注释
# ========================
#
# 本文件是自定义视频训练的主脚本，主要用于训练 VQ-VAE2 模型（或其变体）用于处理视频数据。
# 文件结构及主要功能说明如下：

# 1. 引入必要的包（头部）：
#    - os, sys 以及 argparse 等基础包用于路径、参数解析等通用功能。
#    - torch 相关用于深度学习模型与训练。
#    - train.videos2.trainVqVae, models.vq_vae.vq_vae2, train.train_utils, train.videos2.video_utils, log_utils.log_utils 等为项目中自定义的训练、模型、日志和数据处理相关模块。

# 2. 加入项目根目录到 sys.path：
#    - 这样做确保直接运行脚本时，项目内其它模块能被正常 import（不受工作目录影响）。

# 3. PYTORCH_CUDA_ALLOC_CONF 环境变量设置：
#    - 提前设置 PyTorch 的显存分配策略（如 expandable_segments），避免显存耗尽相关问题。

# 4. train_videos(device_override=None) 训练入口函数：
#    - 该函数封装了完整的训练步骤，包含但不限于：
#      a. 加载参数配置（参数通过 video_utils.params 提供，实际用到 data_args、train_args、model_args）。
#      b. 支持通过命令行指定训练设备类型（CPU/GPU/自动），并能处理显卡GPU编号覆盖（通过环境变量）。
#      c. 自动补全 VQ-GAN 相关超参 (disc_factor, disc_start 等) 以保证兼容性。
#      d. 初始化日志系统，包括 wandb 记录（可选），并传递完整训练配置用于追踪。
#      e. 构建数据加载器：使用 video_pipe 构造训练和验证数据加载器，并根据配置决定是否启用验证集。
#      f. 构建标准化及反标准化变换（normalize, unnormalize）方便模型前后处理。
#      g. 创建模型、分析架构和参数量。
#      h. 实例化核心训练控制器 TrainVqVae，并配置其所需参数和日志。
#      i. 执行主训练循环，包括异常处理与日志管理，确保即使中断（CTRL+C 或出错）也能妥善收尾。

# 5. 主程序入口（if __name__ == '__main__':）：
#    - 通过 argparse 实现命令行参数（如 --device, --help-device, --gpu-index），增强脚本灵活性与通用性。
#    - 提供帮助信息（设备选择说明），便于用户理解如何切换 CPU/GPU，甚至支持多 GPU 环境手动指定索引。
#    - 预检测可用 GPU 数量并打印，方便用户了解当前运行环境。
#    - 透传 gpu_index 到内部训练逻辑，实现“外部指定 GPU” 跟配置文件间的协同。
#    - 最终调用 train_videos() 开始所有训练流程。

# 6. 其他说明：
#    - 该脚本假设项目结构和所依赖模块（如 video_pipe, TrainVqVae, LogManager 等）均已实现，并能够被正确 import。
#    - 所有关键步骤（模型初始化、数据加载、日志记录、训练/验证流程等）均体现了高内聚、低耦合的设计思路。
#    - 丰富的命令行功能确保脚本既支持批量自动化，也能应付实验性调试；

# ========================



def train_videos(device_override=None, config_path='param.json'):
    """
    主训练函数
    
    执行以下步骤：
    1. 加载配置参数
    2. 初始化wandb（如果启用）
    3. 创建模型
    4. 分析并保存模型架构信息
    5. 创建数据加载器
    6. 初始化训练对象并开始训练
    
    Args:
        device_override: 可选的设备选择，会覆盖配置文件中的设置
                        可选值: 'cpu', 'cuda', 'auto', None
        config_path: 配置文件路径，默认为 'param.json'
    """
    # 加载配置文件
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    # 如果配置中没有 data_process，尝试从 video_utils 获取（向后兼容）
    if 'data_process' not in params:
        from video_utils import params as video_params
        if 'data_process' in video_params:
            params['data_process'] = video_params['data_process']
    
    # 设置数据文件列表
    if 'data_process' in params:
        params['data_args']['training_data_files'] = list_videos(params["data_process"]["train"])
        params['data_args']['validation_data_files'] = list_videos(params["data_process"]["vaild"])

    data_args = params['data_args']
    train_args = params['train_args']
    model_args = params['model_args']

    
    # 如果提供了命令行参数，覆盖配置文件中的设备设置
    if device_override is not None:
        train_args['device'] = device_override
    # 若通过环境变量传入 GPU 序号，则覆盖 train_args 中的 gpu_index
    gpu_env = os.environ.get('VQ_VAE_GPU_INDEX')
    if gpu_env is not None:
        train_args['gpu_index'] = gpu_env



    # ========== 初始化日志系统 ==========
    # log_dir要加上原先的json配置文件的名字
    log_dir = train_args['folder_name']+config_path.split('/')[-1].split('.')[0]
    project_name = params.get('project_name', 'dalle_train_vae')
    log_mgr = LogManager(
        log_dir=log_dir,
        use_wandb=params.get('use_wandb', False),
        project_name=project_name,
        config=params,
        resume=train_args['checkpoint_path'] is not None,
        id_name=build_id_name(params)
    )
    logger = log_mgr.logger
    logger.info("=" * 80)
    logger.info("VQ-VAE2 视频训练程序启动")
    logger.info("=" * 80)
    logger.info(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # ========== 记录配置信息 ==========
    # logger.info("【训练配置】")
    # logger.info(f"模型参数: {model_args}")
    # logger.info(f"数据参数: {data_args}")
    # logger.info(f"训练参数: {train_args}")
    # logger.info("")

    # ========== 检测并设置设备 ==========
    # 从配置中获取设备选择，如果没有指定则自动检测
    device = train_args.get('device', 'auto')  # 可选: 'cpu', 'cuda', 'auto'
    preferred_gpu = train_args.get('gpu_index', None)
    if isinstance(preferred_gpu, str) and preferred_gpu.isdigit():
        preferred_gpu = int(preferred_gpu)

    # if device_choice == 'cpu':
    #     device = 'cpu'
    #     logger.info("用户指定使用 CPU")
    # else:
    #     # auto 或 cuda
    #     if torch.cuda.is_available():
    #         device = choose_gpu(logger, preferred=preferred_gpu)
    #         torch.cuda.set_device(device)
    #         logger.info(f"最终使用设备: {device}")
    #         # 记录所选 GPU 信息
    #         idx = int(device.split(':')[-1]) if ':' in device else 0
    #         props = torch.cuda.get_device_properties(idx)
    #         logger.info(f"GPU名称: {props.name}")
    #         logger.info(f"GPU内存: {props.total_memory / 1024**3:.2f} GB")
    #     else:
    #         device = 'cpu'
    #         logger.warning("未检测到可用 GPU，回退到 CPU")
    # logger.info("")

    # ========== 创建模型 ==========
    logger.info("正在创建模型...")
    # 确保 model_args 包含 sequence_length（从 data_args 获取）
    if 'sequence_length' not in model_args:
        model_args['sequence_length'] = data_args['sequence_length']
        logger.info(f"将 sequence_length={data_args['sequence_length']} 添加到 model_args")
    elif model_args['sequence_length'] != data_args['sequence_length']:
        logger.warning(
            f"model_args 中的 sequence_length={model_args['sequence_length']} "
            f"与 data_args 中的 sequence_length={data_args['sequence_length']} 不一致，"
            f"使用 data_args 中的值"
        )
        model_args['sequence_length'] = data_args['sequence_length']
    
    # 使用模型适配器创建模型（适配器会从配置文件读取模型类型）
    model = create_model(model_args=model_args, config_path=config_path)
    model = model.to(device)
    total_params = get_model_size(model)
    logger.info(f"模型创建完成")
    logger.info(f"可训练参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    logger.info("")

    # ========== 分析并保存模型架构 ==========
    logger.info("正在分析模型架构...")
    # 计算输入形状: (batch*sequence, channels, height, width)
    # VQ-VAE2 的输入是 (batch*sequence, channels, height, width)
    batch_size = data_args['batch_size']
    sequence_length = data_args['sequence_length']
    
    # 从不同模型类型的配置中获取 input_channels
    # 对于 taming 模型，从 ddconfig.in_channels 获取
    # 对于其他模型，从 model_args.input_channels 或 model_args.in_channels 获取
    if 'ddconfig' in model_args and 'in_channels' in model_args['ddconfig']:
        # Taming 模型
        input_channels = model_args['ddconfig']['in_channels']
    elif 'input_channels' in model_args:
        # 传统模型（VQVAE2 等）
        input_channels = model_args['input_channels']
    elif 'in_channels' in model_args:
        # 某些模型使用 in_channels
        input_channels = model_args['in_channels']
    else:
        # 默认值
        input_channels = 3
        logger.warning(f"无法从配置中获取 input_channels，使用默认值 {input_channels}")
    
    # 使用原始图像尺寸（下采样在模型内部进行）
    # 通常视频帧的尺寸是 256x256 或 128x128
    # 对于 taming 模型，可以从 ddconfig.resolution 获取
    if 'ddconfig' in model_args and 'resolution' in model_args['ddconfig']:
        estimated_h = estimated_w = model_args['ddconfig']['resolution']
    else:
        estimated_h, estimated_w = 256, 256  # 默认值
    
    # VQ-VAE2 的输入是 (batch*sequence, channels, height, width)
    input_shape = (batch_size * sequence_length, input_channels, estimated_h, estimated_w)
    logger.info(f"用于架构分析的输入形状: {input_shape} (batch*seq={batch_size}*{sequence_length}, channels, h, w)")
    architecture_file = os.path.join(log_dir, 'model_architecture.txt')
    analyze_model_architecture(model, input_shape, architecture_file, logger)
    logger.info("")

    # ========== 创建数据加载器 ==========
    logger.info("正在创建数据加载器...")
    logger.info(f"视频文件数量: {len(data_args['training_data_files'])}")
    logger.info(f"批次大小: {data_args['batch_size']}")
    logger.info(f"序列长度: {data_args['sequence_length']}")
    logger.info(f"工作线程数: {data_args['num_threads']}")
    
    # ========== 自动计算数据集的均值和方差 ==========
    logger.info("正在计算数据集的均值和方差...")
    # try:
    #     # 从配置中获取计算参数，如果没有则使用默认值
    #     normalize_args = data_args.get('normalize_compute_args', {})
    #     max_videos = normalize_args.get('max_videos', 100)  # 限制处理的视频数量以加快速度
    #     max_frames_per_video = normalize_args.get('max_frames_per_video', 50)
    #     sample_frames = normalize_args.get('sample_frames', True)
    #     num_samples = normalize_args.get('num_samples', 10)
        
    #     mean, std = compute_normalize_params(
    #         training_files=data_args['training_data_files'],
    #         validation_files=data_args.get('validation_data_files'),
    #         max_videos=max_videos,
    #         max_frames_per_video=max_frames_per_video,
    #         target_size=(256, 256),
    #         sample_frames=sample_frames,
    #         num_samples=num_samples
    #     )
    #     logger.info(f"计算得到的均值: {mean}")
    #     logger.info(f"计算得到的标准差: {std}")
    # except Exception as e:
    #     logger.warning(f"计算数据集统计信息时出错: {e}")
    #     logger.warning("使用 ImageNet 默认值")
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]


    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = transforms.Normalize(mean=mean, std=std)
    unnormalize = NormalizeInverse(mean=mean, std=std)
    
    training_loader = video_pipe(
        filenames=data_args['training_data_files'],
        batch_size=data_args['batch_size'],
        num_threads=data_args['num_threads'],
        device_id=data_args['device_id'],
        sequence_length=data_args['sequence_length'],
        shard_id=data_args['shard_id'],
        num_shards=data_args['num_shards'],
        initial_prefetch_size=data_args['initial_prefetch_size'],
        seed=data_args['seed'],
        random_shuffle=False,
        config_path=config_path
    )
    validation_loader = None
    if data_args.get('validation_data_files'):
        validation_loader = video_pipe(
            filenames=data_args['validation_data_files'],
            batch_size=data_args['batch_size'],
            num_threads=data_args['num_threads'],
            device_id=data_args['device_id'],
            sequence_length=data_args['sequence_length'],
            shard_id=data_args['shard_id'],
            num_shards=data_args['num_shards'],
            initial_prefetch_size=data_args['initial_prefetch_size'],
            seed=data_args['seed'],
            random_shuffle=False,
            config_path=config_path
        )
    logger.info(f"数据加载器创建完成，每个epoch有 {len(training_loader.loader)} 个批次")
    logger.info("")

    # ========== 创建训练对象并开始训练 ==========
    logger.info("初始化训练对象...")
    # TrainVqVae 会从 param.json 配置文件中读取大部分参数，只需要传递必要的参数
    train_object = TrainVqVae(
        model=model, 
        training_loader=training_loader, 
        validation_loader=validation_loader,
        log_mgr=log_mgr,
        normalize=normalize, 
        unnormalize=unnormalize,
        logger=logger,  # 传递logger,
        default_gpu=device,
        config_path=config_path
    )
    # 确保模型在正确的设备上
    logger.info(f"模型已移动到设备: {next(model.parameters()).device}")
    logger.info("")
    
    try:
        logger.info("开始训练循环...")
        logger.info("")
        train_object.train()
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
        raise
    finally:
        log_mgr.finish()
        logger.info("训练程序结束")


if __name__ == '__main__':
    # 解析命令行参数

    gpu_count = torch.cuda.device_count()
    print(f"✅ 发现 {gpu_count} 个GPU")

    parser = argparse.ArgumentParser(description='VQ-VAE2 视频训练脚本')
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        help='选择训练设备: cpu (强制使用CPU), cuda (强制使用GPU), auto (自动检测，默认)'
    )
    parser.add_argument(
        '--help-device',
        action='store_true',
        help='显示设备选择帮助信息'
    )
    parser.add_argument(
        '--gpu-index',
        type=int,
        default=None,
        help='指定使用的GPU序号，如0或1；不指定则运行时列出可选GPU'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='指定配置文件路径（默认为 param.json，或从环境变量 TRAIN_CONFIG_PATH 读取）'
    )
    
    args = parser.parse_args()
    
    # 确定配置文件路径：命令行参数 > 环境变量 > 默认值
    config_path = args.config or os.environ.get('TRAIN_CONFIG_PATH', 'param.json')
    
    if args.help_device:
        print("=" * 80)
        print("设备选择说明")
        print("=" * 80)
        print("--device 参数选项:")
        print("  cpu  : 强制使用CPU进行训练（即使有GPU可用）")
        print("  cuda : 强制使用GPU进行训练（如果GPU不可用会回退到CPU）")
        print("  auto : 自动检测，如果有GPU则使用GPU，否则使用CPU（默认）")
        print("")
        print("也可以在配置文件的 train_args 中设置 'device' 参数")
        print("=" * 80)
        sys.exit(0)
    
    # 运行训练
    # 将命令行 gpu-index 注入 device_override 逻辑
    if args.gpu_index is not None:
        # 在 train_videos 内部优先从 train_args 读取 gpu_index，这里通过环境变量传递
        os.environ['VQ_VAE_GPU_INDEX'] = str(args.gpu_index)
    train_videos(device_override=args.device, config_path=config_path)
