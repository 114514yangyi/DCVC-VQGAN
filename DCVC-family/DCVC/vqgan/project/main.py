import json
import os
import subprocess
from pathlib import Path
import torch


PARAM_PATH = Path("param.json")
MULTITASK_DIR = Path("multitask")
TRAIN_CMD = ["python", "train/train_custom_videos.py"]
DEFAULT_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def load_params(config_path="param.json"):
    """加载配置文件"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到 {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prompt_yes_no(msg: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{msg} {suffix}: ").strip().lower()
    if ans == "":
        return default
    return ans in ("y", "yes")


def choose_mode() -> str:
    print("选择运行方式：")
    print("1) 前台运行（直接显示日志）")
    print("2) 后台静默（nohup 输出到 output.log）")
    choice = input("请输入 1 或 2 [默认2]: ").strip()
    return "foreground" if choice == "1" else "background"


def get_available_tasks():
    """获取可用的任务配置文件"""
    if not MULTITASK_DIR.exists():
        return []
    tasks = []
    # 支持所有 .json 文件（不仅仅是 task*.json）
    for task_file in sorted(MULTITASK_DIR.glob("*.json")):
        # 排除 config.json（配置文件）
        if task_file.name != "config.json":
            tasks.append(task_file.stem)  # 例如 "task1" 或 "taming"
    return tasks


def get_available_gpus():
    """获取可用的GPU数量"""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def load_task_gpu_mapping():
    """从 config.json 读取任务到GPU的映射"""
    config_path = MULTITASK_DIR / "config.json"
    
    if not config_path.exists():
        print(f"未找到配置文件: {config_path}")
        print("请创建 multitask/config.json 文件，格式如下：")
        print('{"taming.json": 3, "task1.json": 0, "task2.json": -1}')
        return None
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        return None
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return None
    
    gpu_count = get_available_gpus()
    if gpu_count == 0:
        print("未检测到可用的GPU")
        return None
    
    task_gpu_map = {}
    skipped_tasks = []
    invalid_tasks = []
    
    for task_file, gpu_idx in config.items():
        # 移除 .json 后缀获取任务名
        task_name = task_file.replace(".json", "")
        
        if gpu_idx == -1:
            # -1 表示跳过该任务
            skipped_tasks.append(task_name)
            continue
        elif isinstance(gpu_idx, int) and 0 <= gpu_idx < gpu_count:
            task_gpu_map[task_name] = f"cuda:{gpu_idx}"
        else:
            invalid_tasks.append((task_name, gpu_idx))
    
    # 显示配置信息
    if skipped_tasks:
        print(f"跳过的任务: {', '.join(skipped_tasks)}")
    if invalid_tasks:
        print("⚠️  无效的GPU配置:")
        for task, gpu in invalid_tasks:
            print(f"  {task}: GPU {gpu} (有效范围: 0-{gpu_count-1} 或 -1)")
    
    if not task_gpu_map:
        print("没有有效的任务需要启动")
        return None
    
    return task_gpu_map


def run_single_task(task_name: str, gpu: str, mode: str):
    """运行单个训练任务"""
    config_path = MULTITASK_DIR / f"{task_name}.json"
    log_file = f"output_{task_name}_{gpu.replace(':', '_')}.log"
    
    # 使用 --config 参数传递配置文件路径，--device 参数传递GPU
    cmd = TRAIN_CMD + ["--device", gpu, "--config", str(config_path)]
    
    env = os.environ.copy()
    env.update(DEFAULT_ENV)
    
    if mode == "foreground":
        print(f"前台启动 {task_name} 在 {gpu}...")
        subprocess.run(cmd, env=env)
    else:
        print(f"后台启动 {task_name} 在 {gpu}，日志输出到 {log_file}")
        with open(log_file, "w") as f:
            subprocess.Popen(
                ["nohup"] + cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setpgrp
            )


def run_multitask_mode():
    """多任务模式：同时启动多个训练任务（从 config.json 读取配置）"""
    print(f"\n从 {MULTITASK_DIR}/config.json 读取任务配置...")
    task_gpu_map = load_task_gpu_mapping()
    if not task_gpu_map:
        print("没有任务需要启动（所有任务都被跳过或没有可用任务）")
        return
    
    print("\n任务分配（从 config.json 读取）：")
    for task, gpu in task_gpu_map.items():
        print(f"  {task} -> {gpu}")
    
    if not prompt_yes_no("\n确认启动这些任务？", default=True):
        print("已取消")
        return
    
    mode = choose_mode()
    
    # 启动所有任务
    for task, gpu in task_gpu_map.items():
        run_single_task(task, gpu, mode)
    
    if mode == "background":
        print("\n所有任务已在后台启动")
        print("可使用以下命令查看日志：")
        for task, gpu in task_gpu_map.items():
            log_file = f"output_{task}_{gpu.replace(':', '_')}.log"
            print(f"  tail -f {log_file}  # {task} 在 {gpu}")


def run_single_task_mode():
    """单任务模式：使用 param.json"""
    params = load_params()
    print("当前参数 (param.json)：")
    print(json.dumps(params, indent=2, ensure_ascii=False))

    if prompt_yes_no("是否修改 param.json 后再继续？", default=False):
        print("请修改 param.json 后重新运行本程序。")
        return

    mode = choose_mode()

    env = os.environ.copy()
    env.update(DEFAULT_ENV)

    if mode == "foreground":
        print("前台启动训练...")
        subprocess.run(TRAIN_CMD, env=env)
    else:
        with open("param.json", "r") as t:
            params = json.load(t)
        default_gpu = params.get('default_gpu', 'cuda:0')
        if isinstance(default_gpu, int):
            default_gpu = f"cuda:{default_gpu}"
        log_file = f"output_{default_gpu.replace(':', '_')}.log"
        print(f"后台启动训练，日志输出到 {log_file}")
        with open(log_file, "w") as f:
            subprocess.Popen(
                ["nohup"] + TRAIN_CMD,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setpgrp
            )
        print("已启动后台进程，可使用 tail -f output.log 查看日志。")


def main():
    print("=" * 80)
    print("VQ-VAE 训练启动器")
    print("=" * 80)
    
    gpu_count = get_available_gpus()
    if gpu_count > 0:
        print(f"✅ 发现 {gpu_count} 个GPU")
    else:
        print("⚠️  未检测到GPU，将使用CPU")
    
    print("\n选择运行模式：")
    print("1) 单任务模式（使用 param.json）")
    print("2) 多任务模式（使用 multitask/ 中的配置文件，可在不同GPU上并行运行）")
    
    choice = input("请输入 1 或 2 [默认1]: ").strip()
    
    if choice == "2":
        run_multitask_mode()
    else:
        run_single_task_mode()


if __name__ == "__main__":
    main()

