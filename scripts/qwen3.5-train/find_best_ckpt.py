import os
import json
import argparse
from pathlib import Path

def find_best_checkpoint(base_dir: str) -> str:
    """
    在指定的训练输出根目录中寻找最佳 checkpoint。
    逻辑：
    1. 寻找最新的运行目录 (类似 vX-YYYYMMDD-HHMMSS 或直接传入的运行目录)
    2. 在所有的 checkpoint-xxx 目录中查找 trainer_state.json
    3. 优先读取 trainer_state.json 中的 best_model_checkpoint
    4. 否则读取 log_history 中的最小 eval_loss
    5. 最后降级返回步数最大（最新）的 checkpoint
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return ""

    # 1. 寻找最新的运行目录
    run_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("v")], 
                      key=lambda x: x.stat().st_mtime)
    target_dir = run_dirs[-1] if run_dirs else base_path
    
    # 2. 获取所有的 checkpoint 目录
    ckpts = [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not ckpts:
        return ""

    best_ckpt_path = None
    best_loss = float('inf')

    for ckpt in ckpts:
        state_file = ckpt / "trainer_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # 优先检查是否直接记录了 best_model_checkpoint
                if state.get("best_model_checkpoint"):
                    best_path = Path(state["best_model_checkpoint"])
                    if best_path.exists():
                        return str(best_path.resolve())
                
                # 从 log_history 中读取最新的 eval_loss
                eval_losses = [log["eval_loss"] for log in state.get("log_history", []) if "eval_loss" in log]
                if eval_losses:
                    # 获取此 checkpoint 生成时的最小 eval_loss
                    min_loss = min(eval_losses)
                    if min_loss < best_loss:
                        best_loss = min_loss
                        best_ckpt_path = ckpt
            except Exception:
                continue

    # 如果通过 loss 找到了最佳的 ckpt
    if best_ckpt_path:
        return str(best_ckpt_path.resolve())

    # 3. 降级方案：按步数寻找最新的 checkpoint
    def get_step(p):
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return -1
            
    best_ckpt_path = max(ckpts, key=get_step)
    return str(best_ckpt_path.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动寻找最新且最佳的 Checkpoint")
    parser.add_argument("dir", type=str, help="训练输出的根目录 (例如: outputs/stage1_grounding/qwen3.5)")
    args = parser.parse_args()
    
    best = find_best_checkpoint(args.dir)
    if best:
        print(best)
    else:
        # 没有找到时返回空字符串，或者 exit 1
        exit(1)
