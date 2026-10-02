# train_yolo11_robot.py
# -*- coding: utf-8 -*-
"""
Train YOLO11 model for Micro Robot Detection
Author: MX
"""

import argparse
import os
from pathlib import Path

def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train the MicroVLA YOLO11 detector")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "micro.yaml")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parent / "models/yolo11m.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    from ultralytics import YOLO
    print("====================================")
    print("   YOLO11 Micro-Robot Training")
    print("====================================")
    cur_path = Path(__file__).resolve().parent
    device = args.device or get_device()
    print(f"[INFO] Using device: {device}")

    # ==========================
    # 加载 YOLO11 预训练模型
    # ==========================
    print("[INFO] Loading YOLO11m pretrained model ...")
    model_path = args.model
    model = YOLO(model_path)
    print("[INFO] Model loaded successfully")
    config_path = args.data
    # ==========================
    # 开始训练
    # ==========================
    model.train(
        data= config_path,    # 数据配置
        epochs=100,              # 训练轮数
        imgsz=960,               # 输入图尺寸（提高小目标检测效果）
        batch=8,                 # 批大小
        device=device,           # 使用 GPU/MPS/CPU
        workers=4,               # 数据加载线程
        optimizer="Adam",         # 优化器，可改 Adam
        patience=20,             # 早停 patience
        save=True,               # 保存模型
        name="robot_yolo11_train" # 输出目录 runs/detect/robot_yolo11_train
    )

    print("\n====================================")
    print("训练结束！结果保存在：")
    print("runs/detect/robot_yolo11_train/")
    print("最重要的文件：weights/best.pt (最佳检测模型)")
    print("====================================\n")


if __name__ == "__main__":
    main()
