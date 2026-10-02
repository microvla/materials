import os
import random
import shutil
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # micro 根目录
sys.path.append(str(ROOT))

def create_dataset_path(base_path):
    # 获取目录列表
    items = os.listdir(base_path)

    # 匹配 dataset, dataset1, dataset2...
    pattern = re.compile(r"^dataset(\d*)$")

    numbers = []
    for item in items:
        match = pattern.match(item)
        if match:
            num = match.group(1)
            numbers.append(int(num) if num.isdigit() else 0)  # dataset -> 0

    # 决定新的folder名
    if numbers:
        new_number = max(numbers) + 1
        new_name = f"dataset{new_number}"
    else:
        new_name = "dataset"  # 如果没有则先建 dataset

    path = os.path.join(base_path, new_name)
    return path

# 原始数据
IMG_DIR = "vision/training_images"
LBL_DIR = "vision/training_labels"

# 输出数据集目录
OUT = create_dataset_path(ROOT)

# 创建目录结构
os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/images/val", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True)
os.makedirs(f"{OUT}/labels/val", exist_ok=True)

# 收集全部图片文件名
images = [f for f in os.listdir(IMG_DIR)
          if f.lower().endswith(".jpg") or f.lower().endswith(".png")]

images.sort()
random.seed(0)      # 固定随机种子，方便复现
random.shuffle(images)

val_ratio = 0.2     # 20% 做验证
val_count = int(len(images) * val_ratio)

for i, img_name in enumerate(images):
    base, ext = os.path.splitext(img_name)
    label_name = base + ".txt"

    src_img = os.path.join(IMG_DIR, img_name)
    src_lbl = os.path.join(LBL_DIR, label_name)

    if not os.path.exists(src_lbl):
        print(f"[WARN] label not found for {img_name}, skip")
        continue

    if i < val_count:
        dst_img = os.path.join(OUT, "images/val", img_name)
        dst_lbl = os.path.join(OUT, "labels/val", label_name)
    else:
        dst_img = os.path.join(OUT, "images/train", img_name)
        dst_lbl = os.path.join(OUT, "labels/train", label_name)

    shutil.copy(src_img, dst_img)
    shutil.copy(src_lbl, dst_lbl)

print(">"*3,f"\n[INFO] Done splitting dataset in {OUT}.\n","<"*3)
