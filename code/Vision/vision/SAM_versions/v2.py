# sam_yolo_annotate_from_folder.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # micro 根目录
sys.path.append(str(ROOT))

import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# ===================== 配置 =====================

TRAIN_IMG_DIR = "training_images"         # 输入图片目录
TRAIN_LABEL_DIR = "training_labels"       # 保存 YOLO txt
TRAIN_VIS_DIR = "training_visualization"  # 保存可视化图片

os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(TRAIN_VIS_DIR, exist_ok=True)

# ===================== device & SAM init =====================

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[INFO] using device: {device}")

checkpoint = str(ROOT / "EdgeTAM/checkpoints/edgetam.pt")
model_cfg = "edgetam.yaml"
print("[INFO] loading EdgeTAM/SAM predictor ...")
predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
print("[INFO] predictor loaded.")

# ===================== 鼠标框选 =====================

drawing = False
start_x, start_y = -1, -1
bbox = None
first_img = None

def draw_bbox(event, x, y, flags, param):
    global drawing, start_x, start_y, bbox, first_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = first_img.copy()
        cv2.rectangle(tmp, (start_x, start_y), (x, y), (0,255,0), 2)
        cv2.imshow("Draw BBOX on first image", tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, x_max = sorted([start_x, x])
        y_min, y_max = sorted([start_y, y])
        bbox = (x_min, y_min, x_max, y_max)
        tmp = first_img.copy()
        cv2.rectangle(tmp, (x_min,y_min),(x_max,y_max),(0,255,0),2)
        cv2.imshow("Draw BBOX on first image", tmp)
        print(f"[INFO] BBOX selected: {bbox}")

# ===================== SAM propagate =====================

def run_sam_on_images(frame_dir, bbox):
    frame_names = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".png")
    ])

    inference_state = predictor.init_state(video_path=frame_dir)
    predictor.reset_state(inference_state)

    box = np.array(bbox, dtype=np.float32)
    with torch.inference_mode():
        _, obj_ids, mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=box,
        )

        video_masks = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            video_masks[frame_idx] = {
                oid:(mask_logits[i] > 0.0).cpu().numpy()
                for i,oid in enumerate(obj_ids)
            }

    print("[INFO] done propagate")
    return frame_names, video_masks

# ===================== YOLO bbox =====================

def mask_to_yolo(mask, w, h):
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        print(f"[WARN] invalid mask shape {mask.shape}")
        return None
    ys, xs = np.where(mask > 0)
    if len(xs)==0: return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = (x_min+x_max)/2.0/w
    cy = (y_min+y_max)/2.0/h
    bw = (x_max-x_min)/w
    bh = (y_max-y_min)/h
    return cx, cy, bw, bh

def save_yolo_labels(label_dir, img_name, class_id, yolo_boxes):
    base, _ = os.path.splitext(img_name)
    label_path = os.path.join(label_dir, base + ".txt")
    with open(label_path, "w") as f:
        for cx,cy,bw,bh in yolo_boxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"[LABEL] {label_path}")

# ===================== 保存可视化图片 =====================

def save_visualization_images(img_dir, vis_dir, frame_names, video_masks):
    for idx, img_name in enumerate(frame_names):
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w = img.shape[:2]

        if idx in video_masks:
            for obj_id, mask in video_masks[idx].items():
                mask = np.squeeze(mask)
                if mask.ndim != 2: continue
                m = mask.astype(bool)

                color = np.array([0,255,0], np.uint8)
                img[m] = (0.6*img[m] + 0.4*color).astype(np.uint8)

                yolo = mask_to_yolo(mask, w, h)
                if yolo:
                    cx,cy,bw,bh = yolo
                    x1 = int((cx-bw/2)*w)
                    y1 = int((cy-bh/2)*h)
                    x2 = int((cx+bw/2)*w)
                    y2 = int((cy+bh/2)*h)
                    cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255), 2)

        out_path = os.path.join(vis_dir, img_name.replace(".jpg","_vis.jpg"))
        cv2.imwrite(out_path, img)
        print("[VIS]", out_path)



# ===================== main =====================

def main():
    global first_img, bbox

    img_names = sorted([
        f for f in os.listdir(TRAIN_IMG_DIR)
        if f.lower().endswith(".jpg") or f.lower().endswith(".png")
    ])
    first_img_path = os.path.join(TRAIN_IMG_DIR, img_names[0])
    first_img = cv2.imread(first_img_path)

    cv2.namedWindow("Draw BBOX on first image")
    cv2.setMouseCallback("Draw BBOX on first image", draw_bbox)
    print("👉 请用鼠标框选机器人，按 s 确认")
    cv2.imshow("Draw BBOX on first image", first_img)
    while True:
        key = cv2.waitKey(10)&0xFF
        if key==ord('q'): return
        if key==ord('s') and bbox is not None:
            cv2.destroyAllWindows()
            break

    frame_names, video_masks = run_sam_on_images(TRAIN_IMG_DIR, bbox)

    for idx, img_name in enumerate(frame_names):
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if idx not in video_masks: continue

        yolo_boxes=[]
        for obj_id, mask in video_masks[idx].items():
            yolo = mask_to_yolo(mask, w, h)
            if yolo: yolo_boxes.append(yolo)

        if yolo_boxes:
            save_yolo_labels(TRAIN_LABEL_DIR, img_name, class_id=0, yolo_boxes=yolo_boxes)

    save_visualization_images(TRAIN_IMG_DIR, TRAIN_VIS_DIR, frame_names, video_masks)


if __name__ == "__main__":
    main()
