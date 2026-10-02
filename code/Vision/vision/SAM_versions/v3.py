# sam_yolo_annotate_from_folder.py
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # micro root
sys.path.append(str(ROOT))

import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# ===================== PATH CONFIG =====================

TRAIN_IMG_DIR = "training_images"         # input images
TRAIN_LABEL_DIR = "training_labels"       # YOLO *.txt output
TRAIN_VIS_DIR = "training_visualization"  # visualization images

os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(TRAIN_VIS_DIR, exist_ok=True)

# ===================== DEVICE & MODEL =====================

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

# ===================== GLOBAL STATE =====================

drawing = False
start_x, start_y = -1, -1
bbox = None
current_img = None

# ===================== MOUSE DRAW CALLBACK =====================

def draw_bbox(event, x, y, flags, param):
    global drawing, start_x, start_y, bbox, current_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = current_img.copy()
        cv2.rectangle(tmp, (start_x, start_y), (x, y), (0,255,0), 2)
        cv2.imshow("Draw BBOX", tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, x_max = sorted([start_x, x])
        y_min, y_max = sorted([start_y, y])
        bbox = (x_min, y_min, x_max, y_max)
        tmp = current_img.copy()
        cv2.rectangle(tmp, (x_min,y_min),(x_max,y_max),(0,255,0), 2)
        cv2.imshow("Draw BBOX", tmp)
        print(f"[INFO] BBOX selected: {bbox}")

# ===================== MULTI FRAME MANUAL ANNOTATION =====================

def manual_multi_frame_annotation(img_names):
    global bbox, current_img
    bboxes = {}

    idx = 0
    while idx < len(img_names):
        img_path = os.path.join(TRAIN_IMG_DIR, img_names[idx])
        current_img = cv2.imread(img_path)
        bbox = None

        cv2.namedWindow("Draw BBOX")
        cv2.setMouseCallback("Draw BBOX", draw_bbox)
        print(f"👉 Frame {idx}/{len(img_names)-1} - 标注 bbox, 按 s 保存, 按 n 跳过, 按 q 完成")
        cv2.imshow("Draw BBOX", current_img)

        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord("s"):
                if bbox is not None:
                    bboxes[idx] = bbox
                    print(f"[SAVE] frame {idx} → {bbox}")
                cv2.destroyAllWindows()
                break
            elif key == ord("n"):
                cv2.destroyAllWindows()
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return bboxes

        idx += 1

    return bboxes

# ===================== SAM PROPAGATION (multi-prompt) =====================

def run_sam_with_multi_prompts(frame_dir, bboxes):
    frame_names = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith(".jpg") or f.lower().endswith(".png")
    ])

    inference_state = predictor.init_state(video_path=frame_dir)
    predictor.reset_state(inference_state)

    with torch.inference_mode():
        for frame_idx, box in bboxes.items():
            box = np.array(box, dtype=np.float32)
            _, obj_ids, mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                box=box,
            )
            print(f"[PROMPT] added bbox at frame {frame_idx}")

        video_masks = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            video_masks[frame_idx] = {
                oid:(mask_logits[i] > 0.0).cpu().numpy()
                for i, oid in enumerate(obj_ids)
            }

    print("[INFO] done propagate with multi prompts!")
    return frame_names, video_masks

# ===================== YOLO BBOX CONVERSION =====================

def mask_to_yolo(mask, w, h):
    mask = np.squeeze(mask)

    # force correct shape
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        print("[WARN] invalid mask:", mask.shape)
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx = (x_min + x_max) / 2.0 / w
    cy = (y_min + y_max) / 2.0 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    return cx, cy, bw, bh

def save_yolo_labels(label_dir, img_name, class_id, yolo_boxes):
    base, _ = os.path.splitext(img_name)
    label_path = os.path.join(label_dir, base + ".txt")
    with open(label_path, "w") as f:
        for cx, cy, bw, bh in yolo_boxes:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"[LABEL] saved → {label_path}")

# ===================== VISUALIZATION =====================

def save_visualization_images(img_dir, vis_dir, frame_names, video_masks):
    for idx, img_name in enumerate(frame_names):
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w = img.shape[:2]

        if idx in video_masks:
            for _, mask in video_masks[idx].items():
                mask = np.squeeze(mask)

                if mask.ndim == 3:
                    mask = mask[0]
                if mask.ndim != 2:
                    print("[WARN] mask shape mismatch:", mask.shape)
                    continue

                if mask.shape != (h, w):
                    print("[WARN] skipping mask mismatch", mask.shape, (h, w))
                    continue

                m = mask.astype(bool)
                img[m] = (0.6 * img[m] + 0.4 * np.array([0,255,0])).astype(np.uint8)

                yolo = mask_to_yolo(mask, w, h)
                if yolo:
                    cx,cy,bw,bh = yolo
                    x1 = int((cx-bw/2)*w)
                    x2 = int((cx+bw/2)*w)
                    y1 = int((cy-bh/2)*h)
                    y2 = int((cy+bh/2)*h)
                    cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255), 2)

        out_path = os.path.join(vis_dir, img_name.replace(".jpg","_vis.jpg"))
        cv2.imwrite(out_path, img)
        print(f"[VIS] {out_path}")

# ===================== MAIN =====================

def main():
    img_names = sorted([
        f for f in os.listdir(TRAIN_IMG_DIR)
        if f.lower().endswith(".jpg") or f.lower().endswith(".png")
    ])

    print("[STEP] 多帧人工标注开始")
    bboxes = manual_multi_frame_annotation(img_names)
    print("[INFO] prompts:", bboxes)

    print("[STEP] SAM propagate 开始")
    frame_names, video_masks = run_sam_with_multi_prompts(TRAIN_IMG_DIR, bboxes)

    print("[STEP] YOLO 输出")
    for idx, img_name in enumerate(frame_names):
        if idx not in video_masks: continue
        img = cv2.imread(os.path.join(TRAIN_IMG_DIR, img_name))
        h, w = img.shape[:2]

        yolo_boxes = []
        for _, mask in video_masks[idx].items():
            yolo = mask_to_yolo(mask, w, h)
            if yolo: yolo_boxes.append(yolo)

        if yolo_boxes:
            save_yolo_labels(TRAIN_LABEL_DIR, img_name, 0, yolo_boxes)

    print("[STEP] 保存可视化")
    save_visualization_images(TRAIN_IMG_DIR, TRAIN_VIS_DIR, frame_names, video_masks)

    print("🎉 ALL DONE!")


if __name__ == "__main__":
    main()
