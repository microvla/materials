# sam3_propagate_with_yolo_prompt.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from sam3.model_builder import build_sam3_video_predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 propagate using YOLO bbox prompts for first N frames (no GUI)"
    )
    parser.add_argument("--img_dir", type=str, default="training_images")
    parser.add_argument("--label_dir", type=str, default="training_labels")
    parser.add_argument("--vis_dir", type=str, default="training_visualization")

    parser.add_argument("--checkpoint", type=str,
                        default=str(Path(__file__).resolve().parent / "checkpoints/sam3.pt"))
    parser.add_argument("--gpus", type=str, default="0")

    parser.add_argument("--n", type=int, default=10,
                        help="Use first N Frames YOLO labels as prompt")

    return parser.parse_args()


# ------------------------- UTILS -----------------------------------
def read_yolo_label(txt_path):
    """read format: class cx cy w h"""
    with open(txt_path, "r") as f:
        line = f.readline().strip().split()
    cls, cx, cy, bw, bh = line
    return float(cx), float(cy), float(bw), float(bh)


def yolo_to_box(yolo, w, h):
    cx, cy, bw, bh = yolo
    x1 = int((cx - bw / 2) * w)
    x2 = int((cx + bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    y2 = int((cy + bh / 2) * h)
    return [x1, y1, x2, y2]


def mask_to_yolo(mask, w, h):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = (x_min + x_max) / 2.0 / w
    cy = (y_min + y_max) / 2.0 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    return cx, cy, bw, bh


def save_yolo(label_dir, img_name, yolo_box):
    base, _ = os.path.splitext(img_name)
    path = os.path.join(label_dir, base + ".txt")
    with open(path, "w") as f:
        cx, cy, bw, bh = yolo_box
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print("[LABEL]", path)


def save_vis(vis_dir, frame_idx, img, mask, yolo_box):
    """Overlay mask and box, write PNG"""
    os.makedirs(vis_dir, exist_ok=True)
    h, w = img.shape[:2]
    img_show = img.copy()

    m = mask.astype(bool)
    img_show[m] = (0.6 * img_show[m] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    cx, cy, bw, bh = yolo_box
    x1 = int((cx - bw / 2) * w)
    x2 = int((cx + bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 0, 255), 2)

    out_path = os.path.join(vis_dir, f"vis_{frame_idx:05d}.png")
    cv2.imwrite(out_path, img_show)
    print("[VIS]", out_path)


# ----------------------- MAIN PIPELINE -----------------------------

def main():
    args = parse_args()

    ROOT = Path(__file__).resolve().parent
    img_dir = os.path.join(ROOT, args.img_dir)
    label_dir = os.path.join(ROOT, args.label_dir)
    vis_dir = os.path.join(ROOT, args.vis_dir)

    img_names = sorted(
        [f for f in os.listdir(img_dir)
         if f.lower().endswith((".jpg", ".png"))]
    )
    if not img_names:
        print("[ERROR] no frames found.")
        sys.exit(1)

    N = min(args.n, len(img_names))
    print(f"[INFO] Using YOLO bboxes for first {N} frames.")

    # ---------------- SAM3 predictor ----------------
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    predictor = build_sam3_video_predictor(
        checkpoint_path=args.checkpoint,
        gpus_to_use=gpu_ids
    )
    print("[INFO] SAM3 predictor loaded")

    # ---------------- Session -----------------------
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=img_dir)
    )
    session_id = response["session_id"]

    predictor.handle_request(
        request=dict(type="reset_session", session_id=session_id)
    )

    # ---------------- Add prompts from YOLO -----------------------
    for frame_idx in range(N):
        img = cv2.imread(os.path.join(img_dir, img_names[frame_idx]))
        h, w = img.shape[:2]

        txt_path = os.path.join(label_dir, img_names[frame_idx].replace(".jpg", ".txt")
                                .replace(".png", ".txt"))

        yolo = read_yolo_label(txt_path)
        box = yolo_to_box(yolo, w, h)

        print(f"[PROMPT] frame {frame_idx}, box={box}")
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                box=box,
                obj_id=1,
            )
        )

    # ---------------- Propagate -----------------------
    print("[SAM3] Propagating ...")
    outputs_per_frame = {}
    for event in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)):
        outputs_per_frame[event["frame_index"]] = event["outputs"]

    # ---------------- Process results -----------------------
    for frame_idx, img_name in enumerate(img_names):
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w = img.shape[:2]

        outputs = outputs_per_frame[frame_idx]
        mask_logits = outputs["mask_logits"]
        mask = (mask_logits[0] > 0).cpu().numpy().squeeze()

        yolo_box = mask_to_yolo(mask, w, h)
        if yolo_box is None:
            print(f"[INFO] frame {frame_idx} no object detected, skip.")
            continue

        save_yolo(label_dir, img_name, yolo_box)
        save_vis(vis_dir, frame_idx, img, mask, yolo_box)

    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )

    print("🎉 DONE: YOLO labels updated & VIS PNG saved.")


if __name__ == "__main__":
    main()
