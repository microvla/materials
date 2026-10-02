# sam_yolo_interactive.py
# -*- coding: utf-8 -*-
# c = continue auto propagate
# e = edit current frame bbox
# n = skip frame
# s = save edited bbox
# q = quit

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

TRAIN_IMG_DIR = "./vision/training_images"
TRAIN_LABEL_DIR = "./vision/training_labels"
TRAIN_VIS_DIR = "./vision/training_visualization"
SAM_CHECKPOINT_DIR = "./EdgeTAM/checkpoints/edgetam.pt"

os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
os.makedirs(TRAIN_VIS_DIR, exist_ok=True)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print("[INFO] Python working in: ", Path(__file__).resolve().parent)
print("[INFO] TRAIN_IMG_DIR       =", Path(TRAIN_IMG_DIR).resolve())
print("[INFO] TRAIN_LABEL_DIR     =", Path(TRAIN_LABEL_DIR).resolve())
print("[INFO] TRAIN_VIS_DIR       =", Path(TRAIN_VIS_DIR).resolve())
print("[INFO] SAM_CHECKPOINT_DIR  =", Path(SAM_CHECKPOINT_DIR).resolve())

# device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[INFO] using device: {device}")
print("\n>>>SAM INIT<<\n")
predictor = build_sam2_video_predictor(
    "edgetam.yaml",
    SAM_CHECKPOINT_DIR,
    device=device
)

drawing = False
start_x, start_y = -1, -1
edit_bbox = None
current_img = None

# ---------------------- mouse draw ------------------------------------
def draw_bbox(event, x, y, flags, param):
    global drawing, start_x, start_y, edit_bbox, current_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        tmp = current_img.copy()
        cv2.rectangle(tmp, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv2.imshow("Annotate / Edit", tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_min, x_max = sorted([start_x, x])
        y_min, y_max = sorted([start_y, y])
        edit_bbox = (x_min, y_min, x_max, y_max)
        tmp = current_img.copy()
        cv2.rectangle(tmp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("Annotate / Edit", tmp)
        print("[EDIT] new bbox:", edit_bbox)


# ---------------------- utils ------------------------------------
def mask_to_yolo(mask, w, h):
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        return None

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


def save_yolo(label_dir, img_name, boxes):
    base, _ = os.path.splitext(img_name)
    path = os.path.join(label_dir, base + ".txt")
    with open(path, "w") as f:
        for cx, cy, bw, bh in boxes:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print("[LABEL]", path)


def visualize(video_masks, img_names, img_dir=TRAIN_IMG_DIR, save_dir=TRAIN_VIS_DIR):
    os.makedirs(save_dir, exist_ok=True)

    for frame_idx, mask in video_masks.items():
        img_name = img_names[frame_idx]
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        m = mask.astype(bool)
        vis = img.copy()
        vis[m] = (0.6 * vis[m] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

        yolo = mask_to_yolo(mask, w, h)
        if yolo is not None:
            cx, cy, bw, bh = yolo
            x1 = int((cx - bw / 2) * w)
            x2 = int((cx + bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        base, _ = os.path.splitext(img_name)
        out_path = os.path.join(save_dir, base + ".jpg")
        cv2.imwrite(out_path, vis)
        print("[VIS]", out_path)


# ---------------------- propagation ------------------------------------
def propagate_with_interactive(img_names, initial_bbox):
    inference_state = predictor.init_state(video_path=TRAIN_IMG_DIR)
    predictor.reset_state(inference_state)

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        box=np.array(initial_bbox, np.float32)
    )

    video_masks = {}

    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
        
        img_path = os.path.join(TRAIN_IMG_DIR, img_names[frame_idx])
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        mask = (mask_logits[0] > 0).cpu().numpy()
        mask = np.squeeze(mask)
        m = mask.astype(bool)

        img_show = img.copy()
        img_show[m] = (0.6 * img_show[m] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

        yolo = mask_to_yolo(mask, w, h)
        if yolo is not None:
            cx, cy, bw, bh = yolo
            x1 = int((cx - bw / 2) * w)
            x2 = int((cx + bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
        img_show,               # 图像
        "Press [e] to refine the result",
        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1, cv2.LINE_AA)
        cv2.putText(
        img_show,               # 图像
        "Press [s] to save and continue",
        (50, 85), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1, cv2.LINE_AA)
        cv2.imshow("Propagating", img_show)
        
        key = cv2.waitKey(0) & 0xFF
        while key not in (ord('e'),ord('s'),ord('q'),ord('n')):
            key = cv2.waitKey(0) & 0xFF
            print("[ERROR] Wrong input, Retry.")
        if key == ord('e'):
            print("[EDIT] enter edit mode")
            global edit_bbox, current_img
            current_img = img.copy()
            cv2.namedWindow("Annotate / Edit")
            cv2.putText(
            current_img,
            "Press [s] to save the refinement",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1.0,(255, 255, 255),2, cv2.LINE_AA)
            cv2.setMouseCallback("Annotate / Edit", draw_bbox)
            cv2.imshow("Annotate / Edit", current_img)
            edit_bbox = None
            h, w = img.shape[:2]
            while True:
                k = cv2.waitKey(10) & 0xFF
                if k == ord('s') and edit_bbox is not None:
                    # 1) 用编辑后的框更新 predictor 状态（影响后续 propagate）
                    predictor.add_new_points_or_box(
                        inference_state, frame_idx, 1,
                        box=np.array(edit_bbox, np.float32)
                    )
                    print("[UPDATE] Correction applied")

                    # 2) 将编辑框转换成 YOLO 标注格式
                    x_min, y_min, x_max, y_max = edit_bbox
                    cx = (x_min + x_max) / 2.0 / w
                    cy = (y_min + y_max) / 2.0 / h
                    bw = (x_max - x_min) / w
                    bh = (y_max - y_min) / h
                    yolo = (cx, cy, bw, bh)
                    # 3) 保存当前帧标签（用新 bbox 覆盖旧的）
                    print("[UPDATE] Overwright old label")
                    save_yolo(TRAIN_LABEL_DIR, img_names[frame_idx], [yolo])

                    # 4) 同时更新 video_masks（用于可视化）
                    corrected_mask = np.zeros((h, w), dtype=np.uint8)
                    corrected_mask[y_min:y_max, x_min:x_max] = 1
                    video_masks[frame_idx] = corrected_mask

                    cv2.destroyWindow("Annotate / Edit")
                    break

        elif key == ord('n'):
            print("[SKIP] frame", frame_idx)
            continue

        elif key == ord('q'):
            print("[QUIT] stop interactive labeling")
            break

        if yolo is not None and key == ord('s'):
            save_yolo(TRAIN_LABEL_DIR, img_names[frame_idx], [yolo])

        video_masks[frame_idx] = mask

    cv2.destroyAllWindows()
    return video_masks


# ---------------------- main ------------------------------------
def main():
    img_names = sorted([f for f in os.listdir(TRAIN_IMG_DIR)
                        if f.lower().endswith(".jpg") or f.lower().endswith(".png")])

    print("# c = continue auto propagate")
    print("# e = edit current frame bbox")
    print("# n = skip frame")
    print("# s = save edited bbox")
    print("# q = quit")

    global current_img, edit_bbox
    current_img = cv2.imread(os.path.join(TRAIN_IMG_DIR, img_names[0]))
    cv2.namedWindow("Annotate / Edit")
    cv2.putText(
    current_img,               # 图像
    "Manually lable the first robot loc. Press [s] to continue",            # 显示内容
    (50, 50),                  # 坐标（左上角位置）
    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
    1.0,                       # 字号
    (255, 255, 255),               # 颜色（B,G,R）
    2,                         # 线宽
    cv2.LINE_AA                # 抗锯齿
    )
    cv2.setMouseCallback("Annotate / Edit", draw_bbox)
    cv2.imshow("Annotate / Edit", current_img)

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s') and edit_bbox is not None:
            break
        if key == ord('q'):
            return
    cv2.destroyAllWindows()

    video_masks = propagate_with_interactive(img_names, edit_bbox)
    visualize(video_masks, img_names)

    print("[FINISH] 可视化已保存到:", TRAIN_VIS_DIR)


if __name__ == "__main__":
    main()
