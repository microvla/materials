# realtime_cam_infer.py
# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from vision.calibration.base import Calib


def main():
    vision_dir = Path(__file__).resolve().parent.parent
    calib_path = vision_dir / "calibration"
    model_path = vision_dir / "yolo" / "models" / "best.pt"

    print("使用模型:", model_path)

    # 加载相机标定
    calibration = Calib(
        kd_path=calib_path / "configs" / "charuco_camera.yaml",
        ext_path=calib_path / "configs" / "plane_extrinsics.yaml"
    )

    device = "mps"  # Mac 硬件加速

    # 加载 YOLO
    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("\n" + ">" * 20 + " START INFERENCE " + "<" * 20)
    print("[info] Device:", device)

    # --------------------------------
    # 主循环
    # --------------------------------
    l_Xw = 0
    l_Yw = 0
    while True:
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        # 去畸变
        undist_frame = cv2.undistort(frame, calibration.K, calibration.D)

        # YOLO 推理
        results = model.predict(
            undist_frame, imgsz=960, device=device, conf=0.5, verbose=False
        )
        annotated_frame = results[0].plot()

        # =======================================
        # 提取检测框并转换为世界坐标
        # =======================================
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                # bbox: x1,y1,x2,y2
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # 中心点像素坐标
                u = int((x1 + x2) / 2)
                v = int((y1 + y2) / 2)
                a = 0.8

                # 像素 → 世界坐标
                world_pos = calibration.pixel_to_world(u, v)
                if world_pos is not None:
                    Xw, Yw = world_pos
                    Xw = a * Xw + (1-a)*l_Xw
                    Yw = a * Yw + (1-a)*l_Yw
                    l_Xw = Xw
                    l_Yw = Yw
                    # 画中心点
                    cv2.circle(annotated_frame, (u, v), 4, (0, 255, 0), -1)
                    # 标注世界坐标
                    cv2.putText(
                        annotated_frame,
                        f"{Xw:.1f}, {Yw:.1f} mm",
                        (u , v + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        # =======================================

        fps = 1 / (time.time() - start)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("YOLO Micro Robot Detection", annotated_frame)

        # time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n>>> END <<<\n")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
