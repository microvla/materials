# realtime_cam_infer.py
# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
print(ROOT)
sys.path.append(str(ROOT))
from vision.calibration.base import Calib

def main():
    vision_dir = Path(__file__).resolve().parent.parent
    calib_path = vision_dir/"calibration"
    model_path = vision_dir/"yolo"/"models"/"best.pt"
    print("model",model_path)
    calibration = Calib(kd_path=calib_path / "configs" / "charuco_camera.yaml",
                        ext_path=calib_path / "configs" / "plane_extrinsics.yaml")
    device = "mps"
    # 加载训练好的权重
    model = YOLO(model_path) 
    model.to(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print(">"*20,"start inferring","<"*20)
    print("[info] Device: ",device)
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break
        # 去畸变
        undist_frame = cv2.undistort(frame, calibration.K, calibration.D)
        # 执行推理
        results = model.predict(undist_frame, imgsz=480, device=device, conf=0.9, verbose=False)
        annotated_frame = results[0].plot()

        fps = 1 / (time.time() - start)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示
        cv2.imshow("YOLO11 Micro Robot Detection", annotated_frame)
        time.sleep(0.01)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n>>>END<<<\n")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
