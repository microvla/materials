# realtime_cam_infer_class.py
# -*- coding: utf-8 -*-
import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import sys

ROOT = Path(__file__).resolve().parent.parent
print(ROOT)
sys.path.append(str(ROOT))

from vision.calibration.base import Calib


class Location:
    """
    Streaming + YOLO 
    """

    def __init__(self, visualize=True, device="mps", cap_freq=1):
        """
        :param visualize: 是否显示带标注的视频
        :param device: YOLO 推理设备
        :param cap_freq: 多少帧推理一次
        """
        self.visualize = visualize
        self.cap_freq = cap_freq
        self.device = device
        self.alpha = 0.8 # 平滑滤波系数
        
        self.l_Xw = 0.0
        self.l_Yw = 0.0

        self.frame_id = 0
        self.last_results = None
        
        self.img_size = 640

        # 路径
        vision_dir = ROOT / "vision"
        calib_path = vision_dir / "calibration"
        model_path = vision_dir / "yolo" / "models" / "best_s.pt"

        print("使用模型:", model_path)

        # === 加载标定 ===
        self.calibration = Calib(
            kd_path=calib_path / "configs" / "charuco_camera.yaml",
            ext_path=calib_path / "configs" / "plane_extrinsics.yaml"
        )

        # === 加载YOLO模型 ===
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
            print("[info] Model fused.")
        except:
            print("[info] Fuse not supported, skipping.")

        # === 打开摄像头 ===
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("摄像头读取失败")

        self.h, self.w = frame.shape[:2]
        print(f"[info] Camera resolution: {self.w} x {self.h}")

        # === 畸变预计算 ===
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.calibration.K, self.calibration.D,
            None, self.calibration.K,
            (self.w, self.h),
            cv2.CV_16SC2
        )
        print(">" * 20 + " Ready for Inference " + "<" * 20)


    def get_robot_pos(self):
        """
        返回检测到的机器人 (Xw, Yw) 世界坐标。
        如果未检测到机器人，返回 None。
        """
        start = time.time()

        ret, frame = self.cap.read()
        if not ret:
            return None

        # 畸变校正
        undist = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

        self.frame_id += 1

        # 每 1 帧推理一次
        if self.frame_id % self.cap_freq == 0 or self.last_results is None:
            results = self.model.predict(
                undist,
                imgsz=self.img_size,
                conf=0.5,
                device=self.device,
                verbose=False
            )
            self.last_results = results[0]

        results = self.last_results

        annotated = undist.copy()
        detected_world = None  # 返回给用户的坐标

        # ------------------------------
        # 检测框与坐标转换
        # ------------------------------
        if results is not None and results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                u = (x1 + x2) // 2
                v = (y1 + y2) // 2

                world_pos = self.calibration.pixel_to_world(u, v)
                if world_pos is not None:
                    Xw, Yw = world_pos
                    Xw = self.alpha * Xw + (1 - self.alpha) * self.l_Xw
                    Yw = self.alpha * Yw + (1 - self.alpha) * self.l_Yw
                    self.l_Xw, self.l_Yw = Xw, Yw

                    detected_world = (Xw, Yw)

                    if self.visualize:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(annotated, (u, v), 5, (0, 255, 0), -1)
                        cv2.putText(
                            annotated,
                            f"{Xw:.1f}, {Yw:.1f} mm",
                            (u, v + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

        # FPS
        fps = 1.0 / (time.time() - start)
        if self.visualize:
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("YOLO Micro Robot Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.close()

        return detected_world

    # -----------------------------------------------------
    # 释放资源
    # -----------------------------------------------------
    def close(self):
        print("\n >>> Camera Closed <<< \n")
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()


vision = Location(visualize=True, device="mps", cap_freq=1)
while True:
    try:
        start = time.time()
        pos = vision.get_robot_pos()
        if pos:
            Xw, Yw = pos
            end = time.time()
            print(
                f"机器人世界坐标：\n"
                f" - X = {Xw:.2f} mm\n"
                f" - Y = {Yw:.2f} mm\n"
                f" - Infer: {1/(end-start):.1f} Hz\n"
            )
    except KeyboardInterrupt:
        print("\n >>>END<<<")
        sys.exit()
        