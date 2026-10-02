import cv2
import numpy as np
from pathlib import Path


class Calib:
    def __init__(self, kd_path: Path, ext_path: Path):
        """加载标定数据与外参"""
        self.K, self.D = self._load_camera_parameters(kd_path)
        self.rvec, self.tvec, self.R = self._load_plane_extrinsics(ext_path)

        self.mouse_u = None
        self.mouse_v = None

    # -------------------------------
    # 加载相机标定
    # -------------------------------
    @staticmethod
    def _load_camera_parameters(path: Path):
        if not path.exists():
            print(f"[ERROR] KD 配置文件未找到: {path}")
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("dist_coeffs").mat()
        fs.release()
        return K, D

    # -------------------------------
    # 加载工作平面外参
    # -------------------------------
    @staticmethod
    def _load_plane_extrinsics(path: Path):
        if not path.exists():
            print(f"[ERROR] ext 配置文件未找到: {path}")
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        rvec = fs.getNode("rvec").mat()
        tvec = fs.getNode("tvec").mat()
        R = fs.getNode("R").mat()
        fs.release()
        return rvec, tvec, R

    # -------------------------------
    # 像素 → 世界坐标（mm）
    # -------------------------------
    def pixel_to_world(self, u, v):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        xn = (u - cx) / fx
        yn = (v - cy) / fy
        ray_cam = np.array([[xn], [yn], [1]])

        R_inv = self.R.T
        Cw = -R_inv @ self.tvec
        dw = R_inv @ ray_cam

        if abs(dw[2]) < 1e-6:
            return None

        lam = -float(Cw[2]) / float(dw[2])
        Pw = Cw + lam * dw
        return float(Pw[0]), float(Pw[1])

    # -------------------------------
    # 鼠标事件处理
    # -------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_u, self.mouse_v = x, y


def main():
    current_dir = Path(__file__).resolve().parent
    converter = Calib(
        kd_path=current_dir / "configs" / "charuco_camera.yaml",
        ext_path=current_dir / "configs" / "plane_extrinsics.yaml"
    )

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("plane")
    cv2.setMouseCallback("plane", converter.on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undist = cv2.undistort(frame, converter.K, converter.D)

        if converter.mouse_u is not None:
            Xw, Yw = converter.pixel_to_world(converter.mouse_u, converter.mouse_v)
            cv2.circle(undist, (converter.mouse_u, converter.mouse_v), 5, (0, 255, 0), -1)
            cv2.putText(undist, f"{Xw:.1f}, {Yw:.1f} mm",
                        (converter.mouse_u + 5, converter.mouse_v - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("plane", undist)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
