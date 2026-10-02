import cv2
import numpy as np

# -----------------------------
# 读取相机内参
# -----------------------------
def load_camera_parameters(path="charuco_camera.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open {path}")
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeffs").mat()
    fs.release()
    return K, D

# -----------------------------
# 构造你的 Charuco 板
# -----------------------------
def build_charuco_board():
    squares_x = 6
    squares_y = 6
    square_length = 16.0   # mm
    marker_length = 12.0   # mm

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=aruco_dict
    )
    return board, aruco_dict


# -----------------------------
# 计算一张图的重投影 RMS
# -----------------------------
def compute_rms_one_frame(gray, board, K, D, detector):
    # 检测 ArUco
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return None

    # 插值 Charuco
    num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if num is None or num < 10:
        return None

    # 姿态估计（新 API 要传 rvec,tvec）
    rvec_guess = np.zeros((3,1), dtype=np.float32)
    tvec_guess = np.zeros((3,1), dtype=np.float32)
    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        K,
        D,
        rvec_guess,
        tvec_guess,
        False
    )
    if not ok:
        return None

    # 构建对应的 3D 棋盘点（Z=0）
    obj_points_all = board.getChessboardCorners()  # list of Point3f
    obj_points = []
    img_points = []

    for i in range(len(charuco_ids)):
        idx = int(charuco_ids[i][0])
        obj_points.append(obj_points_all[idx])
        img_points.append(charuco_corners[i][0])

    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    # 用当前 K,D,rvec,tvec 投影回去
    proj, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)

    # 计算重投影误差
    diff = img_points - proj
    err = np.linalg.norm(diff, axis=1)  # 每个点的像素误差
    rms = np.sqrt(np.mean(err**2))
    return rms


def main():
    K, D = load_camera_parameters()
    board, aruco_dict = build_charuco_board()
    detector = cv2.aruco.ArucoDetector(
        aruco_dict, cv2.aruco.DetectorParameters()
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头打不开")
        return

    print("=======================================")
    print(" 标定健康检查 Demo")
    print(" 对着 Charuco 棋盘，按 'c' 检查一次")
    print(" 按 'q' 退出")
    print("=======================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 用标定参数先做去畸变
        undist = cv2.undistort(frame, K, D)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        cv2.imshow("health_check", undist)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            rms = compute_rms_one_frame(gray, board, K, D, detector)
            if rms is None:
                print("⚠ 没检测到足够的 Charuco 角点，换个角度再试")
            else:
                print(f"当前 RMS 重投影误差: {rms:.3f} 像素")
                # 你可以根据自己系统精度需要调整这个阈值
                if rms < 0.3:
                    print("✅ 标定状态良好，不需要重标定")
                elif rms < 0.7:
                    print("🟡 勉强可用，如果要做高精度测量建议重标定")
                else:
                    print("❌ 误差很大，建议重新做相机标定（K,D）")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
