import cv2
import numpy as np


# ========================================================
# 1. 读取相机内参
# ========================================================
def load_camera_parameters(path="charuco_camera.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open camera file: {path}")
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeffs").mat()
    fs.release()
    return K, D


# ========================================================
# 2. Charuco 板定义（必须与打印一致）
# ========================================================
def build_charuco_board():
    squares_x = 6
    squares_y = 6
    square_length = 15.5   # mm
    marker_length = 11.04   # mm

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_5X5_100
    )
    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=aruco_dict
    )
    return board, aruco_dict


# ========================================================
# 3. 平均 R,t（使用 SVD 使旋转矩阵合法）
# ========================================================
def average_poses(rvecs, tvecs):
    R_acc = np.zeros((3, 3), dtype=np.float64)
    for rvec in rvecs:
        R, _ = cv2.Rodrigues(rvec)
        R_acc += R

    U, S, Vt = np.linalg.svd(R_acc)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    t_stack = np.hstack(tvecs)  # 3×N
    t_avg = np.mean(t_stack, axis=1, keepdims=True)

    rvec_avg, _ = cv2.Rodrigues(R_avg)
    return rvec_avg, t_avg, R_avg


# ========================================================
# 4. Charuco 3D 姿态采集（手动按 s 保存，按 c 求平均）
# ========================================================
def calibrate_plane_extrinsics(
        camera_id=0,
        min_frames=15
):
    K, D = load_camera_parameters()
    board, aruco_dict = build_charuco_board()

    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        cv2.aruco.DetectorParameters()
    )

    rvec_list = []
    tvec_list = []

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("==============================================")
    print("  基于 Charuco 的 3D 姿态标定 + 平面外参求解")
    print("==============================================")
    print("操作说明：")
    print("  * 确保 Charuco 板平放在工作平面上，位置不要动")
    print("  * 相机固定不动")
    print("  * 按 's' 保存当前 R,t")
    print("  * 按 'c' 求平均外参")
    print("  * 按 'q' 退出")
    print("==============================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        undist = cv2.undistort(frame, K, D)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        # 检测 ArUco
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        rvec, tvec = None, None

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(undist, marker_corners, marker_ids)

            # 插值 Charuco 角点
            num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

            if num is not None and num > 10:
                # ⭐ 新 API：必须提供 rvec_guess, tvec_guess
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
                if ok:
                    cv2.drawFrameAxes(undist, K, D, rvec, tvec, 20)
                    cv2.putText(undist, "Pose OK", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                else:
                    cv2.putText(undist, "Pose FAILED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(undist, "Few Charuco corners", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        else:
            cv2.putText(undist, "No markers", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(
            undist,
            f"Saved poses: {len(rvec_list)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )

        cv2.imshow("Plane Extrinsics", undist)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            if rvec is not None:
                rvec_list.append(rvec)
                tvec_list.append(tvec)
                print(f"✓ 保存姿态 {len(rvec_list)}")
            else:
                print("⚠ 当前帧姿态无效，未保存")

        elif key == ord('c'):
            if len(rvec_list) < min_frames:
                print(f"⚠ 姿态不足 {len(rvec_list)}/{min_frames}")
                continue

            print("▶ 计算平均外参 R,t ...")
            rvec_avg, tvec_avg, R_avg = average_poses(rvec_list, tvec_list)

            print("===== 结果 =====")
            print("rvec_avg:\n", rvec_avg)
            print("tvec_avg:\n", tvec_avg)
            print("R_avg:\n", R_avg)

            fs = cv2.FileStorage("plane_extrinsics.yaml", cv2.FILE_STORAGE_WRITE)
            fs.write("rvec", rvec_avg)
            fs.write("tvec", tvec_avg)
            fs.write("R", R_avg)
            fs.release()

            print("📁 已保存到 plane_extrinsics.yaml")

    cap.release()
    cv2.destroyAllWindows()


# ========================================================
# 5. 加载外参
# ========================================================
def load_plane_extrinsics(path="plane_extrinsics.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    R = fs.getNode("R").mat()
    fs.release()
    return rvec, tvec, R


# ========================================================
# 6. 像素 → world(mm)（基于 3D 几何，不是 Homography）
# ========================================================
def pixel_to_world(u, v, K, R, tvec):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    d_cam = np.array([[x_n], [y_n], [1.0]])

    R_inv = R.T
    Cw = -R_inv @ tvec
    dw = R_inv @ d_cam

    n = np.array([[0.0], [0.0], [1.0]])
    denom = float(n.T @ dw)
    if abs(denom) < 1e-8:
        return None

    lam = -float(n.T @ Cw) / denom
    Pw = Cw + lam * dw

    return float(Pw[0]), float(Pw[1])


# ========================================================
# 7. world(mm) → pixel
# ========================================================
def world_to_pixel(X_mm, Y_mm, K, R, tvec):
    Pw = np.array([[X_mm], [Y_mm], [0.0]])
    Pc = R @ Pw + tvec

    x_n = Pc[0,0] / Pc[2,0]
    y_n = Pc[1,0] / Pc[2,0]

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    u = fx * x_n + cx
    v = fy * y_n + cy

    return float(u), float(v)


# ========================================================
# 8. 测试示例
# ========================================================
def demo_convert_one_point():
    K, D = load_camera_parameters()
    rvec, tvec, R = load_plane_extrinsics()

    u, v = 640, 512
    X, Y = pixel_to_world(u, v, K, R, tvec)
    print("Pixel -> World:", X, Y)

    u2, v2 = world_to_pixel(X, Y, K, R, tvec)
    print("World -> Pixel:", u2, v2)


# ========================================================
if __name__ == "__main__":
    calibrate_plane_extrinsics()
    # demo_convert_one_point()
