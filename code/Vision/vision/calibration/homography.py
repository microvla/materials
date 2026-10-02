import cv2
import numpy as np


# --------------------------------------------------------------
# 1. 你的相机标定文件（charuco_camera.yaml）
# --------------------------------------------------------------
def load_camera_parameters(path="charuco_camera.yaml"):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeffs").mat()
    return K, D


# --------------------------------------------------------------
# 2. 构建与你打印的一样的 Charuco 板
# --------------------------------------------------------------
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


# --------------------------------------------------------------
# 3. Homography 标定主程序
# --------------------------------------------------------------
def calibrate_homography(camera_id=0):
    K, D = load_camera_parameters()
    board, aruco_dict = build_charuco_board()

    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        cv2.aruco.DetectorParameters()
    )

    print("============================================")
    print(" 🔧 工作平面 Homography 标定（新 API）")
    print("============================================")
    print("操作说明：")
    print("  1. 将 Charuco 板平放在机器人工作平面上")
    print("  2. 按 's' 拍一张用于标定")
    print("  3. 按 'q' 退出")
    print("--------------------------------------------")

    cap = cv2.VideoCapture(camera_id)

    H = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        undist = cv2.undistort(frame, K, D)
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

        # 寻找 ArUco
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(undist, marker_corners, marker_ids)

            # 插值 Charuco 角点
            count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=gray,
                board=board
            )

            if count > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    undist, charuco_corners, charuco_ids
                )

        cv2.imshow("Homography Calibration", undist)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            if count < 10:
                print("⚠ Charuco 点太少，无法标定")
                continue

            print("✓ 收集到 Charuco 点，开始求 Homography...")

            # 2D 像素坐标
            img_pts = charuco_corners.reshape(-1, 2)

            # 2D 世界平面坐标 (单位：mm)
            obj_pts = []
            chessboard = board.getChessboardCorners()  # 3D 角点（Z=0）

            for cid in charuco_ids.flatten():
                obj_pts.append(chessboard[cid][:2])

            obj_pts = np.array(obj_pts, dtype=np.float32)

            # 求 Homography
            H, mask = cv2.findHomography(img_pts, obj_pts, cv2.RANSAC)

            if H is None:
                print("❌ Homography 求解失败")
                continue

            print("✓ Homography 求解成功")
            print(H)

            # 保存
            fs = cv2.FileStorage("homography.yaml", cv2.FILE_STORAGE_WRITE)
            fs.write("H", H)
            fs.release()

            print("\n📁 已保存 H → homography.yaml")
            print("============================================\n")

    cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------
# 4. Pixel ↔ mm 转换函数（你后面会用）
# --------------------------------------------------------------
def pixel_to_mm(u, v, H):
    p = np.array([u, v, 1.0])
    world = H @ p
    world /= world[2]
    return float(world[0]), float(world[1])  # Xmm, Ymm


def mm_to_pixel(x, y, H):
    Hinv = np.linalg.inv(H)
    p = np.array([x, y, 1.0])
    img = Hinv @ p
    img /= img[2]
    return float(img[0]), float(img[1])


# --------------------------------------------------------------
# main
# --------------------------------------------------------------
if __name__ == "__main__":
    calibrate_homography()
