import cv2
import time

# ----------- 打开摄像头（建议指定 AVFoundation 后端）-------------
camera_index = 0  # 如果你有多个摄像头，把 0 改为 1 或 2
cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit()

# ----------- 打印摄像头信息 -------------
print("=== 摄像头基本信息 ===")
props = {
    "Frame Width": cv2.CAP_PROP_FRAME_WIDTH,
    "Frame Height": cv2.CAP_PROP_FRAME_HEIGHT,
    "FPS": cv2.CAP_PROP_FPS,
    "Brightness": cv2.CAP_PROP_BRIGHTNESS,
    "Contrast": cv2.CAP_PROP_CONTRAST,
    "Saturation": cv2.CAP_PROP_SATURATION,
    "Hue": cv2.CAP_PROP_HUE,
    "Gain": cv2.CAP_PROP_GAIN,
    "Exposure": cv2.CAP_PROP_EXPOSURE,
    "FourCC": cv2.CAP_PROP_FOURCC,
}

for name, prop in props.items():
    value = cap.get(prop)
    if prop == cv2.CAP_PROP_FOURCC:
        # 转换为 FOURCC 字符串
        fourcc = "".join([chr(int(value) >> 8*i & 0xFF) for i in range(4)])
        print(f"{name}: {fourcc}")
    else:
        print(f"{name}: {value}")

print("\n按 q 退出程序\n")

# ----------- FPS 计算 -------------
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取画面")
        break

    # ---- 计算 FPS ----
    now_time = time.time()
    fps = 1 / (now_time - prev_time)
    prev_time = now_time

    # ---- 将 FPS 显示到画面上 ----
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
