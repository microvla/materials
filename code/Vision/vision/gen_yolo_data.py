# Generate the training dataset and labels of MICRO robot for location.
import cv2
import os
from pathlib import Path
from calibration.base import Calib
import os
current_dir = Path(__file__).resolve().parent
print("Python working in path:", current_dir)

IMG_DIR = os.path.join(current_dir,"training_images")
Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
last_idx = max([int(os.path.splitext(f)[0]) for f in os.listdir(IMG_DIR) if os.path.splitext(f)[0].isdigit()], default=-1)
frame_id = last_idx + 1

SAVE_DIR = os.path.join(current_dir,"training_images")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

calib = Calib(
    kd_path=current_dir/"calibration"/"configs"/"charuco_camera.yaml",
    ext_path=current_dir/"calibration"/"configs"/"plane_extrinsics.yaml"
)

cap = cv2.VideoCapture(0)
print("Camera opened. Press 'S' to save frame, 'Q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera")
        break

    undist = cv2.undistort(frame, calib.K, calib.D)
    disp = undist.copy()

    cv2.putText(disp, f"Saved: {frame_id-1}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(disp, "Press S to save, Q to quit", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Training Data Collector", disp)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("SAVED in:", SAVE_DIR)
        img_path = os.path.join(SAVE_DIR, f"{frame_id:05d}.jpg")
        success = cv2.imwrite(img_path, undist)

        if success:
            print(f"[SAVED] {img_path}")
            frame_id += 1
        else:
            print(f"[ERROR] Failed to save {img_path}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()