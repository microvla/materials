import cv2
import numpy as np
from io import BytesIO

def create_charuco_board_10cm(
        squares_x=6,
        squares_y=6,
        square_length=16.0,   # mm
        marker_length=12.0,   # mm
        output_png="charuco_96mm.png",
        output_pdf="charuco_A4_96mm.pdf"
):
    # Dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

    # New API (OpenCV 4.10+)
    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=float(square_length),
        markerLength=float(marker_length),
        dictionary=aruco_dict
    )

    # Draw image at high resolution
    img = board.generateImage(outSize=(3000, 3000))
    cv2.imwrite(output_png, img)
    print("PNG saved:", output_png)

    # ---- Generate A4 PDF (exact 96mm x 96mm) ----
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.utils import ImageReader

        # convert image to PNG bytes
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        buffer = BytesIO()
        ok, encoded = cv2.imencode(".png", img_rgb)
        buffer.write(encoded.tobytes())
        buffer.seek(0)

        img_reader = ImageReader(buffer)

        # A4 page size
        a4_width, a4_height = A4

        # convert 96 mm → PDF pt
        mm_to_pt = 72 / 25.4
        board_size_mm = squares_x * square_length    # = 96 mm
        board_size_pt = board_size_mm * mm_to_pt

        # center on A4
        x = (a4_width - board_size_pt) / 2
        y = (a4_height - board_size_pt) / 2

        c = canvas.Canvas(output_pdf, pagesize=A4)
        c.drawImage(img_reader, x, y, width=board_size_pt, height=board_size_pt)
        c.showPage()
        c.save()

        print("PDF saved:", output_pdf)

    except ImportError:
        print("reportlab not installed; PDF skipped.")


if __name__ == "__main__":
    create_charuco_board_10cm()
