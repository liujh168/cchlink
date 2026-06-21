import sys

sys.path.insert(0, r"i:\cchlink")

import cv2

from src.preprocess.board_detector import detect_board_corners, detect_grid_homography
from src.preprocess.perspective import warp_board
from src.segmentation.grid_splitter import split_board

problem_ids = ["004", "007", "009", "012", "015"]

for test_id in problem_ids:
    print(f"\n{'=' * 60}")
    print(f"=== test_{test_id} ===")
    print(f"{'=' * 60}")
    img_path = rf"i:\cchlink\data\raw\eval\test_{test_id}.png"
    img = cv2.imread(img_path)
    if img is None:
        print("  Cannot read image")
        continue
    h, w = img.shape[:2]
    print(f"  Image size: {w}x{h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  Gray mean: {gray.mean():.1f}, std: {gray.std():.1f}")
    print(f"  Gray min: {gray.min()}, max: {gray.max()}")

    corners = detect_board_corners(img)
    print(f"  Corner detection: {'OK' if corners is not None else 'FAILED'}")
    if corners is not None:
        print(f"  Corners shape: {corners.shape}")
        area = cv2.contourArea(corners.reshape(-1, 1, 2))
        print(f"  Board area: {area:.0f} / {h * w} = {area / (h * w) * 100:.1f}%")

        board = warp_board(img, corners)
        gray_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        print(f"  Warped board: {board.shape}")
        print(f"  Board mean: {gray_board.mean():.1f}")

        cells = split_board(board)
        if cells:
            sample = cells[45]
            print(f"  Sample cell (5,0): shape={sample.shape}, mean={sample.mean():.1f}")

    H = detect_grid_homography(img)
    print(f"  Homography detection: {'OK' if H is not None else 'FAILED'}")

    print()
