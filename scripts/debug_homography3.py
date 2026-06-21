import sys

sys.path.insert(0, r"i:\cchlink")

import cv2

from src.pipeline import _extract_cells_by_homography
from src.preprocess.board_detector import detect_board_corners, detect_grid_homography
from src.preprocess.perspective import WARP_PAD, warp_board
from src.segmentation.grid_splitter import split_board

problem_ids = ["004", "007", "009", "012", "015"]

for test_id in problem_ids:
    print(f"\n=== test_{test_id} ===")
    img_path = rf"i:\cchlink\data\raw\eval\test_{test_id}.png"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    corners = detect_board_corners(img)
    H = detect_grid_homography(img)

    print(f"  Corner detection: {'OK' if corners is not None else 'FAIL'}")
    print(f"  Homography detection: {'OK' if H is not None else 'FAIL'}")

    # Corner-based path
    if corners is not None:
        board = warp_board(img, corners)
        board = board[WARP_PAD : board.shape[0] - WARP_PAD, WARP_PAD : board.shape[1] - WARP_PAD]
        cells_corner = split_board(board)
        corner_means = [float(c.mean()) for c in cells_corner]
        corner_sizes = set((c.shape[0], c.shape[1]) for c in cells_corner)
        print(f"  Corner path cells: min={min(corner_means):.1f} max={max(corner_means):.1f}")
        print(f"  Corner cell shapes: {corner_sizes}")

    # Homography path
    if H is not None:
        cells_hom = _extract_cells_by_homography(img, H)
        hom_means = [float(cells_hom[r, c].mean()) for r in range(10) for c in range(9)]
        print(f"  Homography path cells: min={min(hom_means):.1f} max={max(hom_means):.1f}")

    print()
