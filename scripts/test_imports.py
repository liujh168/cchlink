import sys
sys.path.insert(0, r"i:\cchlink")

print("1. preprocess...")
from src.preprocess.board_detector import detect_board_corners
from src.preprocess.perspective import warp_board
print("   OK")

print("2. segmentation...")
from src.segmentation.grid_splitter import split_board, split_board_with_positions
print("   OK")

print("3. recognition...")
from src.recognition.model import build_model, save_model, load_model, NUM_CLASSES, CLASS_NAMES
from src.recognition.dataset import PieceDataset, CLASS_TO_IDX, TRAIN_TRANSFORM, VAL_TRANSFORM
from src.recognition.predictor import PiecePredictor
print("   OK")

print("4. fen...")
from src.fen.fen_builder import build_fen, IDX_TO_FEN, IDX_TO_NAME
print("   OK")

print("5. pipeline...")
from src.pipeline import Pipeline
print("   OK")

print()
print(f"   NUM_CLASSES = {NUM_CLASSES}")
print(f"   CLASS_NAMES = {CLASS_NAMES}")
print(f"   IDX_TO_FEN  = {IDX_TO_FEN}")
print(f"   CLASS_TO_IDX = {CLASS_TO_IDX}")
