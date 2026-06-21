import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("1. preprocess...")
print("   OK")

print("2. segmentation...")
print("   OK")

print("3. recognition...")
from src.recognition.dataset import CLASS_TO_IDX
from src.recognition.model import CLASS_NAMES, NUM_CLASSES

print("   OK")

print("4. fen...")
from src.fen.fen_builder import IDX_TO_FEN

print("   OK")

print("5. pipeline...")
print("   OK")

print()
print(f"   NUM_CLASSES = {NUM_CLASSES}")
print(f"   CLASS_NAMES = {CLASS_NAMES}")
print(f"   IDX_TO_FEN  = {IDX_TO_FEN}")
print(f"   CLASS_TO_IDX = {CLASS_TO_IDX}")
