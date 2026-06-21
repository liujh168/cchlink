import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fen.fen_builder import build_fen

# 模拟一个初始棋盘布局: 第10行(黑方底线)到第1行(红方底线)
# 黑方将初始在 (0,4) = idx 9*0+4=4, 红方帅在 (9,4)=85
# 黑车 (0,0)=0, (0,8)=8
# etc.

initial_board = [
    # row 0 (黑底线): 车 马 象 士 将 士 象 马 车
    10,
    11,
    9,
    8,
    7,
    8,
    9,
    11,
    10,
    # row 1: 空 炮 空 空 空 空 空 炮 空
    14,
    12,
    14,
    14,
    14,
    14,
    14,
    12,
    14,
    # row 2: 卒 空 卒 空 卒 空 卒 空 卒
    13,
    14,
    13,
    14,
    13,
    14,
    13,
    14,
    13,
    # row 3: 全空
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    # row 4: 全空
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    # row 5: 全空
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    # row 6: 全空
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    14,
    # row 7: 兵 空 兵 空 兵 空 兵 空 兵
    6,
    14,
    6,
    14,
    6,
    14,
    6,
    14,
    6,
    # row 8: 空 炮 空 空 空 空 空 炮 空
    14,
    5,
    14,
    14,
    14,
    14,
    14,
    5,
    14,
    # row 9 (红底线): 俥 马 相 仕 帅 仕 相 马 俥
    3,
    4,
    2,
    1,
    0,
    1,
    2,
    4,
    3,
]

fen = build_fen(initial_board)
print("FEN:", fen)

expected = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
# Wait, we use different FEN mapping. Let me verify:
# 车(黑)=r, 马(黑)=n, 象(黑)=b, 士(黑)=a, 将(黑)=k
# 炮(黑)=c, 卒(黑)=p
# 俥(红)=R, 马(红)=N, 相(红)=B, 仕(红)=A, 帅(红)=K
# 炮(红)=C, 兵(红)=P

# row 0: rnbakabnr
# row 1: 空-炮-空-空-空-空-空-炮-空 = 1c5c1 -> wait, check: empty=14, 炮(黑c)=12
#        [14,12,14,14,14,14,14,12,14] = 1c5c1 ✓
# row 2: [13,14,13,14,13,14,13,14,13] = p1p1p1p1p ✓
# row 3-6: 全空 = 9 ✓
# row 7: [6,14,6,14,6,14,6,14,6] = P1P1P1P1P ✓
# row 8: [14,5,14,14,14,14,14,5,14] = 1C5C1 ✓
# row 9: [3,4,2,1,0,1,2,4,3] = RNBAKABNR ✓

expected_fen = "rnbakabnr/1c5c1/p1p1p1p1p/9/9/9/9/P1P1P1P1P/1C5C1/RNBAKABNR"
print("Expected:", expected_fen)
print("Match:", fen == expected_fen)
