import os
from PIL import Image

base = r"i:\cchlink\data\pieces"

# 检查红帅样本
red_dir = os.path.join(base, "红帅")
black_dir = os.path.join(base, "黑将")
empty_dir = os.path.join(base, "空")

red_img = Image.open(os.path.join(red_dir, "00000.png"))
black_img = Image.open(os.path.join(black_dir, "00000.png"))
empty_img = Image.open(os.path.join(empty_dir, "00000.png"))

print(f"红帅: size={red_img.size}, mode={red_img.mode}, center_pixel={red_img.getpixel((20, 20))}")
print(f"黑将: size={black_img.size}, mode={black_img.mode}, center_pixel={black_img.getpixel((20, 20))}")
print(f"空:   size={empty_img.size}, mode={empty_img.mode}, center_pixel={empty_img.getpixel((20, 20))}")

# 验证同一类的多样性
pixels = []
for f in sorted(os.listdir(red_dir))[:5]:
    img = Image.open(os.path.join(red_dir, f))
    pixels.append(img.getpixel((20, 20)))
print(f"红帅 前5张中心像素: {pixels}")
print(f"多样性 (是否不同): {len(set(pixels))} / {len(pixels)}")

# 验证所有类别都存在，且数量正确
print()
for cls in sorted(os.listdir(base)):
    d = os.path.join(base, cls)
    if os.path.isdir(d):
        cnt = len([f for f in os.listdir(d) if f.endswith(".png")])
        status = "OK" if cnt == 400 else f"MISMATCH ({cnt})"
        print(f"  {cls:6s}: {cnt} 张  {status}")
