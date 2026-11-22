"""Vẽ icon ứng dụng (túi mua sắm + kính lúp) -> app/assets/app.ico (16..256 px). Chạy lại được, ghi đè tệp."""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "app" / "assets" / "app.ico"
S = 512  # vẽ lớn rồi thu nhỏ cho mịn


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((16, 16, S - 16, S - 16), radius=110, fill="#0F1117")
    # túi
    d.arc((172, 92, 316, 236), start=180, end=360, fill="#E6E8EB", width=26)
    d.rounded_rectangle((112, 160, 376, 420), radius=36, fill="#4C8BF5")
    # kính lúp
    d.ellipse((262, 252, 422, 412), outline="#E6E8EB", width=30, fill="#0F1117")
    d.line((398, 392, 460, 454), fill="#E6E8EB", width=40)
    return img


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    draw().save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(OUT)
