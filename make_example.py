"""Render an original fictional customer message; no network or personal data."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", required=True, help="Path to a local TrueType font")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / "examples"
    case = json.loads((folder / "support_ticket.json").read_text())
    image = Image.new("RGB", case["image_size"], "#f1f5f9")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((48, 48, 1232, 672), radius=22, fill="white", outline="#d8e0ea", width=2)
    draw.text((90, 90), "Customer message", font=ImageFont.truetype(args.font, 42), fill="#0f172a")
    draw.line((90, 160, 1190, 160), fill="#d8e0ea", width=2)
    font = ImageFont.truetype(args.font, 35)
    for index, line in enumerate(case["message_lines"]):
        draw.text((90, 208 + 61 * index), line, font=font, fill="#1e293b")
    image.save(folder / case["image"])


if __name__ == "__main__":
    main()
