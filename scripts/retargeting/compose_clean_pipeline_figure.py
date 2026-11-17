"""Compose a text-light 2x4 qualitative retargeting figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLUE = (52, 120, 184)
AMBER = (199, 122, 27)
GREEN = (57, 132, 97)
BLUE_BG = (220, 238, 255)
AMBER_BG = (255, 237, 207)
GREEN_BG = (220, 244, 230)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", type=Path,
                        default=Path("LocoMuJoCo_blender/retargeting_shape_fit_grid"))
    parser.add_argument("--stage2", type=Path,
                        default=Path("LocoMuJoCo_blender/retargeting_pipeline_stage2_gr1_dance"))
    parser.add_argument("--stage3", type=Path,
                        default=Path("LocoMuJoCo_blender/retargeting_pipeline_stage3_booster_walk"))
    parser.add_argument("--output", type=Path,
                        default=Path("LocoMuJoCo_blender/retargeting_pipeline_full/retargeting_pipeline_clean.png"))
    return parser.parse_args()


def _font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _paste_panel(canvas, image, xy, accent, label, font):
    x, y = xy
    radius = 42
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, image.width - 1, image.height - 1), radius=radius, fill=255
    )
    canvas.paste(image, xy, mask)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (x, y, x + image.width - 1, y + image.height - 1),
        radius=radius, outline=accent, width=7,
    )
    badge = (x + 24, y + 24, x + 174, y + 120)
    draw.rounded_rectangle(badge, radius=48, fill=accent)
    draw.text(((badge[0] + badge[2]) / 2, (badge[1] + badge[3]) / 2 + 1),
              f"({label})", font=font, fill="white", anchor="mm")


def _arrow(draw, start, end, color, width=10):
    draw.line((start, end), fill=color, width=width)
    x, y = end
    if start[0] == end[0]:
        points = ((x, y), (x - 18, y - 26), (x + 18, y - 26))
    else:
        points = ((x, y), (x - 26, y - 18), (x - 26, y + 18))
    draw.polygon(points, fill=color)


def _stage_badge(draw, xy, number, accent, font):
    x, y = xy
    box = (x, y, x + 370, y + 116)
    draw.rounded_rectangle(box, radius=58, fill=accent)
    draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + 1),
              f"STAGE {number}", font=font, fill="white", anchor="mm")


def main():
    args = _parse_args()
    stage1 = args.stage1.resolve()
    stage2 = args.stage2.resolve()
    stage3 = args.stage3.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        Image.open(stage1 / "before.png").convert("RGB"),
        Image.open(stage2 / "stage2_smpl.png").convert("RGB"),
        Image.open(stage2 / "stage2_targets.png").convert("RGB"),
        Image.open(stage2 / "stage2_robot.png").convert("RGB"),
        Image.open(stage1 / "after.png").convert("RGB"),
        Image.open(stage3 / "stage3_source.png").convert("RGB"),
        Image.open(stage3 / "stage3_smpl.png").convert("RGB"),
        Image.open(stage3 / "stage3_target.png").convert("RGB"),
    ]
    width, height = panels[0].size
    if any(panel.size != (width, height) for panel in panels):
        raise ValueError("All eight panels must have identical dimensions.")

    # With 28 px of section padding on each side, a 102 px column gutter
    # leaves the same 46 px gap as the vertical Stage 2/3 separation.
    margin, gutter, row_gap, stage_header = 42, 102, 70, 145
    top_y = margin + stage_header
    bottom_y = top_y + height + row_gap + stage_header
    canvas = Image.new(
        "RGB",
        (2 * margin + 4 * width + 3 * gutter,
         bottom_y + height + margin),
        "white",
    )
    positions = [
        (margin + col * (width + gutter), top_y if row == 0 else bottom_y)
        for row in range(2) for col in range(4)
    ]
    accents = (BLUE, AMBER, AMBER, AMBER, BLUE, GREEN, GREEN, GREEN)
    font = _font(60)
    stage_font = _font(72)
    draw = ImageDraw.Draw(canvas)
    columns = [margin + col * (width + gutter) for col in range(4)]
    section_pad = 28
    draw.rounded_rectangle(
        (12, 12, columns[0] + width + section_pad, canvas.height - 12),
        radius=46, fill=BLUE_BG,
    )
    draw.rounded_rectangle(
        (columns[1] - section_pad, 12, canvas.width - 12, top_y + height + 24),
        radius=46, fill=AMBER_BG,
    )
    draw.rounded_rectangle(
        (columns[1] - section_pad, top_y + height + row_gap,
         canvas.width - 12, canvas.height - 12),
        radius=46, fill=GREEN_BG,
    )
    _stage_badge(draw, (columns[0] + 12, 27), 1, BLUE, stage_font)
    _stage_badge(draw, (columns[1] + 12, 27), 2, AMBER, stage_font)
    _stage_badge(
        draw, (columns[1] + 12, top_y + height + row_gap + 15),
        3, GREEN, stage_font,
    )
    for panel, position, accent, label in zip(
        panels, positions, accents, "abcdefgh"
    ):
        _paste_panel(canvas, panel, position, accent, label, font)

    # Stage 1 flows vertically; Stages 2 and 3 flow horizontally.
    first_x = positions[0][0] + width / 2
    _arrow(
        draw,
        (first_x, positions[0][1] + height + 15),
        (first_x, positions[4][1] - 15),
        BLUE,
    )
    for row, accent in ((0, AMBER), (1, GREEN)):
        y = positions[row * 4 + 1][1] + height / 2
        for col in (1, 2):
            x0 = positions[row * 4 + col][0] + width + 15
            x1 = positions[row * 4 + col + 1][0] - 15
            _arrow(draw, (x0, y), (x1, y), accent)

    canvas.save(output)
    for panel in panels:
        panel.close()
    print(output)


if __name__ == "__main__":
    main()
