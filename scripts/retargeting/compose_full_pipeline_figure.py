"""Combine eight retargeting panels into a two-row by four-column figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1", type=Path,
        default=Path("LocoMuJoCo_blender/retargeting_shape_fit_grid"),
    )
    parser.add_argument(
        "--stage2", type=Path,
        default=Path("LocoMuJoCo_blender/retargeting_pipeline_stage2_gr1_dance"),
    )
    parser.add_argument(
        "--stage3", type=Path,
        default=Path("LocoMuJoCo_blender/retargeting_pipeline_stage3_booster_walk"),
    )
    parser.add_argument("--stage2-target-label", default="GR-1")
    parser.add_argument("--stage3-target-label", default="Booster T1")
    parser.add_argument(
        "--output", type=Path,
        default=Path("LocoMuJoCo_blender/retargeting_pipeline_full/retargeting_pipeline_stages_1_2_3.png"),
    )
    return parser.parse_args()


def _font(size, bold=False):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _rounded_paste(canvas, image, xy, radius=40):
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, image.width - 1, image.height - 1), radius=radius, fill=255
    )
    canvas.paste(image, xy, mask)


def _arrow(draw, start, end, color=(82, 91, 104), width=7):
    draw.line((start, end), fill=color, width=width)
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    length = 24
    for delta in (2.55, -2.55):
        point = (
            end[0] + length * np.cos(angle + delta),
            end[1] + length * np.sin(angle + delta),
        )
        draw.line((end, point), fill=color, width=width)


def _horizontal_transition(draw, col, y, text, margin, width, gutter, font):
    x0 = margin + (col + 1) * width + col * gutter + 20
    x1 = x0 + gutter - 40
    _arrow(draw, (x0, y), (x1, y))
    draw.text(((x0 + x1) / 2, y - 34), text, font=font,
              fill=(82, 91, 104), anchor="ms")


def _stage_heading(draw, xy, number, description, stage_font, description_font):
    x, y = xy
    stage_text = f"Stage {number}"
    draw.text((x, y), stage_text, font=stage_font, fill=(29, 35, 45), anchor="la")
    stage_width = draw.textlength(stage_text, font=stage_font)
    draw.text((x + stage_width + 24, y + 8), f"— {description}",
              font=description_font, fill=(29, 35, 45), anchor="la")


def main():
    args = _parse_args()
    stage1 = args.stage1.resolve()
    stage2 = args.stage2.resolve()
    stage3 = args.stage3.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    before = Image.open(stage1 / "before.png").convert("RGB")
    after = Image.open(stage1 / "after.png").convert("RGB")
    top = [
        before,
        *(Image.open(stage2 / f"{name}.png").convert("RGB")
          for name in ("stage2_smpl", "stage2_targets", "stage2_robot")),
    ]
    bottom = [
        after,
        *(Image.open(stage3 / f"{name}.png").convert("RGB")
          for name in ("stage3_source", "stage3_smpl", "stage3_target")),
    ]
    width, height = top[0].size
    if any(image.size != (width, height) for image in (*top, *bottom)):
        raise ValueError("All eight source panels must have identical dimensions.")
    metrics = np.load(stage1 / "shape_fitting_states.npz")

    margin, gutter, header, row_gap = 70, 120, 190, 210
    canvas_width = 2 * margin + 4 * width + 3 * gutter
    top_y = margin + header
    bottom_y = top_y + height + row_gap + header
    canvas_height = bottom_y + height + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    stage_font = _font(70, bold=True)
    description_font = _font(42, bold=True)
    label_font = _font(37, bold=True)
    metric_font = _font(28)
    arrow_font = _font(25)

    columns = [margin + index * (width + gutter) for index in range(4)]
    section_radius = 58
    section_pad = 34
    stage_right_x = canvas_width - section_pad
    draw.rounded_rectangle(
        (section_pad, 18, columns[0] + width + section_pad, canvas_height - 18),
        radius=section_radius, fill=(220, 238, 255),
    )
    draw.rounded_rectangle(
        (columns[1] - section_pad, 18, stage_right_x, top_y + height + 74),
        radius=section_radius, fill=(255, 237, 207),
    )
    draw.rounded_rectangle(
        (columns[1] - section_pad, top_y + height + row_gap - 28,
         stage_right_x, canvas_height - 18),
        radius=section_radius, fill=(220, 244, 230),
    )
    _stage_heading(draw, (columns[0], margin), 1, "Shape fitting",
                   stage_font, description_font)
    _stage_heading(draw, (columns[1], margin), 2, "SMPL → robot",
                   stage_font, description_font)
    _stage_heading(draw, (columns[1], top_y + height + row_gap + 12),
                   3, "Robot → SMPL → robot", stage_font, description_font)

    top_labels = (
        "Original SMPL shape", "Original SMPL motion",
        "Target-shaped SMPL skeleton", f"{args.stage2_target_label} trajectory",
    )
    bottom_labels = (
        "G1 Full-fitted SMPL shape", "Source H1 v2 trajectory",
        "H1 v2-shaped SMPL skeleton", f"Target {args.stage3_target_label} trajectory",
    )
    for col, image in enumerate(top):
        x = columns[col]
        draw.text((x + width / 2, top_y - 50), top_labels[col], font=label_font,
                  fill=(56, 63, 74), anchor="ms")
        _rounded_paste(canvas, image, (x, top_y))
    for col, image in enumerate(bottom):
        x = columns[col]
        draw.text((x + width / 2, bottom_y - 50), bottom_labels[col], font=label_font,
                  fill=(56, 63, 74), anchor="ms")
        _rounded_paste(canvas, image, (x, bottom_y))

    draw.text((columns[0] + width / 2, top_y - 10),
              f"Mean site error: {float(metrics['before_error_mm']):.1f} mm",
              font=metric_font, fill=(82, 91, 104), anchor="ms")
    draw.text((columns[0] + width / 2, bottom_y - 10),
              f"Mean site error: {float(metrics['after_error_mm']):.1f} mm",
              font=metric_font, fill=(82, 91, 104), anchor="ms")

    vertical_x = columns[0] + width / 2
    vertical_start = top_y + height + 55
    vertical_end = bottom_y - 130
    _arrow(draw, (vertical_x, vertical_start), (vertical_x, vertical_end))
    draw.text((vertical_x + 28, (vertical_start + vertical_end) / 2),
              "optimize β, scale, and site offsets", font=arrow_font,
              fill=(82, 91, 104), anchor="lm")

    _horizontal_transition(draw, 1, top_y - 47, "apply fitted shape",
                           margin, width, gutter, arrow_font)
    _horizontal_transition(draw, 2, top_y - 47, "soft weld constraints",
                           margin, width, gutter, arrow_font)
    _horizontal_transition(draw, 1, bottom_y - 47, "inverse site fitting",
                           margin, width, gutter, arrow_font)
    _horizontal_transition(draw, 2, bottom_y - 47, "reuse Stage 2",
                           margin, width, gutter, arrow_font)

    canvas.save(output)
    for image in (*top, *bottom):
        image.close()
    print(output)


if __name__ == "__main__":
    main()
