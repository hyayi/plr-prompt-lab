"""preprocess — image pre-processing applied BEFORE the model call.

Part of the model's INPUT surface (like the prompt bytes and the injected
vocabulary): changing anything here changes what the model sees, so this
module is a parity/promotion component alongside plr_prompts / plr_schema.

Currently one step: the yellow target corner marker. run_plr callers can
skip it via `_pre_marked=True` (the lab's experiment configs expose that as
`preprocess.marker: false`).
"""
from __future__ import annotations

from PIL import Image, ImageDraw


def draw_target_marker(
    pil: Image.Image,
    box_ratio: float = 0.65,
    color: tuple[int, int, int] = (255, 230, 0),  # bright yellow
    line_width: int | None = None,
    arm_ratio: float = 0.20,
) -> Image.Image:
    """Draw four yellow corner brackets around the centre of the crop.

    v1.1_cot used a full rectangle — outline closure activated an
    "enclosure" semantic (uniform / one-piece / jumpsuit hallucinations
    accounted for 24% of upper_type regressions on the 50-object PoC).
    Corner brackets keep the "look here" cue without enclosing the
    subject, removing the false-containment prior.

    Args:
      box_ratio: side length of the conceptual box (corners are placed
        on its perimeter) as a fraction of the shorter crop dimension.
      color: RGB of the bracket strokes. Yellow (255, 230, 0) is
        uncommon on Korean street clothing and car paint.
      line_width: stroke thickness. None → auto from shorter side.
      arm_ratio: length of each bracket arm as a fraction of box_side.
    """
    w, h = pil.size
    side = min(w, h)
    if side < 16:                    # tiny thumbnail — skip, marker would dominate
        return pil
    if line_width is None:
        line_width = max(2, side // 80)
    box_side = int(side * box_ratio)
    arm = max(line_width * 2, int(box_side * arm_ratio))
    x0 = (w - box_side) // 2
    y0 = (h - box_side) // 2
    x1 = x0 + box_side
    y1 = y0 + box_side
    out = pil.copy()
    draw = ImageDraw.Draw(out)
    for cx, cy, dx, dy in (
        (x0, y0, +1, +1),    # top-left  L-shape opens down-right
        (x1, y0, -1, +1),    # top-right
        (x0, y1, +1, -1),    # bottom-left
        (x1, y1, -1, -1),    # bottom-right
    ):
        draw.line((cx, cy, cx + dx * arm, cy), fill=color, width=line_width)
        draw.line((cx, cy, cx, cy + dy * arm), fill=color, width=line_width)
    return out
