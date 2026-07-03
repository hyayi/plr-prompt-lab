"""preprocess — 모델 호출 "이전"에 적용되는 이미지 전처리.

프롬프트 바이트·주입 어휘와 마찬가지로 모델 "인풋 표면"의 일부다:
여기를 바꾸면 모델이 보는 것이 바뀌므로 plr_prompts/plr_schema와 같은
parity/승격 대상 컴포넌트.

현재 단계는 하나 — 노란 타깃 코너 마커. run_plr 호출부가
`_pre_marked=True`로 생략 가능 (lab 실험 config의 `preprocess.marker: false`).

입력/출력 예) draw_target_marker(PIL 100×150 크롭)
  → 같은 크기 PIL, 중앙 65% 박스 모서리에 노란 L자 획 4개가 그려진 사본
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
    """크롭 중앙에 노란 코너 브래킷 4개를 그린다 — 다인물 크롭에서 "누구를
    라벨할지" 지정하는 장치 (모델 이미지 인풋의 일부).

    Draw four yellow corner brackets around the centre of the crop.

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
