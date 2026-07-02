"""gallery — self-contained HTML of crops vs labels (visual eval evidence).

Renders every crop in a dataset as a card: thumbnail (base64-inlined JPEG),
prediction vs human label, a correct/wrong badge, and the margin / quality
scores when present. Wrong predictions sort FIRST (lowest margin first) so
the file opens on the failure cases — this is the primary evidence source
for the improve-prompt skill and for humans reviewing an experiment.

GPU-free, network-free, no external assets: the HTML works from a file://
open or any static server.
"""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

_THUMB_MAX = 168  # px, longest side


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _thumb_b64(crop_path: Path) -> str:
    img = Image.open(crop_path).convert("RGB")
    img.thumbnail((_THUMB_MAX, _THUMB_MAX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt(v: Any) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


_CSS = """
body{margin:0;padding:24px;background:#0b1120;color:#dbe4f5;
     font-family:'Pretendard','Noto Sans KR','Malgun Gothic','Segoe UI',sans-serif}
h1{font-size:20px;margin:0 0 4px} .sub{color:#8fa1c0;font-size:13px;margin-bottom:18px}
.filters{margin:0 0 16px;display:flex;gap:8px;flex-wrap:wrap}
.filters button{background:#16213b;border:1px solid #263452;color:#b8c6e4;border-radius:8px;
                padding:5px 12px;font-size:12.5px;cursor:pointer}
.filters button.on{border-color:#38bdf8;color:#e8f2ff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
.card{background:#111a2e;border:1px solid #263452;border-radius:12px;padding:10px;text-align:center}
.card.wrong{border-color:rgba(248,113,113,.65)}
.card img{max-width:100%;border-radius:8px;background:#000}
.oid{font-family:ui-monospace,monospace;font-size:11px;color:#8fa1c0;margin-top:6px}
.pl{font-size:12.5px;margin-top:3px}
.pl b{color:#e8f2ff}
.badge{display:inline-block;font-size:10.5px;font-weight:700;border-radius:5px;
       padding:1px 8px;margin-top:5px}
.badge.ok{background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.4)}
.badge.no{background:rgba(248,113,113,.15);color:#f87171;border:1px solid rgba(248,113,113,.45)}
.badge.unl{background:rgba(148,163,184,.15);color:#94a3b8;border:1px solid rgba(148,163,184,.4)}
.sc{font-size:11px;color:#8fa1c0;margin-top:3px}
"""

_JS = """
function flt(mode, btn){
  document.querySelectorAll('.filters button').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('.card').forEach(c=>{
    c.style.display = (mode==='all' || c.dataset.k===mode
                       || c.dataset.label===mode || c.dataset.pred===mode) ? '' : 'none';
  });
}
"""


def build_gallery(
    dataset_dir: str | Path,
    out_path: str | Path | None = None,
) -> str:
    """Build <dataset>/gallery.html (or out_path). Returns the output path.

    Needs labels.jsonl + crops/; predictions.jsonl enriches each card with
    pred / correct-wrong / margin / quality when present (cards without a
    prediction render as unlabeled-by-model).
    """
    ds = Path(dataset_dir)
    out = Path(out_path) if out_path else ds / "gallery.html"

    labels = {r["obj_id"]: r.get("label") for r in _jsonl(ds / "labels.jsonl")}
    preds = {r["obj_id"]: r for r in _jsonl(ds / "predictions.jsonl")}
    obj_ids = sorted(set(labels) | set(preds))
    if not obj_ids:
        raise FileNotFoundError(
            f"No labels.jsonl / predictions.jsonl content in {ds} — nothing to render."
        )

    cards: list[tuple[tuple, str]] = []
    n_ok = n_wrong = n_nolabel = 0
    classes: set[str] = set()
    for oid in obj_ids:
        crop = ds / "crops" / f"{oid}.jpg"
        if not crop.exists():
            continue
        label = labels.get(oid)
        p = preds.get(oid) or {}
        pred = p.get("pred")
        margin, quality = p.get("margin"), p.get("quality")
        if label is None or pred is None or label == "unknown":
            kind, badge, cls = "unlabeled", '<span class="badge unl">UNSCORED</span>', ""
            n_nolabel += 1
            rank = (2, 0.0)
        elif pred == label:
            kind, badge, cls = "correct", '<span class="badge ok">CORRECT</span>', ""
            n_ok += 1
            rank = (1, margin if isinstance(margin, (int, float)) else 1.0)
        else:
            kind, badge, cls = "wrong", '<span class="badge no">WRONG</span>', " wrong"
            n_wrong += 1
            rank = (0, margin if isinstance(margin, (int, float)) else 0.0)
        if label:
            classes.add(str(label))
        card = (
            f'<div class="card{cls}" data-k="{kind}" '
            f'data-label="{html.escape(str(label or ""))}" data-pred="{html.escape(str(pred or ""))}">'
            f'<img src="data:image/jpeg;base64,{_thumb_b64(crop)}" alt="{html.escape(oid)}">'
            f'<div class="oid">{html.escape(oid)}</div>'
            f'<div class="pl">pred <b>{html.escape(str(pred or "—"))}</b> · '
            f'label <b>{html.escape(str(label or "—"))}</b></div>'
            f"{badge}"
            f'<div class="sc">margin {_fmt(margin)} · quality {_fmt(quality)}</div>'
            f"</div>"
        )
        cards.append((rank, card))

    cards.sort(key=lambda t: t[0])  # wrong first, low margin first
    n = n_ok + n_wrong
    acc = f"{n_ok / n:.3f}" if n else "—"

    filters = ['<button class="on" onclick="flt(\'all\',this)">전체</button>',
               '<button onclick="flt(\'wrong\',this)">오답만</button>',
               '<button onclick="flt(\'correct\',this)">정답만</button>']
    for c in sorted(classes):
        filters.append(f'<button onclick="flt(\'{html.escape(c)}\',this)">label={html.escape(c)}</button>')

    doc = (
        "<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>"
        f"<title>gallery — {html.escape(ds.name)}</title><style>{_CSS}</style></head><body>"
        f"<h1>크롭 시각화 — {html.escape(str(ds))}</h1>"
        f'<div class="sub">scored {n} (correct {n_ok} · wrong {n_wrong} · unscored {n_nolabel})'
        f" · accuracy {acc} · 오답이 먼저, 저-margin 순</div>"
        f'<div class="filters">{"".join(filters)}</div>'
        f'<div class="grid">{"".join(c for _r, c in cards)}</div>'
        f"<script>{_JS}</script></body></html>"
    )
    out.write_text(doc, encoding="utf-8")
    return str(out)
