#!/usr/bin/env python3
"""
Generate SVG diagrams for Post 2: GPU batching visualization.

Produces four diagrams in posts/images/:
  diagram-batch-sequential.svg — 2.0: 5 users, Post 1 behaviour — 5 separate tensors, processed one by one
  diagram-batch-single.svg     — 2.1: single user, 1×N tensor
  diagram-batch-ragged.svg     — 2.2: 5 users, no padding — rows differ in length, not a matrix
  diagram-batch-padded.svg     — 2.3: 5 users, padded — uniform 5×N matrix, GPU can process

Usage:
    uv run python scripts/generate_post02_diagrams.py
"""

from pathlib import Path

# --- Colors (consistent with benchmark charts) ---
BLUE      = "#4C78A8"   # real tokens
PAD_CLR   = "#C8D8E8"   # padding tokens
GREEN     = "#4E9A51"   # valid / success
RED       = "#E45756"   # invalid / failure
MUTED     = "#AAAAAA"   # column headers
ARROW_CLR = "#CCCCCC"   # arrows
USER_FILL = "#EEF3FA"   # user icon background
USER_STK  = "#9BB5D5"   # user icon stroke
PROM_FILL = "#FFF8E7"   # prompt box background
PROM_STK  = "#F0C040"   # prompt box stroke

# --- Token dimensions ---
TW, TH, TG = 18, 18, 3
TS = TW + TG  # stride = 21px per token

# --- Users: (label, prompt text, real token count) ---
USERS = [
    ("U1", '"Tell me about AI"',    5),
    ("U2", '"Hi!"',                  2),
    ("U3", '"What is the meaning"', 7),
    ("U4", '"Explain this"',         4),
    ("U5", '"Why?"',                 3),
]
MAX_T = max(u[2] for u in USERS)  # 7

# --- Layout constants ---
X_USER  = 32    # center x of user icons
X_PROM  = 58    # left x of prompt boxes (width=115)
X_TOK   = 188   # left x of token rows
Y_TITLE = 18    # y of diagram title
Y_HDR   = 36    # y of column headers
Y_FIRST = 62    # y-center of first row
ROW_H   = 44    # vertical spacing between rows
BPAD    = 6     # padding around tensor bounding box
CW      = 380   # canvas width

OUTPUT = Path("posts/images")


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------

def title(cx, y, txt):
    return (
        f'<text x="{cx}" y="{y}" text-anchor="middle" '
        f'font-family="system-ui,sans-serif" font-size="12" font-weight="600" fill="#444">'
        f'{txt}</text>'
    )


def col_header(x, y, txt, anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="system-ui,sans-serif" font-size="9" fill="{MUTED}" letter-spacing="0.8">'
        f'{txt.upper()}</text>'
    )


def arrow(x1, y, x2):
    return (
        f'<line x1="{x1}" y1="{y}" x2="{x2-7}" y2="{y}" '
        f'stroke="{ARROW_CLR}" stroke-width="1.5" stroke-linecap="round"/>'
        f'<polygon points="{x2},{y} {x2-7},{y-3.5} {x2-7},{y+3.5}" fill="{ARROW_CLR}"/>'
    )


def user_icon(cx, y_top, label):
    """Person silhouette centered at cx, with top edge at y_top."""
    return (
        f'<circle cx="{cx}" cy="{y_top+8}" r="7" '
        f'fill="{USER_FILL}" stroke="{USER_STK}" stroke-width="1.3"/>'
        f'<path d="M{cx-9},{y_top+24} Q{cx},{y_top+14} {cx+9},{y_top+24}" '
        f'fill="none" stroke="{USER_STK}" stroke-width="2" stroke-linecap="round"/>'
        f'<text x="{cx}" y="{y_top+36}" text-anchor="middle" '
        f'font-family="system-ui,sans-serif" font-size="9" fill="{MUTED}">{label}</text>'
    )


def prompt_box(x, y, text, w=115, h=24):
    cx = x + w // 2
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" '
        f'fill="{PROM_FILL}" stroke="{PROM_STK}" stroke-width="1.2"/>'
        f'<text x="{cx}" y="{y+16}" text-anchor="middle" '
        f'font-family="system-ui,sans-serif" font-size="9" fill="#666">{text}</text>'
    )


def token_row(x, y, n_real, n_total):
    """Left-padded row: (n_total - n_real) grey padding tokens on the LEFT,
    then n_real blue real tokens on the RIGHT.
    This matches how transformers tokenizes batches — padding_side='left' —
    so all real tokens are flush-right and generation appends at the same position."""
    n_pad = n_total - n_real
    parts = []
    for i in range(n_total):
        tx = x + i * TS
        is_pad = i < n_pad
        color = PAD_CLR if is_pad else BLUE
        parts.append(
            f'<rect x="{tx}" y="{y}" width="{TW}" height="{TH}" rx="2.5" '
            f'fill="{color}" opacity="0.9"/>'
        )
        if is_pad:
            parts.append(
                f'<text x="{tx + TW//2}" y="{y + TH//2 + 4}" text-anchor="middle" '
                f'font-family="monospace" font-size="7" fill="#94AABB">P</text>'
            )
    return "\n".join(parts)


def tensor_box(x_tok, y_top, height, label, valid, n_cols=MAX_T):
    """
    Bounding box around the token area, with a label below.
    valid=True → solid green border + ✓
    valid=False → dashed red border + ✗
    """
    color = GREEN if valid else RED
    dash  = "" if valid else ' stroke-dasharray="5,3"'
    sym   = "✓" if valid else "✗"
    bx = x_tok - BPAD
    by = y_top - BPAD
    bw = n_cols * TS - TG + BPAD * 2
    bh = height + BPAD * 2
    cx = x_tok + (n_cols * TS - TG) // 2
    label_y = by + bh + 16
    return (
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="5" '
        f'fill="none" stroke="{color}" stroke-width="2"{dash}/>'
        f'<text x="{cx}" y="{label_y}" text-anchor="middle" '
        f'font-family="system-ui,sans-serif" font-size="10" font-weight="600" fill="{color}">'
        f'{sym} {label}</text>'
    )


def legend(x, y):
    """Small colour legend: blue = real token, grey = padding."""
    return (
        f'<rect x="{x}" y="{y-9}" width="11" height="11" rx="2" fill="{BLUE}" opacity="0.9"/>'
        f'<text x="{x+15}" y="{y}" font-family="system-ui,sans-serif" font-size="9" fill="#888">real token</text>'
        f'<rect x="{x+80}" y="{y-9}" width="11" height="11" rx="2" fill="{PAD_CLR}" opacity="0.9"/>'
        f'<text x="{x+95}" y="{y}" font-family="system-ui,sans-serif" font-size="9" fill="#888">padding</text>'
    )


# ---------------------------------------------------------------------------
# Diagram builders
# ---------------------------------------------------------------------------

def diagram_21():
    """Single user → prompt → tokens → 1×N tensor."""
    lbl, ptxt, n_tok = USERS[0]
    yc = Y_FIRST
    y_icon = yc - 20
    H = 120

    svg = [
        f'<svg width="{CW}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        title(CW // 2, Y_TITLE, "Single request — one sequence through the model"),
        col_header(X_USER, Y_HDR, "User"),
        col_header(X_PROM + 57, Y_HDR, "Prompt"),
        col_header(X_TOK + n_tok * TS // 2, Y_HDR, "Tokens"),
        user_icon(X_USER, y_icon, lbl),
        arrow(X_USER + 12, yc, X_PROM),
        prompt_box(X_PROM, yc - 12, ptxt),
        arrow(X_PROM + 115, yc, X_TOK),
        token_row(X_TOK, yc - TH // 2, n_tok, n_tok),
        tensor_box(X_TOK, yc - TH // 2, TH, f"1 × {n_tok} tensor — valid", True, n_tok),
        '</svg>',
    ]
    return "\n".join(svg)


def diagram_22():
    """5 users → 5 prompts → ragged token rows → not a matrix."""
    n = len(USERS)
    ycs = [Y_FIRST + i * ROW_H for i in range(n)]
    y_top = ycs[0] - TH // 2
    y_bot = ycs[-1] + TH // 2
    H = y_bot + BPAD + 30

    svg = [
        f'<svg width="{CW}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        title(CW // 2, Y_TITLE, "5 requests, no padding — rows have different lengths"),
        col_header(X_USER, Y_HDR, "Users"),
        col_header(X_PROM + 57, Y_HDR, "Prompts"),
        col_header(X_TOK + MAX_T * TS // 2, Y_HDR, "Tokens"),
    ]
    for i, (lbl, ptxt, n_tok) in enumerate(USERS):
        yc = ycs[i]
        svg += [
            user_icon(X_USER, yc - 20, lbl),
            arrow(X_USER + 12, yc, X_PROM),
            prompt_box(X_PROM, yc - 12, ptxt),
            arrow(X_PROM + 115, yc, X_TOK),
            token_row(X_TOK, yc - TH // 2, n_tok, n_tok),  # no padding
        ]
    svg += [
        tensor_box(X_TOK, y_top, y_bot - y_top, "Not a matrix — GPU cannot process", False),
        '</svg>',
    ]
    return "\n".join(svg)


def diagram_23():
    """5 users → 5 prompts → padded token rows → valid 5×MAX_T matrix."""
    n = len(USERS)
    ycs = [Y_FIRST + i * ROW_H for i in range(n)]
    y_top = ycs[0] - TH // 2
    y_bot = ycs[-1] + TH // 2
    H = y_bot + BPAD + 40  # extra room for legend

    svg = [
        f'<svg width="{CW}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        title(CW // 2, Y_TITLE, f"5 requests, with padding — uniform 5 × {MAX_T} matrix"),
        col_header(X_USER, Y_HDR, "Users"),
        col_header(X_PROM + 57, Y_HDR, "Prompts"),
        col_header(X_TOK + MAX_T * TS // 2, Y_HDR, "Tokens (padded)"),
    ]
    for i, (lbl, ptxt, n_tok) in enumerate(USERS):
        yc = ycs[i]
        svg += [
            user_icon(X_USER, yc - 20, lbl),
            arrow(X_USER + 12, yc, X_PROM),
            prompt_box(X_PROM, yc - 12, ptxt),
            arrow(X_PROM + 115, yc, X_TOK),
            token_row(X_TOK, yc - TH // 2, n_tok, MAX_T),  # padded to MAX_T
        ]
    svg += [
        tensor_box(X_TOK, y_top, y_bot - y_top, f"5 × {MAX_T} matrix — GPU can process", True),
        legend(X_TOK, H - 10),
        '</svg>',
    ]
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Diagram 2.0 — sequential Gantt chart (Post 1 behaviour with 5 users)
# ---------------------------------------------------------------------------

def diagram_sequential():
    """
    Left side: same 5 users, prompts, and ragged token rows as diagram_22.
    Right side: Gantt chart showing the GPU processes each 1×N tensor one at a time.
    Waiting time shown as grey dashed area before each processing bar.
    """
    n = len(USERS)
    ycs = [Y_FIRST + i * ROW_H for i in range(n)]
    y_top = ycs[0] - TH // 2
    y_bot = ycs[-1] + TH // 2

    DIVIDER_X  = 355
    GANTT_X    = 368
    PX_PER_TOK = 6       # pixels per token — scales bar width to token count
    IDLE_W     = 8       # small gap between consecutive bars
    BAR_H      = TH      # match token row height

    tok_counts = [u[2] for u in USERS]
    bar_widths = [t * PX_PER_TOK for t in tok_counts]

    # Sequential start positions: each bar starts after the previous finishes
    bar_starts = []
    x = GANTT_X
    for i, bw in enumerate(bar_widths):
        bar_starts.append(x)
        x += bw + (IDLE_W if i < n - 1 else 0)
    gantt_end = x

    CW_S = gantt_end + 30
    H    = y_bot + BPAD + 42

    svg = [
        f'<svg width="{CW_S}" height="{H}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="white"/>',
        title(CW_S // 2, Y_TITLE,
              "Post 1: 5 requests → 5 separate tensors → GPU processes one at a time"),
        # Left section headers (same as ragged diagram)
        col_header(X_USER,          Y_HDR, "Users"),
        col_header(X_PROM + 57,     Y_HDR, "Prompts"),
        col_header(X_TOK + 40,      Y_HDR, "Tokens"),
        # Section divider
        f'<line x1="{DIVIDER_X}" y1="28" x2="{DIVIDER_X}" y2="{y_bot + 6}" '
        f'stroke="#E8E8E8" stroke-width="1"/>',
        # Gantt header
        col_header(GANTT_X + (gantt_end - GANTT_X) // 2, Y_HDR, "GPU timeline"),
        # Vertical "all arrive at t=0" marker
        f'<line x1="{GANTT_X}" y1="{y_top - 4}" x2="{GANTT_X}" y2="{y_bot + 4}" '
        f'stroke="#CCC" stroke-width="1" stroke-dasharray="3,2"/>',
        f'<text x="{GANTT_X}" y="{y_top - 7}" text-anchor="middle" '
        f'font-family="system-ui,sans-serif" font-size="8" fill="{MUTED}">t=0</text>',
    ]

    for i, (lbl, ptxt, n_tok) in enumerate(USERS):
        yc    = ycs[i]
        bw    = bar_widths[i]
        bx    = bar_starts[i]
        y_bar = yc - BAR_H // 2

        # ── Left section (users, prompts, token rows) ──────────────────────
        row_w = n_tok * TS - TG
        svg += [
            user_icon(X_USER, yc - 20, lbl),
            arrow(X_USER + 12, yc, X_PROM),
            prompt_box(X_PROM, yc - 12, ptxt),
            arrow(X_PROM + 115, yc, X_TOK),
            token_row(X_TOK, y_bar, n_tok, n_tok),   # no padding — ragged
            # Individual tensor bounding box + "1×N" label right-aligned to divider
            f'<rect x="{X_TOK - 2}" y="{y_bar - 2}" width="{row_w + 4}" height="{BAR_H + 4}" rx="3" '
            f'fill="none" stroke="{BLUE}" stroke-width="1" opacity="0.55"/>',
            f'<text x="{DIVIDER_X - 6}" y="{yc + 4}" text-anchor="end" '
            f'font-family="monospace" font-size="8" fill="{BLUE}" opacity="0.8">1×{n_tok}</text>',
        ]

        # ── Gantt right section ─────────────────────────────────────────────
        # Render done area FIRST so bar and label paint on top of it
        done_x = bx + bw + (IDLE_W if i < n - 1 else 0)
        done_w = gantt_end - done_x
        if done_w > 0:
            svg.append(
                f'<rect x="{done_x}" y="{y_bar}" width="{done_w}" height="{BAR_H}" rx="2" '
                f'fill="#F8F8F8" stroke="#EEEEEE" stroke-width="0.5"/>'
            )

        # Waiting area (from t=0 to when this request's bar starts)
        wait_w = bx - GANTT_X
        if wait_w > 0:
            svg.append(
                f'<rect x="{GANTT_X}" y="{y_bar}" width="{wait_w}" height="{BAR_H}" rx="2" '
                f'fill="#F5F5F5" stroke="#DDD" stroke-width="0.5" stroke-dasharray="2,2"/>'
            )

        # Processing bar + label (both rendered after done/wait areas so they're on top)
        svg.append(
            f'<rect x="{bx}" y="{y_bar}" width="{bw}" height="{BAR_H}" rx="2" '
            f'fill="{BLUE}" opacity="0.85"/>'
        )
        bar_label = f"1×{n_tok}"
        if bw >= 22:
            svg.append(
                f'<text x="{bx + bw//2}" y="{yc + 4}" text-anchor="middle" '
                f'font-family="monospace" font-size="8" fill="white">{bar_label}</text>'
            )
        else:
            svg.append(
                f'<text x="{bx + bw + 3}" y="{yc + 4}" '
                f'font-family="monospace" font-size="8" fill="{BLUE}">{bar_label}</text>'
            )

    # Time axis
    axis_y = y_bot + 8
    svg += [
        f'<line x1="{GANTT_X}" y1="{axis_y}" x2="{gantt_end + 6}" y2="{axis_y}" '
        f'stroke="#CCC" stroke-width="1"/>',
        f'<polygon points="{gantt_end+10},{axis_y} {gantt_end+4},{axis_y-3} '
        f'{gantt_end+4},{axis_y+3}" fill="#CCC"/>',
        # Legend
        f'<rect x="{GANTT_X}" y="{axis_y+14}" width="10" height="10" rx="2" '
        f'fill="{BLUE}" opacity="0.85"/>',
        f'<text x="{GANTT_X+14}" y="{axis_y+23}" font-family="system-ui,sans-serif" '
        f'font-size="8" fill="#888">GPU processing</text>',
        f'<rect x="{GANTT_X+100}" y="{axis_y+14}" width="10" height="10" rx="2" '
        f'fill="#F5F5F5" stroke="#DDD" stroke-width="0.5"/>',
        f'<text x="{GANTT_X+114}" y="{axis_y+23}" font-family="system-ui,sans-serif" '
        f'font-size="8" fill="#888">waiting in queue</text>',
        '</svg>',
    ]
    return "\n".join(svg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, svg in [
        ("diagram-batch-sequential.svg", diagram_sequential()),
        ("diagram-batch-single.svg",     diagram_21()),
        ("diagram-batch-ragged.svg",     diagram_22()),
        ("diagram-batch-padded.svg",     diagram_23()),
    ]:
        path = OUTPUT / filename
        path.write_text(svg)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
