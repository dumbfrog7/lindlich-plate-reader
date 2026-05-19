"""
Plate Reader Growth Curve Analyzer
==================================
A Streamlit app for visualizing OD growth curves from Biotek 96-well
plate reader exports.

Run with:    streamlit run app.py
"""

import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Plotly is used for the *interactive* version of the comparison plot
# (hover tooltips, zoom, pan). Matplotlib is still used for the static
# PNG/SVG downloads. If plotly isn't installed, we fall back to the static
# matplotlib display — the app keeps working, just without hover.
try:
    import plotly.graph_objects as _plotly_go  # noqa: F401
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Plate Reader Analyzer · Lindlich Lab",
    page_icon="🧫",
    layout="wide",
)

LAB_WEBSITE = "https://www.lindlich-lab.com"
LOGO_PATH = Path(__file__).parent / "logo.png"

# Persistent top-left logo with click-through to the lab website
if LOGO_PATH.exists():
    try:
        st.logo(str(LOGO_PATH), link=LAB_WEBSITE, size="large")
    except TypeError:
        st.logo(str(LOGO_PATH))

# Custom CSS: white main area, lab-blue-tinted sidebar, larger header logo
st.markdown(
    """
    <style>
    /* Main area: clean white */
    .stApp {
        background: #ffffff;
    }

    /* Sidebar: subtle lab-blue tint */
    [data-testid="stSidebar"] {
        background: rgba(99, 104, 174, 0.07);
    }

    /* ============== Logo sizing ==============
       `st.logo()` renders the logo image with data-testid="stLogo".
       It appears in two places: inside the SIDEBAR header
       (data-testid="stSidebarHeader") — we want this LARGE — and
       inside the PAGE header (data-testid="stHeader") — we keep this
       small/medium. Streamlit's default sizing renders the image at
       ~32px height as an inline style, so `max-height` alone isn't
       enough; we have to force enlargement with `min-height`. */

    /* Sidebar header container — tall enough to accommodate the big logo */
    [data-testid="stSidebarHeader"] {
        height: auto !important;
        min-height: 140px !important;
        padding: 14px 16px 10px !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }
    /* The <a> wrapping the logo when link= is set — fill available space */
    [data-testid="stSidebarHeader"] a,
    [data-testid="stLogoLink"] {
        display: inline-block !important;
        max-width: 100% !important;
        height: auto !important;
        line-height: 0 !important;
        flex: 1 1 auto !important;
    }
    /* The actual logo image inside the sidebar — FORCED large via min-height.
       Multiple selectors cover slight version differences in Streamlit's DOM. */
    section[data-testid="stSidebar"] [data-testid="stLogo"],
    section[data-testid="stSidebar"] img[data-testid="stLogo"],
    [data-testid="stSidebarHeader"] [data-testid="stLogo"],
    [data-testid="stSidebarHeader"] img,
    [data-testid="stSidebarHeader"] a img {
        height: auto !important;
        width: auto !important;
        min-height: 100px !important;
        max-height: 130px !important;
        max-width: 100% !important;
        object-fit: contain !important;
        display: block !important;
    }

    /* Page header (top of page, only really visible when sidebar collapsed) —
       keep the logo small here so it doesn't dominate the header bar. */
    [data-testid="stHeader"] {
        background: transparent !important;
        height: auto !important;
        min-height: 60px !important;
    }
    [data-testid="stHeader"] [data-testid="stLogo"],
    [data-testid="stHeader"] img[data-testid="stLogo"] {
        max-height: 40px !important;
        min-height: 24px !important;
        height: auto !important;
        width: auto !important;
    }

    /* Tighter, more modern headings */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.015em;
    }

    /* Nicer tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 8px 18px;
        font-weight: 500;
    }

    /* Rounder corners on inputs/widgets for a softer look */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox > div,
    .stMultiSelect > div {
        border-radius: 10px !important;
    }

    /* Compact plate grid buttons — keys start with "_w_".
       Subtle gradient + inset shadow gives the wells a soft "real well" feel.
       Selected wells override the background via the dynamic CSS injection
       in render_plate_selector, but the rounded shape + shadow remain. */
    [class*="st-key-_w_"] button {
        min-height: 24px !important;
        height: 24px !important;
        padding: 0 2px !important;
        font-size: 0.66rem !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #fafafa, #ececec) !important;
        border: 1px solid #d8d8d8 !important;
        color: #666 !important;
        box-shadow:
            inset 0 1px 1px rgba(255,255,255,0.7),
            inset 0 -1px 1px rgba(0,0,0,0.04) !important;
        transition: transform 0.08s !important;
    }
    [class*="st-key-_w_"] button:hover {
        transform: scale(1.08);
        z-index: 5;
    }
    /* Tighter vertical gap between grid rows */
    [class*="st-key-_w_"] {
        margin-bottom: -10px !important;
    }

    /* === Real "96-well plate" frame ===========================
       Wraps the column header row + 8 well rows. Uses clip-path to
       cut TWO corners (bottom-left, bottom-right) to mimic the
       cut/notched corners of a real SBS-format plate (top edge
       stays square, matching the user's actual plate). drop-shadow
       on filter is used instead of box-shadow because filter respects
       clip-path, while box-shadow would be clipped away. */
    [class*="st-key-_plate_frame"] {
        background:
            linear-gradient(180deg, #f7f7f3 0%, #ececea 100%) !important;
        padding: 18px 18px 14px 14px !important;
        position: relative !important;
        margin-bottom: 8px !important;
        clip-path: polygon(
            0% 0%,
            100% 0%,
            100% calc(100% - 18px),
            calc(100% - 18px) 100%,
            18px 100%,
            0% calc(100% - 18px)
        ) !important;
        filter: drop-shadow(0 5px 12px rgba(0,0,0,0.15)) !important;
        /* Inset shadow follows the clipped outline so it traces the
           notched corners just like a real plate's plastic edge */
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.7),
            inset 0 0 0 2px rgba(0,0,0,0.07),
            inset 0 -2px 0 rgba(0,0,0,0.04) !important;
    }

    /* Headers: row letters (A→…H→) and column numbers (1↓…12↓).
       Stronger gray + flat "printed on the plate" look. Solid
       border-bottom on column headers and border-right on row
       letters acts as a visible separator from the wells. */
    [class*="st-key-_plate_frame"] [class*="st-key-_col_"] button,
    [class*="st-key-_plate_frame"] [class*="st-key-_row_"] button {
        background: transparent !important;
        border: none !important;
        color: #3d3d3d !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        box-shadow: none !important;
        min-height: 24px !important;
        height: 24px !important;
        padding: 0 4px !important;
        letter-spacing: 0.03em !important;
        border-radius: 0 !important;
        white-space: nowrap !important;
    }
    /* Column-header row: visible bottom border to separate header
       from the wells below */
    [class*="st-key-_plate_frame"] [class*="st-key-_col_"] button {
        border-bottom: 1.5px solid rgba(0,0,0,0.22) !important;
        margin-bottom: 6px !important;
    }
    /* Row-letter column: visible right border to separate header
       from the wells alongside */
    [class*="st-key-_plate_frame"] [class*="st-key-_row_"] button {
        border-right: 1.5px solid rgba(0,0,0,0.22) !important;
        margin-right: 4px !important;
    }
    [class*="st-key-_plate_frame"] [class*="st-key-_col_"] button:hover,
    [class*="st-key-_plate_frame"] [class*="st-key-_row_"] button:hover {
        background: rgba(99,104,174,0.10) !important;
        color: #3d4080 !important;
    }

    /* Color-coded action buttons (Select all = green, Clear = coral,
       Invert = lab blue). Subtle tints, matching the lab palette. */
    [class*="st-key-_sel_all"] button {
        background: rgba(89,187,137,0.10) !important;
        background-color: rgba(89,187,137,0.10) !important;
        border: 1px solid rgba(89,187,137,0.45) !important;
        color: #2a7a52 !important;
        font-weight: 600 !important;
        min-height: 34px !important;
        height: 34px !important;
    }
    [class*="st-key-_sel_all"] button:hover {
        background: rgba(89,187,137,0.22) !important;
        background-color: rgba(89,187,137,0.22) !important;
        border-color: rgba(89,187,137,0.7) !important;
    }
    [class*="st-key-_sel_clear"] button {
        background: rgba(231,111,81,0.10) !important;
        background-color: rgba(231,111,81,0.10) !important;
        border: 1px solid rgba(231,111,81,0.45) !important;
        color: #a04030 !important;
        font-weight: 600 !important;
        min-height: 34px !important;
        height: 34px !important;
    }
    [class*="st-key-_sel_clear"] button:hover {
        background: rgba(231,111,81,0.22) !important;
        background-color: rgba(231,111,81,0.22) !important;
        border-color: rgba(231,111,81,0.7) !important;
    }
    [class*="st-key-_sel_invert"] button {
        background: rgba(99,104,174,0.10) !important;
        background-color: rgba(99,104,174,0.10) !important;
        border: 1px solid rgba(99,104,174,0.45) !important;
        color: #3d4080 !important;
        font-weight: 600 !important;
        min-height: 34px !important;
        height: 34px !important;
    }
    [class*="st-key-_sel_invert"] button:hover {
        background: rgba(99,104,174,0.22) !important;
        background-color: rgba(99,104,174,0.22) !important;
        border-color: rgba(99,104,174,0.7) !important;
    }

    /* === Vintage accent: typewriter font on sidebar section headings ====
       Scoped narrowly so it never touches icon elements, button text, or
       body labels. Sidebar h-tags = the "1. Load data", "2. Plot settings"
       etc. labels rendered by st.header(). */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,700;0,9..144,800;1,9..144,700;1,9..144,800&family=Special+Elite&display=swap');

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Special Elite', 'Courier New', 'Consolas',
                     monospace !important;
        letter-spacing: 0.01em !important;
        color: #2a2540 !important;
    }

    .app-title-wrap {
        margin: 4px 0 22px 0;
    }
    h1.app-title,
    h1.app-title .curves,
    h1.app-title .dots,
    .app-title-sub {
        font-family: 'Fraunces', 'Playfair Display', Georgia, serif !important;
    }
    h1.app-title {
        font-weight: 800 !important;
        font-size: clamp(1.9rem, 4vw, 2.9rem) !important;
        letter-spacing: -0.022em !important;
        line-height: 1.05 !important;
        color: #2a2540 !important;
        margin: 0 0 10px 0 !important;
        padding: 0 !important;
    }
    h1.app-title .curves {
        font-style: italic;
        font-weight: 800;
        background: linear-gradient(95deg,
            #FCB03B 0%,
            #FCB03B 22%,
            #6368AE 52%,
            #59BB89 82%,
            #59BB89 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent;
    }
    h1.app-title .dots {
        color: #c8c4d8;
        font-style: italic;
        font-weight: 700;
        margin: 0 0.05em;
    }
    /* Underline: soft-ended gradient bar in the three lab colors */
    .app-title-rule {
        height: 5px;
        width: 100%;
        max-width: 640px;
        border-radius: 999px;
        background: linear-gradient(90deg,
            rgba(252,176,59,0)    0%,
            #FCB03B               8%,
            #FCB03B              30%,
            #6368AE              52%,
            #59BB89              74%,
            #59BB89              92%,
            rgba(89,187,137,0)  100%);
    }
    .app-title-sub {
        margin-top: 8px !important;
        color: #6b6680;
        font-size: 0.95rem;
        font-family: 'Fraunces', Georgia, serif !important;
        font-style: italic;
    }

    /* Row/column label buttons (clickable headers) — slightly bigger so the
       arrow hint is readable. The arrow itself is added to the button label
       in render_plate_selector. */
    [class*="st-key-_row_"] button,
    [class*="st-key-_col_"] button {
        background: transparent !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: #666 !important;
        font-weight: 700 !important;
        padding: 0 !important;
        min-height: 22px !important;
        height: 22px !important;
        font-size: 0.78rem !important;
        line-height: 1 !important;
        border-radius: 5px !important;
        transition: background 0.12s, color 0.12s, border 0.12s, transform 0.08s;
    }
    [class*="st-key-_row_"] button:hover,
    [class*="st-key-_col_"] button:hover {
        background: rgba(99,104,174,0.12) !important;
        background-color: rgba(99,104,174,0.12) !important;
        color: #6368AE !important;
        border-color: rgba(99,104,174,0.40) !important;
        transform: scale(1.06);
    }
    /* Quadrant-select buttons — slightly fancier than the row/col labels */
    [class*="st-key-_quad_"] button {
        background: rgba(99,104,174,0.06) !important;
        background-color: rgba(99,104,174,0.06) !important;
        border: 1px solid rgba(99,104,174,0.25) !important;
        color: #4a4880 !important;
        font-weight: 700 !important;
        min-height: 26px !important;
        height: 26px !important;
        padding: 0 6px !important;
        font-size: 0.74rem !important;
        line-height: 1 !important;
        border-radius: 6px !important;
    }
    [class*="st-key-_quad_"] button:hover {
        background: rgba(99,104,174,0.16) !important;
        background-color: rgba(99,104,174,0.16) !important;
        border-color: rgba(99,104,174,0.5) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLATE_ROWS = list("ABCDEFGH")
WELL_PATTERN = re.compile(r"^([A-H])0?([1-9]|1[0-2])$")

# Lab brand colors (first three are the official lab palette)
LAB_YELLOW = "#FCB03B"
LAB_BLUE = "#6368AE"
LAB_GREEN = "#59BB89"

# Extended palette for plots with more than 3 conditions.
# First three = lab colors, then 5 complementary tones — 8 colors total.
# In "Solid" mode, once all 8 colors are used, the next 8 lines reuse the
# same colors but with a dotted line style, then dashed, then dash-dot.
PLOT_PALETTE = [
    LAB_YELLOW,    # 1 — lab yellow
    LAB_BLUE,      # 2 — lab blue
    LAB_GREEN,     # 3 — lab green
    "#E76F51",     # 4 — coral
    "#9D4EDD",     # 5 — purple
    "#264653",     # 6 — dark teal
    "#B5179E",     # 7 — magenta
    "#F4A261",     # 8 — amber
]

# Only the three lab colors — used in "Cycle styles" mode
LAB_3_COLORS = [LAB_YELLOW, LAB_BLUE, LAB_GREEN]

# Matplotlib line styles cycled after the 8-color palette is exhausted.
# Order: solid → dotted → dashed → dash-dot (matches user's preference).
LINE_STYLE_CYCLE = ["-", ":", "--", "-."]

STYLE_MODES = ("Solid", "Cycle styles", "Fade opacity")


def get_line_props(i, style_mode):
    """Compute (color, linestyle, alpha) for the i-th line based on style_mode.

    - Solid: 8 colors. Lines 1-8 solid; 9-16 dotted (same colors reused);
      17-24 dashed; 25-32 dash-dot. So with 8 distinct colors you still get
      32 visually-distinguishable lines.
    - Cycle styles: only the 3 lab colors, cycling line style every 3 lines
      (1-3 solid → 4-6 dotted → 7-9 dashed → 10-12 dash-dot).
    - Fade opacity: 8 colors, all solid, opacity decreases linearly.
    """
    if style_mode == "Cycle styles":
        color = LAB_3_COLORS[i % 3]
        ls = LINE_STYLE_CYCLE[(i // 3) % len(LINE_STYLE_CYCLE)]
        return color, ls, 1.0
    if style_mode == "Fade opacity":
        color = PLOT_PALETTE[i % len(PLOT_PALETTE)]
        alpha = max(0.35, 1.0 - 0.10 * i)
        return color, "-", alpha
    # default "Solid": 8 colors, then cycle line styles
    n_colors = len(PLOT_PALETTE)
    color = PLOT_PALETTE[i % n_colors]
    ls = LINE_STYLE_CYCLE[(i // n_colors) % len(LINE_STYLE_CYCLE)]
    return color, ls, 1.0


def palette_preview_html(style_mode, n=16):
    """Multi-row SVG preview — shows enough lines to make the cycling visible."""
    line_w = 30
    gap = 4
    per_row = 8
    rows = (n + per_row - 1) // per_row
    row_h = 22
    total_w = (line_w + gap) * per_row
    total_h = row_h * rows + 4
    parts = [
        f"<svg width='{total_w}' height='{total_h}' "
        f"style='max-width:100%; display:block;'>"
    ]
    for i in range(n):
        r = i // per_row
        c = i % per_row
        color, ls, alpha = get_line_props(i, style_mode)
        dash = {"-": "none", "--": "5,3", ":": "2,3", "-.": "5,3,2,3"}[ls]
        x = c * (line_w + gap)
        y = r * row_h + 8
        parts.append(
            f"<line x1='{x}' y1='{y}' x2='{x + line_w}' y2='{y}' "
            f"stroke='{color}' stroke-width='3' "
            f"stroke-dasharray='{dash}' opacity='{alpha:.2f}'/>"
        )
        parts.append(
            f"<text x='{x + line_w / 2}' y='{y + 12}' "
            f"text-anchor='middle' font-size='9' fill='#666'>{i + 1}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


# ============================================================
# Helpers
# ============================================================

def normalize_well_name(name):
    """Turn 'A01' into 'A1', leave 'A1' alone. Returns None if not a well name."""
    m = WELL_PATTERN.match(str(name).strip())
    return f"{m.group(1)}{int(m.group(2))}" if m else None


def to_numeric_smart(series):
    """Convert a column to numeric, handling both '.' and ',' as decimal separator."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def parse_gsheet_url(url):
    """Extract sheet ID and gid from a Google Sheets URL."""
    sheet_id_match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    gid_match = re.search(r"[#&?]gid=([0-9]+)", url)
    if not sheet_id_match:
        return None, None
    return sheet_id_match.group(1), (gid_match.group(1) if gid_match else "0")


def gsheet_csv_url(sheet_id, gid):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"


@st.cache_data(ttl=60, show_spinner=False)
def load_from_gsheet(url):
    sheet_id, gid = parse_gsheet_url(url)
    if not sheet_id:
        raise ValueError("Couldn't parse the Google Sheets URL.")
    return pd.read_csv(gsheet_csv_url(sheet_id, gid), header=None)


@st.cache_data(show_spinner=False)
def load_from_upload(file_bytes, filename):
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(file_bytes), header=None)
    return pd.read_csv(BytesIO(file_bytes), header=None)


# ============================================================
# Data processing
# ============================================================

@st.cache_data(show_spinner=False)
def process_data(raw):
    """Turn raw sheet into a clean DataFrame indexed by time in hours.

    Expects:
      - row 0:        column headers
      - column 0:     time in seconds
      - column 1:     temperature (ignored)
      - columns 2+:   well names (A1, B2, …) with OD values
    """
    df = raw.copy()

    # Promote first row to header
    df.columns = df.iloc[0].astype(str).str.strip().tolist()
    df = df.iloc[1:].reset_index(drop=True)

    # First column = time
    time_col = df.columns[0]
    df[time_col] = to_numeric_smart(df[time_col])
    df = df.dropna(subset=[time_col]).reset_index(drop=True)

    # Find well columns
    well_cols = {}  # normalized name -> original column
    for col in df.columns[1:]:
        norm = normalize_well_name(col)
        if norm:
            well_cols[norm] = col
    if not well_cols:
        raise ValueError(
            "No well columns found. Expected headers like A1, B2, etc."
        )

    well_data = pd.DataFrame(index=df[time_col].values / 3600.0)  # seconds → hours
    well_data.index.name = "Time (h)"

    for norm_name, orig_col in well_cols.items():
        well_data[norm_name] = to_numeric_smart(df[orig_col]).values

    well_data = well_data[sorted(well_data.columns, key=lambda w: (w[0], int(w[1:])))]
    return well_data


@st.cache_data(show_spinner=False)
def process_layout(raw):
    """Turn the layout sheet into a {well_name: condition_string} mapping.

    Expects an 8x12 grid with optional 1-12 header row and optional A-H label column.
    """
    df = raw.copy()

    # Header row of 1..12?
    first_row = df.iloc[0].astype(str).str.strip().tolist()
    n_numeric = sum(1 for x in first_row if x.isdigit() and 1 <= int(x) <= 12)
    has_header_row = n_numeric >= 8

    # Label column of A..H?
    first_col = df.iloc[:, 0].astype(str).str.strip().tolist()
    n_letters = sum(1 for x in first_col if x in "ABCDEFGH")
    has_label_col = n_letters >= 4

    if has_header_row:
        df = df.iloc[1:].reset_index(drop=True)
    if has_label_col:
        df = df.iloc[:, 1:].reset_index(drop=True)

    df = df.iloc[:8, :12]

    well_to_cond = {}
    for r_idx in range(len(df)):
        row_letter = PLATE_ROWS[r_idx]
        for c_idx in range(len(df.columns)):
            cell = df.iloc[r_idx, c_idx]
            if pd.notna(cell) and str(cell).strip().lower() not in ("", "nan"):
                well = f"{row_letter}{c_idx + 1}"
                well_to_cond[well] = str(cell).strip()
    return well_to_cond


def apply_calibration(well_data, normalize=False, scale=False, inc_od=0.03):
    """Normalize and/or scale OD data — mirrors the old cal() function.

    normalize: subtract each well's minimum value so the baseline starts at 0.
               (equivalent to: data -= data.min() per column)

    scale:     convert from plate reader OD units to cuvette-equivalent OD.
               The 96-well plate has a shorter light path than a cuvette, so
               raw plate OD ≠ cuvette OD. Division by 0.23 is the empirical
               path-length correction factor used in the lab (Lindlich Lab
               standard). After scaling, inc_od (the biological starting OD
               in cuvette units, typically 0.03) is added back.
               Formula:  OD_cuvette = OD_plate / 0.23 + inc_od

    When both are enabled, normalization is applied first (same order as
    the original cal() function in the lab's Colab scripts).
    """
    data = well_data.copy()
    if normalize:
        # Subtract the per-well minimum so every curve starts near zero
        data = data.subtract(data.min(axis=0), axis=1)
    if scale:
        # Path-length correction + add starting inoculation OD
        data = data / 0.23 + inc_od
    return data


# ============================================================
# Plots
# ============================================================

def make_plate_overview(well_data, max_od=None, log_scale=False,
                        figsize=(14, 9), selection_colors=None,
                        excluded_wells=None, font_family=None,
                        show_titles=True):
    """8×12 grid of mini OD curves at each well's physical position.

    Optional decoration for the Compare-tab thumbnail:
      - selection_colors: {well_name: hex_color} — give the mini-plot a thick
        colored border so the user sees which wells are in the comparison.
      - excluded_wells: iterable of well names — draw their border dashed in
        the same color (or default gray if not in selection_colors) to mark
        them as "excluded from mean".
    """
    selection_colors = selection_colors or {}
    excluded_wells = set(excluded_wells or [])

    rc = {"font.family": font_family} if font_family else {}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(8, 12, figsize=figsize,
                                 sharex=True, sharey=not log_scale)
        time_h = well_data.index.values
        overall_max = (float(np.nanmax(well_data.values))
                       if well_data.size else 1.0)
        y_top = max_od if max_od else overall_max * 1.05

        for r_idx, row_letter in enumerate(PLATE_ROWS):
            for c_idx in range(12):
                ax = axes[r_idx, c_idx]
                well = f"{row_letter}{c_idx + 1}"
                if well in well_data.columns:
                    ax.plot(time_h, well_data[well].values,
                            color=LAB_BLUE, linewidth=1.0)
                    ax.set_facecolor("white")
                else:
                    ax.set_facecolor("#f5f5f5")  # empty well = gray

                if log_scale:
                    ax.set_yscale("log")
                    # Avoid weird auto-scaling when data has zeros: pin a
                    # sensible range based on positive values.
                    if well in well_data.columns:
                        vals = well_data[well].values
                        pos = vals[vals > 0]
                        if pos.size:
                            ax.set_ylim(max(pos.min() * 0.5, 1e-3),
                                        max(pos.max() * 1.5, 0.1))
                else:
                    ax.set_ylim(0, y_top)

                # Hide BOTH major and minor tick marks/labels — log scale
                # auto-adds minor ticks (10⁻¹, 10⁻², …) that would otherwise
                # bleed through these tiny subplots and look like garbage.
                ax.tick_params(axis='both', which='both',
                               bottom=False, left=False,
                               labelbottom=False, labelleft=False)
                ax.set_xticks([])
                ax.set_xticks([], minor=True)
                ax.set_yticks([])
                ax.set_yticks([], minor=True)

                # Border styling: highlight selected + excluded wells
                sel_color = selection_colors.get(well)
                if sel_color and well in excluded_wells:
                    # Selected but excluded → dashed colored border
                    for spine in ax.spines.values():
                        spine.set_color(sel_color)
                        spine.set_linewidth(1.5)
                        spine.set_linestyle((0, (2.5, 2)))
                elif sel_color:
                    # Active in selection → solid thick colored border
                    for spine in ax.spines.values():
                        spine.set_color(sel_color)
                        spine.set_linewidth(2.2)
                else:
                    for spine in ax.spines.values():
                        spine.set_color("#cccccc")
                        spine.set_linewidth(0.5)

                if show_titles and r_idx == 0:
                    ax.set_title(str(c_idx + 1), fontsize=10)
                if show_titles and c_idx == 0:
                    ax.set_ylabel(row_letter, fontsize=10, rotation=0,
                                  labelpad=12, va="center")

        if show_titles:
            fig.suptitle("Plate Overview — all 96 wells", fontsize=12)
        fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False, max_entries=4)
def get_plate_overview_png(well_data, max_od, log_scale, plot_font_key,
                           figsize=(14, 9), dpi=160):
    """Cached PNG bytes for the Plate Overview tab.

    `make_plate_overview` builds 96 matplotlib subplots — expensive (~200-
    400 ms). Because the tab body re-runs on every interaction (including
    clicks on the Compare tab), without caching this re-renders constantly
    for nothing. Caching the PNG bytes means we only re-render when one of
    the actual inputs (data, log scale, max OD, font) changes.

    `plot_font_key` is a hashable handle (str/tuple) so the cache key
    stays stable — Streamlit hashes lists fine, but tuples are simpler.
    """
    font_family = list(plot_font_key) if plot_font_key else None
    fig = make_plate_overview(
        well_data, max_od=max_od, log_scale=log_scale,
        figsize=figsize, font_family=font_family,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_plate_thumbnail_svg(well_data, selection_colors=None,
                             excluded_wells=None, max_od=None,
                             log_scale=False):
    """Lightweight SVG-only plate thumbnail with tiny sparklines per well.

    Replaces the matplotlib 96-subplot thumbnail that was rendered into
    the Compare tab. The matplotlib version was the single biggest
    bottleneck on every well click (~200 ms × every rerun); this version
    is pure string building + a polyline per well, ~5-10 ms total. No
    figure objects, no axes, no RAM held between reruns.

    Visual semantics match the old matplotlib thumbnail:
      - colored border + colored sparkline = active selection
      - diagonal stripes + faded sparkline = excluded replicate
      - light gray fill = empty well (no data)
      - unselected wells: white background + gray sparkline
    """
    selection_colors = selection_colors or {}
    excluded_wells = set(excluded_wells or [])
    available_wells = set(well_data.columns)

    # Layout
    cell_w, cell_h = 28, 22
    gap = 2
    pad_left, pad_top = 18, 16
    width = pad_left + 12 * cell_w + 11 * gap + 4
    height = pad_top + 8 * cell_h + 7 * gap + 4

    # Y-range for sparklines (shared across wells unless log_scale)
    if well_data.size:
        overall_max = float(np.nanmax(well_data.values))
    else:
        overall_max = 1.0
    y_top = max_od if max_od else max(overall_max * 1.05, 1e-3)

    time_vals = well_data.index.values
    if len(time_vals) < 2:
        # Degenerate case: no time series to draw
        return f"<svg width='{width}' height='{height}'></svg>"
    t_min, t_max = float(time_vals.min()), float(time_vals.max())
    t_range = max(t_max - t_min, 1e-9)

    # Downsample to at most ~40 points per sparkline (massive SVG-size win
    # for long runs; visual difference is invisible at 28×22 px per cell)
    n_t = len(time_vals)
    step = max(1, n_t // 40)
    idx_sub = np.arange(0, n_t, step)
    t_sub = time_vals[idx_sub]

    parts = [
        f"<svg viewBox='0 0 {width} {height}' "
        f"xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%; max-width:{width}px; height:auto; "
        f"display:block; margin:auto;'>",
    ]

    # Stripe pattern defs — one per unique excluded color
    excluded_colors = {selection_colors[w] for w in excluded_wells
                       if w in selection_colors}
    if excluded_colors:
        parts.append("<defs>")
        for color in excluded_colors:
            pat_id = f"stripe_{color.lstrip('#')}"
            parts.append(
                f"<pattern id='{pat_id}' patternUnits='userSpaceOnUse' "
                f"width='4' height='4' patternTransform='rotate(45)'>"
                f"<rect width='2' height='4' fill='{color}'/>"
                f"<rect x='2' width='2' height='4' fill='white' "
                f"opacity='0.55'/></pattern>"
            )
        parts.append("</defs>")

    # Column labels 1-12
    for c in range(1, 13):
        cx = pad_left + (c - 1) * (cell_w + gap) + cell_w / 2
        parts.append(
            f"<text x='{cx:.1f}' y='11' text-anchor='middle' "
            f"font-size='9' fill='#888' "
            f"font-family='sans-serif'>{c}</text>"
        )
    # Row labels A-H
    for r_idx, row_letter in enumerate(PLATE_ROWS):
        cy = pad_top + r_idx * (cell_h + gap) + cell_h / 2 + 3
        parts.append(
            f"<text x='9' y='{cy:.1f}' text-anchor='middle' "
            f"font-size='9' fill='#888' "
            f"font-family='sans-serif'>{row_letter}</text>"
        )

    # Cells + sparklines
    for r_idx, row_letter in enumerate(PLATE_ROWS):
        for c_idx in range(12):
            well = f"{row_letter}{c_idx + 1}"
            x = pad_left + c_idx * (cell_w + gap)
            y = pad_top + r_idx * (cell_h + gap)

            color = selection_colors.get(well)
            is_excl = well in excluded_wells

            # ---- Background cell ----
            if well not in available_wells:
                parts.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w}' "
                    f"height='{cell_h}' fill='#f5f5f5' "
                    f"stroke='#e0e0e0' stroke-width='0.5' rx='2'/>"
                )
                continue

            if color and is_excl:
                pat_id = f"stripe_{color.lstrip('#')}"
                parts.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w}' "
                    f"height='{cell_h}' fill='url(#{pat_id})' "
                    f"stroke='{color}' stroke-width='1.4' rx='2'/>"
                )
            elif color:
                parts.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w}' "
                    f"height='{cell_h}' fill='white' "
                    f"stroke='{color}' stroke-width='1.8' rx='2'/>"
                )
            else:
                parts.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w}' "
                    f"height='{cell_h}' fill='white' "
                    f"stroke='#d0d0d0' stroke-width='0.5' rx='2'/>"
                )

            # ---- Sparkline ----
            vals = well_data[well].values[idx_sub]
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                continue
            t_v = t_sub[valid]
            v = vals[valid]

            if log_scale:
                v = np.where(v > 1e-3, v, 1e-3)
                v_log = np.log10(v)
                vmin, vmax = float(v_log.min()), float(v_log.max())
                vr = max(vmax - vmin, 1e-9)
                v_norm = (v_log - vmin) / vr
            else:
                v_norm = np.clip(v, 0.0, y_top) / y_top

            plot_pad = 2.5
            px = x + plot_pad + (t_v - t_min) / t_range * \
                (cell_w - 2 * plot_pad)
            py = (y + cell_h - plot_pad
                  - v_norm * (cell_h - 2 * plot_pad))

            # Build the polyline string
            pts = " ".join(
                f"{px[i]:.1f},{py[i]:.1f}" for i in range(len(px))
            )
            spark_color = color if color else "#6368AE"
            spark_op = 0.45 if is_excl else (0.9 if color else 0.55)
            spark_w = 1.2 if color else 0.9
            parts.append(
                f"<polyline points='{pts}' fill='none' "
                f"stroke='{spark_color}' stroke-width='{spark_w}' "
                f"stroke-opacity='{spark_op}' "
                f"stroke-linejoin='round'/>"
            )

    parts.append("</svg>")
    return "".join(parts)


def make_comparison_plot(well_data, selections, settings):
    """Main comparison plot for selected conditions / wells."""
    data = well_data.copy()

    blank_wells = settings.get("blank_wells") or []
    blank_present = [w for w in blank_wells if w in data.columns]
    blank_applied = False
    if blank_present:
        blank_mean = data[blank_present].mean(axis=1)
        for col in data.columns:
            data[col] = data[col] - blank_mean
        blank_applied = True

    font_family = settings.get("plot_font")
    rc = {"font.family": font_family} if font_family else {}

    legend_pos = settings.get("legend_position", "right")
    # Figure size is adjusted per legend position so the panel has room
    # without squashing the axes. bbox_inches="tight" on save ensures
    # the legend box is never cropped.
    if legend_pos == "right":
        figsize = (12, 6)
    elif legend_pos == "bottom":
        figsize = (10, 7.5)
    elif legend_pos == "top":
        figsize = (10, 7.5)
    else:  # inside ("best")
        figsize = (10, 6)

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        time_h = data.index.values

        style_mode = settings.get("style_mode", "Solid")

        for i, sel in enumerate(selections):
            wells = [w for w in sel["wells"] if w in data.columns]
            if not wells:
                continue
            color, linestyle, alpha = get_line_props(i, style_mode)

            if settings.get("average", True) and len(wells) > 1:
                mean = data[wells].mean(axis=1)
                ax.plot(time_h, mean, label=sel["label"],
                        color=color, linestyle=linestyle, alpha=alpha,
                        linewidth=2)
                if settings.get("error_bars", False):
                    std = data[wells].std(axis=1)
                    ax.fill_between(time_h, mean - std, mean + std,
                                    color=color, alpha=alpha * 0.2,
                                    linewidth=0)
            else:
                for j, w in enumerate(wells):
                    lbl = (f"{sel['label']} ({w})"
                           if len(wells) > 1 else sel["label"])
                    ax.plot(time_h, data[w].values, label=lbl,
                            color=color, linestyle=linestyle,
                            alpha=alpha * (1.0 if j == 0 else 0.6),
                            linewidth=1.5)

        if settings.get("log_scale"):
            ax.set_yscale("log")
        if settings.get("max_od"):
            ax.set_ylim(top=settings["max_od"])

        ax.set_xlabel("Time (hours)")
        # OD$_{600}$ uses matplotlib's mathtext for the subscript "600"
        ax.set_ylabel(
            "OD$_{600}$ (blank-subtracted)" if blank_applied
            else "OD$_{600}$"
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # ---- Legend placement ----
        n_items = len(selections)
        legend_kwargs = dict(
            frameon=True, fancybox=True, framealpha=0.96,
            fontsize=9, borderpad=0.6, labelspacing=0.5,
            handlelength=2.2,
        )
        if legend_pos == "right":
            leg = ax.legend(loc="upper left",
                            bbox_to_anchor=(1.02, 1.0),
                            **legend_kwargs)
            # Leave 22% of horizontal space on the right for the legend
            fig.subplots_adjust(left=0.08, right=0.78,
                                top=0.95, bottom=0.10)
        elif legend_pos == "bottom":
            ncol = min(n_items, 4) if n_items > 0 else 1
            leg = ax.legend(loc="upper center",
                            bbox_to_anchor=(0.5, -0.12),
                            ncol=ncol,
                            **legend_kwargs)
            fig.subplots_adjust(left=0.08, right=0.97,
                                top=0.95, bottom=0.25)
        elif legend_pos == "top":
            ncol = min(n_items, 4) if n_items > 0 else 1
            leg = ax.legend(loc="lower center",
                            bbox_to_anchor=(0.5, 1.03),
                            ncol=ncol,
                            **legend_kwargs)
            fig.subplots_adjust(left=0.08, right=0.97,
                                top=0.80, bottom=0.10)
        else:  # inside, matplotlib picks best location
            leg = ax.legend(loc="best", **legend_kwargs)
            fig.tight_layout()

        # Style the legend box subtly so it reads as its own panel
        if leg is not None:
            frame = leg.get_frame()
            frame.set_edgecolor("#cccccc")
            frame.set_facecolor("#fafafa")
            frame.set_linewidth(0.8)

    return fig


def _rgba_from_hex(hex_color, alpha):
    """Convert '#RRGGBB' + alpha → 'rgba(r,g,b,a)' for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def make_comparison_plot_plotly(well_data, selections, settings):
    """Interactive Plotly version of the comparison plot.

    Same data + styling logic as `make_comparison_plot` (so the screen view
    matches the downloadable PNG/SVG visually), but using Plotly so the
    user gets hover tooltips with the curve name + (time, OD) values, plus
    zoom and pan for free.

    Returns a plotly.graph_objects.Figure, or None if Plotly isn't installed.
    """
    if not PLOTLY_AVAILABLE:
        return None
    import plotly.graph_objects as go

    data = well_data.copy()
    blank_wells = settings.get("blank_wells") or []
    blank_present = [w for w in blank_wells if w in data.columns]
    blank_applied = False
    if blank_present:
        blank_mean = data[blank_present].mean(axis=1)
        for col in data.columns:
            data[col] = data[col] - blank_mean
        blank_applied = True

    style_mode = settings.get("style_mode", "Solid")
    legend_pos = settings.get("legend_position", "right")

    # Map matplotlib line styles to Plotly's dash strings
    DASH_MAP = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}

    fig = go.Figure()
    time_h = data.index.values

    for i, sel in enumerate(selections):
        wells = [w for w in sel["wells"] if w in data.columns]
        if not wells:
            continue
        color, linestyle, alpha = get_line_props(i, style_mode)
        dash = DASH_MAP.get(linestyle, "solid")

        if settings.get("average", True) and len(wells) > 1:
            mean = data[wells].mean(axis=1).values
            fig.add_trace(go.Scatter(
                x=time_h, y=mean,
                name=sel["label"],
                mode="lines",
                line=dict(color=color, width=2.5, dash=dash),
                opacity=alpha,
                hovertemplate=(f"<b>{sel['label']}</b><br>"
                               f"Time: %{{x:.2f}} h<br>"
                               f"OD: %{{y:.4f}}"
                               f"<extra></extra>"),
            ))
            if settings.get("error_bars", False):
                std = data[wells].std(axis=1).values
                upper = mean + std
                lower = mean - std
                # ±SD as filled band; two transparent edge traces with the
                # same legendgroup so they toggle with the mean line.
                fig.add_trace(go.Scatter(
                    x=time_h, y=upper,
                    line=dict(color=color, width=0),
                    showlegend=False, hoverinfo="skip",
                    legendgroup=sel["label"],
                ))
                fig.add_trace(go.Scatter(
                    x=time_h, y=lower,
                    line=dict(color=color, width=0),
                    fill="tonexty",
                    fillcolor=_rgba_from_hex(color, alpha * 0.18),
                    showlegend=False, hoverinfo="skip",
                    legendgroup=sel["label"],
                ))
        else:
            for j, w in enumerate(wells):
                lbl = (f"{sel['label']} ({w})"
                       if len(wells) > 1 else sel["label"])
                fig.add_trace(go.Scatter(
                    x=time_h, y=data[w].values,
                    name=lbl,
                    mode="lines",
                    line=dict(color=color, width=1.8, dash=dash),
                    opacity=alpha * (1.0 if j == 0 else 0.6),
                    hovertemplate=(f"<b>{lbl}</b><br>"
                                   f"Time: %{{x:.2f}} h<br>"
                                   f"OD: %{{y:.4f}}"
                                   f"<extra></extra>"),
                ))

    # ---- Layout ----
    font_family = settings.get("plot_font")
    font_dict = {}
    if font_family:
        if isinstance(font_family, list):
            font_dict["family"] = ", ".join(font_family)
        else:
            font_dict["family"] = font_family

    # Plotly accepts HTML for axis titles → <sub>600</sub> renders as
    # subscript. Same in the hover tooltips above (kept short as "OD").
    y_title = ("OD<sub>600</sub> (blank-subtracted)"
               if blank_applied else "OD<sub>600</sub>")
    layout_kwargs = dict(
        xaxis=dict(title="Time (hours)",
                   gridcolor="rgba(0,0,0,0.08)",
                   zeroline=False),
        yaxis=dict(title=y_title,
                   gridcolor="rgba(0,0,0,0.08)",
                   zeroline=False),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=font_dict if font_dict else dict(family="sans-serif"),
        margin=dict(l=60, r=40, t=30, b=50),
        height=520,
    )
    if settings.get("log_scale"):
        layout_kwargs["yaxis"]["type"] = "log"
    if settings.get("max_od"):
        if settings.get("log_scale"):
            import math
            layout_kwargs["yaxis"]["range"] = [
                math.log10(1e-3),
                math.log10(settings["max_od"]),
            ]
        else:
            layout_kwargs["yaxis"]["range"] = [0, settings["max_od"]]

    # Legend placement matches the matplotlib version's intent
    if legend_pos == "right":
        layout_kwargs["legend"] = dict(
            orientation="v", x=1.02, y=1.0,
            xanchor="left", yanchor="top",
            bgcolor="#fafafa", bordercolor="#cccccc", borderwidth=1,
        )
        layout_kwargs["margin"]["r"] = 200
    elif legend_pos == "bottom":
        layout_kwargs["legend"] = dict(
            orientation="h", x=0.5, y=-0.18,
            xanchor="center", yanchor="top",
            bgcolor="#fafafa", bordercolor="#cccccc", borderwidth=1,
        )
        layout_kwargs["margin"]["b"] = 120
    elif legend_pos == "top":
        layout_kwargs["legend"] = dict(
            orientation="h", x=0.5, y=1.15,
            xanchor="center", yanchor="bottom",
            bgcolor="#fafafa", bordercolor="#cccccc", borderwidth=1,
        )
        layout_kwargs["margin"]["t"] = 90
    else:  # "best" → inside, top-right corner
        layout_kwargs["legend"] = dict(
            x=0.99, y=0.99, xanchor="right", yanchor="top",
            bgcolor="rgba(250,250,250,0.85)",
            bordercolor="#cccccc", borderwidth=1,
        )

    fig.update_layout(**layout_kwargs)
    return fig


def _handle_well_click(well, gk, group_wells_fn):
    """Apply the click rules for the plate selector.

    See render_plate_selector's docstring for the full behavior spec.
    Mutates st.session_state.group_order and st.session_state.group_excluded
    in place.
    """
    group_order = st.session_state.group_order
    group_excluded = st.session_state.group_excluded

    if gk not in group_order:
        # New group — select it, all replicates active, no exclusions
        group_order.append(gk)
        group_excluded[gk] = set()
        return

    excl_set = group_excluded.setdefault(gk, set())
    if well in excl_set:
        # Currently excluded → re-include
        excl_set.discard(well)
        return

    # Currently active. Try to exclude — unless this is the last active well,
    # in which case clicking it deselects the entire group.
    all_wells = group_wells_fn(gk)
    active_count = sum(1 for w in all_wells if w not in excl_set)
    if active_count <= 1:
        group_order.remove(gk)
        group_excluded.pop(gk, None)
    else:
        excl_set.add(well)


def render_plate_selector(well_data, well_to_cond, cond_to_wells,
                          style_mode="Solid", show=("actions", "grid")):
    """8x12 clickable plate with replicate-aware group selection.

    The `show` argument lets the caller render only parts of the UI, so
    different layout zones can be split across columns:

      - "actions" → Select-all/Clear/Invert + Q1-Q4 + count display
      - "grid"    → 8x12 well buttons + clickable row/column headers +
                    mini-legend

    Both modes share the same session_state; calling with `show=("grid",)`
    in one place and `show=("actions",)` in another renders a consistent
    selection across both areas.

    Selection state lives in two session_state keys:

      - st.session_state.group_order: list of group keys (= condition string
        if the well has a layout entry, otherwise the well name itself) in
        the order they were FIRST CLICKED. This order drives color
        assignment, so once a group has a color it keeps it — adding more
        groups afterwards no longer shifts existing colors.
      - st.session_state.group_excluded: dict mapping group_key -> set of
        excluded wells (replicates the user has X-ed out, e.g. due to
        contamination — they stay visible in the grid but are dropped from
        averaging, plotting, and CSV export).

    Click behavior:

      1. Click any well of an unselected group → the whole group becomes
         active (all replicates share the group's color in the grid).
      2. Click an *active* well of an already-selected group → that well is
         excluded (X-marked, strikethrough, striped background). Useful for
         dropping a contaminated replicate from the mean.
      3. Click an *excluded* well → it's re-included.
      4. Click the *last* remaining active well of a group → the entire
         group is deselected (otherwise you couldn't undo a selection).
      5. Singletons (wells without a layout entry) toggle on/off normally.
    """
    available = set(well_data.columns)

    # ---- State init ----
    if "group_order" not in st.session_state:
        st.session_state.group_order = []
    if "group_excluded" not in st.session_state:
        st.session_state.group_excluded = {}

    # ---- Helpers ----
    def group_key(well):
        """The group identifier for a well — its condition or its own name."""
        return well_to_cond.get(well, well)

    def group_wells(gk):
        """All wells belonging to a group, restricted to wells we have data for."""
        if gk in cond_to_wells:
            return [w for w in cond_to_wells[gk] if w in available]
        # singleton (no layout entry) — group key IS the well name
        return [gk] if gk in available else []

    # ---- Prune stale groups (e.g. after reloading data with a different layout) ----
    st.session_state.group_order = [
        gk for gk in st.session_state.group_order if group_wells(gk)
    ]
    for gk in list(st.session_state.group_excluded.keys()):
        if gk not in st.session_state.group_order:
            del st.session_state.group_excluded[gk]
        else:
            st.session_state.group_excluded[gk] = (
                st.session_state.group_excluded[gk] & set(group_wells(gk))
            )

    group_order = st.session_state.group_order
    group_excluded = st.session_state.group_excluded

    # ---- Color assignment: FIXED by click order (no alphabetical re-sorting) ----
    gk_to_props = {
        gk: get_line_props(i, style_mode) for i, gk in enumerate(group_order)
    }

    # ---- Inject per-well CSS ----
    if group_order:

        def _fill_for_linestyle(color, ls):
            """Background CSS whose visual rhythm mirrors the plot line style.

            Two groups that happen to share a color (e.g. lines 1 and 9 in
            Solid mode, where the 8-color palette wraps) get different fill
            patterns here, matching how their lines look in the plot:

              "-"  solid    → solid color block
              ":"  dotted   → small white dots on color (polka-dot)
              "--" dashed   → horizontal color/white bands (stack of dashes)
              "-." dash-dot → alternating long+short horizontal bands

            45° diagonals are deliberately NOT used so they stay reserved
            for the universal "excluded" overlay below.
            """
            if ls == ":":
                return (
                    f"  background-color: {color} !important;"
                    f"  background-image: radial-gradient(circle,"
                    f" rgba(255,255,255,0.85) 1.6px, transparent 1.9px)"
                    f" !important;"
                    f"  background-size: 6px 6px !important;"
                )
            if ls == "--":
                return (
                    f"  background-image: repeating-linear-gradient(0deg,"
                    f" {color} 0 5px,"
                    f" rgba(255,255,255,0.65) 5px 8px) !important;"
                    f"  background-color: {color} !important;"
                )
            if ls == "-.":
                return (
                    f"  background-image: repeating-linear-gradient(0deg,"
                    f" {color} 0 6px,"
                    f" rgba(255,255,255,0.65) 6px 8px,"
                    f" {color} 8px 10px,"
                    f" rgba(255,255,255,0.65) 10px 12px) !important;"
                    f"  background-color: {color} !important;"
                )
            # solid (default)
            return (
                f"  background: {color} !important;"
                f"  background-color: {color} !important;"
            )

        css_rules = []
        for gk in group_order:
            color, ls, alpha = gk_to_props[gk]
            excl_set = group_excluded.get(gk, set())
            for w in group_wells(gk):
                if w in excl_set:
                    # Excluded: 45° diagonal stripes + strikethrough + slight
                    # fade. Same overlay regardless of line style — universal
                    # "won't be plotted" indicator; group identity is still
                    # readable from the border color and the legend.
                    stripe = (
                        f"repeating-linear-gradient(45deg,"
                        f" {color} 0px, {color} 4px,"
                        f" rgba(255,255,255,0.55) 4px,"
                        f" rgba(255,255,255,0.55) 8px)"
                    )
                    css_rules.append(
                        f".st-key-_w_{w} button {{"
                        f"  background: {stripe} !important;"
                        f"  background-color: {color} !important;"
                        f"  border: 1.5px solid {color} !important;"
                        f"  color: #ffffff !important;"
                        f"  text-decoration: line-through !important;"
                        f"  text-decoration-thickness: 2px !important;"
                        f"  text-decoration-color: rgba(0,0,0,0.6) !important;"
                        f"  opacity: 0.92 !important;"
                        f"  font-weight: 700 !important;"
                        f"}}"
                    )
                else:
                    # Active: fill mirrors the plot's line style so two
                    # groups sharing the same color (lines 1 & 9, etc.) are
                    # visually distinct in the grid too.
                    css_rules.append(
                        f".st-key-_w_{w} button {{"
                        f"{_fill_for_linestyle(color, ls)}"
                        f"  border-color: {color} !important;"
                        f"  color: white !important;"
                        f"  opacity: {alpha:.2f} !important;"
                        f"}}"
                    )
        st.markdown(
            "<style>" + "".join(css_rules) + "</style>",
            unsafe_allow_html=True,
        )

    # ---- Bulk-add helper: add the GROUP of each well to the selection,
    #      without touching exclusions. Used by Q1-Q4, row, and column
    #      shortcuts. Defined unconditionally because both the "actions"
    #      block (quadrants) and the "grid" block (row/col headers) use it.
    def bulk_add(wells):
        for w in wells:
            if w not in available:
                continue
            gk_w = group_key(w)
            if gk_w not in st.session_state.group_order:
                st.session_state.group_order.append(gk_w)
                st.session_state.group_excluded.setdefault(gk_w, set())

    # =======================================================================
    # PART 1: Actions (Select-all/Clear/Invert + Q1-Q4 + count + tip)
    # =======================================================================
    if "actions" in show:
        # Side-by-side layout:
        #   LEFT half  = 3 stacked actions (Select all / Clear / Invert)
        #   RIGHT half = 2×2 quadrant grid (Q1 Q2 / Q3 Q4)
        # The 2×2 grid on the right visually mirrors the four physical
        # quadrants of the plate, so Q1 sits top-left, Q2 top-right, etc.
        act_col, quad_col = st.columns([1, 1], gap="small")

        with act_col:
            if st.button("✓ Select all", use_container_width=True,
                         key="_sel_all"):
                existing = set(st.session_state.group_order)
                new_gks = []
                for gk in sorted(cond_to_wells.keys()):
                    if gk not in existing and group_wells(gk):
                        new_gks.append(gk)
                for w in sorted(available,
                                key=lambda x: (x[0], int(x[1:]))):
                    if w not in well_to_cond and w not in existing:
                        new_gks.append(w)
                st.session_state.group_order = (
                    list(st.session_state.group_order) + new_gks
                )
                st.rerun()
            if st.button("✗ Clear", use_container_width=True,
                         key="_sel_clear"):
                st.session_state.group_order = []
                st.session_state.group_excluded = {}
                st.rerun()
            if st.button("⇄ Invert", use_container_width=True,
                         key="_sel_invert"):
                all_gks = {group_key(w) for w in available}
                current = set(st.session_state.group_order)
                st.session_state.group_order = sorted(all_gks - current)
                st.session_state.group_excluded = {}
                st.rerun()

        with quad_col:
            QUADRANTS = [
                ("Q1", "A1–D6",
                 [f"{r}{c}" for r in "ABCD" for c in range(1, 7)]),
                ("Q2", "A7–D12",
                 [f"{r}{c}" for r in "ABCD" for c in range(7, 13)]),
                ("Q3", "E1–H6",
                 [f"{r}{c}" for r in "EFGH" for c in range(1, 7)]),
                ("Q4", "E7–H12",
                 [f"{r}{c}" for r in "EFGH" for c in range(7, 13)]),
            ]
            qrow1 = st.columns(2)
            qrow2 = st.columns(2)
            cols_in_order = list(qrow1) + list(qrow2)
            # Short label "Q1" only — the well range goes in the tooltip;
            # the narrower column wouldn't fit "Q1 (A1–D6)" comfortably.
            for col, (qname, qrange, qwells) in zip(
                    cols_in_order, QUADRANTS):
                with col:
                    if st.button(qname,
                                 key=f"_quad_{qname}",
                                 use_container_width=True,
                                 help=f"Select all wells in {qname} "
                                      f"({qrange})."):
                        bulk_add(qwells)
                        st.rerun()

        # Count + tip
        n_active = sum(
            len([w for w in group_wells(gk)
                 if w not in group_excluded.get(gk, set())])
            for gk in group_order
        )
        n_excl = sum(len(s) for s in group_excluded.values())
        n_g = len(group_order)
        bits = [
            f"<b>{n_active}</b> well{'s' if n_active != 1 else ''} active"
        ]
        if n_excl:
            bits.append(
                f"<span style='color:#c45a45;'>"
                f"<b>{n_excl}</b> excluded</span>"
            )
        if n_g:
            bits.append(f"<b>{n_g}</b> group{'s' if n_g != 1 else ''}")
        st.markdown(
            "<div style='margin-top: 8px; color:#555; "
            "font-size: 0.85rem;'>"
            + " · ".join(bits) + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#888; font-size: 0.78rem; "
            "margin-top: 4px;'>"
            "💡 Click a <b>row letter</b> or <b>column number</b> in the "
            "grid to select that whole row/column."
            "</div>",
            unsafe_allow_html=True,
        )

    # =======================================================================
    # PART 2: Grid (column headers + 8 rows of well buttons + mini-legend)
    # =======================================================================
    if "grid" in show:
        # Wrap the column header row + 8 well rows in a styled
        # container that visually looks like a real 96-well plate
        # (rounded plastic frame, soft shadow, A1-corner notch). The
        # mini-legend below stays OUTSIDE the frame — it summarises
        # the selection, it isn't part of the plate itself.
        with st.container(key="_plate_frame"):
            # ---- Column number header row (clickable) ----
            # Row-letter column is wider (1.1 vs the old 0.7) so the
            # "A →" / "B →" label fits on a single line with breathing
            # room for the arrow next to the letter.
            head = st.columns([1.1] + [1] * 12)
            head[0].markdown("&nbsp;", unsafe_allow_html=True)
            for i in range(1, 13):
                with head[i]:
                    col_wells = [f"{r}{i}" for r in PLATE_ROWS]
                    # Arrow ↓ hints clicking selects DOWN that column
                    if st.button(f"{i} ↓", key=f"_col_{i}",
                                 use_container_width=True,
                                 help=f"Select all wells in column {i}"):
                        bulk_add(col_wells)
                        st.rerun()

            # ---- 8 rows of well buttons (with clickable row letter) ----
            for row_letter in PLATE_ROWS:
                row_cols = st.columns([1.1] + [1] * 12)
                with row_cols[0]:
                    row_wells = [f"{row_letter}{c}" for c in range(1, 13)]
                    # Arrow → hints clicking selects ACROSS that row
                    if st.button(f"{row_letter} →",
                                 key=f"_row_{row_letter}",
                                 use_container_width=True,
                                 help=f"Select all wells in row {row_letter}"):
                        bulk_add(row_wells)
                        st.rerun()
                for c in range(1, 13):
                    well = f"{row_letter}{c}"
                    with row_cols[c]:
                        if well not in available:
                            # Empty well — render a small dot, no button
                            st.markdown(
                                "<div style='text-align:center; "
                                "color:#d0d0d0; padding: 4px 0; "
                                "font-size: 1rem; line-height: 1;'>·</div>",
                                unsafe_allow_html=True,
                            )
                            continue

                        cond = well_to_cond.get(well, "")
                        gk = group_key(well)
                        excl_set = group_excluded.get(gk, set())
                        is_excluded = well in excl_set
                        in_selection = gk in group_order

                        # Tooltip reflects the current interactive state
                        if cond:
                            if is_excluded:
                                tooltip = (f"{well} — {cond}  ✕ excluded "
                                           f"(click to re-include)")
                            elif in_selection:
                                n_active_in_group = sum(
                                    1 for w in group_wells(gk)
                                    if w not in excl_set
                                )
                                if n_active_in_group > 1:
                                    tooltip = (
                                        f"{well} — {cond}  "
                                        f"(click to exclude from mean)"
                                    )
                                else:
                                    tooltip = (
                                        f"{well} — {cond}  "
                                        f"(last active — "
                                        f"click to deselect group)"
                                    )
                            else:
                                n_rep = len(group_wells(gk))
                                if n_rep > 1:
                                    tooltip = (
                                        f"{well} — {cond}  "
                                        f"(click selects all "
                                        f"{n_rep} replicates)"
                                    )
                                else:
                                    tooltip = f"{well} — {cond}"
                        else:
                            tooltip = well

                        # Diagonal stripes + strikethrough handle the
                        # visual; no extra ✕ glyph needed in the label.
                        label = well

                        if st.button(label, key=f"_w_{well}",
                                     help=tooltip,
                                     use_container_width=True):
                            _handle_well_click(well, gk, group_wells)
                            st.rerun()

    # =======================================================================
    # PART 3: Legend (mini selection summary — rendered separately so the
    # caller can place it AFTER the "Show Chosen Plots" button instead of
    # squeezed between the plate and the button)
    # =======================================================================
    if "legend" in show:
        # ---- Selection summary (mini-legend, in click order) ----
        if group_order:
            legend_html = [
                "<div style='margin: 14px 0 4px 0; padding: 12px 16px; ",
                "background: rgba(99,104,174,0.05); border-radius: 8px; ",
                "border-left: 3px solid #6368AE; font-size: 0.88rem;'>",
                "<div style='font-weight: 600; margin-bottom: 6px; "
                "color:#333;'>",
                "Preview — these groups will be plotted "
                "<span style='font-weight:400; color:#888;'>"
                "(in click order)</span>:</div>",
            ]
            for gk in group_order:
                color, ls, alpha = gk_to_props[gk]
                dash = {
                    "-": "none", "--": "5,3", ":": "2,3", "-.": "5,3,2,3"
                }[ls]
                line_svg = (
                    f"<svg width='42' height='10' "
                    f"style='vertical-align: middle; flex-shrink: 0;'>"
                    f"<line x1='1' y1='5' x2='41' y2='5' "
                    f"stroke='{color}' stroke-width='3' "
                    f"stroke-dasharray='{dash}' opacity='{alpha:.2f}'/>"
                    f"</svg>"
                )
                all_wells = sorted(group_wells(gk))
                excl_set = group_excluded.get(gk, set())
                active = [w for w in all_wells if w not in excl_set]
                excluded = [w for w in all_wells if w in excl_set]

                note_parts = []
                if len(active) > 1:
                    note_parts.append(f"{len(active)} replicates → mean")
                if excluded:
                    note_parts.append(
                        f"<span style='color:#c45a45;'>"
                        f"✕ excluded: {', '.join(excluded)}</span>"
                    )
                note_html = (
                    f" <span style='color:#999;'>"
                    f"({' · '.join(note_parts)})</span>"
                    if note_parts else ""
                )
                wells_str = ", ".join(active) if active else "—"
                legend_html.append(
                    f"<div style='display: flex; align-items: center; "
                    f"gap: 10px; margin-top: 4px; flex-wrap: wrap;'>"
                    f"{line_svg}"
                    f"<span style='font-weight: 500; color: #333;'>"
                    f"{gk}</span>"
                    f"<span style='color: #888; font-size: 0.82rem;'>"
                    f"[{wells_str}]</span>"
                    f"{note_html}"
                    f"</div>"
                )
            legend_html.append("</div>")
            st.markdown("".join(legend_html), unsafe_allow_html=True)


# ============================================================
# UI
# ============================================================

st.markdown(
    f"""
    <div class="app-title-wrap">
        <h1 class="app-title">
            Well, Well, Well<span class="dots">…</span>
            <span class="curves">Look at those Curves!</span>
        </h1>
        <div class="app-title-rule"></div>
        <div class="app-title-sub">
            Visualize OD growth curves from your Biotek 96-well plate exports ·
            <a href="{LAB_WEBSITE}" style="color:#6368AE; text-decoration:none;
                font-weight:600;">Lindlich Lab</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----- Sidebar: data + global settings -----
with st.sidebar:
    st.header("1. Load data")
    source = st.radio("Source", ["Google Sheets", "Upload file"], horizontal=True)

    if source == "Google Sheets":
        data_url = st.text_input(
            "Data tab URL",
            help="In Google Sheets, click your DATA tab, then copy the URL "
                 "from your browser's address bar.",
        )
        layout_url = st.text_input(
            "Layout tab URL (optional)",
            help="The URL of your layout_flat tab. Leave blank if you don't have one.",
        )
        if st.button("📥 Load from Sheets", use_container_width=True):
            try:
                if data_url:
                    st.session_state["raw_data"] = load_from_gsheet(data_url)
                    st.success(f"Loaded data ({len(st.session_state['raw_data'])} rows)")
                if layout_url:
                    st.session_state["raw_layout"] = load_from_gsheet(layout_url)
                    st.success("Loaded layout")
                else:
                    st.session_state.pop("raw_layout", None)
            except Exception as e:
                st.error(f"Couldn't load: {e}")

    else:
        data_file = st.file_uploader("Data file", type=["csv", "xlsx", "xls"])
        layout_file = st.file_uploader("Layout file (optional)",
                                       type=["csv", "xlsx", "xls"])
        if data_file is not None:
            try:
                st.session_state["raw_data"] = load_from_upload(
                    data_file.read(), data_file.name)
                st.success(f"Loaded data ({len(st.session_state['raw_data'])} rows)")
            except Exception as e:
                st.error(f"Couldn't load data: {e}")
        if layout_file is not None:
            try:
                st.session_state["raw_layout"] = load_from_upload(
                    layout_file.read(), layout_file.name)
                st.success("Loaded layout")
            except Exception as e:
                st.error(f"Couldn't load layout: {e}")

    st.divider()
    st.header("2. Plot settings")
    log_scale = st.checkbox("Log-scale Y-axis", value=True)
    use_max_od = st.checkbox("Cap Y-axis maximum")
    max_od = None
    if use_max_od:
        max_od = st.number_input(
            "Max OD", min_value=0.01, max_value=20.0,
            value=1.5, step=0.1, label_visibility="collapsed")

    # Plot font choice — matplotlib uses its own font handling, separate from
    # the typewriter style used in the app body. "Default" leaves matplotlib's
    # built-in DejaVu Sans alone (recommended for readability of axis labels).
    PLOT_FONT_OPTIONS = {
        "Default (sans-serif)": None,
        "Serif (Times-like)":   ["serif"],
        "Monospace (Courier)":  ["monospace"],
        "Helvetica / Arial":    ["Helvetica", "Arial", "sans-serif"],
        "Times New Roman":      ["Times New Roman", "Times", "serif"],
        "Courier New":          ["Courier New", "Courier", "monospace"],
        "Comic Sans MS":        ["Comic Sans MS", "Chalkboard SE", "sans-serif"],
    }
    plot_font_label = st.selectbox(
        "Plot font",
        list(PLOT_FONT_OPTIONS.keys()),
        index=0,
        help="Font family for axis labels, legend, and titles inside the "
             "matplotlib plots. The typewriter look of the app interface "
             "is *not* applied here by default — small axis text can become "
             "hard to read in heavy display fonts. Pick another option to "
             "match the rest of the app, or leave on Default."
    )
    plot_font = PLOT_FONT_OPTIONS[plot_font_label]

    LEGEND_POSITIONS = {
        "Right (outside, boxed)":  "right",
        "Bottom (outside, boxed)": "bottom",
        "Top (outside, boxed)":    "top",
        "Inside the plot (auto)":  "best",
    }
    legend_position_label = st.selectbox(
        "Legend position",
        list(LEGEND_POSITIONS.keys()),
        index=0,
        help="Where to place the legend on the comparison plot:\n"
             "• **Right / Bottom / Top (outside)** — legend sits in its "
             "own boxed panel outside the axes so it never overlaps the "
             "curves. Default is Right.\n"
             "• **Inside the plot** — matplotlib picks the best spot "
             "inside the axes. Compact but can hide data when the legend "
             "is large."
    )
    legend_position = LEGEND_POSITIONS[legend_position_label]

    st.divider()
    st.header("3. Comparison options")
    average = st.checkbox("Average replicates", value=True,
                         help="Group wells with the same condition label "
                              "into one mean line.")
    error_bars = st.checkbox(
        "Show error bars (±SD)", value=False, disabled=not average,
        help="Only active when averaging replicates."
    )
    blank_sub = st.checkbox(
        "Subtract negative control",
        help="Pick the condition that represents your blank/negative control. "
             "Its mean OD is subtracted from every plotted line.")
    # blank_wells_sel is filled in after we know the layout (later in the script)

    st.markdown("**Calibration**")
    normalize = st.checkbox(
        "Normalize (subtract minimum per well)",
        value=True,
        help="Subtracts each well's minimum OD value so every curve starts "
             "near zero. Useful for removing baseline drift between wells. "
             "Applied before scaling if both are active.",
    )
    scale_to_cuvette = st.checkbox(
        "Scale to cuvette OD (÷ 0.23)",
        value=True,
        help="Converts plate reader OD units to cuvette-equivalent OD using "
             "the lab's empirical path-length correction factor (0.23). "
             "Formula: OD_cuvette = OD_plate ÷ 0.23 + starting OD",
    )
    inc_od = 0.03
    if scale_to_cuvette:
        inc_od = st.number_input(
            "Starting OD (cuvette units)",
            min_value=0.0, max_value=1.0,
            value=0.03, step=0.005, format="%.3f",
            help="The inoculation OD in cuvette units added back after "
                 "scaling. Default 0.03 matches the lab standard (inc_OD).",
        )

    st.divider()
    st.header("4. Line styles & colors")
    style_mode = st.selectbox(
        "Style mode",
        STYLE_MODES,
        index=0,
        help="How to distinguish lines:\n"
             "• **Solid** — 8 distinct colors with solid lines. Beyond 8 "
             "conditions, the same colors repeat but with dotted, then "
             "dashed, then dash-dot lines (gives 32 distinguishable lines).\n"
             "• **Cycle styles** — only the 3 lab colors, but the line style "
             "cycles every 3 conditions (solid → dotted → dashed → dash-dot). "
             "Great for grouping related conditions visually.\n"
             "• **Fade opacity** — 8 colors, all solid, each next line "
             "slightly more transparent."
    )
    st.caption("**Preview** (order conditions will be drawn):")
    st.markdown(palette_preview_html(style_mode, n=16), unsafe_allow_html=True)
    st.caption(
        "💡 Colors are assigned in the **order you click** conditions on the "
        "plate. Once a group has a color, it keeps it — clicking more wells "
        "won't reshuffle existing colors."
    )

# ----- Main area -----
if "raw_data" not in st.session_state:
    st.info("👈 Load your data from the sidebar to get started.")
    st.markdown(
        """
        **Quick start**
        1. Pick **Google Sheets** or **Upload file** on the left.
        2. (Optional) Add a `layout_flat` tab/file — an 8×12 grid where each cell
           holds the condition name for that well. Wells with the same name are
           treated as replicates.
        3. Click **Load**, then switch between the **Plate Overview** and
           **Compare Conditions** tabs.
        """
    )
    st.stop()

try:
    well_data = process_data(st.session_state["raw_data"])
except Exception as e:
    st.error(f"Couldn't process data: {e}")
    st.stop()

# Apply normalization and/or scaling if enabled in the sidebar.
# This mirrors the old cal() function from the lab's Colab scripts.
# normalize=True  → subtract per-well minimum (baseline to zero)
# scale_to_cuvette=True → divide by 0.23 path-length factor + add inc_od
if normalize or scale_to_cuvette:
    well_data = apply_calibration(well_data,
                                  normalize=normalize,
                                  scale=scale_to_cuvette,
                                  inc_od=inc_od)

well_to_cond = {}
cond_to_wells = {}
if "raw_layout" in st.session_state:
    try:
        well_to_cond = process_layout(st.session_state["raw_layout"])
        for w, c in well_to_cond.items():
            cond_to_wells.setdefault(c, []).append(w)
    except Exception as e:
        st.warning(f"Couldn't process layout: {e}")

# ----- Sidebar part 2: blank selector (needs layout) + footer -----
blank_wells_sel = []
with st.sidebar:
    if blank_sub:
        if cond_to_wells:
            blank_cond = st.selectbox(
                "Choose blank condition", sorted(cond_to_wells.keys()),
                key="_blank_cond")
            blank_wells_sel = cond_to_wells.get(blank_cond, [])
            st.caption(f"Subtracting mean of: {', '.join(blank_wells_sel)}")
        else:
            blank_wells_sel = st.multiselect(
                "Choose blank well(s)", sorted(well_data.columns),
                help="No layout loaded — pick wells manually.",
                key="_blank_wells")

    st.divider()
    st.caption(
        f"Runs locally · your data never leaves this computer  \n"
        f"Made for [Lindlich Lab]({LAB_WEBSITE}) 💜"
    )

# Summary banner
n_wells = len(well_data.columns)
n_t = len(well_data)
duration_h = float(well_data.index.max()) if n_t else 0.0
banner = (f"**{n_wells}** wells · **{n_t}** time points · "
          f"**{duration_h:.1f} h** duration")
if cond_to_wells:
    banner += f" · **{len(cond_to_wells)}** conditions"
banner += " · **1** sexy plot"
st.markdown(banner)

tab_overview, tab_compare = st.tabs(["Plate Overview", "Compare Conditions"])

with tab_overview:
    # Cached PNG bytes — avoids re-running 96 matplotlib subplots on
    # every rerun (tab bodies in Streamlit ALWAYS re-execute, even
    # when the tab isn't visible). Cache key is the data + log scale
    # + max OD + font choice.
    plot_font_key = tuple(plot_font) if plot_font else None
    png_bytes = get_plate_overview_png(
        well_data,
        max_od=max_od if use_max_od else None,
        log_scale=log_scale,
        plot_font_key=plot_font_key,
    )
    st.image(png_bytes, use_container_width=True)
    st.download_button("⬇️ Download PNG", png_bytes,
                       "plate_overview.png", "image/png")

with tab_compare:
    # ---- Prune stale state (data may have changed since last rerun) ----
    _available = set(well_data.columns)
    st.session_state.setdefault("group_order", [])
    st.session_state.setdefault("group_excluded", {})

    def _group_wells(gk):
        if gk in cond_to_wells:
            return [w for w in cond_to_wells[gk] if w in _available]
        return [gk] if gk in _available else []

    st.session_state.group_order = [
        gk for gk in st.session_state.group_order if _group_wells(gk)
    ]
    for _gk in list(st.session_state.group_excluded.keys()):
        if _gk not in st.session_state.group_order:
            del st.session_state.group_excluded[_gk]
        else:
            st.session_state.group_excluded[_gk] = (
                st.session_state.group_excluded[_gk] & set(_group_wells(_gk))
            )

    # ---- Compute selection colors for the thumbnail ----
    _selection_colors = {}
    _excluded_wells = set()
    for _i, _gk in enumerate(st.session_state.group_order):
        _color, _, _ = get_line_props(_i, style_mode)
        for _w in _group_wells(_gk):
            _selection_colors[_w] = _color
        _excluded_wells.update(st.session_state.group_excluded.get(_gk, set()))

    # ---- Side-by-side layout: selector (left, wide) + thumbnail (right) ----
    # Side-by-side layout:
    #   LEFT (wide)   = grid only — visible at the very top of the page
    #   RIGHT (narrow)= thumbnail + actions + how-to-use expander
    # The grid is what the user does most work in, so it gets the top
    # of the screen with no instructions or buttons stacked above it.
    grid_col, side_col = st.columns([7, 3], gap="medium")

    with side_col:
        # 1. Action buttons + Q1-Q4 + count (now at top — most-used controls
        #    sit prominently above the thumbnail)
        render_plate_selector(well_data, well_to_cond, cond_to_wells,
                              style_mode=style_mode, show=("actions",))

        st.markdown(
            "<div style='margin-top: 14px;'></div>",
            unsafe_allow_html=True,
        )

        # 2. Mini plate thumbnail (at-a-glance reference of current
        #    selection). Pure SVG — no matplotlib, no figure objects,
        #    no 96-subplot tight_layout pass. Renders in ~5-10 ms
        #    instead of ~200 ms, so clicking wells stays snappy.
        thumb_svg = make_plate_thumbnail_svg(
            well_data,
            selection_colors=_selection_colors,
            excluded_wells=_excluded_wells,
            max_od=max_od if use_max_od else None,
            log_scale=log_scale,
        )
        st.markdown(thumb_svg, unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.75rem; color:#888; "
            "margin: 4px 0 8px 0; text-align: center;'>"
            "<i>At a glance: colored border = selected · "
            "dashed = excluded</i>"
            "</div>",
            unsafe_allow_html=True,
        )

        # 3. Collapsed how-to-use (stays at the bottom)
        with st.expander("How to use the plate selector",
                         expanded=False):
            st.markdown(
                "**Click a well** to select its condition — all "
                "replicates in the layout light up in the same color "
                "automatically. **Click an already-selected replicate "
                "again** to mark it as excluded (useful for dropping a "
                "contaminated well from the mean) — excluded wells "
                "appear with diagonal stripes and a strikethrough. "
                "Click an excluded well to re-include it. Click the "
                "*last* active well of a group to drop the whole "
                "group.\n\n"
                "Use the **Q1–Q4** buttons for quadrants, or click a "
                "**row letter / column number** to select that whole "
                "row or column at once.\n\n"
                "When you're happy with the selection, hit *Show "
                "Chosen Plots* below. Empty wells (no data) are shown "
                "as small dots."
            )

    with grid_col:
        # The 8×12 well grid — the main interaction surface
        render_plate_selector(well_data, well_to_cond, cond_to_wells,
                              style_mode=style_mode, show=("grid",))

    # ---- Show / Hide plot buttons sit DIRECTLY under the grid so the user
    # doesn't have to scroll past the preview legend to click "Show Plots".
    # The preview legend renders BELOW the buttons (it confirms what will be
    # plotted — useful after clicking the button, not before).
    b1, b2 = st.columns([3, 1])
    with b1:
        show_clicked = st.button(
            "📊 Show Chosen Plots",
            type="primary", use_container_width=True,
            key="_show_plots_btn",
            help="Generate the plot from the currently selected wells. "
                 "The plot is frozen until you click this again — so picking "
                 "more wells afterward won't slow things down.",
        )
    with b2:
        hide_clicked = st.button(
            "🗑️ Hide plot",
            use_container_width=True,
            key="_hide_plot_btn",
            help="Remove the current plot from view.",
        )

    # Preview legend — what will be plotted, in click order
    render_plate_selector(well_data, well_to_cond, cond_to_wells,
                          style_mode=style_mode, show=("legend",))

    st.markdown("---")

    if hide_clicked:
        st.session_state.pop("_frozen_plot", None)

    if show_clicked:
        group_order = st.session_state.get("group_order", [])
        group_excluded = st.session_state.get("group_excluded", {})

        # Build ordered selections, dropping excluded wells. Group order
        # matches click order, so plot line colors match grid button colors.
        selections = []
        for gk in group_order:
            if gk in cond_to_wells:
                wells_in_group = [w for w in cond_to_wells[gk]
                                  if w in well_data.columns]
            else:
                wells_in_group = ([gk] if gk in well_data.columns else [])
            excl = group_excluded.get(gk, set())
            active = sorted(w for w in wells_in_group if w not in excl)
            if active:
                selections.append({"label": gk, "wells": active})

        n_active = sum(len(s["wells"]) for s in selections)

        if not selections:
            st.warning("No active wells — click some on the plate first.")
        else:
            settings = {
                "log_scale": log_scale,
                "max_od": max_od if use_max_od else None,
                "average": average,
                "error_bars": error_bars,
                "blank_wells": blank_wells_sel if blank_sub else None,
                "style_mode": style_mode,
                "plot_font": plot_font,
                "legend_position": legend_position,
            }

            # Render the matplotlib version once for the PNG/SVG downloads,
            # AND the Plotly version for the interactive on-screen display.
            # Both freeze into session_state so reruns are instant.
            fig_cmp = make_comparison_plot(well_data, selections, settings)
            png_buf = BytesIO()
            fig_cmp.savefig(png_buf, format="png", dpi=160,
                            bbox_inches="tight")
            svg_buf = BytesIO()
            fig_cmp.savefig(svg_buf, format="svg", bbox_inches="tight")
            plt.close(fig_cmp)

            # Plotly interactive version — JSON for portability across reruns.
            plotly_json = None
            if PLOTLY_AVAILABLE:
                fig_plotly = make_comparison_plot_plotly(
                    well_data, selections, settings
                )
                if fig_plotly is not None:
                    plotly_json = fig_plotly.to_json()

            # Build CSV from selections + settings
            export_df = pd.DataFrame(index=well_data.index)
            d = well_data.copy()
            if settings["blank_wells"]:
                bm = d[settings["blank_wells"]].mean(axis=1)
                for col in d.columns:
                    d[col] = d[col] - bm
            for sel in selections:
                wells = [w for w in sel["wells"] if w in d.columns]
                if not wells:
                    continue
                if average and len(wells) > 1:
                    export_df[f"{sel['label']} mean"] = d[wells].mean(axis=1)
                    export_df[f"{sel['label']} sd"] = d[wells].std(axis=1)
                else:
                    for w in wells:
                        export_df[f"{sel['label']} [{w}]"] = d[w]
            csv_bytes = export_df.to_csv().encode("utf-8")

            st.session_state["_frozen_plot"] = {
                "png": png_buf.getvalue(),
                "svg": svg_buf.getvalue(),
                "csv": csv_bytes,
                "plotly_json": plotly_json,
                "n_lines": len(selections),
                "n_wells": n_active,
            }

    # Display the frozen plot (if any)
    frozen = st.session_state.get("_frozen_plot")
    if frozen:
        # Prefer the interactive Plotly version for on-screen display so
        # the user gets hover tooltips with the curve name. Fall back to
        # the static PNG if plotly isn't installed.
        if frozen.get("plotly_json"):
            import plotly.io as pio
            fig_plotly = pio.from_json(frozen["plotly_json"])
            st.plotly_chart(fig_plotly, use_container_width=True,
                            config={"displaylogo": False,
                                    "modeBarButtonsToRemove":
                                        ["lasso2d", "select2d"]})
        else:
            st.image(frozen["png"], use_container_width=True)
            if not PLOTLY_AVAILABLE:
                st.caption("💡 Install `plotly` to get interactive hover "
                           "tooltips on the plot: `pip install plotly`")
        st.caption(
            f"Showing **{frozen['n_lines']}** "
            f"line{'s' if frozen['n_lines'] != 1 else ''} "
            f"from **{frozen['n_wells']}** "
            f"well{'s' if frozen['n_wells'] != 1 else ''}. "
            "This view is frozen — pick more wells and hit "
            "*Show Chosen Plots* again to refresh."
        )
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button("⬇️ PNG", frozen["png"], "comparison.png",
                               "image/png", use_container_width=True)
        with e2:
            st.download_button("⬇️ SVG", frozen["svg"], "comparison.svg",
                               "image/svg+xml", use_container_width=True)
        with e3:
            st.download_button("⬇️ CSV", frozen["csv"], "comparison.csv",
                               "text/csv", use_container_width=True)
