"""
====================================================================
Student Performance Assessment System
====================================================================

Module:
    utils/styles.py

Purpose:
    Central color palette and global CSS theme for the Streamlit
    dashboard.

Usage
-----
Call load_theme() once, right after st.set_page_config(), and
inject it with st.markdown(..., unsafe_allow_html=True):

    st.markdown(load_theme(), unsafe_allow_html=True)

Custom CSS classes available for use in st.markdown(...,
unsafe_allow_html=True) calls on individual pages:

    <div class="mode-card">...</div>       -- card container
    <div class="success-box">...</div>     -- green callout
    <div class="warning-box">...</div>     -- amber callout
    <div class="info-box">...</div>        -- blue callout

These are optional -- st.success()/st.warning()/st.info() remain
the default on every page; the *-box classes are available if a
page wants a styled callout instead (e.g. inside a custom-HTML
card where a plain st.success() wouldn't fit visually).

Version:
    6.0
====================================================================
"""

from __future__ import annotations


# ============================================================
# COLOR PALETTE
# ============================================================

PRIMARY = "#2E7D32"          # Main Green

PRIMARY_DARK = "#1B5E20"

PRIMARY_LIGHT = "#66BB6A"

SUCCESS = "#388E3C"

SUCCESS_LIGHT = "#E8F5E9"

WARNING = "#F9A825"

WARNING_LIGHT = "#FFF8E1"

INFO = "#1565C0"

INFO_LIGHT = "#E3F2FD"

ERROR = "#C62828"

ERROR_LIGHT = "#FFEBEE"

BACKGROUND = "#F8FAFC"

CARD = "#FFFFFF"

TEXT = "#1F2937"

TEXT_LIGHT = "#6B7280"

BORDER = "#DCE3E8"


# ============================================================
# GLOBAL THEME
# ============================================================

def load_theme() -> str:
    """
    Return the <style> block for the dashboard's global theme.

    Deliberately does NOT override .stApp's background, the
    sidebar, st.metric(), or generic text color -- those should
    keep following whichever theme the user has selected (Light
    / Dark / System) via Streamlit's own "⋮" menu. Only our own
    custom components are styled here (buttons, .mode-card,
    .success-box, .warning-box, .info-box), and each defines its
    own background *and* text color together, so they stay
    readable regardless of which theme they're sitting in
    without needing to fight Streamlit's native theming.
    """

    return f"""
    <style>

    .stButton > button {{

        width: 100%;

        background: {PRIMARY};

        color: white;

        border: none;

        border-radius: 8px;

        font-weight: 600;

        height: 48px;

        transition: 0.3s;
    }}

    .stButton > button:hover {{

        background: {PRIMARY_DARK};

        color: white;

        border: none;
    }}

    .mode-card {{

        background: {CARD};

        color: {TEXT};

        border: 1px solid {BORDER};

        border-radius: 12px;

        padding: 25px;

        box-shadow: 0 3px 10px rgba(0,0,0,0.06);

        height: 100%;
    }}

    /* Solid accent background + high-contrast text: readable
       regardless of theme, since each box carries its own
       background rather than relying on the page behind it. */
    .success-box {{

        background: {SUCCESS};

        color: white;

        padding: 12px;

        border-radius: 8px;

        border-left: 5px solid {PRIMARY_DARK};
    }}

    .warning-box {{

        background: {WARNING};

        color: {TEXT};

        padding: 12px;

        border-radius: 8px;

        border-left: 5px solid #9E6C00;
    }}

    .info-box {{

        background: {INFO};

        color: white;

        padding: 12px;

        border-radius: 8px;

        border-left: 5px solid {PRIMARY_DARK};
    }}

    </style>
    """
