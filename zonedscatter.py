import json
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------- Config ----------
st.set_page_config(page_title="Scatter Zone Plotter", layout="wide")

DEFAULT_COLORS = {
    "scatter": "#4C72B0",
    "line_x": "#000000",
    "line_y": "#000000",
    "zone1": "#FF0000",
    "zone2": "#FFA500",
    "zone3": "#008080",
    "zone4": "#1E90FF",
    "labels": "#000000",
}

# ---------- Helpers ----------
def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def compute_cut(mode: str, series: pd.Series, manual: float) -> float:
    mode = (mode or "").lower()
    if mode == "mean":
        return float(series.mean())
    if mode == "median":
        return float(series.median())
    return float(manual or 0.0)

def zone_name(x, y, cut_x, cut_y) -> str:
    if x < cut_x and y < cut_y:
        return "Zone I"
    elif x < cut_x and y >= cut_y:
        return "Zone II"
    elif x >= cut_x and y >= cut_y:
        return "Zone III"
    else:
        return "Zone IV"

def build_figure(
    df: pd.DataFrame,
    x: str,
    y: str,
    label_col: str | None,
    include_labels: bool,
    scale_mode: str,
    line_mode_x: str,
    line_mode_y: str,
    manual_x: float,
    manual_y: float,
    title: str,
    xlabel: str,
    ylabel: str,
    colors: dict,
    template: str = "plotly_white",
) -> tuple[go.Figure, float, float, pd.DataFrame]:
    # ---- scale ----
    scale_mode = (scale_mode or "none").lower()
    if scale_mode == "lakhs":
        scale, unit = 1e5, " (in Lakhs)"
    elif scale_mode == "crores":
        scale, unit = 1e7, " (in Crores)"
    else:
        scale, unit = 1, ""

    # Prepare numeric data
    df = df.copy()
    df["_x"] = to_numeric_series(df[x]) / scale
    df["_y"] = to_numeric_series(df[y]) / scale
    df = df.dropna(subset=["_x", "_y"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No numeric rows after conversion. Check your column selections.")

    # Cuts
    cut_x = compute_cut(line_mode_x, df["_x"], manual_x / scale if scale != 1 else manual_x)
    cut_y = compute_cut(line_mode_y, df["_y"], manual_y / scale if scale != 1 else manual_y)

    # Axis ranges
    x0, x1 = float(df["_x"].min()), float(df["_x"].max())
    y0, y1 = float(df["_y"].min()), float(df["_y"].max())
    xpad = 0.05 * (x1 - x0 or 1)
    ypad = 0.05 * (y1 - y0 or 1)
    x0, x1, y0, y1 = x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad

    # Scatter
    if include_labels and label_col:
        mode = "markers+text"
        text_vals = df[label_col].astype(str)
    else:
        mode = "markers"
        text_vals = None

    scatter = go.Scatter(
        x=df["_x"],
        y=df["_y"],
        mode=mode,
        text=text_vals,
        textposition="top center" if text_vals is not None else None,
        marker=dict(color=colors["scatter"], size=9, line=dict(color="white", width=0.7)),
        name="Data",
    )

    # Divider lines
    line_x_trace = go.Scatter(
        x=[cut_x, cut_x], y=[y0, y1], mode="lines",
        line=dict(color=colors["line_x"], width=2, dash="dash"),
        name=f"X line ({line_mode_x.capitalize()} = {cut_x:,.2f})"
    )
    line_y_trace = go.Scatter(
        x=[x0, x1], y=[cut_y, cut_y], mode="lines",
        line=dict(color=colors["line_y"], width=2, dash="dash"),
        name=f"Y line ({line_mode_y.capitalize()} = {cut_y:,.2f})"
    )

    # Quadrant shapes
    shapes = [
        dict(type="rect", xref="x", yref="y", x0=x0, x1=cut_x, y0=y0, y1=cut_y, fillcolor=colors["zone1"], opacity=0.12, line_width=0),
        dict(type="rect", xref="x", yref="y", x0=x0, x1=cut_x, y0=cut_y, y1=y1, fillcolor=colors["zone2"], opacity=0.12, line_width=0),
        dict(type="rect", xref="x", yref="y", x0=cut_x, x1=x1, y0=cut_y, y1=y1, fillcolor=colors["zone3"], opacity=0.12, line_width=0),
        dict(type="rect", xref="x", yref="y", x0=cut_x, x1=x1, y0=y0, y1=cut_y, fillcolor=colors["zone4"], opacity=0.12, line_width=0),
    ]

    # Labels
    annotations = [
        dict(x=(x0+cut_x)/2, y=(y0+cut_y)/2, text="Zone I", showarrow=False, font=dict(size=14, color=colors["zone1"])),
        dict(x=(x0+cut_x)/2, y=(cut_y+y1)/2, text="Zone II", showarrow=False, font=dict(size=14, color=colors["zone2"])),
        dict(x=(cut_x+x1)/2, y=(cut_y+y1)/2, text="Zone III", showarrow=False, font=dict(size=14, color=colors["zone3"])),
        dict(x=(cut_x+x1)/2, y=(y0+cut_y)/2, text="Zone IV", showarrow=False, font=dict(size=14, color=colors["zone4"])),
    ]

    fig = go.Figure([scatter, line_x_trace, line_y_trace])
    fig.update_layout(
        template=template,
        title=dict(text=title, x=0.5),
        xaxis=dict(title=xlabel + unit, range=[x0, x1]),
        yaxis=dict(title=ylabel + unit, range=[y0, y1]),
        shapes=shapes,
        annotations=annotations,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=60, r=40, t=60, b=60),
    )
    # 1:1 aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Add zones to DF
    out = df.copy()
    out["Zone"] = [zone_name(px, py, cut_x, cut_y) for px, py in zip(out["_x"], out["_y"])]
    return fig, cut_x, cut_y, out


def export_image(fig: go.Figure, fmt: str, width_in: float | None, height_in: float | None, dpi: int | None) -> bytes:
    """Export with Kaleido. For PNG/JPG we pass pixel size; for vectors ignore DPI."""
    try:
        if fmt in ("png", "jpg", "jpeg"):
            dpi = dpi or 300
            width_in = width_in or 6.0
            height_in = height_in or 6.0
            width_px = int(width_in * dpi)
            height_px = int(height_in * dpi)
            return fig.to_image(format=fmt, width=width_px, height=height_px, engine="kaleido")
        else:
            return fig.to_image(format=fmt, engine="kaleido")
    except Exception as e:
        # Fallback: return an HTML download instead
        html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
        raise RuntimeError(
            "Image export failed (Kaleido not available or misconfigured). "
            "You can still download an interactive HTML below."
        ) from e


# ---------- Sidebar Navigation ----------
page = st.sidebar.radio("Navigation", ["Scatter Zone Plotter", "Documentation & Links"])

# ---------- Main ----------
if page == "Scatter Zone Plotter":
    # Header with dark mode toggle
    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.title("Scatter Zone Plotter (Web)")
    with c2:
        # Modern toggle (Streamlit >= 1.31); fallback to checkbox if needed
        try:
            dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False), key="dark_mode")
        except Exception:
            dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.get("dark_mode", False), key="dark_mode")

    # Choose Plotly template + optional minimal CSS for dark mode
    template = "plotly_dark" if dark_mode else "plotly_white"
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp { background-color: #111111; color: #EEEEEE; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ---- Sidebar: Data -> Variables -> Options -> Save/Load ----
    # Data
    with st.sidebar.expander("📁 Data", expanded=True):
        up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        df = None
        if up is not None:
            if up.name.endswith((".xlsx", ".xls")):
                try:
                    import openpyxl  # ensure the engine exists
                    xls = pd.ExcelFile(up, engine="openpyxl")
                except Exception as e:
                    st.error("Excel support requires `openpyxl`. Add it to requirements.txt or install it locally.")
                    st.stop()
                sheet = st.selectbox("Sheet", xls.sheet_names)
                df = xls.parse(sheet)
            else:
                df = pd.read_csv(up)

        if df is None or df.empty:
            st.info("Upload a dataset to continue.")
            st.stop()

    cols = df.columns.tolist()

    # Variables
    with st.sidebar.expander("🔢 Variables", expanded=True):
        x_col = st.selectbox("X Column", cols, key="x_col")
        y_col = st.selectbox("Y Column", cols, key="y_col")
        xlabel = st.text_input("X Label", x_col, key="xlabel")
