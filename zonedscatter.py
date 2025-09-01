import json
import io
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Scatter Zone Plotter", layout="wide")

# ---------- Helpers ----------
DEFAULT_COLORS = {
    "scatter": "#4C72B0",
    "line_x": "#000000",
    "line_y": "#000000",
    "zone1": "#FF0000",  # I
    "zone2": "#FFA500",  # II
    "zone3": "#008080",  # III
    "zone4": "#1E90FF",  # IV
    "labels": "#000000",
}


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric and return."""
    return pd.to_numeric(s, errors="coerce")


def compute_cut(mode: str, series: pd.Series, manual: float) -> float:
    mode = (mode or "").lower()
    if mode == "mean":
        return float(series.mean())
    if mode == "median":
        return float(series.median())
    return float(manual or 0.0)


def zone_name(x, y, cut_x, cut_y) -> str:
    # 1. x < cut_x, y < cut_y -> Zone I
    # 2. x < cut_x, y >= cut_y -> Zone II
    # 3. x >= cut_x, y >= cut_y -> Zone III
    # 4. x >= cut_x, y < cut_y -> Zone IV
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
    colors: Dict[str, str],
) -> Tuple[go.Figure, float, float, pd.DataFrame]:
    # ---- scale ----
    scale_mode = (scale_mode or "none").lower()
    if scale_mode == "lakhs":
        scale = 1e5
        unit = " (in Lakhs)"
    elif scale_mode == "crores":
        scale = 1e7
        unit = " (in Crores)"
    else:
        scale = 1.0
        unit = ""

    work = df.copy()

    x_num = to_numeric_series(work[x]) / scale
    y_num = to_numeric_series(work[y]) / scale
    work = work.assign(_x=x_num, _y=y_num)
    work = work.dropna(subset=["_x", "_y"]).reset_index(drop=True)

    if work.empty:
        raise ValueError("No valid numeric rows after conversion. Check your column selections.")

    # ---- cuts ----
    cut_x = compute_cut(line_mode_x, work["_x"], manual_x / scale if scale != 1 else manual_x)
    cut_y = compute_cut(line_mode_y, work["_y"], manual_y / scale if scale != 1 else manual_y)

    # ---- axis ranges (with margin) ----
    x_min, x_max = float(work["_x"].min()), float(work["_x"].max())
    y_min, y_max = float(work["_y"].min()), float(work["_y"].max())
    x_pad = 0.05 * (x_max - x_min or 1)
    y_pad = 0.05 * (y_max - y_min or 1)
    x0, x1 = x_min - x_pad, x_max + x_pad
    y0, y1 = y_min - y_pad, y_max + y_pad

    # ---- zones as shapes ----
    shapes = [
        # Zone I
        dict(type="rect", xref="x", yref="y", x0=x0, x1=cut_x, y0=y0, y1=cut_y,
             fillcolor=colors["zone1"], opacity=0.12, line_width=0),
        # Zone II
        dict(type="rect", xref="x", yref="y", x0=x0, x1=cut_x, y0=cut_y, y1=y1,
             fillcolor=colors["zone2"], opacity=0.12, line_width=0),
        # Zone III
        dict(type="rect", xref="x", yref="y", x0=cut_x, x1=x1, y0=cut_y, y1=y1,
             fillcolor=colors["zone3"], opacity=0.12, line_width=0),
        # Zone IV
        dict(type="rect", xref="x", yref="y", x0=cut_x, x1=x1, y0=y0, y1=cut_y,
             fillcolor=colors["zone4"], opacity=0.12, line_width=0),
    ]

    # ---- scatter (optionally with text) ----
    if include_labels and label_col:
        text_vals = work[label_col].astype(str)
        mode = "markers+text"
        textposition = "top center"
    else:
        text_vals = None
        mode = "markers"
        textposition = None

    scatter = go.Scatter(
        x=work["_x"],
        y=work["_y"],
        mode=mode,
        text=text_vals,
        textposition=textposition,
        marker=dict(color=colors["scatter"], size=9, line=dict(color="white", width=0.7)),
        name="Data",
        hovertemplate=f"{x}: %{ 'x' }<br>{y}: %{ 'y' }<extra></extra>",
    )

    # ---- divider lines (as traces so they appear in legend) ----
    line_x_trace = go.Scatter(
        x=[cut_x, cut_x], y=[y0, y1],
        mode="lines",
        line=dict(color=colors["line_x"], width=2, dash="dash"),
        name=f"X line ({line_mode_x.capitalize()} = {cut_x:,.2f})",
        hoverinfo="skip",
    )
    line_y_trace = go.Scatter(
        x=[x0, x1], y=[cut_y, cut_y],
        mode="lines",
        line=dict(color=colors["line_y"], width=2, dash="dash"),
        name=f"Y line ({line_mode_y.capitalize()} = {cut_y:,.2f})",
        hoverinfo="skip",
    )

    # ---- annotations for zone labels ----
    annotations = [
        dict(x=(x0 + cut_x) / 2, y=(y0 + cut_y) / 2, text="Zone I",
             showarrow=False, font=dict(size=14, color=colors["zone1"])),
        dict(x=(x0 + cut_x) / 2, y=(cut_y + y1) / 2, text="Zone II",
             showarrow=False, font=dict(size=14, color=colors["zone2"])),
        dict(x=(cut_x + x1) / 2, y=(cut_y + y1) / 2, text="Zone III",
             showarrow=False, font=dict(size=14, color=colors["zone3"])),
        dict(x=(cut_x + x1) / 2, y=(y0 + cut_y) / 2, text="Zone IV",
             showarrow=False, font=dict(size=14, color=colors["zone4"])),
    ]

    fig = go.Figure(data=[scatter, line_x_trace, line_y_trace])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis=dict(title=xlabel + unit, range=[x0, x1], gridcolor="#e6e6e6"),
        yaxis=dict(title=ylabel + unit, range=[y0, y1], gridcolor="#e6e6e6"),
        shapes=shapes,
        annotations=annotations,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=40, t=60, b=60),
    )

    # add zones to a DF copy for download
    out = work.copy()
    out["Zone"] = [zone_name(px, py, cut_x, cut_y) for px, py in zip(out["_x"], out["_y"])]

    return fig, cut_x, cut_y, out


def download_bytes(fig: go.Figure, fmt: str = "png", scale: int = 2) -> bytes:
    # Requires "kaleido" in requirements
    return fig.to_image(format=fmt, scale=scale)


# ---------- UI ----------
st.title("Scatter Zone Plotter (Web)")
st.caption("Upload a CSV, pick X/Y, choose how to split zones (Mean/Median/Manual), "
           "optionally scale to Lakhs/Crores, and download the plot and processed CSV.")

with st.sidebar:
    st.header("1) Upload CSV")
    up = st.file_uploader("CSV file", type=["csv"])
    st.markdown("---")
    st.header("2) Settings I")
    scale_mode = st.radio("Scale values", ["None", "Lakhs", "Crores"], horizontal=True, index=0)
    st.markdown("— **Zone lines** —")
    line_mode_x = st.radio("Vertical (X) line", ["Mean", "Median", "Manual"], horizontal=True, index=1)
    manual_x = st.number_input("Manual X value (use scaled units if scaling is enabled)", value=0.0)
    line_mode_y = st.radio("Horizontal (Y) line", ["Mean", "Median", "Manual"], horizontal=True, index=1)
    manual_y = st.number_input("Manual Y value (use scaled units if scaling is enabled)", value=0.0)
    st.markdown("---")
    st.header("3) Settings II")
    title = st.text_input("Title", "Scatter Plot with Custom Zones")

# content area
if up is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# read CSV
try:
    df = pd.read_csv(up)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if df.empty:
    st.warning("CSV is empty.")
    st.stop()

# column selectors
cols = df.columns.tolist()
left, right = st.columns([1, 2], vertical_alignment="top")

with left:
    st.subheader("Variables & Labels")
    x_col = st.selectbox("X Column", cols)
    y_col = st.selectbox("Y Column", cols)
    xlabel = st.text_input("X Axis Label", x_col)
    ylabel = st.text_input("Y Axis Label", y_col)

    include_labels = st.checkbox("Include point labels", value=False)
    label_col = st.selectbox("Label column", cols, index=0, disabled=not include_labels)

    st.subheader("Colors")
    c1, c2 = st.columns(2)
    with c1:
        color_scatter = st.color_picker("Scatter", DEFAULT_COLORS["scatter"])
        color_line_x = st.color_picker("Line X", DEFAULT_COLORS["line_x"])
        color_zone1 = st.color_picker("Zone1", DEFAULT_COLORS["zone1"])
        color_zone3 = st.color_picker("Zone3", DEFAULT_COLORS["zone3"])
    with c2:
        color_line_y = st.color_picker("Line Y", DEFAULT_COLORS["line_y"])
        color_labels = st.color_picker("Labels", DEFAULT_COLORS["labels"])
        color_zone2 = st.color_picker("Zone2", DEFAULT_COLORS["zone2"])
        color_zone4 = st.color_picker("Zone4", DEFAULT_COLORS["zone4"])

    colors = {
        "scatter": color_scatter,
        "line_x": color_line_x,
        "line_y": color_line_y,
        "zone1": color_zone1,
        "zone2": color_zone2,
        "zone3": color_zone3,
        "zone4": color_zone4,
        "labels": color_labels,
    }

    st.subheader("Save / Load Settings")
    # Download settings
    settings = {
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "scale_mode": scale_mode.lower(),
        "line_mode_x": line_mode_x.lower(),
        "line_mode_y": line_mode_y.lower(),
        "manual_x": float(manual_x),
        "manual_y": float(manual_y),
        "include_labels": bool(include_labels),
        "label_column": label_col if include_labels else "",
        "colors": colors,
        "x_col": x_col,
        "y_col": y_col,
    }
    settings_bytes = json.dumps(settings, indent=2).encode("utf-8")
    st.download_button("Download settings (.json)", data=settings_bytes, file_name="scatter_zone_settings.json",
                       mime="application/json")

    st.markdown("Upload a settings JSON to auto-fill controls:")
    up_settings = st.file_uploader("Load settings JSON", type=["json"], key="load_settings_json")
    if up_settings is not None:
        try:
            new_s = json.load(up_settings)
            # Apply settings into session state, then rerun so the widgets pick them up
            st.session_state.update({
                "Scale values": {"None":0,"Lakhs":1,"Crores":2}.get(new_s.get("scale_mode","none").capitalize(), 0),
            })
            # For Streamlit widgets, easiest is to re-render values directly below (not full reactive set).
            st.success("Settings file loaded. Please set controls to match as needed.")
        except Exception as e:
            st.error(f"Failed to load settings: {e}")

with right:
    st.subheader("Preview")
    try:
        fig, cut_x, cut_y, plot_df = build_figure(
            df=df,
            x=x_col, y=y_col,
            label_col=(label_col if include_labels else None),
            include_labels=include_labels,
            scale_mode=scale_mode.lower(),
            line_mode_x=line_mode_x.lower(),
            line_mode_y=line_mode_y.lower(),
            manual_x=float(manual_x),
            manual_y=float(manual_y),
            title=title, xlabel=xlabel, ylabel=ylabel, colors=colors,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Downloads: processed CSV (with Zone)
        dl_csv = plot_df.copy()
        # Rename scaled cols to the original column names for clarity
        dl_csv.rename(columns={"_x": x_col, "_y": y_col}, inplace=True)
        csv_bytes = dl_csv.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed CSV (with Zone)", data=csv_bytes,
                           file_name="processed_with_zones.csv", mime="text/csv")

        # Downloads: image
        fmt = st.selectbox("Download plot format", ["png", "jpg", "pdf", "eps"], index=0)
        img_bytes = download_bytes(fig, fmt=fmt, scale=2)
        st.download_button(f"Download plot as .{fmt}", data=img_bytes,
                           file_name=f"scatter_zones.{fmt}",
                           mime=("application/pdf" if fmt == "pdf"
                                 else "image/jpeg" if fmt in ("jpg", "jpeg")
                                 else "application/postscript" if fmt == "eps"
                                 else "image/png"))
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.write("Tip: Hover points to see values; change options on the left to update the plot.")
