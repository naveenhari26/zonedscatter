import io
import json
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Matplotlib only for exporting (no Kaleido / Chrome needed)
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

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
    m = mode.lower()
    if m == "mean":
        return float(series.mean())
    if m == "median":
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

def scale_info(scale_mode: str):
    m = scale_mode.lower()
    if m == "lakhs":
        return 1e5, " (in Lakhs)"
    if m == "crores":
        return 1e7, " (in Crores)"
    return 1, ""

def build_plotly_figure(df, x, y, label_col, include_labels, scale_mode,
                        line_mode_x, line_mode_y, manual_x, manual_y,
                        title, xlabel, ylabel, colors):
    # scale
    scale, unit = scale_info(scale_mode)
    df = df.copy()
    df["_x"] = to_numeric_series(df[x]) / scale
    df["_y"] = to_numeric_series(df[y]) / scale
    df = df.dropna(subset=["_x", "_y"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No numeric rows after conversion. Check your data.")

    cut_x = compute_cut(line_mode_x, df["_x"], manual_x / scale if scale != 1 else manual_x)
    cut_y = compute_cut(line_mode_y, df["_y"], manual_y / scale if scale != 1 else manual_y)

    # axis ranges
    x0, x1 = df["_x"].min(), df["_x"].max()
    y0, y1 = df["_y"].min(), df["_y"].max()
    xpad, ypad = 0.05 * (x1 - x0 or 1), 0.05 * (y1 - y0 or 1)
    x0, x1, y0, y1 = x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad

    # scatter
    if include_labels and label_col:
        mode, text_vals = "markers+text", df[label_col].astype(str)
    else:
        mode, text_vals = "markers", None

    scatter = go.Scatter(
        x=df["_x"], y=df["_y"], mode=mode,
        text=text_vals, textposition="top center" if text_vals is not None else None,
        marker=dict(color=colors["scatter"], size=9, line=dict(color="white", width=0.7)),
        name="Data"
    )

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

    # zones
    shapes = [
        dict(type="rect", x0=x0, x1=cut_x, y0=y0, y1=cut_y,
             fillcolor=colors["zone1"], opacity=0.12, line_width=0),
        dict(type="rect", x0=x0, x1=cut_x, y0=cut_y, y1=y1,
             fillcolor=colors["zone2"], opacity=0.12, line_width=0),
        dict(type="rect", x0=cut_x, x1=x1, y0=cut_y, y1=y1,
             fillcolor=colors["zone3"], opacity=0.12, line_width=0),
        dict(type="rect", x0=cut_x, x1=x1, y0=y0, y1=cut_y,
             fillcolor=colors["zone4"], opacity=0.12, line_width=0),
    ]
    annotations = [
        dict(x=(x0+cut_x)/2, y=(y0+cut_y)/2, text="Zone I", showarrow=False,
             font=dict(size=14, color=colors["zone1"])),
        dict(x=(x0+cut_x)/2, y=(cut_y+y1)/2, text="Zone II", showarrow=False,
             font=dict(size=14, color=colors["zone2"])),
        dict(x=(cut_x+x1)/2, y=(cut_y+y1)/2, text="Zone III", showarrow=False,
             font=dict(size=14, color=colors["zone3"])),
        dict(x=(cut_x+x1)/2, y=(y0+cut_y)/2, text="Zone IV", showarrow=False,
             font=dict(size=14, color=colors["zone4"])),
    ]

    fig = go.Figure([scatter, line_x_trace, line_y_trace])
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title=xlabel + unit, range=[x0, x1]),
        yaxis=dict(title=ylabel + unit, range=[y0, y1]),
        shapes=shapes, annotations=annotations,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=40, t=60, b=60),
        height=720  # taller; not 1:1
    )

    df_out = df.copy()
    df_out["Zone"] = [zone_name(px, py, cut_x, cut_y) for px, py in zip(df["_x"], df["_y"])]
    return fig, cut_x, cut_y, df_out, unit

def export_with_matplotlib(
    df_proc: pd.DataFrame,
    cut_x: float,
    cut_y: float,
    title: str,
    xlabel: str,
    ylabel: str,
    colors: dict,
    include_labels: bool,
    label_col: str | None,
    fmt: str,
    dpi: int | None,
    width_in: float | None,
    height_in: float | None,
) -> bytes:
    if width_in is None or height_in is None:
        width_in, height_in = 7.0, 6.0

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=(dpi or 100))

    x = df_proc["_x"].to_numpy()
    y = df_proc["_y"].to_numpy()

    # Axis padding
    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()
    xpad, ypad = 0.05 * (x1 - x0 or 1), 0.05 * (y1 - y0 or 1)
    x0, x1, y0, y1 = x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad

    # Quadrants
    ax.add_patch(plt.Rectangle((x0, y0), cut_x - x0, cut_y - y0, color=colors["zone1"], alpha=0.12))
    ax.add_patch(plt.Rectangle((x0, cut_y), cut_x - x0, y1 - cut_y, color=colors["zone2"], alpha=0.12))
    ax.add_patch(plt.Rectangle((cut_x, cut_y), x1 - cut_x, y1 - cut_y, color=colors["zone3"], alpha=0.12))
    ax.add_patch(plt.Rectangle((cut_x, y0), x1 - cut_x, cut_y - y0, color=colors["zone4"], alpha=0.12))

    # Scatter
    ax.scatter(x, y, s=45, c=colors["scatter"], edgecolors="white", linewidths=0.7)

    # Labels
    if include_labels and label_col and (label_col in df_proc.columns):
        for _, r in df_proc.iterrows():
            ax.text(r["_x"], r["_y"], str(r[label_col]), fontsize=8,
                    color=colors["labels"], ha="left", va="bottom")

    # Lines
    ax.axvline(cut_x, color=colors["line_x"], linestyle="--", linewidth=1.8,
               label=f"X line ({cut_x:,.2f})")
    ax.axhline(cut_y, color=colors["line_y"], linestyle="--", linewidth=1.8,
               label=f"Y line ({cut_y:,.2f})")

    # Zone names
    ax.text((x0 + cut_x) / 2, (y0 + cut_y) / 2, "Zone I", color=colors["zone1"],
            fontsize=11, fontweight="bold", ha="center", va="center")
    ax.text((x0 + cut_x) / 2, (cut_y + y1) / 2, "Zone II", color=colors["zone2"],
            fontsize=11, fontweight="bold", ha="center", va="center")
    ax.text((cut_x + x1) / 2, (cut_y + y1) / 2, "Zone III", color=colors["zone3"],
            fontsize=11, fontweight="bold", ha="center", va="center")
    ax.text((cut_x + x1) / 2, (y0 + cut_y) / 2, "Zone IV", color=colors["zone4"],
            fontsize=11, fontweight="bold", ha="center", va="center")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(loc="upper left", fontsize=9)

    buf = io.BytesIO()
    save_kwargs = {"format": fmt.lower()}
    if fmt.lower() in ("jpg", "jpeg"):
        save_kwargs["facecolor"] = "white"
    if fmt.lower() in ("png", "jpg", "jpeg"):
        save_kwargs["dpi"] = dpi or 300

    fig.tight_layout()
    fig.savefig(buf, **save_kwargs, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ---------- Navigation ----------
page = st.sidebar.radio("Navigation", ["Scatter Zone Plotter", "Documentation & Links"])

if page == "Scatter Zone Plotter":
    st.title("Scatter Zone Plotter")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Data
        with st.expander("📂 Data", expanded=True):
            up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="uploader")
            df = None
            if up is not None:
                if up.name.endswith((".xlsx", ".xls")):
                    xls = pd.ExcelFile(up)
                    sheet = st.selectbox("Choose sheet", xls.sheet_names, key="sheet_select")
                    df = xls.parse(sheet)
                else:
                    df = pd.read_csv(up)

            if df is None:
                st.info("Upload a dataset to start.")
                st.stop()

        # Variables
        with st.expander("🔢 Variables", expanded=True):
            cols = df.columns.tolist()
            x_col = st.selectbox("X Column", cols, key="xcol_select")
            y_col = st.selectbox("Y Column", cols, key="ycol_select")
            xlabel = st.text_input("X Label", x_col, key="xlabel_input")
            ylabel = st.text_input("Y Label", y_col, key="ylabel_input")
            include_labels = st.checkbox("Include labels", key="labels_cb")
            label_col = st.selectbox("Label column", cols, disabled=not include_labels, key="labelcol_select")

        # Options
        with st.expander("⚙️ Options", expanded=True):
            scale_mode = st.radio("Scale values", ["None", "Lakhs", "Crores"], index=0, key="scale_radio")
            line_mode_x = st.radio("Vertical (X) line", ["Mean", "Median", "Manual"], index=1, key="xline_radio")
            manual_x = st.number_input("Manual X", value=0.0, key="manual_x")
            line_mode_y = st.radio("Horizontal (Y) line", ["Mean", "Median", "Manual"], index=1, key="yline_radio")
            manual_y = st.number_input("Manual Y", value=0.0, key="manual_y")
            title = st.text_input("Title", "Scatter Plot with Custom Zones", key="title_input")

        # Colors
        with st.expander("🎨 Colors", expanded=False):
            st.caption("Pick colors for each element")
            color_specs = [
                ("Scatter", "scatter"),
                ("Line X", "line_x"),
                ("Line Y", "line_y"),
                ("Labels", "labels"),
                ("Zone I", "zone1"),
                ("Zone II", "zone2"),
                ("Zone III", "zone3"),
                ("Zone IV", "zone4"),
            ]
            colors = {}
            for i in range(0, len(color_specs), 4):
                row = st.columns(4)
                for col, (label, key) in zip(row, color_specs[i:i+4]):
                    colors[key] = col.color_picker(label, DEFAULT_COLORS[key], key=f"color_{key}")

        # Save/Load
        with st.expander("💾 Save/Load Settings", expanded=False):
            settings = dict(
                title=title, xlabel=xlabel, ylabel=ylabel,
                scale_mode=scale_mode, line_mode_x=line_mode_x.lower(),
                line_mode_y=line_mode_y.lower(),
                manual_x=float(manual_x), manual_y=float(manual_y),
                include_labels=include_labels, label_column=label_col if include_labels else "",
                colors=colors, x_col=x_col, y_col=y_col,
            )
            st.download_button(
                "Save settings (.json)",
                data=json.dumps(settings, indent=2),
                file_name="settings.json",
                mime="application/json",
                key="save_json_btn"
            )

            st.markdown("---")
            loaded = st.file_uploader("Load settings (.json)", type=["json"], key="load_json")
            if loaded is not None:
                try:
                    cfg = json.load(loaded)
                    # Apply to session state then rerun
                    st.session_state["title_input"] = cfg.get("title", title)
                    st.session_state["xlabel_input"] = cfg.get("xlabel", xlabel)
                    st.session_state["ylabel_input"] = cfg.get("ylabel", ylabel)
                    st.session_state["scale_radio"] = cfg.get("scale_mode", scale_mode)
                    st.session_state["xline_radio"] = cfg.get("line_mode_x", line_mode_x).capitalize()
                    st.session_state["yline_radio"] = cfg.get("line_mode_y", line_mode_y).capitalize()
                    st.session_state["manual_x"] = float(cfg.get("manual_x", manual_x))
                    st.session_state["manual_y"] = float(cfg.get("manual_y", manual_y))
                    st.session_state["labels_cb"] = bool(cfg.get("include_labels", include_labels))
                    if cfg.get("label_column") in cols:
                        st.session_state["labelcol_select"] = cfg["label_column"]
                    if cfg.get("x_col") in cols:
                        st.session_state["xcol_select"] = cfg["x_col"]
                    if cfg.get("y_col") in cols:
                        st.session_state["ycol_select"] = cfg["y_col"]
                    for k, v in cfg.get("colors", {}).items():
                        if k in DEFAULT_COLORS:
                            st.session_state[f"color_{k}"] = v
                    st.success("Settings loaded. UI updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Invalid settings file: {e}")

    with col2:
        st.subheader("Preview")
        try:
            fig, cut_x, cut_y, plot_df, unit = build_plotly_figure(
                df, x_col, y_col, label_col if include_labels else None, include_labels,
                scale_mode, line_mode_x, line_mode_y, manual_x, manual_y,
                title, xlabel, ylabel, colors
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Export options ---
            st.markdown("### Export Options")
            fmt = st.selectbox("File format", ["png", "jpg", "pdf", "eps", "svg"], index=0, key="fmt_select")

            if fmt in ["png", "jpg"]:
                dpi = st.number_input("DPI (resolution)", value=300, step=50,
                                      min_value=72, max_value=600, key="dpi_input")
                width_in = st.number_input("Width (inches)", value=7.0, step=0.5, key="width_input")
                height_in = st.number_input("Height (inches)", value=6.0, step=0.5, key="height_input")
            else:
                dpi = None
                width_in = st.number_input("Width (inches)", value=7.0, step=0.5, key="v_width_input")
                height_in = st.number_input("Height (inches)", value=6.0, step=0.5, key="v_height_input")
                st.info("DPI is not applicable for vector formats.")

            # Prepare static image with Matplotlib
            img_bytes = export_with_matplotlib(
                plot_df, cut_x, cut_y,
                title=title,
                xlabel=xlabel + unit,
                ylabel=ylabel + unit,
                colors=colors,
                include_labels=include_labels,
                label_col=(label_col if include_labels else None),
                fmt=fmt,
                dpi=dpi,
                width_in=width_in,
                height_in=height_in
            )

            # Generate interactive HTML once (used for download + embed instructions)
            fig_html = fig.to_html(full_html=False)
            html_bytes = fig_html.encode("utf-8")

            # Buttons aligned horizontally
            colA, colB, colC = st.columns(3)
            with colA:
                st.download_button(
                    f"Download plot as .{fmt}",
                    data=img_bytes,
                    file_name=f"scatter_zones.{fmt}",
                    mime=("application/pdf" if fmt == "pdf"
                          else "image/svg+xml" if fmt == "svg"
                          else "application/postscript" if fmt == "eps"
                          else "image/png" if fmt == "png"
                          else "image/jpeg"),
                    key="img_download"
                )
            with colB:
                st.download_button(
                    "Download interactive HTML",
                    data=html_bytes,
                    file_name="scatter_zones.html",
                    mime="text/html",
                    key="html_download"
                )
            with colC:
                csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download processed CSV",
                    data=csv_bytes,
                    file_name="processed_with_zones.csv",
                    mime="text/csv",
                    key="csv_dl2"
                )

            # Responsive embed instructions (lightweight, no srcdoc here)
            st.markdown("### Embed on your website (responsive)")
            st.markdown(
                "1) Upload **`scatter_zones.html`** (downloaded above) to any public URL (e.g. your site, GitHub Pages).  \n"
                "2) Paste this snippet where you want the chart:"
            )
            responsive_embed = """<iframe
  src="https://YOUR-DOMAIN/path/scatter_zones.html"
  loading="lazy"
  style="width: 100%; aspect-ratio: 16 / 10; border: 0;">
</iframe>"""
            st.code(responsive_embed, language="html")

        except Exception as e:
            st.error(str(e))

elif page == "Documentation & Links":
    st.title("Documentation")
    st.markdown("""
    ### What is this?
    A tool to classify scatter data into 4 quadrants (zones) using mean / median / manual cut lines.

    ### Use cases
    - Municipal finance: tax vs non-tax revenue
    - Hospital/operations: BOR vs TOR style plots
    - Any 2D metric segmentation

    ### Features
    - CSV/Excel upload with sheet selection
    - Scaling (None, Lakhs, Crores)
    - Flexible cut lines (Mean, Median, Manual)
    - Custom colors (responsive grid)
    - Save/Load settings (.json)
    - Interactive preview (Plotly)
    - Static export **without Kaleido** (PNG/JPG/PDF/EPS/SVG via Matplotlib)
    - Export processed CSV with assigned Zone

    ### Tips
    - If labels overlap, zoom with the Plotly toolbar or temporarily hide labels before export.
    - Vector formats (PDF/SVG/EPS) are great for publication; raster (PNG/JPG) suits slides.

    ### Links
    - [Streamlit Docs](https://docs.streamlit.io)
    - [Plotly Python](https://plotly.com/python/)
    """)
