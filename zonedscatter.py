import json
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

def build_figure(df, x, y, label_col, include_labels, scale_mode,
                 line_mode_x, line_mode_y, manual_x, manual_y,
                 title, xlabel, ylabel, colors):

    # ---- scale ----
    scale_mode = scale_mode.lower()
    if scale_mode == "lakhs":
        scale, unit = 1e5, " (in Lakhs)"
    elif scale_mode == "crores":
        scale, unit = 1e7, " (in Crores)"
    else:
        scale, unit = 1, ""

    df = df.copy()
    df["_x"] = to_numeric_series(df[x]) / scale
    df["_y"] = to_numeric_series(df[y]) / scale
    df = df.dropna(subset=["_x", "_y"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No numeric rows after conversion. Check your data.")

    cut_x = compute_cut(line_mode_x, df["_x"], manual_x / scale if scale != 1 else manual_x)
    cut_y = compute_cut(line_mode_y, df["_y"], manual_y / scale if scale != 1 else manual_y)

    # ---- axis ranges ----
    x0, x1 = df["_x"].min(), df["_x"].max()
    y0, y1 = df["_y"].min(), df["_y"].max()
    xpad, ypad = 0.05 * (x1 - x0 or 1), 0.05 * (y1 - y0 or 1)
    x0, x1, y0, y1 = x0 - xpad, x1 + xpad, y0 - ypad, y1 + ypad

    # ---- scatter ----
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

    # ---- zones ----
    shapes = [
        dict(type="rect", x0=x0, x1=cut_x, y0=y0, y1=cut_y, fillcolor=colors["zone1"], opacity=0.12, line_width=0),
        dict(type="rect", x0=x0, x1=cut_x, y0=cut_y, y1=y1, fillcolor=colors["zone2"], opacity=0.12, line_width=0),
        dict(type="rect", x0=cut_x, x1=x1, y0=cut_y, y1=y1, fillcolor=colors["zone3"], opacity=0.12, line_width=0),
        dict(type="rect", x0=cut_x, x1=x1, y0=y0, y1=cut_y, fillcolor=colors["zone4"], opacity=0.12, line_width=0),
    ]
    annotations = [
        dict(x=(x0+cut_x)/2, y=(y0+cut_y)/2, text="Zone I", showarrow=False, font=dict(size=14, color=colors["zone1"])),
        dict(x=(x0+cut_x)/2, y=(cut_y+y1)/2, text="Zone II", showarrow=False, font=dict(size=14, color=colors["zone2"])),
        dict(x=(cut_x+x1)/2, y=(cut_y+y1)/2, text="Zone III", showarrow=False, font=dict(size=14, color=colors["zone3"])),
        dict(x=(cut_x+x1)/2, y=(y0+cut_y)/2, text="Zone IV", showarrow=False, font=dict(size=14, color=colors["zone4"])),
    ]

    fig = go.Figure([scatter, line_x_trace, line_y_trace])
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title=xlabel+unit, range=[x0, x1]),
        yaxis=dict(title=ylabel+unit, range=[y0, y1]),
        shapes=shapes, annotations=annotations,
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=60, r=40, t=60, b=60),
        height=650
    )

    df["Zone"] = [zone_name(px, py, cut_x, cut_y) for px, py in zip(df["_x"], df["_y"])]
    return fig, cut_x, cut_y, df


# ---------- Navigation ----------
page = st.sidebar.radio("Navigation", ["Scatter Zone Plotter", "Documentation & Links"])

if page == "Scatter Zone Plotter":
    st.title("Scatter Zone Plotter (Web)")

    # Layout: Left (controls), Right (plot)
    col1, col2 = st.columns([1, 2])

    with col1:
        # ---- Data ----
        st.header("📂 Data")
        up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        df = None
        if up is not None:
            if up.name.endswith((".xlsx", ".xls")):
                xls = pd.ExcelFile(up)
                sheet = st.selectbox("Choose sheet", xls.sheet_names)
                df = xls.parse(sheet)
            else:
                df = pd.read_csv(up)

        if df is None:
            st.info("Upload a dataset to start.")
            st.stop()

        # ---- Variables ----
        st.header("🔢 Variables")
        cols = df.columns.tolist()
        x_col = st.selectbox("X Column", cols)
        y_col = st.selectbox("Y Column", cols)
        xlabel = st.text_input("X Label", x_col)
        ylabel = st.text_input("Y Label", y_col)
        include_labels = st.checkbox("Include labels")
        label_col = st.selectbox("Label column", cols, disabled=not include_labels)

        # ---- Options ----
        st.header("⚙️ Options")
        scale_mode = st.radio("Scale values", ["None", "Lakhs", "Crores"], index=0)
        line_mode_x = st.radio("Vertical (X) line", ["Mean", "Median", "Manual"], index=1)
        manual_x = st.number_input("Manual X", value=0.0)
        line_mode_y = st.radio("Horizontal (Y) line", ["Mean", "Median", "Manual"], index=1)
        manual_y = st.number_input("Manual Y", value=0.0)
        title = st.text_input("Title", "Scatter Plot with Custom Zones")

        # --- Colors compact grid ---
        st.markdown("**🎨 Colors**")
        c1, c2, c3 = st.columns(3)
        with c1:
            scatter_c = st.color_picker("Scatter", DEFAULT_COLORS["scatter"])
            line_x_c = st.color_picker("Line X", DEFAULT_COLORS["line_x"])
            zone1_c = st.color_picker("Zone I", DEFAULT_COLORS["zone1"])
        with c2:
            line_y_c = st.color_picker("Line Y", DEFAULT_COLORS["line_y"])
            labels_c = st.color_picker("Labels", DEFAULT_COLORS["labels"])
            zone2_c = st.color_picker("Zone II", DEFAULT_COLORS["zone2"])
        with c3:
            zone3_c = st.color_picker("Zone III", DEFAULT_COLORS["zone3"])
            zone4_c = st.color_picker("Zone IV", DEFAULT_COLORS["zone4"])
        colors = {"scatter": scatter_c, "line_x": line_x_c, "line_y": line_y_c,
                  "zone1": zone1_c, "zone2": zone2_c, "zone3": zone3_c, "zone4": zone4_c, "labels": labels_c}

        # ---- Save/Load ----
        st.header("💾 Save/Load Settings")
        settings = dict(title=title, xlabel=xlabel, ylabel=ylabel,
                        scale_mode=scale_mode, line_mode_x=line_mode_x.lower(),
                        line_mode_y=line_mode_y.lower(),
                        manual_x=float(manual_x), manual_y=float(manual_y),
                        include_labels=include_labels, label_column=label_col if include_labels else "",
                        colors=colors, x_col=x_col, y_col=y_col)
        st.download_button("Save settings (.json)", json.dumps(settings, indent=2),
                           file_name="settings.json", mime="application/json")

    with col2:
        st.subheader("Preview")
        try:
            fig, cut_x, cut_y, plot_df = build_figure(
                df, x_col, y_col, label_col if include_labels else None, include_labels,
                scale_mode, line_mode_x.lower(), line_mode_y.lower(),
                manual_x, manual_y, title, xlabel, ylabel, colors
            )
            st.plotly_chart(fig, use_container_width=True)

            csv_bytes = plot_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download processed CSV", csv_bytes, "processed_with_zones.csv", "text/csv")

            # --- Export options ---
            st.markdown("### Export Options")
            fmt = st.selectbox("File format", ["png", "jpg", "pdf", "eps", "svg"], index=0)

            if fmt in ["png", "jpg"]:
                dpi = st.number_input("DPI (resolution)", value=300, step=50, min_value=72, max_value=600)
                width_in = st.number_input("Width (inches)", value=7.0, step=0.5)
                height_in = st.number_input("Height (inches)", value=6.0, step=0.5)

                try:
                    img_bytes = fig.to_image(
                        format=fmt,
                        width=int(width_in * dpi),
                        height=int(height_in * dpi),
                        scale=1,
                        engine="kaleido"
                    )
                except Exception:
                    img_bytes = pio.to_image(fig, format=fmt,
                                             width=int(width_in * dpi),
                                             height=int(height_in * dpi))
            else:
                st.info("DPI/dimensions not applicable for vector formats (PDF, EPS, SVG).")
                try:
                    img_bytes = fig.to_image(format=fmt, engine="kaleido")
                except Exception:
                    img_bytes = pio.to_image(fig, format=fmt)

            st.download_button(
                f"Download plot as .{fmt}",
                data=img_bytes,
                file_name=f"scatter_zones.{fmt}",
                mime=("application/pdf" if fmt == "pdf"
                      else "image/svg+xml" if fmt == "svg"
                      else "application/postscript" if fmt == "eps"
                      else "image/png" if fmt == "png"
                      else "image/jpeg"),
            )

        except Exception as e:
            st.error(str(e))

elif page == "Documentation & Links":
    st.title("Documentation")
    st.markdown("""
    ### What is this?
    A tool to classify scatter data into 4 quadrants (zones) based on mean/median/manual thresholds.

    ### Use cases
    - Municipal finance: classify tax vs non-tax revenue
    - Economics: compare expenditure categories
    - Business: identify high/low performers

    ### Features
    - CSV/Excel upload (with sheet selection)
    - Scaling (None, Lakhs, Crores)
    - Flexible line positioning (Mean, Median, Manual)
    - Custom colors (compact sidebar grid)
    - Save/Load settings
    - Export plot (PNG/JPG with DPI & size, PDF/EPS/SVG vector)
    - Export processed CSV with Zone labels

    ### Links
    - [Streamlit Documentation](https://docs.streamlit.io)
    - [Plotly Graphing](https://plotly.com/python/)
    """)
