            # --- Export options (Matplotlib backend, no Kaleido) ---
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

            # Generate HTML & embed code
            html_bytes = fig.to_html(full_html=False).encode("utf-8")
            embed_code = f"""
<iframe srcdoc='{fig.to_html(full_html=False).replace("'", "&apos;")}' 
        width="800" height="600" style="border:none;">
</iframe>
"""

            # Align buttons horizontally
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
                    key="csv_dl2"   # different key from earlier csv_dl
                )

            # Show embed option
            st.markdown("### Embed on your website")
            st.code(embed_code, language="html")
