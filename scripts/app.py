import streamlit as st
from PIL import Image
import tempfile
import os
import io
from gsfusion import run_pansharpening
from IHS import run_ihs_technique

st.set_page_config(page_title="Pan-sharpening App", layout="centered")
st.title("Pan-sharpening using Multispectral & Panchromatic Images")

# Upload UI
multi_file = st.file_uploader("Upload Multispectral Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="multi")
panchro_file = st.file_uploader("Upload Panchromatic Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="panchro")

# Check both files uploaded
if multi_file and panchro_file:
    st.success("Both images uploaded successfully!")

    # Save images temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_multi:
        tmp_multi.write(multi_file.read())
        multi_path = tmp_multi.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_panchro:
        tmp_panchro.write(panchro_file.read())
        panchro_path = tmp_panchro.name

    # Run backend processing
    with st.spinner("Running Pan-sharpening..."):
        pan_result_img = run_pansharpening(multi_path, panchro_path)
        ihs_result_img = run_ihs_technique(multi_path, panchro_path)

    st.success("Processing Complete!")

    # --- Let user choose the format ---
    format_option = st.selectbox(
        "Select download format", 
        options=["TIFF", "PNG", "JPEG"], 
        index=0
    )
    ext_map = {"TIFF": "tif", "PNG": "png", "JPEG": "jpg"}
    mime_map = {"TIFF": "image/tiff", "PNG": "image/png", "JPEG": "image/jpeg"}
    file_ext = ext_map[format_option]
    mime_type = mime_map[format_option]

    # --- Gram-Schmidt Result ---
    st.subheader("🔍 Pan-sharpened Image (Gram-Schmidt)")
    st.image(pan_result_img, use_container_width=True)

    # Convert to bytes
    pan_byte_arr = io.BytesIO()
    pan_result_img.save(pan_byte_arr, format=format_option)
    pan_bytes = pan_byte_arr.getvalue()

    st.download_button(
        label=f"Download Gram-Schmidt Image ({format_option})",
        data=pan_bytes,
        file_name=f"pansharpened_gramschmidt.{file_ext}",
        mime=mime_type
    )

    # --- IHS Result ---
    st.subheader("🎨 Pan-sharpened Image (IHS Technique)")
    st.image(ihs_result_img, use_container_width=True)

    ihs_byte_arr = io.BytesIO()
    ihs_result_img.save(ihs_byte_arr, format=format_option)
    ihs_bytes = ihs_byte_arr.getvalue()

    st.download_button(
        label=f"Download IHS Image ({format_option})",
        data=ihs_bytes,
        file_name=f"pansharpened_ihs.{file_ext}",
        mime=mime_type
    )
