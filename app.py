import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io

from cartoonizer import (
    apply_bilateral_filter,
    detect_edges,
    dilate_edges
)
from quantization import quantize_image

# -----------------------------
# Saturation Adjustment Function
# -----------------------------
def adjust_saturation(img: np.ndarray, saturation_scale: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(page_title="Image Cartoonizer", layout="centered")
st.title("🎨 Image Cartoonizer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------
# Sidebar Branding & Controls
# -----------------------------
st.sidebar.header("🛠️ Cartoonization Controls")

# 🔝 Logos: Cartoonile + Nile University
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    st.image("assets/cartoonile_logo.png", width=80)
with col2:
    st.image("assets/nile_logo.png", width=80)

st.sidebar.markdown("## 🎓 Cartoonile – Image Processing Project")

# 🎛️ Default best settings
BEST_SETTINGS = {
    "slider_stylization": 9,
    "slider_edge_block": 13,
    "slider_edge_c": 2,
    "slider_thickness": 2,
}

# 🔁 Reset to Best Settings
if st.sidebar.button("🔁 Reset to Best Settings"):
    for key, value in BEST_SETTINGS.items():
        st.session_state[key] = value

# 🎛️ Sliders with session_state
stylization_intensity = st.sidebar.slider(
    "Stylization Intensity", 1, 10,
    st.session_state.get("slider_stylization", BEST_SETTINGS["slider_stylization"]),
    key="slider_stylization"
)
edge_block = st.sidebar.slider(
    "Edge Detection Block Size", 3, 25,
    st.session_state.get("slider_edge_block", BEST_SETTINGS["slider_edge_block"]),
    step=2,
    key="slider_edge_block"
)
edge_c = st.sidebar.slider(
    "Edge Threshold Constant (C)", -10, 10,
    st.session_state.get("slider_edge_c", BEST_SETTINGS["slider_edge_c"]),
    key="slider_edge_c"
)
line_thickness = st.sidebar.slider(
    "Line Thickness", 1, 7,
    st.session_state.get("slider_thickness", BEST_SETTINGS["slider_thickness"]),
    key="slider_thickness"
)

# 👥 Project Team
st.sidebar.markdown("### 👨‍💻 Developed By")
row1_col1, row1_col2 = st.sidebar.columns(2)
with row1_col1:
    st.image("assets/ahmed.jpeg", width=100)
    st.markdown("**Ahmed Khaled**  \nID: 232000046")
with row1_col2:
    st.image("assets/gannat.jpg", width=100)
    st.markdown("**Gannat Sharaf El-Deen**  \nID: NU20XXXXX")

# 👨‍🏫 Supervisor & TA
st.sidebar.markdown("### 👨‍🏫 Supervised By")
row2_col1, row2_col2 = st.sidebar.columns(2)
with row2_col1:
    st.image("assets/professor.jpg", width=100)
    st.markdown("**Prof. Dr. Walid Al-Attabany**  \nCourse Instructor")
with row2_col2:
    st.image("assets/ta.jpg", width=100)
    st.markdown("**Eng. Shahenda Hatem**  \nTeaching Assistant")

# -----------------------------
# Image Processing Logic
# -----------------------------
# Derived parameters from stylization
k = int(np.interp(stylization_intensity, [1, 10], [12, 4]))
saturation_scale = np.interp(stylization_intensity, [1, 10], [1.0, 2.5])

if uploaded_file:
    # Load and convert image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Step 1: Quantize and adjust saturation
    quantized = quantize_image(image_bgr, k=k)
    saturated = adjust_saturation(quantized, saturation_scale=saturation_scale)
    saturated_rgb = cv2.cvtColor(saturated, cv2.COLOR_BGR2RGB)

    # Step 2: Edge detection and dilation
    smoothed = apply_bilateral_filter(image_bgr)
    edges = detect_edges(smoothed, block_size=edge_block, C=edge_c)
    edges = dilate_edges(edges, thickness=line_thickness)

    # Step 3: Merge cartoon image
    cartoon = cv2.bitwise_and(saturated, saturated, mask=edges)
    cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
    cartoon_pil = Image.fromarray(cartoon_rgb)

    # -----------------------------
    # Output Display
    # -----------------------------
    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)

    st.subheader("🎨 Quantized + Saturated")
    st.image(Image.fromarray(saturated_rgb), use_container_width=True)

    st.subheader("✏️ Contour Edges")
    st.image(edges, use_container_width=True, clamp=True)

    st.subheader("🏁 Final Cartoonized Image")
    st.image(cartoon_pil, use_container_width=True)

    # Download final cartoon image
    img_buffer = io.BytesIO()
    cartoon_pil.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    st.download_button(
        label="📥 Download Cartoon Image",
        data=img_buffer,
        file_name="cartoonized.png",
        mime="image/png"
    )
else:
    st.info("📤 Upload an image to begin cartoonization.")
