import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# ========== Load Models ==========
classification_model = load_model("Classification_model.keras", compile=False)
segmentation_model = load_model("Segmentation_model.keras", compile=False)

# ========== Page Layout ==========
st.set_page_config(page_title="Brain Tumor Detector", layout="wide")
st.title("🧠 Brain Tumor Detection & Segmentation")
st.markdown("Upload a brain MRI image and choose whether to **classify** it or **segment** the tumor.")

# ========== Split Layout ==========
col_left, col_right = st.columns([1, 2])

# ========== Shared State ==========
uploaded_file = None
img_array = None

# ========== Left Column ==========
with col_left:
    st.subheader("⚙️ Options")
    task = st.radio("Choose the model task:", ["Classification", "Segmentation"])
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png", "tif"])

# ========== Right Column ==========
with col_right:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # إذا الصورة فيها 4 قنوات (RGBA)، بنحولها إلى RGB
        if img_array.ndim == 3 and img_array.shape[-1] == 4:
            image = image.convert("RGB")
            img_array = np.array(image)

        st.subheader("📷 Uploaded Image")

        # 🔽 تصغير للعرض فقط
        resized_display_img = cv2.resize(img_array, (300, 300))  
        st.image(resized_display_img, caption="📷 Uploaded Image", use_container_width=False)

        # تأكد من 3 قنوات
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        elif img_array.shape[-1] == 1:
            img_array = np.concatenate([img_array] * 3, axis=-1)

        if task == "Segmentation":
            seg_img = cv2.resize(img_array, (128, 128)) / 255.0
            seg_input = np.expand_dims(seg_img, axis=0)

            mask = segmentation_model.predict(seg_input)[0]
            mask = (mask > 0.5).astype(np.uint8).squeeze()

            # نسخة من الصورة لرسم حدود الورم
            overlay_img = cv2.resize(img_array, (128, 128)).copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 2)

            st.subheader("🩻 Segmentation Map")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(seg_img, caption="Processed Image", use_container_width=True)
            with col2:
                st.image(overlay_img, caption="Tumor Boundary (Red)", use_container_width=True)
            with col3:
                st.image(mask * 255, caption="Predicted Mask", use_container_width=True, clamp=True)

# ========== Show Classification Result in Left Column ==========
with col_left:
    if task == "Classification" and uploaded_file is not None:
        clf_img = cv2.resize(img_array, (224, 224))
        clf_input = np.expand_dims(clf_img, axis=0)

        prediction = classification_model.predict(clf_input)[0][0]
        label = "Tumor" if prediction > 0.5 else "No Tumor"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.subheader("🧠 Classification Result")
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence:.2f}")