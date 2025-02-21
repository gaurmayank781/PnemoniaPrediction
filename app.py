import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import time
import gdown
import os

# Google Drive file ID
file_id = "1szyaLTqUuLXU6a6mgwxYZYn9Qo54q9GB"
output = "chest_xray.h5"

# Download the model from Google Drive only if it doesn't exist
if not os.path.exists(output):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output, quiet=False)
    except Exception as e:
        st.error(f"Failed to download model. Error: {e}")
        st.warning("If the download fails, manually download the model from the link below and place it in the same folder as this script.")
        st.markdown(f"[📥 Download Model Manually](https://drive.google.com/uc?export=download&id={file_id})")
        st.stop()  # Stop execution if the model can't be downloaded

# Load trained model
model = load_model(output)

# Function to preprocess image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Grad-CAM function
def get_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

# Overlay heatmap function
def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

# Streamlit UI
st.set_page_config(page_title="AI X-ray Health Scanner", layout="centered")
st.title("🩺 AI-Powered X-ray Health Scanner")
st.write("### Upload Your X-ray to Get Instant Results")
st.image("xray.jpeg", use_container_width=True)

uploaded_file = st.file_uploader("Browse File", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    st.write("**Scan Status:** Upload Complete ✅")
    
    # Show AI Processing animation
    with st.spinner("AI Processing...🔄"):
        time.sleep(3)
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0][0]
        confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)
        result = "At Risk" if prediction > 0.5 else "Healthy"
    
    st.subheader(f"🧑‍⚕️ **Result: {result}**")
    st.write(f"### 🔢 Confidence Level: {confidence:.2f}%")
    
    # Confidence animation
    progress_bar = st.progress(0)
    for i in range(int(confidence)):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    # Generate health report
    if result == "Healthy":
        report_text = (
            "AI X-ray Health Scanner Report\n"
            "----------------------------------------\n"
            "🩺 **Diagnosis: Healthy**\n"
            "✅ Your heart is in great condition!\n"
            "✔ Maintain a balanced diet, regular exercise, and routine checkups.\n"
            "----------------------------------------\n"
            "Confidence Level: " + str(confidence) + "%\n"
        )
    else:
        report_text = (
            "AI X-ray Health Scanner Report\n"
            "----------------------------------------\n"
            "🩺 **Diagnosis: At Risk**\n"
            "⚠️ Your X-ray indicates possible concerns with heart health.\n"
            "📌 Please consult a cardiologist for further evaluation.\n"
            "----------------------------------------\n"
            "Confidence Level: " + str(confidence) + "%\n"
        )
    
    st.success("💬 **Personalized AI Health Report:**")
    st.text(report_text)
    
    # Grad-CAM Heatmap Analysis
    if st.button("🔍 View X-ray Heatmap Analysis"):
        with st.spinner("Generating Heatmap...🔄"):
            time.sleep(2)
            heatmap = get_gradcam(model, processed_img, "block5_conv3")
            superimposed_img = overlay_heatmap(image, heatmap)
            st.image(superimposed_img, caption="AI Heatmap Analysis", use_container_width=True)
    
    # Doctor's Advice
    st.subheader("🩺 Doctor's Advice")
    if result == "Healthy":
        st.success("✔ Maintain a balanced diet, regular exercise, and routine checkups for optimal health.")
    else:
        st.error("⚠️ Please consult a cardiologist for further examination. Early detection leads to better outcomes!")
    
    # Download Full Report Button
    st.download_button("📥 Download Full Report", report_text, file_name="AI_Health_Report.txt")
