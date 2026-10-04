import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    with open("config.json") as f:
        cfg = json.load(f)
    return model, cfg

model, cfg = load_model()
class_names = cfg["class_names"]
img_size = tuple(cfg["img_size"])

st.title("🩺 Skin Cancer Detection")
st.caption("Model: " + cfg["model_name"] + "  |  Test accuracy: " + format(cfg["test_accuracy"], ".2%"))
st.warning("Educational demo only - NOT a medical diagnosis. Please consult a dermatologist.")

file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    arr = np.expand_dims(np.asarray(img.resize(img_size), dtype=np.float32), 0)
    probs = model.predict(arr, verbose=0)[0]
    top = int(np.argmax(probs))
    st.subheader("Prediction: " + class_names[top])
    st.write("Confidence: " + format(float(probs[top]), ".2%"))
    st.bar_chart({class_names[i]: float(p) for i, p in enumerate(probs)})
