# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import joblib
from PIL import Image

st.set_page_config(page_title="Clasificador de Dígitos", layout="centered")
st.title("🧠 Clasificación de Dígitos - Naive Bayes")

# Carga de modelos
modelo_1 = joblib.load("modelo_gaussian_nb.pkl")
modelo_2 = joblib.load("modelo_bernoulli_nb.pkl")
modelos = {"GaussianNB": modelo_1, "BernoulliNB": modelo_2}

modelo_seleccionado = st.selectbox("Selecciona un modelo:", list(modelos.keys()))
modelo = modelos[modelo_seleccionado]

st.markdown("Dibuja un número del 0 al 9 en el recuadro")
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    if st.button("🧪 Clasificar dibujo"):
        img = cv2.resize(img.astype("uint8"), (28, 28))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_invertida = 255 - img_gray
        img_norm = img_invertida / 255.0
        img_flat = img_norm.reshape(1, -1)

        pred = modelo.predict(img_flat)[0]
        try:
            probas = modelo.predict_proba(img_flat)
            confianza = np.max(probas[0]) * 100
            st.success(f"✏️ Predicción: **{pred}** con **{confianza:.2f}%** de confianza")
        except:
            st.success(f"✏️ Predicción: **{pred}** (este modelo no entrega probabilidad)")

        st.image(img_gray, caption="Dibujo procesado (28x28)", width=150)

st.markdown("---")
st.markdown("Desarrollado por Jorge Gutierrez R. | Dataset: MNIST | Modelos: Naive Bayes")

