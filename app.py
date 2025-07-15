import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import joblib
import io

# Cargar los modelos
model1 = joblib.load(f'{best_models[0]}.pkl')
model2 = joblib.load(f'{best_models[1]}.pkl')

st.title('Clasificación de Dígitos con Naive Bayes')
st.write('Dibuja un dígito (0-9) en el lienzo y los modelos lo clasificarán')

# Crear el lienzo para dibujar
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Preprocesar la imagen dibujada
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img = img.convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28))  # Redimensionar a 28x28
    
    # Mostrar la imagen procesada
    st.image(img, caption='Imagen procesada (28x28)', width=100)
    
    # Convertir a array numpy y aplanar
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, -1)
    
    # Realizar predicciones
    pred1 = model1.predict(img_array)
    prob1 = model1.predict_proba(img_array)
    conf1 = np.max(prob1) * 100
    
    pred2 = model2.predict(img_array)
    prob2 = model2.predict_proba(img_array)
    conf2 = np.max(prob2) * 100
    
    # Mostrar resultados
    st.subheader("Resultados de clasificación:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Modelo 1: {best_models[0]}**")
        st.write(f"Predicción: {pred1[0]}")
        st.write(f"Confianza: {conf1:.2f}%")
    
    with col2:
        st.write(f"**Modelo 2: {best_models[1]}**")
        st.write(f"Predicción: {pred2[0]}")
        st.write(f"Confianza: {conf2:.2f}%")
