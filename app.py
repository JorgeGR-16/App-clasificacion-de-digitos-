import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
import numpy as np
from PIL import Image
import os

# Configuración de la página
st.set_page_config(page_title="Clasificador de Dígitos", layout="wide")

# Título de la aplicación
st.title("🖍️ Clasificador de Dígitos con Naive Bayes")
st.markdown("---")

# Verificación de archivos de modelo
MODEL_DIR = "modelos"
REQUIRED_MODELS = ["GaussianNB.pkl", "MultinomialNB.pkl"]

# Verificar si los modelos existen
missing_models = [m for m in REQUIRED_MODELS if not os.path.exists(os.path.join(MODEL_DIR, m))]

if missing_models:
    st.error(f"❌ Error: Faltan los siguientes archivos de modelo en la carpeta '{MODEL_DIR}': {', '.join(missing_models)}")
    st.stop()

# Cargar modelos
try:
    modelo1 = joblib.load(os.path.join(MODEL_DIR, "GaussianNB.pkl"))
    modelo2 = joblib.load(os.path.join(MODEL_DIR, "MultinomialNB.pkl"))
except Exception as e:
    st.error(f"❌ Error al cargar los modelos: {str(e)}")
    st.stop()

# Crear dos columnas
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Lienzo para dibujar")
    # Lienzo para dibujar
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Fondo negro
        stroke_width=15,
        stroke_color="rgba(255, 255, 255, 1)",  # Trazo blanco
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.header("Resultados")
    
    if canvas.image_data is not None:
        # Preprocesamiento de la imagen
        img = Image.fromarray(canvas.image_data.astype('uint8'))
        img = img.resize((28, 28)).convert('L')  # Redimensionar y convertir a escala de grises
        
        # Mostrar imagen procesada
        st.subheader("Imagen procesada (28×28 px)")
        st.image(img, width=150)
        
        # Convertir a array numpy
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, -1)
        
        # Realizar predicciones
        try:
            pred1 = modelo1.predict(img_array)
            prob1 = modelo1.predict_proba(img_array)
            conf1 = np.max(prob1) * 100
            
            pred2 = modelo2.predict(img_array)
            prob2 = modelo2.predict_proba(img_array)
            conf2 = np.max(prob2) * 100
            
            # Mostrar resultados
            st.subheader("Predicciones")
            
            st.metric(label=f"Modelo GaussianNB", 
                     value=f"{pred1[0]}", 
                     delta=f"{conf1:.1f}% de confianza")
            
            st.metric(label=f"Modelo MultinomialNB", 
                     value=f"{pred2[0]}", 
                     delta=f"{conf2:.1f}% de confianza")
            
        except Exception as e:
            st.error(f"Error durante la predicción: {str(e)}")

# Información adicional
st.markdown("---")
st.info("ℹ️ Dibuja un dígito del 0 al 9 en el lienzo y los modelos harán su predicción.")

# Nota: Asegúrate de tener los archivos .pkl en la carpeta 'modelos'
