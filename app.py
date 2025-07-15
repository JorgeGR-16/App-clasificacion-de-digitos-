import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
import numpy as np
from PIL import Image
import os

# Configuración
st.set_page_config(page_title="Clasificador de Dígitos NB", layout="wide")
st.title("🔢 Clasificador de Dígitos (4 Modelos Naive Bayes)")

# Lista de modelos a cargar
MODELOS = {
    "GaussianNB": "modelos/GaussianNB.pkl",
    "MultinomialNB": "modelos/MultinomialNB.pkl", 
    "ComplementNB": "modelos/ComplementNB.pkl",
    "BernoulliNB": "modelos/BernoulliNB.pkl"
}

# Función para cargar modelos
@st.cache_resource
def cargar_modelos():
    modelos_cargados = {}
    for nombre, ruta in MODELOS.items():
        try:
            modelos_cargados[nombre] = joblib.load(ruta)
        except Exception as e:
            st.error(f"❌ Error cargando {nombre}: {str(e)}")
            return None
    return modelos_cargados

# Carga de modelos
with st.spinner("Cargando modelos..."):
    modelos = cargar_modelos()
    
if not modelos:
    st.error("No se pudieron cargar los modelos. Verifica los archivos.")
    st.stop()

# Lienzo para dibujar
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Dibuja tu dígito")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

# Procesamiento y predicción
if canvas.image_data is not None:
    with col2:
        st.header("Resultados")
        
        # Preprocesamiento
        img = Image.fromarray(canvas.image_data.astype('uint8'))
        img = img.resize((28, 28)).convert('L')
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, -1)
        
        # Mostrar imagen procesada
        st.subheader("Imagen procesada")
        st.image(img, width=100)
        
        # Predicciones
        st.subheader("Predicciones")
        
        for nombre, modelo in modelos.items():
            try:
                pred = modelo.predict(img_array)[0]
                proba = modelo.predict_proba(img_array)
                confianza = np.max(proba) * 100
                
                st.metric(
                    label=f"Modelo {nombre}",
                    value=pred,
                    delta=f"{confianza:.1f}% confianza"
                )
            except Exception as e:
                st.error(f"Error en {nombre}: {str(e)}")

# Sección informativa
st.markdown("---")
st.info("""
**Instrucciones:**
1. Dibuja un dígito (0-9) en el lienzo
2. Los 4 modelos harán sus predicciones
3. Compara los resultados y niveles de confianza
""")
