import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# 🌻 Mostrar versión de Python
st.write("🌞 Versión de Python:", platform.python_version())

# 🌻 Cargar modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 🌻 Estilo de la página
st.markdown("""
<style>
body {
    background-color: #fff8dc;
    color: #4a3000;
}
h1, h2, h3 {
    color: #d4a017;
    text-align: center;
    font-family: 'Comic Sans MS', cursive;
}
.stButton>button {
    background-color: #f4d03f;
    color: #4a3000;
    border-radius: 10px;
    border: 2px solid #d4a017;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #f1c40f;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# 🌻 Título de la app
st.title("🌻 Reconocimiento de Imágenes entre Girasoles 🌞")
st.markdown("Usando un modelo entrenado con Teachable Machine para identificar posiciones o gestos 🌼")

# 🌻 Imagen inicial
image = Image.open('girasol.jpg')
st.image(image, width=350, caption="🌻 Imagen de ejemplo")

# 🌻 Barra lateral informativa
with st.sidebar:
    st.subheader("🌼 Cómo usar la app")
    st.write("1. Toma una foto con la cámara. 📸")
    st.write("2. El modelo analizará la imagen y dará resultados con probabilidad.")
    st.write("3. Observa los resultados y prueba con diferentes posiciones. 🌻")

# 🌻 Captura desde la cámara
img_file_buffer = st.camera_input("Toma una Foto 🌞")

if img_file_buffer is not None:
    # 🌻 Preprocesar imagen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # 🌻 Inferencia
    prediction = model.predict(data)
    st.markdown("### 🌻 Resultados de la Predicción")
    #print(prediction)  # opcional en consola

    # 🌞 Mostrar resultados según probabilidad
    if prediction[0][0] > 0.5:
        st.header(f'⬅️ Izquierda, con Probabilidad: {prediction[0][0]:.2f} 🌻')
    if prediction[0][1] > 0.5:
        st.header(f'⬆️ Arriba, con Probabilidad: {prediction[0][1]:.2f} 🌼')
    # Si se tiene más categorías, se pueden agregar aquí
    # if prediction[0][2] > 0.5:
    #     st.header(f'➡️ Derecha, con Probabilidad: {prediction[0][2]:.2f} 🌞')
