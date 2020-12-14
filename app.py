import streamlit as st
import numpy as np
import functions
from PIL import Image

# App Title
st.image(logo, use_column_width=True)
st.title("Curae PnuemoApp")

# Introduction text
logo=Image.open('REALAI.jpg')

st.header("AI supported Pneumonia Detection App!")
st.markdown(unsafe_allow_html=True, body="<p>Pneumonia is an infection that inflames the alveoli in one or both lungs. 
						"The alveoli may fill with pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing." 
						"In general, pneumonia is caused by a virus, a bacteria or fungi."
						"Pneumonia can range in seriousness from mild to life-threatening."
						"It is most serious for infants and young children, people older than age 65,"
						"and people with health problems or weakened immune systems."
						"Chest X-Ray images can be useful in the diagnosis of Pneumonia. A doctor can diagnose it, 
						"know the extent and the location of the infection by looking at an X-Ray image withing a few seconds or minutes."
						"The aim of this work is to automatize the detection of Pneumonia by using Deep Learning algorithms and only X-Ray images."
 						"Different techniques will be used in order to achieve the best possible results. "
						"If this task can be sucessfully achieved by a machine, then it will help doctors to save a lot of time that could be used in other actions.<p>")


st.markdown("Load an X-Ray Chest image.")

# Loading model

# Img uploader
img = st.file_uploader(label="Load X-Ray Chest image", type=['jpeg', 'jpg', 'png'], key="xray")

if img is not None:
    # Preprocessing Image
    p_img = functions.preprocess_image(img)

    if st.checkbox('Zoom image'):
        image = np.array(Image.open(img))
        st.image(image, use_column_width=True)
    else:
        st.image(p_img)

    # Loading model
    loading_msg = st.empty()
    loading_msg.text("Predicting...")
    model = functions.load_model()

    # Predicting result
    prob, prediction = functions.predict(model, p_img)

    loading_msg.text('')

    if prediction:
        st.markdown(unsafe_allow_html=True, body="<span style='color:red; font-size: 50px'><strong><h4>Pneumonia! :slightly_frowning_face:</h4></strong></span>")
    else:
        st.markdown(unsafe_allow_html=True, body="<span style='color:green; font-size: 50px'><strong><h3>Healthy! :smile: </h3></strong></span>")

    st.text(f"*Probability of pneumonia is {round(prob[0][0] * 100, 2)}%")

