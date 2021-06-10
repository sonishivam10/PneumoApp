import keras
import numpy as np
import streamlit as st
from keras import layers, models, optimizers  # modeling
from PIL import Image

MODEL = "best_model_multiclass_128.h5"


@st.cache(allow_output_mutation=True)
def load_model():
    print("loading model")
    model = keras.models.load_model(f"model/{MODEL}", compile=True)

    return model


def preprocess_image(img):
    image = Image.open(img).convert("RGB")
    p_img = image.resize((128,128))

    return np.array(p_img) / 255.0


def predict(model, img):
    prob = model.predict(np.reshape(img, [1, 128, 128, 3]))
    #if prob > 0.5:
    #    prediction = True
    #else:
    #    prediction = False
    #return prob, prediction
    if  prob==0:
        prediction="Normal"
    elif prob==1:
        prediction="Viral"
    else:
        prediction="Bacterial"

