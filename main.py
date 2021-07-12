import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
import io
import keras
from skimage import transform


primaryColor="#F63366"
backgroundColor="#8FF6F2"
secondaryBackgroundColor="#F0F2F6"
textColor="#000000"
font="sans serif"

model = keras.models.load_model('tumor_model.h5')

st.set_page_config(layout="centered")

st.title('Brain Tumor Detection')
st.subheader('Find some MRI images here : https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection')

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

#image augmenting for predicting
def import_and_predict(image_data, model):
    np_image = image_data
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    prediction = model.predict(np_image)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    im = Image.open(file)
    st.image(im, use_column_width=True)
    prediction = import_and_predict(im, model)

    if prediction[[0]] < 0.5:
        st.subheader("Ans. - No Tumor in the given MRI scan.")
    elif prediction[[0]] > 0.5:
        st.subheader("Ans. - Tumor present in the given MRI scan.")
    
    st.warning("Probability 0-0.5 : No tumor , 0.5-1 : Tumor.")
    st.error(prediction)