import streamlit as st 
from PIL import Image 
import tensorflow as tf
from keras.preprocessing import image
import pandas as pd
import numpy as np
import cv2 as cv 
st.title(":green[MNIST Classifier]")

st.header("**Upload a picture of a number from an MNIST-like format.**")
st.sidebar.title(":blue[The model was trained using a CNN and has an accuracy of 92.2%.]")
st.sidebar.title("")
st.sidebar.title("MNIST is a classic dataset of handwritten images. The images are 28x28 pixels and are grayscale. The images are of handwritten digits from 0-9. The goal is to classify the image as one of the 10 digits.")


st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
st.sidebar.title(" ")
uploaded_file = st.sidebar.file_uploader("Choose a file")
model = tf.keras.models.load_model('MNIST_better_e100.h5')
if uploaded_file is not None:

    st.image(uploaded_file,width=400)
    

    test_image=image.load_img(uploaded_file,
                         target_size=(28,28))
    
    test_image=cv.cvtColor(np.array(test_image), cv.COLOR_BGR2GRAY)
    
    test_image=np.reshape(image.img_to_array(test_image).flatten(),(1,28,28))
    result1= model.predict([test_image])
    st.warning("The number is:")
    st.warning(np.array(result1).argmax(axis=1)[0])
 
