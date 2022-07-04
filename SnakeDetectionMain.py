import streamlit as st
from streamlit_option_menu import option_menu

import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import PIL

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import load_model

def main():
    st.title("Snake Species Classification")
    st.header("this is header")
    st.subheader('This is a subheader')
    st.caption('This is a string that explains something above.')
    upload_option = st.sidebar.selectbox("How you want to make photo",
                                         ('Upload','Shoot Photo'))

    if upload_option == 'Upload':
        photo = upload_photo()
    else:
        photo = shoot_photo()
    if st.button("Classifiy snake"):
        print("Classifing the snake photo in the model")
       
        class_names = ['Nerodia sipedon - Nothern Watersnake','Thamnophis sirtalis - Common Garter snake','Storeria dekayi -DeKay\'s Brown snake', 'Patherophis obsoletus - Black Rat snake', 'Cortalus atrox - Western Diamondback rattlesnake']
        with st.spinner("Classifying snake specie"):
           Ans = prediction(photo,"snake_species.h5")
        string=class_names[np.argmax(Ans)]
        st.header(string)
        st.write(Ans)
        
        



def prediction(img, weights_file):#this is copy file of road sign project
    # Load the model
    model = keras.models.load_model(weights_file)
   
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
   # prediction_percentage = model.predict(data)
    #prediction = prediction_percentage.round()
    return prediction

    #return prediction, prediction_percentage





     

def shoot_photo():
    snake_image = st.camera_input("Shoot photo to calssify snake")
    if snake_image is not None:
        st.sidebar.success("Photo Shooted successfully")
        img = Image.open(snake_image)
        st.sidebar.image(img)
        return img

def upload_photo():
    snake_image = st.file_uploader("Upload the file")
    if snake_image is not None:
        st.sidebar.success("Photo Uploaded successfully")
        img = Image.open(snake_image)
        st.sidebar.image(img)
        return img

main()
