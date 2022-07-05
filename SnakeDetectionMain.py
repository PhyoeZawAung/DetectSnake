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
     st.set_page_config(
        page_title="Snake Classification By Team PeniCol",
        page_icon="🐍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={

            'About': """This app is made especially for farmer and can use for educational purpose. 
            This app can classify the major 5 snake species.
            We also develop for the rest of species. We never stop develop.""",
        }
    )
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
        
    if st.sidebar.button("Contect Developer"):
        contact()
        
def contact():

    form_submit = """<form action="https://formsubmit.co/phyoezawaung9696@gmail.com" method="POST">
     <input type="text" name="name" placeholder=" Name "required>
     <input type="email" name="email" placeholder="Email Address">
     <textarea id="subject" name="subject" placeholder="Write something.." style="height:200px"></textarea>
     <input type="hidden" name="_captcha" value="false">
     <button type="submit">Send</button>
     </form>
     <style>

input[type=text],input[type=email], select, textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-sizing: border-box;
  margin-top: 6px;
  margin-bottom: 16px;
  resize: vertical;
}

button[type=submit] 
{
  background-color: #04AA6D;
  color: white;
  padding: 12px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button[type=submit]:hover
{
  background-color: #45a049;
}


</style>
     """
    #st.markdown(form_submit,unsafe_allow_html=True) this is not work css of button class
    components.html(form_submit,height =500)


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
