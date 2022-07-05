import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import load_model


def main():
    st.set_page_config(
        page_title="မြွေအမျိုးအစားခွဲခြားခြင်း ",
        page_icon="🐍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={

            'About': """လယ်သမားများနှင့် ပညာရေး အတွက် အသုံးပြုရန် ဖြစ်ပါသည်။
            မြွေအမျိုးအစား ၅ မျိုး ကိူ ခွဲခြားပေးနိုင်ပါသည်။
            ကျန်သောအမျိုးအစားများ အတွက်လည်း ဆက်လက်လုပ်ဆောင်နေပါသည်။
           """,
        }
    )
    upload_option = st.sidebar.selectbox("ပုံတင်ရန်ရွေးချယ်ပါ‌",
                                         ('ပုံကိုတင်မည် ', 'ပုံရိုက်မည် '))

    if upload_option == 'ပုံကိုတင်မည် ':
        photo = upload_photo()
    else:
        photo = shoot_photo()
    if st.button("ခွဲခြားမည် "):
        if photo is not None:
            print("Classifing the snake photo in the model")
            with st.spinner("ခွဲခြားနေသည်......"):
                Ans = prediction(photo, "snake_species (1).h5")


            res = Ans['name'].iat[0];
            st.header(res)

            links = {
            'Nerodia sipedon - Northern Watersnake': 'https://en.wikipedia.org/wiki/Common_watersnake',
            'Thamnophis sirtalis - Common Garter snake': 'https://en.wikipedia.org/wiki/Common_garter_snake',
            "Storeria dekayi - DeKay's Brown snake": 'https://en.wikipedia.org/wiki/DeKay%27s_brown_snake',
            'Patherophis obsoletus - Black Rat snake': 'https://en.wikipedia.org/wiki/Pantherophis_obsoletus',
            'Crotalus atrox - Western Diamondback rattlesnake': 'https://en.wikipedia.org/wiki/Western_diamondback_rattlesnake',
            'Others': '',
            }

            is_venomous = {
            'Nerodia sipedon - Northern Watersnake': "<span style='color: green'>Non-venomous</span>",
            'Thamnophis sirtalis - Common Garter snake': "<span style='color: darkorange'>Mildly Venomous</span>",
            "Storeria dekayi - DeKay's Brown snake": "<span style='color: green'>Non-venomous</span>",
            'Patherophis obsoletus - Black Rat snake': "<span style='color: green'>Non-venomous</span>",
            'Crotalus atrox - Western Diamondback rattlesnake': "<span style='color: red'>Venomous</span>",
            'Others': 'ဝမ်းနည်းပါတယ်... ပုံထဲက မျိုးစိတ်တွေကို ကျွန်ုပ်တို့ရဲ့ မော်ဒယ်က ရှာမတွေ့နိုင်ပါဘူး။',
            }
    

            st.markdown(is_venomous[res], unsafe_allow_html=True)
            if res != 'Others':
                st.write("နောက်ထပ်အကြောင်းအရာများသိလိုလျှင်[link]("+links[res]+")")
            st.markdown("<span style='color: darkorange'>Warning: ‌မြွေကိုက်ခံရလျှင် ဆေးရုံသိူ့ အမြန်ဆုံးသွားသင့်ပါသည် ။ မြွေကိုက်ခံရခြင်းသည် အဆိပ်ရှိမရှိ လူနာအား ဆေးရုံတင်ထားရပြီး အနီးကပ်စောင့်ကြည့်ရန် လိုအပ်ပါသည်။.</span>", unsafe_allow_html=True)

            st.markdown("""---""")
            st.header("Confidence Chart")
            fig, ax = plt.subplots()

            ax  = sns.barplot(y = 'name',x='values', data = Ans,order = Ans.sort_values('values',ascending=False).name)

            ax.set(xlabel='Confidence %', ylabel='Species')

            for i in ax.containers:
                ax.bar_label(i,)

            st.pyplot(fig)
        else:
            st.warning("ပုံတင်ခြင်း (သို့မဟုတ်) ပုံရိုက်ခြင်းကို အရင်ပြုလုပ်ပါ ")
    if st.sidebar.button("ဆက်သွယ်ရန် "):
        contact()


def contact():
    form_submit = """<form action="https://formsubmit.co/phyoezawaung9696@gmail.com" method="POST">
     <input type="text" name="name" placeholder="အမည် "required>
     <input type="email" name="email" placeholder="အီးမေး">
     <textarea id="subject" name="subject" placeholder="အကြောင်းအရာ......." style="height:200px"></textarea>
     <input type="hidden" name="_captcha" value="false">
     <button type="submit">ပေးပို့မည် </button>
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
    # st.markdown(form_submit,unsafe_allow_html=True)
    components.html(form_submit, height=500)


def prediction(img, weights_file):#this is copy file of road sign project

    class_names = [
            "Nerodia sipedon - Northern Watersnake",
            "Thamnophis sirtalis - Common Garter snake",
            "Storeria dekayi - DeKay's Brown snake", 
            "Patherophis obsoletus - Black Rat snake", 
            "Crotalus atrox - Western Diamondback rattlesnake",
            'Others'
        ]
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
    prediction = model.predict(data)*100

    prediction = pd.DataFrame(np.round(prediction,1),columns = class_names).transpose()

    prediction.columns = ['values']

    prediction  = prediction.nlargest(6, 'values')

    prediction = prediction.reset_index()

    prediction.columns = ['name', 'values']

   # prediction_percentage = model.predict(data)
    #prediction = prediction_percentage.round()
    return prediction

    #return prediction, prediction_percentage

def shoot_photo():
    snake_image = st.camera_input("မြွေကို အလယ်တွင်ထား ရိုက်ပါ")
    if snake_image is not None:
        st.sidebar.success("ပုံရိုက်ခြင်းအောင်မြင်ပါသည")
        img = Image.open(snake_image)
        st.sidebar.image(img)
        return img


def upload_photo():
    snake_image = st.file_uploader("ဖိုင်ရွေးချယ်၍ တင်ပါ")
    if snake_image is not None:
        st.sidebar.success("ဖိုင်တင်ခြင်းအောင်မြင်ပါသည်")
        img = Image.open(snake_image)
        st.image(img)
        return img


main()
