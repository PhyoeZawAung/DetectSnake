import streamlit as st
from PIL import Image


def main():
    upload_option = st.sidebar.selectbox("ပုံတင်ရန်ရွေးချယ်ပါ‌",
                                         ('ပုံကိုတင်မည် ', 'ပုံရိုက်မည် '))

    if upload_option == 'ပုံကိုတင်မည် ':
        upload_photo()
    else:
        shoot_photo()


def shoot_photo():
    snake_image = st.camera_input("မြွေကို အလယ်တွင်ထား ရိုက်ပါ")
    if snake_image is not None:
        st.sidebar.success("ပုံရိုက်ခြင်းအောင်မြင်ပါသည")
        img = Image.open(snake_image)
        st.sidebar.image(img)
    return snake_image


def upload_photo():
    snake_image = st.file_uploader("ဖိုင်ရွေးချယ်၍ တင်ပါ")
    if snake_image is not None:
        st.sidebar.success("ဖိုင်တင်ခြင်းအောင်မြင်ပါသည်")
        img = Image.open(snake_image)
        st.sidebar.image(img)
    return snake_image


main()