import streamlit as st
from PIL import Image
def main():
    upload_option = st.sidebar.selectbox("How you want to make photo",
                                         ('Upload','Shoot Photo'))

    if upload_option == 'Upload':
        upload_photo()
    else:
        shoot_photo()







     

def shoot_photo():
    snake_image = st.camera_input("Shoot photo to calssify snake")
    if snake_image is not None:
        st.sidebar.success("Photo Shooted")
        img = Image.open(snake_image)
        st.sidebar.image(img)
    return snake_image

def upload_photo():
    snake_image = st.file_uploader("Upload the file")
    if snake_image is not None:
        st.sidebar.success("Photo Uploaded")
        img = Image.open(snake_image)
        st.sidebar.image(img)
    return snake_image

main()