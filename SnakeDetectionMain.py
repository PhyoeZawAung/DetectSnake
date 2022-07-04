import streamlit as st
from PIL import Image
import tensorflow
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
        photo =shoot_photo()
    if st.button("Classifiy snake"):
        print("Classifing the snake photo in the model")
        st.write("Classifying.....")
        prediction(photo,"snake_species.h5")



def prediction(img, weights_file):#this is copy file of road sign project
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 244, 244, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (244, 244)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255)

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    #prediction_percentage = model.predict(data)
    #prediction = prediction_percentage.round()
    return prediction

    #return prediction, prediction_percentage





     

def shoot_photo():
    snake_image = st.camera_input("Shoot photo to calssify snake")
    if snake_image is not None:
        st.sidebar.success("Photo Shooted successfully")
        img = Image.open(snake_image)
        st.sidebar.image(img)
    return snake_image

def upload_photo():
    snake_image = st.file_uploader("Upload the file")
    if snake_image is not None:
        st.sidebar.success("Photo Uploaded successfully")
        img = Image.open(snake_image)
        st.sidebar.image(img)
    return snake_image

main()
