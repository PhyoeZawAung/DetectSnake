import streamlit as st
def main():
    upload_option = st.sidebar.selectbox("ပုံတင်ရန်ရွေးချယ်ပါ‌",
                                         ('ပုံကိုတင်မည် ', 'ပုံရိုက်မည် '))

    if upload_option == 'ပုံကိုတင်မည် ':
        upload_photo()
    else:
        shoot_photo()
    if st.button("ခွဲခြားမည် "):
        print("Classifing the snake photo in the model")
        st.write("ခွဲခြားနေသည်........")
        class_names = ['class-1','class-2','class-3', 'class-4', 'class-5']
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
        st.sidebar.image(img)
        return img


main()
