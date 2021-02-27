from preprocessing import resize_img, hair_removal, classifier
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import urllib

# @st.cache
st.title("SkinCare Recommender System")
st.write("""
    ## Which One is Best ! 
    ### Based on 8 Dataset's classes
""")

uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)

selective_arch = st.sidebar.selectbox("Select Architcture: ", ("Simpler Method", "Vgg16", "Resnet-50", "EffecientNet", "Inception"))
st.write("You have selected "+selective_arch+" To check The Accuracy of Model")

# component for toggling code
show_code = st.sidebar.checkbox("Show Code")
st.write("""
        ### Step-I
        #### Uploaded Image:
    """)
if uploaded_file:
    st.image(uploaded_file)
    bytes_data = uploaded_file.read()
    st.write(BytesIO(bytes_data))
    image = Image.open(BytesIO(bytes_data))
    image = np.array(image)
    open_cv_image = image[:, :, ::-1].copy() 
    st.write("""
        ### Step-II
        #### Preprocessing:
        ##### Resize Image:

    """)
    img = resize_img(open_cv_image)
    st.image(img)
    st.write("""
        ##### Noise Remove:
    """)
    img = hair_removal(img)
    st.image(img)
    st.write("""
        ##### Prediction:
    """)
    if selective_arch == "Simpler Method":
        prediction = classifier(img)
        class_name = ["Malignant", "Benign", "Basal Cell"]
        string="This Image Mostly like to: "+class_name[np.argmax(prediction)]
        st.write(prediction)
        st.success(string)
    else:
        st.write("Models are Not Available yet")
else:
    st.write("NILL")

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    # path_file = "preprocessing.py"
    url = "https://github.com/MujtabaAhmad0928/SCRS/blob/main/"+path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

if show_code:
    st.code(get_file_content_as_string("main.py"))



