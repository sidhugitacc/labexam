import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import re

@st.cache_data()
def load_data():
    data = pd.read_csv("preprocessed_data.csv")
    return data

def grayscale(image):
    grayscale_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return grayscale_image

def rotate(image, angle):
    rotated_image = np.array(image.rotate(angle))
    return rotated_image

def resize_image(image, width, height):
    resized_image = image.resize((width, height))
    return resized_image

def crop_image(image, x, y, width, height):
    cropped_image = image.crop((x, y, x + width, y + height))
    return cropped_image



def main():
    st.title("Women's Clothing E-Commerce")

 
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select Option", ["3D Plot Visualisation", "Image Processing", "Text Similarity Analysis"])

    if option == "3D Plot Visualisation":
    
        data = load_data()


        st.write("Loaded Data:")
        st.write(data)

        # Sidebar filters and visualization code...
        st.sidebar.subheader("Filters")
        age_range = st.sidebar.slider("Select Age Range", int(data["Age"].min()), int(data["Age"].max()), (int(data["Age"].min()), int(data["Age"].max())))
        rating = st.sidebar.selectbox("Select Rating", [1, 2, 3, 4, 5])
        recommended = st.sidebar.selectbox("Select Recommendation", ["Recommended", "Not Recommended"])
        division_names = st.sidebar.multiselect("Select Division Name", data["Division Name"].unique())
        department_names = st.sidebar.multiselect("Select Department Name", data["Department Name"].unique())
        class_names = st.sidebar.multiselect("Select Class Name", data["Class Name"].unique())

        # Print filtering conditions
        st.write("Filtering Conditions:")
        st.write("Age Range:", age_range)
        st.write("Rating:", rating)
        st.write("Recommended:", recommended)
        st.write("Division Names:", division_names)
        st.write("Department Names:", department_names)
        st.write("Class Names:", class_names)

        # Filter data
        filtered_data = data[
            (data["Age"] >= age_range[0]) & (data["Age"] <= age_range[1]) &
            (data["Rating"] == rating) &
            ((data["Recommended IND"] == 1) if recommended == "Recommended" else (data["Recommended IND"] == 0)) &
            (data["Division Name"].isin(division_names)) &
            (data["Department Name"].isin(department_names)) &
            (data["Class Name"].isin(class_names))
        ]

        # Print number of rows in filtered dataset
        st.write("Number of Rows in Filtered Data:", len(filtered_data))

        # Show filtered data
        st.write("### Filtered Data")
        st.write(filtered_data)

        st.subheader("3D Plot: Age, Rating, and Positive Feedback Count")
        fig = px.scatter_3d(filtered_data, x='Age', y='Rating', z='Positive Feedback Count', color='Rating')
        st.plotly_chart(fig)

    elif option == "Image Processing":
        # Image processing
        st.sidebar.subheader("Image Processing")
        uploaded_images = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_images:
            for i, uploaded_image in enumerate(uploaded_images):
                image = Image.open(uploaded_image)
                st.image(image, caption=f"Uploaded Image {i+1}", use_column_width=True)

                # Image operations
                st.subheader(f"Image Operations for Image {i+1}")
                grayscale_option = st.checkbox(f"Convert to Grayscale {i+1}", key=f"grayscale_{i}")
                blur_option = st.checkbox(f"Apply Gaussian Blur {i+1}", key=f"blur_{i}")
                rotation_option = st.checkbox(f"Rotate Image {i+1}", key=f"rotate_{i}")
                resize_option = st.checkbox(f"Resize Image {i+1}", key=f"resize_{i}")
                crop_option = st.checkbox(f"Crop Image {i+1}", key=f"crop_{i}")



                if grayscale_option:
                    grayscale_image = grayscale(image)
                    st.image(grayscale_image, caption=f"Grayscale Image {i+1}", use_column_width=True)

                if rotation_option:
                    rotation_angle = st.slider(f"Select Rotation Angle (degrees) {i+1}", -180, 180, 0, key=f"angle_{i}")
                    rotated_image = rotate(image, rotation_angle)
                    st.image(rotated_image, caption=f"Rotated Image (Angle: {rotation_angle} degrees) Image {i+1}", use_column_width=True)
                
                if resize_option:
                    new_width = st.slider(f"Select New Width {i+1}", 1, image.width, image.width, key=f"width_{i}")
                    new_height = st.slider(f"Select New Height {i+1}", 1, image.height, image.height, key=f"height_{i}")
                    resized_image = resize_image(image, new_width, new_height)
                    st.image(resized_image, caption=f"Resized Image {i+1}", use_column_width=True)

                if crop_option:
                    crop_x = st.slider(f"Select X coordinate for cropping {i+1}", 0, image.width, 0, key=f"crop_x_{i}")
                    crop_y = st.slider(f"Select Y coordinate for cropping {i+1}", 0, image.height, 0, key=f"crop_y_{i}")
                    crop_width = st.slider(f"Select Width for cropping {i+1}", 1, image.width, image.width, key=f"crop_width_{i}")
                    crop_height = st.slider(f"Select Height for cropping {i+1}", 1, image.height, image.height, key=f"crop_height_{i}")
                    cropped_image = crop_image(image, crop_x, crop_y, crop_width, crop_height)
                    st.image(cropped_image, caption=f"Cropped Image {i+1}", use_column_width=True)



if __name__ == "__main__":
    main()






# import streamlit as st
# import pandas as pd

# # Load data
# @st.cache_data()
# def load_data():
#     data = pd.read_csv("WomensClothingE-CommerceReviews.csv")
#     return data

# def preprocess_data(data):
#     # Replace missing titles with "None"
#     data["Title"].fillna("None", inplace=True)
#     return data

# def main():
#     st.title("Women's Clothing E-Commerce Dashboard")

#     # Load data
#     data = load_data()

#     # Preprocess data
#     data = preprocess_data(data)

#     # Save preprocessed data to a new CSV file
#     data.to_csv("preprocessed_data.csv", index=False)

#     # Print loaded and preprocessed data
#     st.write("Loaded Data:")
#     st.write(data)

#     # Sidebar filters and visualization code...

# if __name__ == "__main__":
#     main()

