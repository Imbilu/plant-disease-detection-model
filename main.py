# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# from streamlit_chat import message
# import os
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.llms import HuggingFacePipeline

# # Check if a GPU is available and set the device accordingly
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the model and tokenizer
# # model_name = "distilgpt2"

# @st.cache_resource
# def load_disease_model():
#     try:
#         model_path = 'plant_disease_model.keras'
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"File not found: {model_path}. Please ensure the file is in the correct location.")

#         disease_model = tf.keras.models.load_model(model_path)
#         return disease_model
#     except Exception as e:
#         st.error(f"Error loading disease model: {e}")
#         return None

# disease_model = load_disease_model()

# class_names = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
#     'Apple___healthy', 'Blueberry___healthy',
#     'Cherry_(including_sour)___Powdery_mildew',
#     'Cherry_(including_sour)___healthy',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#     'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
#     'Corn_(maize)___healthy', 'Grape___Black_rot',
#     'Grape___Esca_(Black_Measles)',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#     'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
#     'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
#     'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
#     'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
#     'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
#     'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
#     'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]

# def model_prediction(image):
#     try:
#         img = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
#         input_arr = tf.keras.preprocessing.image.img_to_array(img)
#         input_arr = np.expand_dims(input_arr, axis=0)
#         prediction = disease_model.predict(input_arr)
#         res_idx = np.argmax(prediction)
#         return res_idx
#     except Exception as e:
#         st.error(f"Error in prediction: {e}")
#         return None

# @st.cache_resource
# def load_chatbot_model():
#     model_name = "distilgpt2"  # Using a smaller model that works well on CPUs
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     return tokenizer, model

# tokenizer, chatbot_model = load_chatbot_model()

# # Create the conversation pipeline
# generator = pipeline("text-generation", model=chatbot_model, tokenizer=tokenizer, device=device, max_new_tokens=100, temperature=0.3)

# # Define the LangChain prompt and chain
# template = """
# You are a helpful plant disease expert. A user just uploaded an image of a plant.
# The model predicted that:
# The plant is a {plant} and its leaves have {disease}. What should I do?

# User: {user_input}
# Expert:
# """
# prompt = PromptTemplate(template=template, input_variables=["plant", "disease", "user_input"])
# llm = HuggingFacePipeline(pipeline=generator)
# llm_chain = LLMChain(prompt=prompt, llm=llm)

# # App logic
# def initialize_chatbot():
#     if "generated" not in st.session_state:
#         st.session_state["generated"] = []
#     if "past" not in st.session_state:
#         st.session_state["past"] = []

# def model_prediction(image):
#     img = tf.keras.preprocessing.image.load_img(image, target_size=(128, 128))
#     input_arr = tf.keras.preprocessing.image.img_to_array(img)
#     input_arr = np.expand_dims(input_arr, axis=0)
#     prediction = disease_model.predict(input_arr)
#     res_idx = np.argmax(prediction)
#     return res_idx

# def get_expert_response(plant, disease, user_input=""):
#     return llm_chain.run(plant=plant, disease=disease, user_input=user_input, max_new_tokens=1024)

# def format_response(raw_response):
#     # Extract the actual advice part from the raw response
#     advice_start = raw_response.find("Expert:") + len("Expert:")
#     formatted_response = raw_response[advice_start:].strip()
#     return formatted_response

# def main():
#     st.sidebar.title("Dashboard")
#     app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Predict & Chat", "Chatbot"])

#     if app_mode == "Home":
#         st.header("Plant Disease Detection")
#         st.image("tomato.jpeg", use_column_width=True)
#         st.markdown('''
#             Welcome to your one-stop shop for healthy plants! Worried about those mysterious spots on your leaves?
#             Wondering what's ailing your favorite flower? Upload a picture here, and our cutting-edge AI will analyze it to identify potential plant diseases.
#             We'll provide you with clear information and helpful guidance to get your greenery back on track!

#             ### Get Started
#             Click on the **Predict & Chat** page in the sidebar to upload an image and get a prediction.
#         ''')

#     elif app_mode == "About":
#         st.header("About")
#         st.markdown('''
#             This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
#             This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes.
#             The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
#             A new directory containing 33 test images is created later for prediction purposes.
#         ''')

#     elif app_mode == "Predict & Chat":
#         st.header("Prediction & Chat")
#         image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

#         if image:
#             st.image(image, use_column_width=True, caption="Uploaded Image")
#             if st.button("Predict"):
#                 res_idx = model_prediction(image)
#                 plant, *disease = class_names[res_idx].split("___")
#                 prediction_message = f"**Model Prediction**  \nPlant: {plant}  \nLeaf Condition: {' '.join(disease)}"
#                 st.success(prediction_message)

#                 # Generate initial chatbot response
#                 formatted_prediction = f"The plant is a {plant} and its leaves have {' '.join(disease)}. What should I do?"
#                 initial_response = get_expert_response(plant, ' '.join(disease))
#                 initial_response = format_response(initial_response)
#                 st.markdown(f"### Expert Response:\n{initial_response}")

#                 # Initialize chatbot
#                 initialize_chatbot()

#                 # Display chat history
#                 with st.expander("Chat History", expanded=True):
#                     for i in range(len(st.session_state['generated'])):
#                         message(st.session_state['generated'][i], key=str(i))
#                         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

#                 # Text Input (using form submit button to prevent automatic reloads)
#                 with st.form(key='chat_form', clear_on_submit=True):
#                     user_input = st.text_input("You: ")
#                     submit_button = st.form_submit_button(label='Send')

#                     if submit_button and user_input:
#                         st.session_state['past'].append(user_input)
#                         response = get_expert_response(plant, ' '.join(disease), user_input)
#                         response = format_response(response)
#                         st.session_state['generated'].append(response)
#                         message(response)
#                         message(user_input, is_user=True)

#             else:
#                 st.error("Failed to get a prediction. Please try again.")
#         else:
#             st.warning("Please upload an image to get a prediction.")

#     elif app_mode == "Chatbot":
#         st.header("Chat with Plant Disease Expert")
#         initialize_chatbot()

# if __name__ == "__main__":
#     main()


import streamlit as st
import requests

st.title('Plant Disease Prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, use_column_width=True, caption="Uploaded Image")

    if st.button("Predict"):
        # Send the image to the backend for prediction
        files = {'file': uploaded_file.getvalue()}
        response = requests.post('https://1eba-34-141-252-88.ngrok-free.app/predict', files=files)
        prediction = response.json()
        st.success(f"Prediction: {prediction['result']}")
        st.write(prediction['description'])

st.header("Chat with Plant Disease Expert")

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You: ")
    submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        response = requests.post('https://1eba-34-141-252-88.ngrok-free.app/chat', json={'message': user_input})
        st.write(f"Expert: {response.json()['reply']}")
