import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Load the MNIST model for image classification
mnist_model = tf.keras.models.load_model(r"D:\streamlit\CNN\mnist_model.h5")  

# Load the CNN model for tumor prediction
tumor_model = tf.keras.models.load_model(r"D:\streamlit\CNN\tumor_model.h5")  
# Load the tokenizer for spam detection
with open(r'D:\streamlit\RNN\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)  

# Load the trained RNN model for spam detection
rnn_model = tf.keras.models.load_model("D:/streamlit/RNN/rnn_spam_model.h5")  

# Load the trained RNN model for spam detection
lstm_spam = tf.keras.models.load_model("D:/streamlit/LSTM/lstm_spam")

# Load the trained LSTM model for sentiment analysis
lstm_model = tf.keras.models.load_model("D:/streamlit/LSTM/imdb_model.h5")  



# Streamlit app
 
st.title('Modelhub - Multimodel App')

option = st.sidebar.selectbox('Choose Analysis Type', ('Image Classification', 'Text Analysis'))

if option == 'Image Classification':
    st.subheader('Image Classification')
    image_option = st.selectbox('Choose Image Task', ('MNIST Handwritten Digits Recognition', 'Tumor Prediction'))

    if image_option == 'MNIST Handwritten Digits Recognition':
        st.subheader('MNIST Handwritten Digits Recognition')
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28))
            img = img.reshape(1, 28, 28, 1)

            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            prediction = mnist_model.predict(img)
            st.write(f"Prediction: {np.argmax(prediction)}")

    elif image_option == 'Tumor Prediction':
        st.subheader('Tumor Prediction')
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            img = cv2.resize(img_array, (128, 128))
            img = img / 255.0  # Normalize the image
            img = np.expand_dims(img, axis=0)

            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            prediction = tumor_model.predict(img)
            if prediction > 0.5:
                st.write("Prediction: Tumor Detected")
            else:
                st.write("Prediction: No Tumor Detected")

elif option == 'Text Analysis':
    st.subheader('Text Analysis')
    text_option = st.selectbox('Choose Text Task', ('Spam Detection-RNN', 'Spam Detection-LSTM','Sentiment Analysis'))

    if text_option == 'Spam Detection-RNN':
        st.subheader('Spam Detection-RNN')
        text_input = st.text_area("Enter text here")
        
        if st.button('Detect Spam'):
            # Tokenize and pad the input text
            sequence = tokenizer.texts_to_sequences([text_input])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=50)
            
            # Make predictions using rnn_model
            prediction = rnn_model.predict(padded_sequence)
            
            # Display prediction result
            st.write("Prediction Result: Spam" if prediction > 0.5 else "Prediction Result: Not Spam")
    
    elif text_option == 'Spam Detection-LSTM':
        st.subheader('Spam Detection-LSTM')
        text_input = st.text_area("Enter text here")
        
        if st.button('Predict'):
            # Tokenize and pad the input text
            sequence = tokenizer.texts_to_sequences([text_input])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, padding='post', maxlen=50)
        
            # Make prediction
            prediction = lstm_spam.predict(padded_sequence)
        
            # Display prediction result
            if prediction[0][0] >= 0.5:
                st.error('Spam!')
            else:
                st.success('Ham!')
    
    elif text_option == 'Sentiment Analysis':
        st.subheader('Sentiment Analysis')
        text_input = st.text_area("Enter text here")

            
        if st.button('Analyze Sentiment'):
            # Tokenize and pad the input text
            sequence = tokenizer.texts_to_sequences([text_input])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=500)
            
            # Make predictions using lstm_model
            prediction = lstm_model.predict(padded_sequence)
            
            # Display prediction result
            st.write("Sentiment Analysis Result: Positive" if prediction > 0.5 else "Sentiment Analysis Result: Negative")
