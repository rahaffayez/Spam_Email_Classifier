import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('tfidf_vectorizer (1).pkl')

# Function to predict on a new message
def predict_message(message, model, vectorizer):
    # Preprocess the message
    message_transformed = vectorizer.transform([message])

    # Predict the class
    predicted_class = model.predict(message_transformed)[0]

    # Determine the label based on the predicted class
    if predicted_class == 1:
        prediction = "This message is spam"
    else:
        prediction = "This message is not spam"

    return prediction

# Streamlit UI elements
st.title("Message Classification App")
st.write("Enter a message to classify it:")

# Text input for the user to enter a message
message = st.text_input("Message")

# When the user clicks the 'Predict' button
if st.button("Predict"):
    if message:
        # Call the prediction function
        prediction = predict_message(message, model, vectorizer)

        # Show the prediction result
        st.write(f"Prediction: {prediction}")
    else:
        st.write("Please enter a message to classify.")





