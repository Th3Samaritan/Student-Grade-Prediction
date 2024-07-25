import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model and preprocessor
with open('student_gpa_model_rf.pkl', 'rb') as f:
    model = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Function to classify GPA
def classify_gpa(gpa):
    if gpa < 2.80:
        return 'Third Class'
    elif gpa < 3.50:
        return 'Second Class Lower'
    elif gpa < 4.50:
        return 'Second Class Upper'
    else:
        return 'First Class'

# Streamlit app
st.title('Student GPA Prediction')

# Input fields
age = st.number_input('Age', min_value=18, max_value=25, value=20)
gender = st.selectbox('Gender', ['Male', 'Female'])
discipline = st.selectbox('Discipline', [
    'Computer Science Technology', 'Science Laboratory Technology',
    'Home and Rural Economic', 'Horticultural Technology', 
    'Fisheries Technology', 'Crop Production Technology', 
    'Wildlife and Ecotourism Management'
])
cognitive_score = st.number_input('Cognitive Score', min_value=0, max_value=100, value=70)
learning_strategy_score = st.number_input('Learning Strategy Score', min_value=0, max_value=100, value=50)

# Prediction button
if st.button('Predict GPA'):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Discipline': [discipline],
        'CognitiveScore': [cognitive_score],
        'LearningStrategyScore': [learning_strategy_score]
    })
    
    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)
    
    # Make prediction
    predicted_gpa = model.predict(input_data_preprocessed)[0]
    gpa_class = classify_gpa(predicted_gpa)
    
    st.write(f'Predicted GPA: {predicted_gpa:.2f}')
    st.write(f'Academic Classification: {gpa_class} ({predicted_gpa:.2f})')
