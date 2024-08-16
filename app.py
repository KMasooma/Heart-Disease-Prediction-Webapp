import streamlit as st
import pandas as pd
import pickle

# Defining Xtreme gradient's selected features as expected columns
expected_columns = [
    'age', 'resting_bp', 'cholestrol', 'Max_heart_rate', 'oldpeak',
    'sex_0', 'chest_pain_0', 'chest_pain_1', 'chest_pain_2', 'chest_pain_3',
    'fasting_bsugar_0', 'resting_ECG_0', 'resting_ECG_1',
    'ex_angina_0', 'slope_0', 'slope_1', 'slope_2',
    'No_maj_vessels_0', 'No_maj_vessels_1', 'No_maj_vessels_2', 'No_maj_vessels_3',
    'thalassemia_1', 'thalassemia_2', 'thalassemia_3'
]

# App title
st.write("""
# Heart Disease Prediction App
This app predicts if a patient has heart disease.
""")

# Function to get user input
def user_input_features():
    st.sidebar.header('User Input Features')

    # Collecting user input features into dataframe
    age = st.sidebar.number_input('Enter Your Age: ')
    sex = st.sidebar.selectbox('Sex: ', (0, 1))
    chest_pain = st.sidebar.selectbox('Chest Pain Type: ', (0, 1, 2, 3))
    resting_bp = st.sidebar.number_input('Resting Blood Pressure: ')
    cholestrol = st.sidebar.number_input('Serum Cholestrol in mg/dl: ')
    fasting_bsugar = st.sidebar.selectbox('Fasting Blood Sugar: ', (0, 1))
    resting_ECG = st.sidebar.selectbox('Resting Electro-Cardiographic Results: ', (0, 1))
    Max_heart_rate = st.sidebar.number_input('Maximum Heart Rate Achieved: ')
    ex_angina = st.sidebar.selectbox('Exercise Induced Angina: ', (0, 1))
    oldpeak = st.sidebar.number_input('Oldpeak: ')
    slope = st.sidebar.selectbox('Slope of the Peak of ST Segment: ', (0, 1, 2))
    No_maj_vessels = st.sidebar.selectbox('Number of Major Vessels', (0, 1, 2, 3))
    thalassemia = st.sidebar.selectbox('Thalassemia', (0, 1, 2))

    # Creating DataFrame
    data = {
        'age': age, 
        'sex': sex, 
        'chest_pain': chest_pain, 
        'resting_bp': resting_bp,
        'cholestrol': cholestrol, 
        'fasting_bsugar': fasting_bsugar, 
        'resting_ECG': resting_ECG,
        'Max_heart_rate': Max_heart_rate, 
        'ex_angina': ex_angina, 
        'oldpeak': oldpeak,
        'slope': slope, 
        'No_maj_vessels': No_maj_vessels, 
        'thalassemia': thalassemia
    }
    features = pd.DataFrame(data, index=[0])
    
    # Creating dummy variables
    features = pd.get_dummies(features, columns=[
        'sex', 'chest_pain', 'fasting_bsugar', 'resting_ECG', 
        'ex_angina', 'slope', 'No_maj_vessels', 'thalassemia'
    ])
    
    # Ensuring all expected columns are present
    for column in expected_columns:
        if column not in features.columns:
            features[column] = 0
    
    # Reordering columns
    features = features[expected_columns]
    
    return features

input_df = user_input_features()

# Loading model
with open('Xtreme_gradient_model.pkl', 'rb') as file:
    load_clf = pickle.load(file)

# Making predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

# Adding an image
st.image('heart_disease.jpg', use_column_width=True)  # Ensure this image is in the same directory


# Displaying prediction results
st.subheader('Prediction')
result = 'The patient **has heart disease**' if prediction[0] == 1 else 'The patient **does not have heart disease**'
st.write(result)

st.subheader('Prediction Probability')
st.write(prediction_proba)

