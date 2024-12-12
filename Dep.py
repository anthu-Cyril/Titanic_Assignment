




import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Titanic Survival Prediction")

# Input features for prediction
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, step=1)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, step=1)
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=1000.0, step=0.1)
sex = st.selectbox("Sex", ["male", "female"])

# Preprocess the input
sex_encoded = 0 if sex == "male" else 1
features = [[pclass, age, sibsp, parch, fare, sex_encoded]]

# Make a prediction
if st.button("Predict"):
    prediction = model.predict(features)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.write(f"The prediction is: {result}")

