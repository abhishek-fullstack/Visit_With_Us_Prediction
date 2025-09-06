import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="aks2022/Visit-With-Us-Prediction-Model", filename="best_visitwithus_prediction_model_v1.joblib")
model = joblib.load(model_path)

#Download and load label encoders
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
label_encoders = {}
for column in categorical_features:
    encoder_path = hf_hub_download(repo_id="aks2022/Visit-With-Us-Prediction-Model", filename=f"{column}_encoder.joblib")
    label_encoders[column] = joblib.load(encoder_path)

# Streamlit UI for Machine Failure Prediction
st.title("Visit With Us Prediction App")
st.write("""
This application predicts the likelihood of a customer choosing to travel with us.
Please enter the customer details below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.number_input("Preferred Property Star Rating", min_value=3.0, max_value=5.0, value=3.0, step=0.5)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number of Trips", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=0.0, step=0.5)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, max_value=100000.0, value=20000.0, step=100.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])
# Apply label encoding to categorical features
for column in categorical_features:
    input_data[column] = label_encoders[column].transform(input_data[column])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Customer will choose to travel with us" if prediction == 1 else "Customer will not choose to travel with us"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
