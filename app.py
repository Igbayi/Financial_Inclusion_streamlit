import streamlit as st
import pickle
import pandas as pd
import sklearn

# Load the trained model
model = pickle.load(open("Financial Inclusion GradBoostmodel.pkl",'rb'))

# App title
st.title("Financial Inclusion")
st.write("Predict which individuals are most likely to have or use a bank account based on their demographics.")

# Sidebar for user input
st.sidebar.header("Demographic Information")

# Input fields
country = st.sidebar.number_input(
    "Country of the interviewee (0 - 3): Kenya - 0, Rwanda - 1, Tanzania - 2, Uganda - 3", 
    min_value=0, max_value=3, value=0
)
year = st.sidebar.number_input(
    "Year of survey (0 - 2): 2016 - 0, 2017 - 1, 2018 - 2",
    min_value=0, max_value=2, value=0
)
location_type = st.sidebar.number_input(
    "Type of location: Rural - 0, Urban - 1",
    min_value=0, max_value=1, value=0
)
cellphone_access = st.sidebar.number_input(
    "interviewee cellphone_access: No - 0, Yes - 1",
    min_value=0, max_value=1, value=0
)
household_size = st.sidebar.number_input("Number of people living in one house", value=0.0)
age_of_respondent = st.sidebar.number_input("The age of the interviewee", value=0.0)
gender_of_respondent = st.sidebar.number_input(
    "Gender: Female - 0, Male - 1",
    min_value=0, max_value=1, value=0
)

relationship_with_head = st.sidebar.number_input(
    "Interviewee's relationship with the head of the house (0 - 5): Child - 0, Head of Household - 1, Other non-relatives - 2, Other relative - 3, Parent - 4, Spouse - 5",
    min_value=0, max_value=5, value=0
)
marital_status = st.sidebar.number_input(
    "Marital status (0 - 4): Divorced/Seperated - 0, Dont know - 1, Married/Living together - 2, Single/Never Married - 3, Widowed - 4",
    min_value=0, max_value=4, value=0
)
education_level = st.sidebar.number_input(
    "Highest level of education (0 - 5): No formal education - 0, Other/Dont know/RTA - 1, Primary education - 2, Secondary education - 3, Tertiary education - 4, Vocational/Specialised training - 5",
    min_value=0, max_value=5, value=0
)
job_type = st.sidebar.number_input(
    "Interviewee's job type (0 - 9): Dont Know/Refuse to answer - 0, Farming and Fishing - 1, Formally employed Government - 2, Formally employed Private - 3, Government Dependent - 4, Informally employed - 5, No Income - 6, Other Income - 7, Remittance Dependent - 8, Self employed -  9",
    min_value=0, max_value=9, value=0
)

# Prepare input data for prediction
input_data = pd.DataFrame({
    "country": [country],
    "year": [year],
    "location_type": [location_type],
    "cellphone_access": [cellphone_access],
    "household_size": [household_size],
    "age_of_respondent": [age_of_respondent],
    "gender_of_respondent": [gender_of_respondent],
    "relationship_with_head": [relationship_with_head],
    "marital_status": [marital_status],
    "education_level": [education_level],
    "job_type": [job_type],
})

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"This individuals is most likely to have a bank account with a probability of {prediction_proba[0][1]:.2f}.")
    else:
        st.success(f"This individuals is unlikely to have a bank account with a probability of {prediction_proba[0][0]:.2f}.")
