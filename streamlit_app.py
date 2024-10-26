import streamlit as st
import pandas as pd
import joblib  # To load the saved model

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')

# Streamlit app title
st.title("Salary Prediction")

# Get user inputs
experience = st.number_input("Enter Years of Experience:", min_value=0)
age = st.number_input("Enter Age:", min_value=0)
education = st.selectbox("Select Education Level:", ["High School", "Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Select Job Title:", ["Software Engineer", "Data Scientist", "Project Manager", "DevOps Engineer", "Business Analyst"])
gender = st.selectbox("Select Gender:", ["Male", "Female"])
location = st.selectbox("Work Location:", ["New York", "San Francisco", "Remote", "Other"])
company_size = st.selectbox("Company Size:", ["Small (<100 employees)", "Medium (100-500 employees)", "Large (>500 employees)"])
industry = st.selectbox("Select Industry:", ["Technology", "Healthcare", "Finance", "Education", "Other"])

# Skills Section
skills = st.multiselect("Select Skills:", ["Python", "Machine Learning", "Data Analysis", "Cloud Computing", "DevOps"])
other_skills = st.text_input("Enter Other Skills (comma-separated):")

# Certifications Section
certifications = st.multiselect("Select Certifications:", ["AWS Certified", "PMP", "Certified Scrum Master"])
other_certifications = st.text_input("Enter Other Certifications (comma-separated):")

years_at_company = st.selectbox("Years at Current Company:", [1, 2, 3, 4, 5])
contract_type = st.selectbox("Select Contract Type:", ["Full-time", "Part-time", "Freelance", "Contract"])
performance_rating = st.number_input("Job Performance Rating (1-5):", min_value=1, max_value=5)

if st.button("Predict Salary"):
    # Combine skills and certifications
    all_skills = skills + [skill.strip() for skill in other_skills.split(',') if skill.strip()]
    all_certifications = certifications + [cert.strip() for cert in other_certifications.split(',') if cert.strip()]

    # Prepare data for prediction
    input_data = pd.DataFrame([[experience, age, education, job_title, gender, location, company_size, industry,
                                ', '.join(all_skills), ', '.join(all_certifications), years_at_company, contract_type, performance_rating]], 
                               columns=['Years of Experience', 'Age', 'Education Level', 'Job Title', 'Gender', 
                                        'Work Location', 'Company Size', 'Industry', 
                                        'Skills', 'Certifications', 'Years at Current Company', 'Contract Type', 'Job Performance Rating'])

    # Make prediction
    predicted_salary = model.predict(input_data)[0]
    st.success(f"Predicted Salary: ${round(predicted_salary, 2)}")

