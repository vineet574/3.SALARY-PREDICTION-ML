from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib  # Import joblib for saving the model

# Load the updated dataset
data = pd.read_csv('updated_salary_data.csv')

# Check for missing values and handle them
if data.isnull().sum().any():
    data = data.dropna()  # Or you can fill them as necessary

# Define the features and target variable (now including Job Performance Rating)
X = data[['Years of Experience', 'Age', 'Education Level', 'Job Title', 'Gender', 
          'Work Location', 'Company Size', 'Industry', 'Skills', 
          'Certifications', 'Years at Current Company', 'Contract Type', 'Job Performance Rating']]
y = data['Salary']

# Create a pipeline to handle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['Years of Experience', 'Age', 'Years at Current Company', 'Job Performance Rating']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Education Level', 'Job Title', 'Gender', 
                                                         'Work Location', 'Company Size', 'Industry', 
                                                         'Contract Type']),
        ('skills_certs', OneHotEncoder(handle_unknown='ignore'), ['Skills', 'Certifications'])  # Optionally, handle Skills and Certifications
    ])

# Create a pipeline that first transforms the data and then applies Linear Regression
model = Pipeline(steps=[('preprocessor', preprocessor),
                         ('regressor', LinearRegression())])

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'salary_prediction_model.pkl')  # Save your model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = float(request.form['experience'])
    age = float(request.form['age'])
    education = request.form['education']
    job_title = request.form['job_title']
    gender = request.form['gender']
    location = request.form['location']
    company_size = request.form['company_size']
    industry = request.form['industry']

    # Get selected skills from checkboxes
    skills = request.form.getlist('skills')

    # Get custom skills from the "Other Skills" input (handle the case when it's empty)
    other_skills = request.form.get('other_skills', '').split(',')

    # Combine selected skills and other custom skills (ignore empty custom entries)
    all_skills = skills + [skill.strip() for skill in other_skills if skill.strip()]

    # Get selected certifications from checkboxes
    certifications = request.form.getlist('certifications')

    # Get custom certifications from the "Other Certifications" input (handle the case when it's empty)
    other_certifications = request.form.get('other_certifications', '').split(',')

    # Combine selected certifications and other custom certifications (ignore empty custom entries)
    all_certifications = certifications + [cert.strip() for cert in other_certifications if cert.strip()]

    years_at_company = float(request.form['years_at_company'])
    contract_type = request.form['contract_type']
    performance_rating = float(request.form['performance_rating'])  # New input for Job Performance Rating

    # Prepare input for the model, including the combined skills and certifications
    input_data = pd.DataFrame([[experience, age, education, job_title, gender, location, company_size, industry,
                                ', '.join(all_skills), ', '.join(all_certifications), years_at_company, contract_type, performance_rating]], 
                               columns=['Years of Experience', 'Age', 'Education Level', 'Job Title', 'Gender', 
                                        'Work Location', 'Company Size', 'Industry', 
                                        'Skills', 'Certifications', 'Years at Current Company', 'Contract Type', 'Job Performance Rating'])

    # Predict salary
    predicted_salary = model.predict(input_data)[0]

    return render_template('index.html', prediction=round(predicted_salary, 2))

if __name__ == '__main__':
    app.run(debug=True)
