# Salary Prediction Model

This project uses machine learning to predict salaries based on various features such as experience, education, and job role. The project includes data files, scripts for model training and prediction, and a Streamlit web app for interactive predictions.

## Project Overview
The salary prediction model leverages a machine learning algorithm trained on salary data to estimate salaries based on user-provided inputs. The project includes data preprocessing, model training, and deployment in a web-based interface.

## Features
- **Data Preprocessing**: Cleans and processes raw salary data.
- **Model Training**: Trains a machine learning model to predict salaries.
- **Streamlit Web App**: Provides an interface for users to input features and get predictions.

## File Descriptions
- **app.py**: Main script for running the application.
- **Salary Data.csv** and **salary_data.csv**: Datasets used for model training and testing.
- **salary_prediction.py**: Script for training the model and making predictions.
- **salary_prediction_model.pkl**: Saved machine learning model used for predictions.
- **streamlit_app.py**: Streamlit web app for the salary prediction model.
- **templates/**: Folder containing templates for the web application.
- **updated_salary_data.csv**: Updated dataset for further analysis or retraining.

## Installation

### Prerequisites
1. Python 3.6 or higher
2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt


Example requirements.txt:
pandas
numpy
scikit-learn
streamlit


Running the Application
Open the terminal in the project directory.
Run the Streamlit app:
streamlit run app.py

Alternatively, to run the second Streamlit application:
streamlit run streamlit_app.py

Usage
Open the Streamlit web app.
Input relevant features (e.g., years of experience, education level, job role).
Click "Predict" to see the estimated salary based on the input features.
Project Structure
app.py: Main application file.
salary_prediction.py: Model training and prediction script.
salary_prediction_model.pkl: Trained model file.
streamlit_app.py: Additional Streamlit application.
templates/: Folder for HTML templates or assets.
data/: Contains datasets used for training and testing.
Example Usage
User Input:

Experience: 5 years
Education Level: Bachelor's Degree
Job Role: Data Scientist
Model Output: Predicted Salary: $85,000.

