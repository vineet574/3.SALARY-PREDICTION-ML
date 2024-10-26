import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('salary_data.csv')

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Describe the dataset for statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Prepare the data for training
X = data[['YearsExperience']]  # Feature
y = data['Salary']              # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Training Complete!")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Make a prediction for a new input
new_experience = pd.DataFrame([[3.5]], columns=['YearsExperience'])  # Example: Years of experience
predicted_salary = model.predict(new_experience)
print("\nPredicted Salary for 3.5 years of experience:", predicted_salary[0])
