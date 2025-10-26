import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- 1. Create a mock dataset for demonstration ---
data = {
    'age': np.random.randint(25, 75, 100),
    'blood_pressure': np.random.randint(90, 180, 100),
    'cholesterol': np.random.randint(150, 300, 100),
    'has_disease': np.random.randint(0, 2, 100)
}
df = pd.DataFrame(data)

# --- 2. Separate features (X) and target variable (y) ---
X = df[['age', 'blood_pressure', 'cholesterol']]
y = df['has_disease']

# --- 3. Split the data and train the model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# --- 4. Get user input ---
print("Please enter the patient's data to predict the possibility of disease.")

try:
    age = int(input("Enter patient's age: "))
    blood_pressure = int(input("Enter patient's blood pressure (e.g., 120): "))
    cholesterol = int(input("Enter patient's cholesterol level (e.g., 200): "))

    # Create a DataFrame from the user's input
    user_data = pd.DataFrame({
        'age': [age],
        'blood_pressure': [blood_pressure],
        'cholesterol': [cholesterol]
    })

    # --- 5. Predict the possibility ---
    # The model predicts the probability of each class (0 and 1)
    probabilities = model.predict_proba(user_data)
    
    # The probability of having the disease (class 1) is at index 1
    disease_probability = probabilities[0][1]

    # --- 6. Output the result ---
    print(f"\nBased on the data, the model predicts a {disease_probability:.2%} probability of the patient having the disease.")

    if disease_probability >= 0.5:
        print("This indicates a high likelihood of disease.")
    else:
        print("This indicates a low likelihood of disease.")

except ValueError:
    print("Invalid input. Please enter numerical values for age, blood pressure, and cholesterol.")