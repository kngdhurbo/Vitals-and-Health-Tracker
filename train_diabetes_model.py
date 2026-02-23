import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load the Dataset
df_diabetes = pd.read_csv('diabetes.csv')
df_diabetes.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

print("Diabetes Dataset loaded successfully.")

# 1. Identify columns where '0' is an impossible value
cols_with_impossible_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# 2. Replace these '0's with NaN (Not a Number) to mark them as missing
for col in cols_with_impossible_zeros:
    df_diabetes[col] = df_diabetes[col].replace(0, np.nan)

# 3. Now, fill these NaN values with the median of each respective column
for col in cols_with_impossible_zeros:
    df_diabetes[col].fillna(df_diabetes[col].median(), inplace=True)

print("Data cleaning complete: '0' values in key columns have been replaced.")

# 4. Prepare Feature Set
features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
target = 'Outcome'

X = df_diabetes[features]
y = df_diabetes[target]

# --- FEATURE SCALING ---
# Scale features to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Split Data for Training and Testing
# We use the scaled data now
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")


# 6. Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2)

print("\nStarting Hyperparameter Tuning for Diabetes Model...")
grid_search.fit(X_train, y_train)

print("\nTuning complete.")
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nIMPROVED Diabetes Model Accuracy: {accuracy * 100:.2f}%")


# 8. Save the Final Model and the Scaler
joblib.dump(best_model, 'diabetes_model.joblib')
joblib.dump(scaler, 'diabetes_scaler.joblib')
print("\nDiabetes model and scaler saved successfully as diabetes_model.joblib and diabetes_scaler.joblib respectively.")