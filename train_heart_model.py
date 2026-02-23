import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df_heart = pd.read_csv('heart_disease.csv')

# 1. Handle Target Column
df_heart['target_binary'] = (df_heart['num'] > 0).astype(int)

# 2. Handle Categorical & Special Value Columns Individually
# Sex
df_heart['sex'] = df_heart['sex'].apply(lambda x: 1 if x == 'Male' else 0)

# CA (Number of major vessels)
df_heart['ca'] = df_heart['ca'].replace('?', df_heart['ca'].mode()[0])
df_heart['ca'] = pd.to_numeric(df_heart['ca'])

# THAL
df_heart['thal'] = df_heart['thal'].replace('?', df_heart['thal'].mode()[0])
thal_dummies = pd.get_dummies(df_heart['thal'], prefix='thal')
df_heart = pd.concat([df_heart, thal_dummies], axis=1)

# CP (Chest Pain Type)
cp_dummies = pd.get_dummies(df_heart['cp'], prefix='cp')
df_heart = pd.concat([df_heart, cp_dummies], axis=1)

# FBS (Fasting Blood Sugar) & EXANG (Exercise Induced Angina)

df_heart['fbs'].fillna(df_heart['fbs'].mode()[0], inplace=True)
df_heart['exang'].fillna(df_heart['exang'].mode()[0], inplace=True)
df_heart['fbs'] = df_heart['fbs'].astype(int)
df_heart['exang'] = df_heart['exang'].astype(int)


# 3. Assemble the Final Feature List
base_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'ca']
cp_feature_names = list(cp_dummies.columns)
thal_feature_names = list(thal_dummies.columns)
features = base_features + cp_feature_names + thal_feature_names

print("--- Using the following fully-numeric features for training ---")
print(features)

target = 'target_binary'
X = df_heart[features]
y = df_heart[target]

# 4. Final check for any remaining missing values in other numeric columns
medians = X.median()
X = X.fillna(medians)

# --- MODEL TRAINING ---

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2)

print("\nStarting Hyperparameter Tuning with GridSearchCV...")
grid_search.fit(X_train, y_train)

print("\nTuning complete.")
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFINAL Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(best_model, 'heart_model.joblib')
print("\nHeart Disease model saved successfully as 'heart_model.joblib'")