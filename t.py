import joblib

model = joblib.load("xgboost.joblib")

# Predict on new patient data (a pandas DataFrame)
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[:, 1]

print("Predicted stroke:", prediction[0])
print("Stroke probability:", probability[0])