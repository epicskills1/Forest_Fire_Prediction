import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("Fire Prediction - Sheet1.csv")

# Encode the categorical 'Area' column
label_encoder = LabelEncoder()
data['Area'] = label_encoder.fit_transform(data['Area'])

# Split features and target
X = data[['Oxygen', 'Temperature', 'Humidity']]
y = data['Fire Occurence']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Save the model
pickle.dump(log_reg, open('model.pkl', 'wb'))

# Evaluate the model
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
