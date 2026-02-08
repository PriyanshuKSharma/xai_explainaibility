import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv('breast-cancer.csv')

# Features and labels
X = data.drop(['id', 'diagnosis'], axis=1)  # Features
y = data['diagnosis']  # Labels (M or B)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)  # M=1, B=0

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'random_forest_model.pkl')

# Save the encoder
joblib.dump(encoder, 'label_encoder.pkl')

print("Model trained and saved.")