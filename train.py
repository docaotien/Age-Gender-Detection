import keras
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import load_dataset
from model import create_model

# Load the dataset
images, ages, genders = load_dataset()

# Split the dataset into training and testing sets
X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(images, ages, genders, test_size=0.2, random_state=42)

# Create the model
model = create_model()

# Train the model on the age and gender labels
model.fit(X_train, {'age_output': y_train_age, 'gender_output': y_train_gender}, epochs=10, batch_size=32, validation_data=(X_test, {'age_output': y_test_age, 'gender_output': y_test_gender}))

# Save the trained model to a file
model.save('model.h5')
