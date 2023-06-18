import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Set the path to the UTKFace dataset
dataset_path = "C:/Users/dcati/OneDrive/Documents/age_gender/UTK/UTKFace"

# Load and preprocess the UTKFace dataset
def load_utkface_data(dataset_path):
    images = []
    ages = []
    genders = []

    # Loop through each file in the dataset directory
    for file in os.listdir(dataset_path):
        # Extract the age and gender from the filename
        age, gender, _ = file.split("_")[:3]
        # Load the image and resize it to 64x64 pixels
        image_path = os.path.join(dataset_path, file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))

        # Add the image, age, and gender to their respective lists
        images.append(image)
        ages.append(int(age))
        genders.append(int(gender))

    # Convert the lists to numpy arrays
    images = np.array(images, dtype=np.float32)
    ages = np.array(ages, dtype=np.float32)
    genders = np.array(genders, dtype=np.float32)

    return images, ages, genders

# Load the UTKFace dataset
images, ages, genders = load_utkface_data(dataset_path)

# Normalize the images and split the dataset into training and testing sets
images /= 255.0
X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(images, ages, genders, test_size=0.2, random_state=42)

# Create the CNN model
def create_cnn_model():
    model = Sequential()

    # Add convolutional and pooling layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output and add dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    return model

# Create separate models for predicting age and gender
age_model = create_cnn_model()
gender_model = create_cnn_model()

# Compile the models
age_model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mae'])
gender_model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks to save the best models during training
age_checkpoint = ModelCheckpoint("age_model.h5", monitor='val_mae', save_best_only=True, mode='min')
gender_checkpoint = ModelCheckpoint("gender_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')

# Train the models
age_model.fit(X_train, y_train_age, validation_data=(X_test, y_test_age), epochs=5, batch_size=16, callbacks=[age_checkpoint])
gender_model.fit(X_train, y_train_gender, validation_data=(X_test, y_test_gender), epochs=5, batch_size=16, callbacks=[gender_checkpoint])

# Save the trained models to disk
age_model.save("age_model.h5")
gender_model.save("gender_model.h5")

# Function to predict the age and gender of an image
def predict_age_gender(image_path, age_model, gender_model):
    # Load the image and preprocess it
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the age and gender using the trained models
    age_prediction = age_model.predict(image)
    gender_prediction = gender_model.predict(image)

    # Convert the age prediction to an integer and the gender prediction to a string
    age = int(age_prediction[0][0])
    gender = "Male" if gender_prediction[0][0] > 0.5 else "Female"

    return age, gender


    # Load the trained models from disk
    age_model = tf.keras.models.load_model("age_model.h5")
    gender_model = tf.keras.models.load_model("gender_model.h5")

# Main function to prompt the user for an image path and predict the age and gender
def main():

    while True:
        # Prompt the user for an image path
        image_path = input('Enter the path to the input image file or press q to exit: ').strip()
        if image_path == "q":
            break

        # Read the input image file
        image = cv2.imread(image_path)

        # Predict the age and gender of the image
        try:
            age, gender = predict_age_gender(image_path, age_model, gender_model)
            print(f"Predicted age: {age}")
            print(f"Predicted gender: {gender}")
        except:
            print("Error predicting age and gender. Please try another image.")

if __name__ == "__main__":
    main()
