import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Give path
dataset_path = "C:/Users/dcati/OneDrive/Documents/age_gender/UTKFace"
# Load and preprocess the UTKFace dataset
def load_utkface_data(dataset_path):
    images = []
    ages = []
    genders = []

    for file in os.listdir(dataset_path):
        age, gender, _ = file.split("_")[:3]
        image_path = os.path.join(dataset_path, file)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))

        images.append(image)
        ages.append(int(age))
        genders.append(int(gender))

    images = np.array(images, dtype=np.float32)
    ages = np.array(ages, dtype=np.float32)
    genders = np.array(genders, dtype=np.float32)

    return images, ages, genders

images, ages, genders = load_utkface_data(dataset_path)

# Normalize the images and split the dataset
images /= 255.0
X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender = train_test_split(images, ages, genders, test_size=0.2, random_state=42)

# Create the CNN model
def create_cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    return model

age_model = create_cnn_model()
gender_model = create_cnn_model()

# Compile the model
age_model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mae'])
gender_model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
age_checkpoint = ModelCheckpoint("age_model.h5", monitor='val_mae', save_best_only=True, mode='min')
gender_checkpoint = ModelCheckpoint("gender_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')

age_history = age_model.fit(X_train, y_train_age, batch_size=32, epochs=20, validation_split=0.2, callbacks=[age_checkpoint])
gender_history = gender_model.fit(X_train, y_train_gender, batch_size=32, epochs=20, validation_split=0.2, callbacks=[gender_checkpoint])

# Evaluate the model
age_model.evaluate(X_test, y_test_age)
gender_model.evaluate(X_test, y_test_gender)

# load trained models
age_model = tf.keras.models.load_model("age_model.h5")
gender_model = tf.keras.models.load_model("gender_model.h5")

# image
def predict_age_gender(image, age_model, gender_model):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    age_prediction = age_model.predict(image)
    gender_prediction = gender_model.predict(image)

    age = int(age_prediction[0][0])
    gender = "M" if gender_prediction[0][0] > 0.5 else "F"

    return age, gender

# webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    age, gender = predict_age_gender(frame, age_model, gender_model)
    cv2.putText(frame, f"Age: {age}, Gender: {gender}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Real-time Age and Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
