import cv2
import numpy as np
import os

# Load the UTKFace dataset
data_dir = 'UTKFace'
images = []
ages = []
genders = []
for filename in os.listdir(data_dir):
    if filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(data_dir, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 200))
        images.append(image)
        age, gender = filename.split('_')[:2]
        ages.append(int(age))
        genders.append(int(gender))

# Convert the data to NumPy arrays
images = np.array(images)
ages = np.array(ages)
genders = np.array(genders)

# Normalize the images
images = images / 255.0

# Convert the gender labels to binary (0 = male, 1 = female)
genders = np.where(genders == 0, 0, 1)
