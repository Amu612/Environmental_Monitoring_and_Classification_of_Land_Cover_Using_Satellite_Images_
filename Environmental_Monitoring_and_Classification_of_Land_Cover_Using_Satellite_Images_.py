#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import requests
import tempfile
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[ ]:


# prompt: /content/dataset.zip extract

get_ipython().system('unzip /content/dataset.zip')


# In[ ]:


# Create an empty dataframe
data = pd.DataFrame(columns=['image_path', 'label'])

# Define the labels/classes

labels = {
    r"/content/dataset/Satellite Image data/cloudy": "Cloudy",
    r"/content/dataset/Satellite Image data/desert": "Desert",
    r"/content/dataset/Satellite Image data/green_area": "Green_Area",
    r"/content/dataset/Satellite Image data/water": "Water",
}

# Validate folder paths
for folder in labels:
    if not os.path.exists(folder):
        print(f"Warning: The folder {folder} does not exist.")
        continue

    # Process each image in the folder
    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)
        if os.path.isfile(image_path):  # Only process files
            label = labels[folder]
            data = pd.concat([data, pd.DataFrame({'image_path': [image_path], 'label': [label]})], ignore_index=True)

# Display the resulting DataFrame
print(data)


# In[ ]:


# prompt: labels = {
#     r"/content/dataset/Satellite Image data/cloudy": "Cloudy",
#     r"/content/dataset/Satellite Image data/desert": "Desert",
#     r"/content/dataset/Satellite Image data/green_area": "Green_Area",
#     r"/content/dataset/Satellite Image data/water": "Water",
# } show sample 10 10 miages of all folder

fig, axes = plt.subplots(len(labels), 10, figsize=(15, len(labels) * 1.5))

for i, (folder, label_name) in enumerate(labels.items()):
    image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    sample_images = np.random.choice(image_files, min(10, len(image_files)), replace=False)

    for j, image_file in enumerate(sample_images):
        img_path = os.path.join(folder, image_file)
        try:
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].set_title(label_name, fontsize=8)
            axes[i, j].axis('off')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            axes[i, j].axis('off') # Turn off axis even if there's an error

for i in range(len(labels)):
    for j in range(len(sample_images), 10):
        axes[i, j].axis('off') # Turn off remaining axes if less than 10 images

plt.tight_layout()
plt.show()


# In[ ]:


data.to_csv('image_dataset.csv', index=False)


# In[ ]:


df = pd.read_csv("image_dataset.csv")


# In[ ]:


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# In[ ]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=45,
                                   vertical_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    x_col="image_path",
                                                    y_col="label",
                                                    target_size=(255, 255),
                                                    batch_size=32,
                                                    class_mode="categorical")

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                  x_col="image_path",
                                                  y_col="label",
                                                  target_size=(255, 255),
                                                  batch_size=32,
                                                  class_mode="categorical")


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(train_generator, epochs=25, validation_data=test_generator)


# In[ ]:





# In[ ]:


num_samples = test_df.shape[0]
score = model.evaluate(test_generator, steps=num_samples//32+1)


# In[ ]:


model.save('Modelenv.v1.h5')


# In[ ]:


model = load_model("Modelenv.v1.h5")


# In[ ]:


class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']


# In[ ]:


def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


predictions = model.predict(test_generator)
actual_labels = test_generator.classes
predicted_labels = np.argmax(predictions, axis=1)


# In[ ]:


cm = confusion_matrix(actual_labels, predicted_labels)


# In[ ]:


plt.figure(figsize=(8, 6))
plot_confusion_matrix(cm, classes=class_names)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




