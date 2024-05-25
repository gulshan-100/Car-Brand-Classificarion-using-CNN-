import numpy as np 
import pandas as pd 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator

# Preprocessing the Dataset 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r"C:\Users\DELL\Downloads\archive (36)\Cars Dataset\train",
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'categorical')

testing_set = test_datagen.flow_from_directory(r"C:\Users\DELL\Downloads\archive (36)\Cars Dataset\test",
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'categorical')


# Build the CNN model
cnn = tf.keras.models.Sequential()

# First convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Third convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening layer
cnn.add(tf.keras.layers.Flatten())

# Fully connected layer with Dropout
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5))

# Output layer
cnn.add(tf.keras.layers.Dense(units=7, activation='softmax'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#Building the CNN Model 
history = cnn.fit(x = training_set, validation_data = testing_set, epochs = 20)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# In[22]:


# Input an Image to check if the model is working fine or not

import numpy as np 
from keras.preprocessing import image 

img = image.load_img('Cars Dataset/test/Audi/1022.jpg', target_size = (64,64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
# Predict the class
result = cnn.predict(img)

# Define class labels (ensure these match the order used during training)
class_labels = ['AUDI', 'HYUNDAI CRETA', 'MAHINDRA SCORPIO', 'ROLLS ROYCE', 'SWIFT', 'TATA SAFARI', 'TOYOTA INNOVA']

# Find the predicted class index
predicted_class_index = np.argmax(result)

# Get the predicted class label
prediction = class_labels[predicted_class_index]

# Print the prediction
print(f'Predicted class: {prediction}')




# In[ ]:





# In[ ]:




