# Convolutional Neural Network
import os
# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing
# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory(r"C:\Users\ASUS PC\OneDrive\Desktop\dataset huggingface\aiornot\.extras\dataset",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r"C:\Users\ASUS PC\OneDrive\Desktop\dataset huggingface\aiornot\.extras\test dataset",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 35)

cnn.save(r'C:\Users\ASUS PC\OneDrive\Desktop\dataset huggingface\aiornot\.extras\train')


import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import keras.utils as image
import pandas as pd
# Convolutional Neural Network
import os
# Importing the libraries
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

model = tf.keras.models.load_model(r'C:\Users\ASUS PC\OneDrive\Desktop\dataset huggingface\aiornot\.extras\train')


# for img_file in l:
#     img_path = os.path.join(test_dir, img_file)
#     img = image.load_img(img_path, target_size=(64, 64,3))
#     img = image.img_to_array(img)
#     test_image = np.expand_dims(img, axis = 0)
#     test_images.append(img)

# test_images= test_images[:10]

# # # Make predictions
# predictions=[]
# for i in range (1):
#     prediction = model.predict(test_images[i])
    

# print(prediction)

# y_pred= model.predict(test_set)


import numpy as np
import keras.utils as image
res= []
count=0
test_dir = r'C:\Users\ASUS PC\OneDrive\Desktop\dataset huggingface\aiornot\.extras\testset\test'
test_images = []
l= len(os.listdir(test_dir))
for i in range(l):
    path =r'C:/Users/ASUS PC/OneDrive/Desktop/dataset huggingface/aiornot/.extras/testset/test/' + str(i) + '.jpg' 
    test_image = image.load_img( path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image) 
    if int(result[0][0]) == 1:
        count+=1
    res.append(int(result[0][0]))


data = {'id' :os.listdir(test_dir), 'label' : res }
df= pd.DataFrame(data)
df.to_excel(r'Submission.xlsx', index=False)
