import cv2
import os
import glob
import numpy as np
import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

# Create Global Lists variables to be used throughout model
fpaths = []
labels = []
image_list = []
image_size = 224

# Retrieve and store Images' file paths
for image_path in tqdm.tqdm(list(
        glob.glob('C:/Users/npshe/PycharmProjects/Clothing-Classification/Cloth Images/**/*.*'))):
    fpaths.append(image_path)

# Read, resize and retrieve labels for each image
for fpath in fpaths:
    img = cv2.imread(fpath, flags=1)
    image_resize = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image_list.append(image_resize)
    labels.append(fpath.split('\\')[-2])

# Store image data into a list and transform array to fix model
data = np.vstack(image_list)
data = data.reshape(-1, 224, 224, 3)
data = data.transpose(0, 3, 1, 2)

# Turn Labels into Categorical Labels
tmp_labels = labels
uniq_labels = set(tmp_labels)  # eliminate duplication
num_cloths = len(Counter(labels))  # number of cloths
# create dictionary and assign number for each labels
uniqu_labels_index = dict((label, i) for i, label in enumerate(uniq_labels))

labels_num = [uniqu_labels_index[label] for i, label in enumerate(labels)]
labels_num = np.array(labels_num)


# Split Data into Test and Train
N = len(image_list)
N_train = int(N * 0.7)
N_test = int(N * 0.2)

X_train, X_tmp, Y_train, Y_tmp = train_test_split(data, labels_num, train_size=N_train)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_tmp, Y_tmp, test_size=N_test)

# Important Output for Apply_Model Script
print(uniqu_labels_index)

# Create the Deep Learning Model
model = Sequential()
K.set_image_dim_ordering('th')

# Load the VGG model
vgg_conv = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(3, 224, 224), pooling=None)

# Image Augmentation Function
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

model.add(vgg_conv)
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(num_cloths))
model.add(Activation('softmax'))

opt = RMSprop(lr=.0001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=8),epochs=200, steps_per_epoch=5,
                              validation_data=(X_validation, Y_validation), workers=1)

score = model.evaluate(X_test, Y_test, batch_size=16)

# Output graphs of Accuracy and Loss over Epochs
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
print(model.metrics_names)
print(score)
model.summary()

# Saved Model into file
model.save('image_model.h5')
