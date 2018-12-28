from keras.models import load_model
import glob
import tqdm
import numpy as np
import pandas as pd
import cv2

# Create Variables Needed for Application
fpaths = []
results = []
labels = []
# Variable needed to keep consistent output when training the model
uniqu_labels_index = {'Dress_Pant_Images': 0, 'Short_Images': 1, 'T_Shirts_images': 2, 'Dress_Shirt_Images': 3}

# Create List of File Location for Test Images
for image_path in tqdm.tqdm(list(glob.glob(
        'C:/Users/npshe/PycharmProjects/Clothing-Classification/Cloth Images/**/*.*'))):
    fpaths.append(image_path)

# Load Model into Variable
model = load_model('image_model.h5')


# Function to create process image file for the model
def preprocessing_images(path):
    image_list = []
    image_size = 224
    img = cv2.imread(path, flags=1)
    image_resize = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image_list.append(image_resize)

    data = np.vstack(image_list)
    data = data.reshape(-1, 224, 224, 3)
    data_transpose = data.transpose(0, 3, 1, 2)
    return data_transpose

# Initialize the webcam on device
# cam = VideoCapture(0)  # 0 -> index of camera
# s, img = cam.read()
# if s:    # frame captured without any errors
#     namedWindow("cam-test",WINDOW_AUTOSIZE)
#     imshow("cam-test",img)
#     waitKey(0)
#     destroyWindow("cam-test")
#     imwrite('/home/nervster/PycharmProjects/Capstone2_Clothing_Classification/data/filename.jpg',img) #save image
#     cam.release()

# Returning Predicted Labels on Images and the Label from File Path
for fpath in fpaths:
    results.append(model.predict_classes(preprocessing_images(fpath))[0])
    labels.append(fpath.split('\\')[-2])

# Convert Categorical Labels into Numerical to match Model's Output
labels_num = [uniqu_labels_index[label] for i, label in enumerate(labels)]
labels_num = np.array(labels_num)

# Store Data from Model and Actual Labels into Dataframe
dataset = pd.DataFrame({'results': results, 'actual_labels_num': labels_num, 'actual_labels': labels})

# Create Pivot table to compare results
dataset['compare'] = dataset['actual_labels_num'] == dataset['results']
pivot = pd.groupby()
print(pivot)
