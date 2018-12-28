# Clothing_Image_Classification

This repository is to create a model to start identifying clothing items. My hope is that this model can be used to create an "economically feasible" (aka cheaper) automated folding machine. Current automated folding machines are tedious and expensive. I hate folding clothes but I not willing to cough up $700+ Other applications can be a clothing inventory like at Goodwill or your closet.

## Getting Started

1. Image_Classification_Model.py: Main model which will output a H5 file (image_model.h5)
2. Model_Test.py: Applying the model to a set of test clothing pictures and output a pivot to understand performance of model
3. Cloth Images.zip: About 550 Images separated by clothing type (folder names will be used for labelling)

### Prerequisites

What things you need to install the software and how to install them

1. Python 3.6
2. Keras 2.2.4
3. OpenCV 3.3.1
4. Windows 10 (can be applied on Linux but Matplotlib was not working for me on Linux)

### Installing

1. Clone Repository and unzip clothing imaging files
2. Rename Clothing Image Source within both python scripts to match your image paths

Simple installation process. I also used opencv2 to use my laptop's webcam in addition to images stored on computer to test model. 

### Approach

1. Images Collection: Only images of cloths (without any human inside the cloths). Important because my model is supposed to be used post dryer step.
2. Model: Transfer learning on VGG16 model and imagenet weights. Then I applied my own layers to consolidate model to detect my labels.
3. Data Augmentation: Based on Step 5 of referenced url, I needed this step because the application would deal with unfolded cloths. Therefore, I needed my model to detect the cloths in a variety of formats. Source: https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
4. Application: Loaded the model to test on separate images.

### Results

![](MiscFiles/Capture.PNG)

## Next Steps

1. Continue adding to images
2. Expand labels to recognize fabrics and additional types of clothing
3. Research on misclassified images to increase accuracy
4. Implement using YOLO so I can create real time image classification

## License

This project is licensed by nobody :) 
