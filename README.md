# Quickdraw
Google's doodle detection using OpenCV

Steps:
1. run dataLoad.py -> loads the training data
2. run trainModel.py -> trains the model on our data. The data consists of 15 classes in .npy format. You can download other .npy files for your classes from google's quickDraw dataset. Link: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
3. run quickDrawApp.py -> opens an OpenCV app where you can draw using a blue marker.( I have set the app to detect blue color as a marker)
