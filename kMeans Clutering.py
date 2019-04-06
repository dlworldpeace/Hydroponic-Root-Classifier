import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import gdal as gdal
import glob
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
%matplotlib inline

## loading the dataset, getting the classification
import os
hairy_path = "C:/Users/green/Downloads/CS3244/hairy_root"
hairy_root = {}
for filename in os.listdir(hairy_path):
    hairy_root[filename] = "hairy"
    
non_hairy_path = "C:/Users/green/Downloads/CS3244/non_hairy_root"
non_hairy_root = {}
for filename in os.listdir(non_hairy_path):
    non_hairy_root[filename] = "non hairy"

## read each image and put them into the dataset containing all = roots_ds
hairy_roots = [plt.imread(file) for file in glob.glob(hairy_path+ "/" + '*jpg')]
roots_ds = [plt.imread(file) for file in glob.glob(hairy_path+ "/" + '*jpg')]
non_hairy_roots = [plt.imread(file) for file in glob.glob(non_hairy_path+ "/" + '*jpg')]
roots_ds.extend(non_hairy_roots)

## 0 means hairy, 1 means non_hairy
labels = [0 for i in hairy_root.items()]
nonhairylabels = [1 for j in non_hairy_root.items()]
labels.extend(nonhairylabels)

## to flatten the array of each image, so that we can use as predictors
for i in range(0, len(roots_ds)):
    roots_ds[i] = roots_ds[i].ravel()

## obtain training and test dataset
X = roots_ds
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

## start the kmeans clustering!
##declaring the model
kmeans = KMeans(n_clusters = 2, random_state = 0)
##fitting the model using the training dataset
kmeans.fit(X_train)

## obtain the fitted values and the predictions for test dataset
train_predicted_label = kmeans.predict(X_train)
test_predicted_label = kmeans.predict(X_test)

## obtain the training MSE (need to divide by len(y_train)) = 977
correct = 0
train_MSE = 0
for i in range(0,len(y_train)):
    actual = y_train[i]
    pred = train_predicted_label[i]
    if (actual - pred == 0):
        correct += 1
    else:
        train_MSE += (actual - pred)^2
print(correct) ##number of correct predictions
print(train_MSE/len(y_train)) ##mean squared error
print(sum(y_train)) ##number of non_hairy_roots in the training dataset

## obtain the test MSE (need to divide by len(y_test))
test_correct = 0
test_MSE = 0
for i in range(0,len(y_train)):
    actual = y_train[i]
    pred = train_predicted_label[i]
    if (actual - pred == 0):
        test_correct += 1
    else:
        test_MSE+=(actual - pred)^2

print(test_correct)##number of correct predictions
print(test_MSE/len(y_test))
