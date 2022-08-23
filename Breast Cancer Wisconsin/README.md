
# Breast Cancer Wisconsin

The purpose of this project is to train an algorithm to accurately and effectively predict whether a tumor is either benign (non-cancerous) or malignant (cancerous).

## Background
Breast cancer is the most common malignancy among women, accounting for nearly 1 in 3 cancers diagnosed among women in the United States, and it is the second leading cause of cancer death among women. Breast Cancer occurs as a results of abnormal growth of cells in the breast tissue, commonly referred to as a Tumor. This is an analysis of the Breast Cancer Wisconsin (Diagnostic) DataSet, obtained from Kaggle. This data set was created by Dr. William H. Wolberg, physician at the University Of Wisconsin Hospital at Madison, Wisconsin,USA

## The Dataset
The Breast Cancer Wisconsin dataset is a multivariate classification problem which contains 570 rows of data with 32 features that are all numeric. Our dependent variable (the attribute we are trying to predict) is "diagnosis", which determines whether a tumor is cancerous or not. 

## Project Layout
This project uses 6 notebooks that are divided by each stage/topic of our analysis. To begin, start at notebook "01_basic_exploration" then work your way up sequentially until you reach the 6th and final notebook that deploys the model. The functions and plots .py files can be found under the "functions" folder.

## Features
a) radius (mean of distances from center to points on the perimeter)<br>
b) texture (standard deviation of gray-scale values)<br>
c) perimeter<br>
d) area<br>
e) smoothness (local variation in radius lengths)<br>
f) compactness (perimeter^2 / area - 1.0)<br>
g) concavity (severity of concave portions of the contour)<br>
h) concave points (number of concave portions of the contour)<br>
i) symmetry<br>
j) fractal dimension ("coastline approximation" - 1)<br>

## Model Deployment 
From the conclusion of our findings the model that scored the best weighted-recall score was the XGB Classifier which had no variance across 30 prediction iterations. You can find the model deployment file inside the repository with the file name "XGBClassifier.pkl"


## Authors

- [@aaron-chew](https://github.com/aaron-chew)


