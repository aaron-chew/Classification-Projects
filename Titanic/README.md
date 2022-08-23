# Titanic

This project aims to train an algorithm to accurately predict whether a passenger either survived or not during the Titanic shipwreck in 1912. Inside this repository you'll find various ML and statistical techniques used to enhance and regularize our hypotheses with the sole objective of training a robust algorithm that can secure the lowest lost function for this given problem. 



## Background
On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

## The Dataset
The data is orginally split into two files, train.csv and test.csv which can be located inside the "raw_data" folder. We only import the "test.csv" dataset in our final notebook "6_final_pipeline" to ensure our model's performance is tested on data it has never seen before.

## Project Layout
This project uses 6 notebooks that are divided by each stage/topic of our analysis. To begin, start at notebook "01_basic_exploration" then work your way up sequentially until you reach the 6th and final notebook that deploys the model. The functions and plots files can be found under the "functions" folder which contains all the functions used inside our notebooks. 

## Features
* Survived: Outcome of survival (0 = No; 1 = Yes)
* Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
* Name: Name of passenger
* Sex: Sex of the passenger
* Age: Age of the passenger (Some entries contain NaN)
* SibSp: Number of siblings and spouses of the passenger aboard
* Parch: Number of parents and children of the passenger aboard
* Ticket: Ticket number of the passenger
* Fare: Fare paid by the passenger
* Cabin Cabin number of the passenger (Some entries contain NaN)
* Embarked: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)

## Model Deployment
From the conclusion and findings from our project, we found out that the Logistic Regression was the best performing model that achieved the lowest loss function from our sample of algorithms. You can find the final pipeline with all the necessary transformers and model inside the pickle file "LogisticRegression.pkl". 


## Authors

- [@aaron-chew](https://github.com/aaron-chew)


