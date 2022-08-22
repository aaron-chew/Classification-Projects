import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split, StratifiedKFold, cross_val_predict 
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score
import seaborn as sns
from pickle import dump
from pickle import load
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import plotly.express as px
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


 # ============================== 3_Feature_Engineering ============================== #


def hypothesis_testing(h0, benchmark_variable, new_benchmark_name):
    # Stores the key inside a variable. 
    old_dict_key = list(benchmark_variable.keys())[0]
    
    # If our h0 scores a better error metric than h1: 
    if h0 > benchmark_variable[list(benchmark_variable.keys())[0]]:
        # Deletes the old key, and replaces it with a new key name, specified by the user.
        benchmark_variable[new_benchmark_name] = benchmark_variable.pop(old_dict_key)
        # The new h0 value is assigned to the new key name. 
        benchmark_variable[new_benchmark_name] = h0
        print("We reject the null hypothesis with the new benchmark for %s: %.4f%%" % (list(benchmark_variable.keys())[0],h0))
    else:
        print("We accept the null hypothesis.")
        
class model_evaluation:
    def __init__(self):
        # Placeholder for parameter grid for grid search. 
        self.params = dict()
        
    def preprocessing(self, data):
        self.numerical = list(data.select_dtypes(['float64']).columns)
        self.categorical = list(data.select_dtypes(exclude=['float64']).columns)
        self.categorical.remove("Survived") # Remove label from data.
        
        self.cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

        # Define numerical pipeline
        self.num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))])

        # Combine categorical and numerical pipelines
        self.preprocessor = ColumnTransformer([
            ('cat', self.cat_pipe, self.categorical),
            ('num', self.num_pipe, self.numerical)])
        
        self.estimators = [('preprocessor', self.preprocessor)]

    def add_pipe_component(self, string, instance):
        # Function to add pipeline steps. 
        self.estimators.append((string, instance))
        
    def cross_validation(self, data):
        # Splitting the data into features and labels. 
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1] 
        
        # Defining estimator with pipeline applied steps.  
        self.pipe = Pipeline(steps=self.estimators)
        self.pipe.fit(self.X, self.y)
        # Setting up cross validation strategy. 
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        # Evaluate results. 
        self.results = cross_val_score(self.pipe, self.X, self.y, cv=self.cv, scoring='f1_weighted')
        # Average out the cv array and display absloute values to remove the "neg". 
        self.cv_result = self.results.mean()*100 
            
    def RFE_cross_validate(self, data, model):
        self.X = data.iloc[:,:-1]
        self.y = data.iloc[:,-1] 
        
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.rfe = RFECV(estimator=model, cv=self.cv, scoring='f1_weighted', n_jobs=-1) 
        self.rfe_result = self.rfe.fit(self.X, self.y)
                
    def grid_search(self, X, y, model): 
        # Setting up cross validation strategy. 
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        # Evaluate results. 
        self.grid = GridSearchCV(estimator=model, param_grid=self.params, cv=self.cv, scoring='f1_weighted', n_jobs=-1)
        self.grid_results = self.grid.fit(X, y)
        
        print('Best: %f using %s' % (self.grid_results.best_score_, self.grid_results.best_params_))
        self.means = self.grid_results.cv_results_['mean_test_score'] 
        self.params = self.grid_results.cv_results_['params']

        #for mean, param in zip(self.means, self.params):
            #print("%f with: %r" % (mean,param))
            
    def add_params_component(self, key, value):
        # Function to add pipeline steps. 
        self.params[key] = value
    
    def overfitting_checker(self, X, y, model):
        # Cross validation strategy, with 5 number of splits. 
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        # We set "return_train_score" to True to see for overfitting. 
        self.results = cross_validate(model, X, y, cv=self.cv, scoring='f1_weighted', return_train_score=True)  
        print('Model scored an Train_F1_Score of: %.2f and a Validation_F1_Score of: %.2f' % 
              (self.results["train_score"].mean(), self.results["test_score"].mean()))
        
def optimal_components(model, dictResults, df):
    for num in range(1, 51):
        pca = model_evaluation()
        pca.preprocessing(df)
        pca.add_pipe_component("pca", PCA(n_components=num))
        pca.add_pipe_component("clf", model)
        pca.cross_validation(df)

        dictResults[str(num)] = pca.cv_result

        
 # ============================== 4_Optimization ============================== #


def grid_search_plot(instance, model_string):
    f1_scores = instance.grid_results.cv_results_['mean_test_score']
    iterations2 = list()
    [iterations2.append(i) for i in range(1,len(f1_scores)+1)];

    # Plot early stopping results. 
    fig = px.line(x=iterations2, y=f1_scores)
    # Best loss score. 
    fig.add_vline(x=list(f1_scores).index(max(f1_scores))+1, line_width=2, line_dash="dash", line_color="black")  
    fig.update_layout(title='Hypertuning the {}'.format(model_string),
                           xaxis_title='No. of Iterations',
                           yaxis_title='f1_score', height=455, width=900)
    fig.show()

    
 # ============================== 5_Regularization ============================== #    
    
    
class Early_Stopping:
    def __init__(self, X, y, model):
        # Split the data into train and validation sets. 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # Create our model instance with optimal hyperparameters. 
        self.eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        self.clf = model
        self.clf.fit(self.X_train, self.y_train, eval_set=self.eval_set, verbose=False)
        # Make predictions for test data.
        self.y_pred = self.clf.predict(self.X_test)

        # Evaluate predictions.
        self.f1 = f1_score(self.y_test, self.y_pred)

        # Retrieve performance metrics.
        self.results = self.clf.evals_result() # Log loss scores. 
        self.epochs = len(self.results['validation_0']['logloss']) # no. of training epochs/no. of trees. 
        self.x_axis = range(0, self.epochs) # x-axis (no. of trees).
        
def LogLoss_Curve(x_axis, y_axis_train, y_axis_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_train,
                        name='Train'))
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_test,
                        name='Test'))

    fig.update_layout(title='XGB Classifier Log Loss Curve',
                       xaxis_title='Epochs',
                       yaxis_title='Log Loss',
                       height=455, width=900)
    fig.show()
      
# A function to convert probability to a class value, based on a threshold.  
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def threshold_manipulation(X, y, model, model_string):
    # Defining our cross validation strategy. 
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # Fitting our data to our model. 
    model.fit(X, y)
    
    # Conducting predictions using proba. 
    yhat = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
    # Extracting probability from results. 
    probs = yhat[:, 1]
    
    # Threshold values we want to test. 
    thresholds = np.arange(0, 1, 0.001)
    # Evaluate each threshold, using the function "to_labels". 
    scores = [f1_score(y, to_labels(probs, t), average="weighted") for t in thresholds] 
    # IMPORTANT: Remember to set average="weighted", because when we use cross_val_score we use "f1_weighted". Otherwise we'll
    # get different results. A "weighted_f1_score" is preferred when using an imbalanced dataset. 
    
    # Get best threshold. 
    ix = np.argmax(scores)
    print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    
    # Plot thresholds curve. 
    fig = px.line(x=thresholds, y=scores)
    fig.add_vline(x=thresholds[ix], line_width=2, line_dash="dash", line_color="black") # Best loss score. 
    fig.update_layout(title='Threshold Manipulation for '+model_string,
                           xaxis_title='Thresholds',
                           yaxis_title='f1_score', height=455, width=900)
    fig.show()

    
 # ============================== 6_Final_Pipeline ============================== #


def create_age_group(df):
    # We use the efficient numpy vecotorization technqiue for efficient array computation. 
    # Setting our conditions.
    conditions = [
        (df["Age"] < 18), 
        ((df["Age"] > 17) & (df["Age"] < 36)),
        ((df["Age"] > 35) & (df["Age"] < 56)),
        (df["Age"] > 55) 
    ]

    # Values if true. 
    ifTrue = [
        "child",
        "young adult",
        "middle aged",
        "senior"
    ]

    df["Age Group"] = np.select(conditions,ifTrue, "replace_for_nan") # If error occurs, set value as "replace_for_nan".
    # Later down the script we will replace "replace_for_nan" with actual NaN values. We'll leave it as a placeholder for now.
    df = df.replace("replace_for_nan", np.nan)
    return df 

def create_marriage_feature(df):
    df["Last Name"] = df["Name"].map(lambda x : x.split(",")[0]) # Extracting the last name from passengers. 
    df["Name Placeholder"] = df["Name"].map(lambda x : x.split(",")[1]) 
    df["Title"] = df["Name Placeholder"].str.extract(r"([a-zA-Z]+.)") # We want to extract passenger title from "Name_Placeholder".
    marriedFemales = df[df["Title"]=="Mrs."] # Creating married women table.  
    males = df[df["Title"]=="Mr."] # Creating males with the title 'Mr.' table.
    femaleLastNames = list(marriedFemales["Last Name"]) # Store all female last names into a list. 
    # Only keep male last names, that match with any value inside our femaleLastNames list.
    filter1 = males["Last Name"].isin(femaleLastNames) 
    marriedMales = males[filter1] # Apply filter to create our Married Males table. 
    marriedMales = marriedMales[marriedMales["Age Group"]!= "child"] # Dropping children from the rows.

    married = pd.concat([marriedMales, marriedFemales]) # Creating the total married
    marriedIDs = list(married.index) # Storing all married passenger's ID's into a list.

    # Creating our "Married" boolean feature. (0=single, 1=married.) 
    conditions = [df.index.isin(marriedIDs)] # Setting our conditions.
    ifTrue = ["Married"] # Values if true. 
    df["Martial Status"] = np.select(conditions,ifTrue, "Not Married")

    df = df.drop(columns=(["Name Placeholder", "Title", "Last Name"])) # Removing unnecessary columns. 
    return df

def holdout_set_evaluation(y_axis, train_data, holdout_features, holdout_labels, model, threshold, model_string):
    numerical = list(train_data.select_dtypes(['float64']).columns)
    categorical = list(train_data.select_dtypes(exclude=['float64']).columns)
    categorical.remove("Survived") # Remove label from data.
    
    for i in range(1, 31):
        cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

        # Define numerical pipeline
        num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))])

        # Combine categorical and numerical pipelines
        preprocessor = ColumnTransformer([
            ('cat', cat_pipe, categorical),
            ('num', num_pipe, numerical)])

        estimators = [('preprocessor', preprocessor), ("clf", model)]

        # Split the train dataset.
        trainX = train_data.iloc[:,:-1]
        trainy = train_data.iloc[:,-1]

        pipe = Pipeline(steps=estimators)
        pipe = pipe.fit(trainX, trainy) 

        yhat = pipe.predict_proba(holdout_features)
        # Keep probabilities for the positive outcome only
        probs = yhat[:, 1]
        classes = to_labels(probs, threshold)

        score = f1_score(holdout_labels, classes,average="weighted")*100
        y_axis.append(score)
    
def saving_model(train_data, model):
    numerical = list(train_data.select_dtypes(['float64']).columns)
    categorical = list(train_data.select_dtypes(exclude=['float64']).columns)
    categorical.remove("Survived") # Remove label from data.

    cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])

    # Define numerical pipeline
    num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))])

    # Combine categorical and numerical pipelines
    preprocessor = ColumnTransformer([
        ('cat', cat_pipe, categorical),
        ('num', num_pipe, numerical)])

    estimators = [('preprocessor', preprocessor), ("clf", model)]
    
    # Split the train dataset.
    trainX = train_data.iloc[:,:-1]
    trainy = train_data.iloc[:,-1]

    pipe = Pipeline(steps=estimators)
    pipe = pipe.fit(trainX, trainy) 
    
    dump(pipe, open("LogisticRegression.pkl", "wb"))
    
def f1_score_plot(LR_f1_scores, GBC_f1_scores, XGB_f1_scores):
    fig = make_subplots(rows=3, cols=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=LR_f1_scores,
        name="Logistic Regression"
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=GBC_f1_scores,
        name="Gradient Boosting Classifier"
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=XGB_f1_scores,
        name="XGB Classifier",
    ), row=3, col=1)


    fig.update_layout(height=550, width=900, title_text="Model Performance over 30 Iterations")

    fig['layout']['yaxis']['title']='f1_score'
    fig['layout']['xaxis3']['title']='Iterations'

    fig.show()
    
def ROC_AUC_curve(test_labels, predictions):
    fpr, tpr, thresholds = roc_curve(test_labels, predictions)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()


    
 
