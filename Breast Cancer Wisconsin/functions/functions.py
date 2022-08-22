# Importing standard libraries. 
import pandas as pd
import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Import transformers.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, OneHotEncoder, FunctionTransformer,  LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Import validation.
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, recall_score, roc_curve, auc, confusion_matrix

# Import feature selection.
from sklearn.feature_selection import RFECV

# Importing pickle library. 
from pickle import dump
from pickle import load


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
        self.estimators = [('si', SimpleImputer())]

    def add_pipe_component(self, string, instance):
        # Function to add pipeline steps. 
        self.estimators.append((string, instance))
        
    def cross_validation(self, X, y):
        # Defining estimator with pipeline applied steps.  
        self.pipe = Pipeline(steps=self.estimators)
        self.pipe.fit(X, y)
        # Setting up cross validation strategy. 
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        # Evaluate results. 
        self.results = cross_val_score(self.pipe, X, y, cv=self.cv, scoring='recall_weighted', n_jobs=-1)
        # Average out the cv array.
        self.cv_result = self.results.mean()*100 
            
    def RFE_cross_validate(self, X, y, model): 
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.rfe = RFECV(estimator=model, cv=self.cv, scoring='recall_weighted', n_jobs=-1) 
        self.rfe_result = self.rfe.fit(X, y)
        
    def grid_search(self, X, y, model): 
        # Setting up cross validation strategy. 
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        # Evaluate results. 
        self.grid = GridSearchCV(estimator=model, param_grid=self.params, cv=self.cv, scoring='recall_weighted', n_jobs=-1)
        self.grid_results = self.grid.fit(X, y)
        
        print('Best: %f using %s' % (self.grid_results.best_score_, self.grid_results.best_params_))
        self.means = self.grid_results.cv_results_['mean_test_score'] 
        self.params = self.grid_results.cv_results_['params']
            
    def add_params_component(self, key, value):
        # Function to add pipeline steps. 
        self.params[key] = value
        
    def overfitting_checker(self, X, y, model):
        # Cross validation strategy, with 5 number of splits. 
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        # We set "return_train_score" to True to see for overfitting. 
        self.results = cross_validate(model, X, y, cv=self.cv, scoring='recall_weighted', return_train_score=True)  
        print('Model scored an Train_Recall_Score of: %.2f%% and a Validation_Recall_Score of: %.2f%%' % 
              (self.results["train_score"].mean()*100, self.results["test_score"].mean()*100))
        
    def overfitting_checker_no_print(self, X, y, model):
        # Cross validation strategy, with 5 number of splits. 
        self.cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        # We set "return_train_score" to True to see for overfitting. 
        self.results = cross_validate(model, X, y, cv=self.cv, scoring='recall_weighted', return_train_score=True) 
        
    def holdout_set_evaluation(self, y_axis, train_features, train_labels, holdout_features, holdout_labels, threshold):
        for i in range(1, 31):
            self.pipe = Pipeline(steps=self.estimators) # Creating pipeline with transformer steps. 
            self.pipe = self.pipe.fit(train_features, train_labels) # Training our estimator. 

            self.yhat = self.pipe.predict_proba(holdout_features) # Predicting our holdoutset probabilities. 
            self.probs = self.yhat[:, 1] # Defining probabilities.  
            self.classes = to_labels(self.probs, threshold) # Classifying our classes. 
            self.score = recall_score(holdout_labels, self.classes,average="weighted")*100 # Calculating recall score. 
            y_axis.append(self.score) # Storing results in list. 
        

 # ============================== 4_Optimization ============================== #


def grid_search_plot(instance, model_string):
    recall = instance.grid_results.cv_results_['mean_test_score']*100
    iterations2 = list()
    [iterations2.append(i) for i in range(1,len(recall)+1)];

    # Plot early stopping results. 
    fig = px.line(x=iterations2, y=recall)
    # Best loss score. 
    fig.add_vline(x=list(recall).index(max(recall))+1, line_width=2, line_dash="dash", line_color="black")  
    fig.update_layout(title='Hypertuning the {}'.format(model_string),
                           xaxis_title='No. of Iterations',
                           yaxis_title='Weighted Recall', height=455, width=900)
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
        self.mae = mean_absolute_error(self.y_test, self.y_pred)

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

    fig.update_layout(title='XGB Classifier Loss Curve',
                       xaxis_title='Epochs',
                       yaxis_title='LogLoss',
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
    scores = [recall_score(y, to_labels(probs, t), average="weighted") for t in thresholds] 
    # IMPORTANT: Remember to set average="weighted", because when we use cross_val_score we use "f1_weighted". Otherwise we'll
    # get different results. A "weighted_f1_score" is preferred when using an imbalanced dataset. 

    # Get best threshold. 
    ix = np.argmax(scores)
    print('Threshold=%.3f, Recall_Score=%.5f' % (thresholds[ix], scores[ix]))
    
    # Plot thresholds curve. 
    fig = px.line(x=thresholds, y=scores)
    fig.add_vline(x=thresholds[ix], line_width=2, line_dash="dash", line_color="black") # Best loss score. 
    fig.update_layout(title='Threshold Manipulation for '+model_string,
                           xaxis_title='Thresholds',
                           yaxis_title='recall_score', height=455, width=900)
    fig.show() 
    

 # ============================== 6_Final_Pipeline ============================== #


class encoding_label:
    def __init__(self):
        x = 1
    def encode(self, df):
        le = LabelEncoder() # LabelEncoder instance. 
        df["diagnosis"] = le.fit_transform(df["diagnosis"]) # Transforming our label using LabelEncoder.

        filterLabel = list(df.columns) # Storing all features into a list.
        filterLabel.remove("diagnosis") # Removing "diagnosis" from our filter. 

        # Splitting the data into features and label. 
        self.X = df[filterLabel] # Filtering dataframe to exclude label. 
        self.y = df["diagnosis"] # Label. 
        
        
def Recall_plot(Recall_scores_GBC, Recall_scores_RFC, Recall_scores_XGB, score_string):
    fig = make_subplots(rows=3, cols=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=Recall_scores_GBC,
        name="Gradient Boosting Classifier"
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=Recall_scores_RFC,
        name="Random Forest Classifier"
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=list(range(1,31)),
        y=Recall_scores_XGB,
        name="XGB Classifier",
    ), row=3, col=1)
    
    fig.update_layout(height=550, width=900, title_text="Model Performance over 30 Iterations")

    fig['layout']['yaxis2']['title']=score_string
    fig['layout']['xaxis3']['title']='Iterations'

    fig.show()
    
def saving_model(trainFeatures, trainLabel, model):
    estimators = [('si', SimpleImputer()), 
                  ('n', Normalizer()), 
                  ("clf", model)]
    
    pipe = Pipeline(steps=estimators)
    pipe = pipe.fit(trainFeatures, trainLabel) 
    
    dump(pipe, open("XGB_Classifier.pkl", "wb"))
    
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
    
    
