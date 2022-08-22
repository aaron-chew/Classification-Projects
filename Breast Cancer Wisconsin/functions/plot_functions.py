import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import validation.
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, recall_score, roc_curve, auc, confusion_matrix


 # ============================== 1_Basic_Exploration ============================== #
    
    
def histogram(df=None):
    fig = make_subplots(rows=4, cols=2, specs=[[{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}],
                                              [{"colspan": 1}, {"colspan": 1}]],
                        subplot_titles=('radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                                       'compactness_mean', 'concavity_mean', 'concave points_mean'))

    fig.add_trace(go.Histogram(x=df["radius_mean"], name="radius_mean", nbinsx=35, showlegend=True, legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Histogram(x=df["texture_mean"], name="texture_mean", nbinsx=35, showlegend=True, legendgroup='1'),row=1, col=2)
    fig.add_trace(go.Histogram(x=df["perimeter_mean"], name="perimeter_mean", nbinsx=35, showlegend=True,legendgroup='1'), row=2, col=1)
    fig.add_trace(go.Histogram(x=df["area_mean"], name="area_mean", nbinsx=35, showlegend=True, legendgroup='1'),row=2, col=2)
    
    fig.add_trace(go.Histogram(x=df["smoothness_mean"], name="smoothness_mean", nbinsx=35, showlegend=True,legendgroup='1'), row=3, col=1)
    fig.add_trace(go.Histogram(x=df["compactness_mean"], name="compactness_mean", nbinsx=35, showlegend=True,legendgroup='1'), row=3, col=2)
    fig.add_trace(go.Histogram(x=df["concavity_mean"], name="concavity_mean", nbinsx=35, showlegend=True,legendgroup='1'), row=4, col=1)
    fig.add_trace(go.Histogram(x=df["concave points_mean"], name="concave points_mean", nbinsx=35, showlegend=True,legendgroup='1'), 
                  row=4, col=2)
        
    fig.update_layout(plot_bgcolor='#F8F8F6',
                  height=950, width=950, bargap=0.2,
                  title_text='Histogram of Numerical Features', title_font_size=20, title_font_family='Arial Black',                      
                  title_x=0, title_y=0.98,
                  margin=dict(l=0, r=20, t=130, b=80))

    fig.update_annotations(yshift=20)
    #fig.update_traces(insidetextfont_size=10, selector=dict(type='pie'))
    fig.update_yaxes(title='Count')  
    fig.show()

def boxplot(df=None):
    fig = make_subplots(rows=4, cols=2) # Creating a matrix of empty subplots. 

    # Adding each plot into our matrix. 
    # Column 1. 
    fig.append_trace(go.Box(x=df["radius_mean"], name="radius"), row=1, col=1)
    fig.append_trace(go.Box(x=df["texture_mean"],name="texture"), row=2, col=1)
    fig.append_trace(go.Box(x=df["perimeter_mean"],name="perimeter"), row=3, col=1)
    fig.append_trace(go.Box(x=df["area_mean"], name="area"), row=4, col=1)

    # Column 2.
    fig.append_trace(go.Box(x=df["smoothness_mean"], name="smoothness"), row=1, col=2)
    fig.append_trace(go.Box(x=df["compactness_mean"],name="compactness"), row=2, col=2)
    fig.append_trace(go.Box(x=df["concavity_mean"],name="concavity"), row=3, col=2)
    fig.append_trace(go.Box(x=df["concave points_mean"], name="concave points"), row=4, col=2)

    # Editing the layout and showing the figure. 
    fig.update_layout(plot_bgcolor='#F8F8F6', title_text="Boxplot of Numerical Features", title_font_size=20, 
                title_font_family='Arial Black', title_x=0, title_y=0.98, height = 750, width = 950)
    fig.show()
    
 # ============================== 2_Base_Models ============================== #

def kfold(scores=None):
    fig = go.Figure() # Creating blank figure. 

    # Adding indivdual boxplots to figure. 
    fig.add_trace(go.Box(y=list(scores[0]), name="CART"))
    fig.add_trace(go.Box(y=list(scores[1]), name="LR"))
    fig.add_trace(go.Box(y=list(scores[2]), name="Per"))
    fig.add_trace(go.Box(y=list(scores[3]), name="KNN"))
    fig.add_trace(go.Box(y=list(scores[4]), name="RFC"))
    fig.add_trace(go.Box(y=list(scores[5]), name="GBC"))
    fig.add_trace(go.Box(y=list(scores[6]), name="XGBC"))
    fig.add_trace(go.Box(y=list(scores[7]), name="SVC"))

    # Configuring layout. 
    fig.update_layout(plot_bgcolor='#F8F8F6',height=450, width=950, title_text="Performance Over 5 Folds",
                      title_font_size=20, title_font_family='Arial Black')
    fig.show()
    
    
 # ============================== 3_Feature_Engineering ============================== # 
    
    
def plot_rfe(xaxis=None, yGBC=None, yRFC=None, yXGB=None):    
    
    # Creating figure. 
    fig = go.Figure()
    # Adding subplots. 
    fig.add_trace(go.Scatter(x=xaxis, y=yGBC,name='Gradient Boosting Classifier')) 
    fig.add_trace(go.Scatter(x=xaxis, y=yRFC,name='Random Forest Classifer'))
    fig.add_trace(go.Scatter(x=xaxis, y=yXGB,name='XGB Classifier'))
    # Defining labels.  
    fig.update_layout(title='Recursive Feature Elimination for Selected Models',
                       xaxis_title='Total Features Selected',
                       yaxis_title='Weighted Recall', width=980, height=450, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    fig.show() # Display figure. 
    
    
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
                           yaxis_title='Weighted Recall', height=400, width=800, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    
    fig.show()
    
      
 # ============================== 5_Regularization ============================== #     
 
    
def LogLoss_Curve(x_axis, y_axis_train, y_axis_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_train,
                        name='Train'))
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_test,
                        name='Test'))

    fig.update_layout(title='XGB Classifier Loss Curve',
                       xaxis_title='Epochs',
                       yaxis_title='LogLoss',
                       height=400, width=800, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    
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
                           yaxis_title='recall_score', height=400, width=800, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    
    fig.show()
    

 # ============================== 6_Final_Pipeline ============================== #


def recall_plot(Recall_scores_GBC, Recall_scores_RFC, Recall_scores_XGB, score_string):
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
    
    fig.update_layout(height=550, width=900, title_text="Model Performance over 30 Iterations", plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')

    fig['layout']['yaxis2']['title']=score_string
    fig['layout']['xaxis3']['title']='Iterations'

    fig.show()
    
    
    