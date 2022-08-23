import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Import validation.
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split, StratifiedKFold, cross_val_predict, GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, recall_score, roc_curve, auc, confusion_matrix, f1_score


 # ============================== 1_Basic_Exploration ============================== #
    
    
def label_dashboard(df=None):
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'domain'}, {'type': 'bar'}], # Setting up our matrix of subplots. 
                                               [{'type': 'bar'}, {'type': 'bar'}]], # Specs for each individual subplot. 
                        subplot_titles=('Survived Count', 'By Embark',
                                        'By Sex', 'By Class')) # Subplot titles. 

    # Aggregate our label "Survived" by count. 
    data_pie = df.groupby('Survived').agg({'Survived': 'count'}).rename(columns={'Survived': 'count'}).reset_index()

    # Add pie char to subplot using our aggregation as data. 
    fig.add_trace(go.Pie(labels=data_pie['Survived'], values=data_pie['count'], textinfo='value+percent',
                         legendgroup='1', legendgrouptitle_text='Survived:'), 1, 1)

    # "Survived" by Embarked.
    for i in df['Survived'].unique():
        # Note: backslash is used to break up a long syntax without causing errors.
        data_bar = df[df['Survived'] == i].groupby('Embarked').agg({'Embarked': 'count'}) \
            .rename(columns={'Embarked': 'count'}).reset_index() # Setting up the data by aggregation 
        fig.add_trace(go.Bar(x=data_bar['Embarked'], y=data_bar['count'], showlegend=True, # Adding subplot. 
                             texttemplate="%{y}", name=i.astype(str), legendgroup='2',
                             legendgrouptitle_text='Survived:'), row=1, col=2)

    # "Survived" by Sex.
    for i in df['Survived'].unique():
        data_bar = df[df['Survived'] == i].groupby('Sex').agg({'Sex': 'count'}) \
            .rename(columns={'Sex': 'count'}).reset_index()
        fig.add_trace(go.Bar(x=data_bar['Sex'], y=data_bar['count'], showlegend=True,
                             texttemplate="%{y}", name=i.astype(str), legendgroup='3',
                             legendgrouptitle_text='Survived:'), row=2, col=1)    

    # "Survived" by Pclass.
    for i in df['Survived'].unique():
        data_bar = df[df['Survived'] == i].groupby('Pclass').agg({'Pclass': 'count'}) \
            .rename(columns={'Pclass': 'count'}).reset_index() 
        fig.add_trace(go.Bar(x=data_bar['Pclass'], y=data_bar['count'], showlegend=True,
                             texttemplate="%{y}", name=i.astype(str), legendgroup='4',
                             legendgrouptitle_text='Survived:'), row=2, col=2)

    fig.update_layout(plot_bgcolor='#F8F8F6', # Background plot colour. 
                      height=800, width=800, # Subplot dimensions. 
                      title_text=f'Analysis of Survival Rate', # Header
                      title_font_size=20, # Font size.
                      title_font_family='Arial Black', # Font style. 
                      title_x=0, title_y=0.95, # Position of Header, by width (title_x) and height (title_y).                                     
                      margin=dict(l=0, r=20, t=130, b=80))

    # More visual configurations. 
    fig.update_annotations(yshift=20)
    fig.update_traces(insidetextfont_size=10, selector=dict(type='pie'))
    fig.update_xaxes(title='Survived')
    fig.update_yaxes(title='Count of passengers')
    fig.update_yaxes(title='Fare', row=3, col=1)
    fig.show()

def numerical_gender(column=None, y_title=None, df=None):
    fig = make_subplots(rows=3, cols=2, specs=[[{"colspan": 2}, None],
                                               [{'type': 'box'}, {'type': 'box'}],
                                               [{'type': 'box'}, {'type': 'box'}]],
                        subplot_titles=(f'{y_title} Distribution to Survivors',
                                        f'{y_title} Distribution', 
                                        f'{y_title} to Survivors',
                                        f'{y_title} of Males to Survivors',
                                        f'{y_title} of Feamles to Survivors'))

    # Plot 1.1-------------------------------------------------
    for i in df['Survived'].unique():
        data_box = df[df['Survived'] == i]
        fig.add_trace(go.Histogram(x=data_box[column], nbinsx=40, showlegend=True, name=i.astype(str), 
                                   legendgroup='1', legendgrouptitle_text='Survived:'), row=1, col=1)

    # Plot 2.1-------------------------------------------------
    fig.add_trace(
        go.Box(y=df[column], showlegend=False, name=''), row=2, col=1)

    # Plot 2.2-------------------------------------------------
    for i in df['Survived'].unique():
        data_box = df[df['Survived'] == i]
        fig.add_trace(go.Box(y=data_box[column], showlegend=True, name=i.astype(str),
                             legendgroup='2', legendgrouptitle_text='Survived:'), row=2, col=2)

    # Plot 3.1-------------------------------------------------
    for i in df['Survived'].unique():         
        data_box = df[(df['Survived'] == i) & (df['Sex'] == 'male')]
        fig.add_trace(go.Histogram(x=data_box[column], nbinsx=35, showlegend=True, name=i.astype(str), 
                                   legendgroup='3', legendgrouptitle_text='Survived:'), row=3, col=1)

    # Plot 3.2-------------------------------------------------
    for i in df['Survived'].unique():         
        data_box = df[(df['Survived'] == i) & (df['Sex'] == 'female')]
        fig.add_trace(go.Histogram(x=data_box[column], nbinsx=30, showlegend=True, name=i.astype(str), 
                                   legendgroup='4', legendgrouptitle_text='Survived:'), row=3, col=2)
        
        
    fig.update_layout(plot_bgcolor='#F8F8F6',
                  height=800, width=800, bargap=0.2,
                  title_text=f'Analysis of {column}', title_font_size=20, title_font_family='Arial Black',                      
                  title_x=0, title_y=0.98,
                  margin=dict(l=0, r=20, t=130, b=80))


    fig.update_annotations(yshift=20)
    fig.update_traces(insidetextfont_size=10, selector=dict(type='pie'))
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(title=f'{column}')
    fig.update_yaxes(title='Count of passengers')
    fig.update_yaxes(title=f'{column}', row=2, col=1)
    fig.update_xaxes(title='', row=2, col=1)
    fig.update_xaxes(title='Survived or not', row=2, col=2)
    fig.update_yaxes(title=f'{column}', row=2, col=2)
    fig.update_yaxes(title=f'{column}', row=2, col=1)    
    fig.show()
    
def numerical_classes(column=None, y_title=None, df=None):
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'box'}, {'type': 'box'}],
                                               [{"colspan": 2}, None]],
                        subplot_titles=(f'{y_title} of 1st Class to Survivors',
                                        f'{y_title} of 2nd Class to Survivors',
                                        f'{y_title} of 3rd Class to Survivors'))

    # Plot 4.1-------------------------------------------------
    for i in df['Survived'].unique():         
        data_box = df[(df['Survived'] == i) & (df['Pclass'] == 1)]
        fig.add_trace(go.Histogram(x=data_box[column], nbinsx=35, showlegend=True, name=i.astype(str), 
                                   legendgroup='5', legendgrouptitle_text='Survived:'), row=1, col=1)

    # Plot 4.2-------------------------------------------------
    for i in df['Survived'].unique():         
        data_box = df[(df['Survived'] == i) & (df['Pclass'] == 2)]
        fig.add_trace(go.Histogram(x=data_box[column], nbinsx=30, showlegend=True, name=i.astype(str), 
                                   legendgroup='6', legendgrouptitle_text='Survived:'), row=1, col=2)

    # Plot 5.1-------------------------------------------------
    for i in df['Survived'].unique():         
        data_box = df[(df['Survived'] == i) & (df['Pclass'] == 3)]
        fig.add_trace(go.Histogram(x=data_box[column], nbinsx=35, showlegend=True, name=i.astype(str), 
                                   legendgroup='7', legendgrouptitle_text='Survived:'), row=2, col=1)
        
        
    fig.update_layout(plot_bgcolor='#F8F8F6',
                  height=650, width=800, bargap=0.2,
                  #itle_text=f'Analysis of {column}', title_font_size=20, title_font_family='Arial Black',                      
                  title_x=0, title_y=0.98,
                  margin=dict(l=0, r=20, t=130, b=80))


    fig.update_annotations(yshift=20)
    fig.update_traces(insidetextfont_size=10, selector=dict(type='pie'))
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(title=f'{column}')
    fig.update_yaxes(title='Count of passengers')  
    fig.show()

    
def empty_values(height, data):
    empty_data = (round(data.isna().sum() / data.shape[0] * 100, 2).to_frame('percent') \
                                                                   .query("percent > 0") \
                                                                   .sort_values('percent', ascending=True))    
    fig = px.bar(empty_data,  
                 orientation='h',
                 height=height,
                 text=[f"{i}%" for i in empty_data['percent']])
        
    
    fig.update_layout(plot_bgcolor='#F8F8F6',                  
                      font_color='#012623',
                      title_font_size=20,
                      title_font_family='Arial Black',
                      xaxis_title='',
                      yaxis_title='',
                      bargap=0.2,
                      showlegend=False,
                      title_text='Percentage of empty values in the dataset', title_x=0,
                      margin=dict(l=0, r=20, t=100, b=20))
    
    fig.show()
    
def age_group_plot(column=None, y_title=None, df=None):
    fig = make_subplots(rows=1, cols=2, specs=[[{"colspan": 1}, {"colspan": 1}]],
                        subplot_titles=(f'Survived Age Group', 'Gender Age Group'))

        # "Survived" by Embarked.
    for i in df['Survived'].unique():
        # Note: backslash is used to break up a long syntax without causing errors.
        data_bar = df[df['Survived'] == i].groupby('Age Group').agg({'Age Group': 'count'}) \
        .rename(columns={'Age Group': 'count'}).reset_index() # Setting up the data by aggregation 
        fig.add_trace(go.Bar(x=data_bar['Age Group'], y=data_bar['count'], showlegend=True, # Adding subplot. 
                             texttemplate="%{y}", name=str(i), legendgroup='1',
                             legendgrouptitle_text='Survived:'), row=1, col=1)
        
    for i in df['Sex'].unique():
        # Note: backslash is used to break up a long syntax without causing errors.
        data_bar = df[df['Sex'] == i].groupby('Age Group').agg({'Age Group': 'count'}) \
        .rename(columns={'Age Group': 'count'}).reset_index() # Setting up the data by aggregation 
        fig.add_trace(go.Bar(x=data_bar['Age Group'], y=data_bar['count'], showlegend=True, # Adding subplot. 
                             texttemplate="%{y}", name=str(i), legendgroup='2',
                             legendgrouptitle_text='Sex:'), row=1, col=2)
        
    fig.update_layout(plot_bgcolor='#F8F8F6',
                  height=400, width=800, bargap=0.2,
                  title_text=f'Analysis of {column}', title_font_size=20, title_font_family='Arial Black',                      
                  title_x=0, title_y=0.98,
                  margin=dict(l=0, r=20, t=130, b=80))


    fig.update_annotations(yshift=20)
    fig.update_traces(insidetextfont_size=10, selector=dict(type='pie'))
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(title=f'{column}')
    fig.update_yaxes(title='Count of passengers')  
    fig.show()
    
def married_distribution(dfMale=None, dfFemale=None):
    fig = make_subplots(rows=1, cols=2, specs=[[{'colspan': 1}, {'colspan': 1}]], # Setting up our matrix of subplots. 
                        subplot_titles=('Married Males Distribution', 'Married Females Distribution')) # Subplot titles. 
    
    fig.add_trace(go.Histogram(x=dfMale["Age"], nbinsx=35, showlegend=True, legendgroup='7', name="males"), row=1, col=1)
    fig.add_trace(go.Histogram(x=dfFemale["Age"], nbinsx=35, showlegend=True, name="females", legendgroup='7'), row=1, col=2)
    
    fig.update_layout(plot_bgcolor='#F8F8F6',
                  height=400, width=800, bargap=0.2,
                  title_text='Analysis of Married Couples', title_font_size=20, title_font_family='Arial Black',                      
                  title_x=0, title_y=0.98,
                  margin=dict(l=0, r=20, t=130, b=80))

    # More visual configurations. 
    fig.update_annotations(yshift=20)
    #fig.update_xaxes(title='Married Males', row=1, col=1)
    #fig.update_xaxes(title='Married Females', row=1, col=2)
    fig.update_yaxes(title='Count of passengers')
    fig.update_yaxes(title='Fare', row=3, col=1)
    fig.show()
    
    
 # ============================== 2_Base_Models ============================== #


def kfold(scores=None):
    fig = go.Figure() # Creating blank figure. 

    # Adding individual boxplots into our subplot.  
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
                      title_font_size=20, title_font_family='Arial Black', title_x=0, title_y=0.95,                                
                          margin=dict(l=0, r=20, t=130, b=80))
    fig.show() # Showing figure. 

    
 # ============================== 3_Feature_Engineering ============================== # 
    
    
def plot_rfe(xaxis=None, yLR=None, yGBC=None, yXGB=None):    
    
    # Creating figure. 
    fig = go.Figure()
    # Adding subplots. 
    fig.add_trace(go.Scatter(x=xaxis, y=yLR,name='Logistic Regression')) 
    fig.add_trace(go.Scatter(x=xaxis, y=yGBC,name='Gradient Boosting Classifer'))
    fig.add_trace(go.Scatter(x=xaxis, y=yXGB,name='XGB Classifier'))
    # Defining labels.  
    fig.update_layout(title='Recursive Feature Elimination for Selected Models',
                       xaxis_title='Total Features Selected',
                       yaxis_title='Weighted f1 score', width=980, height=450, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    fig.show() # Display figure. 
    

  # ============================== 4_Optimization ============================== #

def n_components_plot(xaxis=None, yLR=None, yGBC=None, yXGB=None):
    # Setting up our figure canvas. 
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(x=xaxis, y=yLR, name="Logistic Regression"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xaxis, y=yGBC, name="Gradient Boosting Classifier"), row=2, col=1) 
    fig.add_trace(go.Scatter(x=xaxis, y=yXGB, name="XGB Classifier"), row=3, col=1)

    # Axes labels. 
    fig['layout']['xaxis3']['title']='n_components'
    fig['layout']['yaxis']['title']='f1_score'

    fig.update_layout(title_text="Finding Optimal N Components", height=550, width=800, plot_bgcolor='#F8F8F6', 
                      title_font_size=20, title_font_family='Arial Black') # Configure dimensions.
    fig.show() # Display plots. 

    
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
                           yaxis_title='f1_score', height=400, width=800, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')

    fig.show()
    
    
 # ============================== 5_Regularization ============================== #     


def LogLoss_Curve(x_axis, y_axis_train, y_axis_test):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_train,
                        name='Train'))
    fig.add_trace(go.Scatter(x=list(x_axis), y=y_axis_test,
                        name='Test'))

    fig.update_layout(title='XGB Classifier Log Loss Curve',
                       xaxis_title='Epochs',
                       yaxis_title='Log Loss',
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
                           yaxis_title='f1_score', height=400, width=800, plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')
    fig.show()
    

 # ============================== 6_Final_Pipeline ============================== #
 

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


    fig.update_layout(height=550, width=900, title_text="Model Performance over 30 Iterations", plot_bgcolor='#F8F8F6', title_font_size=20, 
                       title_font_family='Arial Black')

    fig['layout']['yaxis']['title']='f1_score'
    fig['layout']['xaxis3']['title']='Iterations'

    fig.show()
    
    
    