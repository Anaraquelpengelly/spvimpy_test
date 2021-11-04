#import required libraries
import pandas as pd
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

#define data path:
data_path = 'clean_data/'

#read data:    
data = pd.read_csv(data_path+'first_diet_df.csv')
#read the variable description:
varis = pd.read_csv(data_path+'expo_variables.csv')

#get data with NAs and no scaling: 
X_train = pd.read_csv(data_path+'X_train_with_NAs.csv')

cols = list(X_train.columns)

#prepare xtrain
X_train.drop(columns=['eid', 'index'], inplace =True)
y_train = pd.read_csv(data_path + 'y_train_cell_count_final_with_NAs.csv')
y_train.drop('eid', axis =1, inplace =True)
y_train = y_train.to_numpy()
y_train = y_train.ravel()

#remove duplicated columns (not sure why they happen)
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
print(f'ytrain shape : {y_train.shape} , and xtrain shape is: {X_train.shape}')
#Now select the variables in the Xtest:

X_test = pd.read_csv(data_path + 'X_test_with_NAs.csv')
X_test.drop(columns=['eid', 'index'], inplace = True)
y_test = pd.read_csv(data_path + 'y_test_cell_count_final_with_NAs.csv')
y_test.drop('eid', axis = 1, inplace = True)
y_test = y_test.to_numpy()
y_test = y_test.ravel()
#remove duplicated columns from the Xtest:
X_test = X_test.loc[:, ~X_test.columns.duplicated()]
print(f'ytest shape : {y_test.shape} , and xtest shape is: {X_test.shape}')
#create the DMatrixes for train and test:
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label = y_test)


#calculate the imbalance estimate with ytrain+ytest:
from collections import Counter
x = np.concatenate([y_train, y_test])
counter = Counter(x)
estimate = counter[0]/counter[1]
print(f'The imbalance estimate is {estimate}.')

#set parameters for xgboost:

param = {'max_depth': 5,
         'scale_pos_weight': estimate, #use for imbalanced datasets.
         'eta': 0.2, #eta: learning rate: shrinks the feature weights to make the boosting process more conservative. TODO: to tune!
         #step size shrinkage used to prevent overfitting. Range is [0,1]
         'objective' : 'binary:logistic',
        'nthread': 4,# also don't know what this is
        'colsample_bytree': 0.3, #percentage of features used per tree. High value can lead to overfitting.
         'eval_metric': 'auc'}

## set evaluation list to watch the performance:

evallist = [(dtest, 'eval'), (dtrain, 'train')]

## Train the model: 

num_round = 10 #rounds of training/number of trees. To tune!
bst = xgb.train(param, dtrain, num_round, evallist)
#get the feature importance
feat_imp = bst.get_score()
feats = list(feat_imp.keys())
imps = list(feat_imp.values())
fi_df= pd.DataFrame(data = imps, index = feats, columns = ['Fscores']).sort_values(by='Fscores', ascending = False)
fi_df = fi_df.reset_index()
fi_df = fi_df.rename(columns ={'index': 'features'})

#get today's date:
from datetime import date
date = date.today()

#make plotly plot:
fig = px.bar(fi_df, x='Fscores', y='features',
             labels={'Fscores':'feature importance (Fscore)', 'features':'features'} )
fig.update_layout(
    font=dict(
        size=18
    ))
fig.show()
fig.write_html('results/'+f'feature_importance_xgboost_001_all_{date}.html')

#feature importance with SHAP values:

import shap

X=pd.concat([X_train, X_test], axis = 0)

explainer = shap.TreeExplainer(bst)
shap_values = explainer(X)




#save plots


shap.initjs()

##summary plot
fig = shap.plots.beeswarm(shap_values, show=False)
plt.savefig('results/'+f'summary_plot_all_{date}.png', dpi=150, bbox_inches='tight')
plt.close(fig=None)
##feature_importance plot:

fig = shap.plots.bar(shap_values, show=False)
plt.savefig('results/'+f'imp_plot_diet_all_{date}.png', dpi=150, bbox_inches='tight')
plt.close(fig=None)
##dependence scatter plot:

###first get the most important feature so you can get the dependence scatter for that feature:
#get feature importance df:
def global_shap_importance(model, data):
    """ Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model 
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance

feat_imp_df = global_shap_importance(bst, X)

#make the figure with the most important feature and strongest interactor:


fig =shap.plots.scatter(shap_values[:,feat_imp_df.iloc[0,0]], color=shap_values, show=False)
plt.savefig('results/'+f'dep_scatter_all_{date}.png', dpi=150, bbox_inches='tight')
print('finished saving the plots.')


#save the shap values for all instances:
##first save as a dataframe:
import time
print('started saving as df') 
t0=time.time()
x=pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
t=time.time()
delta=t-t0
print(f'finished saving as df, it took {delta/60} minutes.')

#now save as pickle: 
x.to_pickle('results/shap_values_overall.pkl')
x.to_csv('results/shap_values_overall.csv', index=False)
# try this next time: 
#https://github.com/slundberg/shap/issues/295




