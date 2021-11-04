#%% real life example  taken from : https://bdwilliamson.github.io/vimp/articles/introduction_to_vimp.html 
#get sourth african heart disease data:
from altair.vegalite.v4.schema.core import InlineData
from numpy.core.numeric import outer
import pandas as pd
import numpy as np
data = pd.read_csv("http://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data")
#%% data cleaning:
heart = data.iloc[0:data.shape[0] , 1:data.shape[1]]
#%%
heart['famhist'] = np.where(heart['famhist'] == 'Present',1, 0 )
#%%
X= heart.drop(columns='chd')
X=X.to_numpy()
y=np.array(heart.chd)
#%%
#Here we try xgboost:
#info here: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

import xgboost as xgb
from xgboost.sklearn import XGBClassifier  
from sklearn.metrics import accuracy_score, roc_auc_score , roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

#%% define function to do model fit:
#Here we use the cross validation function of xgb. we give the function an algorithm ( the xgb classifier), the train dataframe, the predictors
# TODO: need to integrate teh test set in there too!! (inside the function fo the split)  
def modelfit(alg, dtrain, y_train, dtest, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        print('using train cv')
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain, y_train, eval_metric='auc')
    print('data fitted')    
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]
    dtest_preds = alg.predict(dtest)
    dtest_predprob = alg.predict_proba(dtest)[:,1]

    #Print model report:
    print (f"\nModel Report\nAccuracy : {accuracy_score(y_train, dtrain_predictions)}\nAUC Score (Train): {roc_auc_score(y_train, dtrain_predprob)}")
    print(f"\nModel Report\nAccuracy : {accuracy_score(y_test, dtest_preds)}\nAUC Score (Test): {roc_auc_score(y_test, dtest_predprob)}")
    fpr, tpr, _ = roc_curve(y_test, dtest_predprob)
    auc=roc_auc_score(y_test, y_pred_proba)
    #Plot the roc curve:
    plt.plot(fpr, tpr, '.-', label='ROC-AUC score'+str(round(auc, 3)))
    plt.xlabel('False positives')
    plt.ylabel('True positives')
    plt.legend(loc='lower right')
    plt.show()

    # feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
    #we remove the above for the moment because we have no feature names (numpy arrays)

#%% 1- Establish a simple model:
xgb0=XGBClassifier(learning_rate = 0.1,
n_estimators=100,
max_depth=4, #max depth of a tree
min_child_weight=3, # weight of leaf? needs to be tuned too high values can lead to underfitting.
gamma=0,
subsample=0.8,
col_sample_bytree=0.75,#prop of obs to be used for each tree. 
reg_alpha= 15, 
objective='binary:logistic',
nthread=4, #used for parallel processing! enter number of cores in the system.
scale_pos_weight=1, # for high class imbalance!helps in fater convergence..
seed=27)
#%%
#lets not use the function 
# modelfit(xgb0, X, y)
#we then first get the data matrix:
data_matrix=xgb.DMatrix(data=X, label=y)
#do the train/test split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123) 
xgb0.fit(X_train,y_train, eval_metric='auc')
preds=xgb0.predict(X_test)
# ypred_proba: probability of each sample to belong to each class.
# for binary outcomes outputs a probability matrix of dimension (N,2). 
# The first index refers to the probability that the data belong to class 0, 
# and the second refers to the probability that the data belong to class 1.


y_pred_proba = xgb0.predict_proba(X_test)[::,1]

print (f"\nModel Report\nAccuracy : {accuracy_score(y_test, preds)}\nAUC Score (Test): {roc_auc_score(y_test, y_pred_proba)}")
#%% cross validation of parameters:
#max_depth, min_child_weight
param_test1 = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,6,1)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4, cv=5)
gsearch1.fit(X_train,y_train)
print(f'best parameters:{gsearch1.best_params_},\nbest score{gsearch1.best_score_}')
gsearch1_res=pd.DataFrame(gsearch1.cv_results_)
#as a note, you can use the gsearch as an object to use for predict, it will use the best parameters!  

#%% Now tune gamma:
# A node is split only when the resulting split gives a positive reduction in the loss function. 
# Gamma specifies the minimum loss reduction required to make a split.
param_test2={ 'gamma': [i/10 for i in range(0,5)]}

gsearch2 =GridSearchCV(estimator=XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=4,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test2, scoring='roc_auc',n_jobs=4, cv=5)
gsearch2.fit(X_train, y_train)
print(f'best parameters:{gsearch2.best_params_},\nbest score{gsearch2.best_score_}')

#make df:
gsearch2_res=pd.DataFrame(gsearch2.cv_results_)

#%%
param_test3={
    'subsample':[i/10 for i in range(6,10)],
    'colsample_by_tree':[i/10 for i in range(6,10)]
}
gsearch3=GridSearchCV(estimator=XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=4,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4, cv=5)
gsearch3.fit(X_train, y_train)
print(f'best parameters:{gsearch3.best_params_},\nbest score{gsearch3.best_score_}')

#make df:
gsearch3_res=pd.DataFrame(gsearch3.cv_results_)
#%%Tune subsample and colsample_bytree
param_test4={
    'subsample':[i/100 for i in range(75,90, 5)],
    'colsample_by_tree':[i/100 for i in range(75,90,5)]
}
gsearch4=GridSearchCV(estimator=XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=4,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4, cv=5)
gsearch4.fit(X_train, y_train)
print(f'best parameters:{gsearch4.best_params_},\nbest score{gsearch4.best_score_}')

#make df:
gsearch4_res=pd.DataFrame(gsearch4.cv_results_)
  

#%% tune regularisation params
param_test5={
    'reg_alpha':list(np.arange(5, 51, 1))
}  
gsearch5=GridSearchCV(estimator=XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=4,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.75,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test5, scoring='roc_auc',n_jobs=4, cv=5)
gsearch5.fit(X_train, y_train)
print(f'best parameters:{gsearch5.best_params_},\nbest score{gsearch5.best_score_}')

#make df:
gsearch5_res=pd.DataFrame(gsearch5.cv_results_)
#updated the model 0 with the bet parameters
#%% reduce the learning rate:
xgb_new =XGBClassifier(learning_rate = 0.1,
n_estimators=100,
max_depth=4, #max depth of a tree
min_child_weight=3, # weight of leaf? needs to be tuned too high values can lead to underfitting.
gamma=0,
subsample=0.8,
col_sample_bytree=0.75,#prop of obs to be used for each tree. 
reg_alpha= 15, 
objective='binary:logistic',
nthread=4, #used for parallel processing! enter number of cores in the system.
scale_pos_weight=1, # for high class imbalance!helps in fater convergence..
seed=27
)
#%%
modelfit(xgb_new, X_train, y_train, X_test, y_test)
#of note, should also do a grid search for the learning rate. 0.1 was better than 0.01 and that 0.3

#%%
ntrees = np.arange(100, 500, 100)
lr = np.arange(0.01, 0.1, 0.05)
param_grid = [{'n_estimators': ntrees, 'learning_rate': lr}]

#set up cv objects
cv_full = GridSearchCV(GradientBoostingClassifier(loss='deviance', max_depth= 1), param_grid=param_grid, cv = 5)
cv_small = GridSearchCV(GradientBoostingClassifier(loss='deviance', max_depth = 1), param_grid=param_grid, cv= 5)

#%%
import vimpy
lm_vim_sbp = vimpy.vim(y = y, x = X , s = 0, 
pred_func=cv_full, measure_type = "r_squared")
#
lm_vim_tob = vimpy.vim(y = y, x = X , s = 1, 
pred_func=cv_full, measure_type = "r_squared")
#
lm_vim_ldl = vimpy.vim(y = y, x = X , s = 2, 
pred_func=cv_full, measure_type = "r_squared")
#
lm_vim_adi = vimpy.vim(y = y, x = X , s = 3, 
pred_func=cv_full, measure_type = "r_squared")
#
lm_vim_fam = vimpy.vim(y = y, x = X , s = 4, 
pred_func=cv_full, measure_type = "r_squared")
#
lm_vim_tpa = vimpy.vim(y = y, x = X , s = 5, 
pred_func=cv_full, measure_type = "r_squared")
#
lm_vim_obe= vimpy.vim(y = y, x = X , s = 6, 
pred_func=cv_full, measure_type = "r_squared")
#
lm_vim_alc = vimpy.vim(y = y, x = X , s = 7, 
pred_func=cv_full, measure_type = "r_squared")
#
lm_vim_age = vimpy.vim(y = y, x = X , s = 8, 
pred_func=cv_full, measure_type = "r_squared")

#%% 
#create function to get all outputs for a function:
def get_output(vimpy_object):
    # get point estimate
    vimpy_object.get_point_est()
    #get influence function estimate
    vimpy_object.get_influence_function()
    #get standard error
    vimpy_object.get_se()
    #get a confidence interval
    vimpy_object.get_ci()
    #do a hypothesis test, compute p-value
    vimpy_object.hypothesis_test(alpha=0.05, delta=0)
    #TODO: what is delta?

get_output(lm_vim_sbp)
get_output(lm_vim_tob)
get_output(lm_vim_ldl)
get_output(lm_vim_adi)
get_output(lm_vim_fam)
get_output(lm_vim_tpa)
get_output(lm_vim_obe)    
get_output(lm_vim_alc)
get_output(lm_vim_age)




#%% get the values and make table with them:
feat=['sbp','tob','ldl', 'adi','fam','tpa','obe','alc','age']
vim=[ lm_vim_sbp.vimp_, 
lm_vim_tob.vimp_, 
lm_vim_ldl.vimp_,
lm_vim_adi.vimp_,
lm_vim_fam.vimp_,
lm_vim_tpa.vimp_,
lm_vim_obe.vimp_,
lm_vim_alc.vimp_,
lm_vim_age.vimp_
 ]
se=[lm_vim_sbp.se_, 
lm_vim_tob.se_, 
lm_vim_ldl.se_,
lm_vim_adi.se_,
lm_vim_fam.se_,
lm_vim_tpa.se_,
lm_vim_obe.se_,
lm_vim_alc.se_,
lm_vim_age.se_ ]

ci=[lm_vim_sbp.ci_,
lm_vim_tob.ci_,
lm_vim_ldl.ci_,
lm_vim_adi.ci_,
lm_vim_fam.ci_,
lm_vim_tpa.ci_,
lm_vim_obe.ci_,
lm_vim_alc.ci_,
lm_vim_age.ci_ ]
p_value=[
lm_vim_sbp.p_value_, 
lm_vim_tob.p_value_,
lm_vim_ldl.p_value_,
lm_vim_adi.p_value_,
lm_vim_fam.p_value_,
lm_vim_tpa.p_value_,
lm_vim_obe.p_value_,
lm_vim_alc.p_value_,
lm_vim_age.p_value_ ]

zipped=list(zip(feat, vim, se, ci, p_value))



df=pd.DataFrame(zipped, columns=['feature', 'vimp', 'se','ci', 'p_value'])
df

#%% get IC upper and lower
df['2.5%'] = [df['ci'].iloc[r][0][0] for r in range(len(df))]
df['97.5%'] = [df['ci'].iloc[r][0][1] for r in range(len(df))]
#%% order the features by importance (even if not significant)
df=df.sort_values(by='vimp', ascending=False, key=abs)#sort by absolute value
#%%Make forest plot

#TODO: need to make the order of the Y axis be the order of the df!!!
import altair as alt

def make_forest(source):
    #preprocessing:
    ##remove spaces from column names:
    source.columns=[col.replace(" ", "") for col in source.columns]
    source['feature'] = source['feature'].astype('string')
    source[['vimp', 'p_value','2.5%', '97.5%']]=source[['vimp', 'p_value','2.5%', '97.5%']].astype('float64')

    #points
    points = alt.Chart(source).mark_point(filled=True).encode(
    x=alt.X('vimp:Q', title='VIMP'),
    y=alt.Y(
        'feature:N', axis=alt.Axis(grid=True)), color=alt.condition('datum.p_value<0.05', alt.ColorValue('red'), alt.ColorValue('black'))
         ).properties(
    width=600,
    height=400)
    
    errorbars =points.mark_errorbar().encode(
    x=alt.X('2.5%:Q', title='VIMP'),
    x2='97.5%:Q',
    y="feature:N"
)
    #line
    line=alt.Chart(source).mark_rule(strokeDash=[5,2]).encode(
    x='a:Q',
    size=alt.value(2),
    color=alt.ColorValue('red'),
    ).transform_calculate(
    a="0")
    return points+errorbars+line


#%%
make_forest(df)


#%%
## ----
## example using precomputed estimates using cross validation
## This is what I need
## ---
np.random.seed(12345)
import random
random.seed(12345)
percent=int(0.5*len(heart))
nums=percent*[1]+(len(heart)-percent)*[0]
random.shuffle(nums)
heart_folds=np.array(nums)
# randomly sample 0 or 1 100 times with replacement probability 0.5
#%%
#fit the full regression
cv_full.fit(X[heart_folds == 1, :], y[heart_folds ==1])
full_fit =cv_full.best_estimator_.predict(X[heart_folds == 1, :])
#%%fit reduced regression
#for sbp
x_small_sbp = np.delete(X[heart_folds == 0, :], 0, 1)#delete the features in s
cv_small.fit(x_small_sbp, y[heart_folds==0])
small_fit_sbp= cv_small.best_estimator_.predict(x_small_sbp)
#%% get variable importance estimates
np.random.seed(12345)
vimp_precompute_sbp = vimpy.vim(y=y, x=X, s=0, f=full_fit, r=small_fit_sbp,
measure_type="r_squared", folds=heart_folds)

#%% get output 
get_output(vimp_precompute_sbp)

#%% look at single results:
print(f'vimp: {vimp_precompute_sbp.vimp_}\nse: {vimp_precompute_sbp.se_}\nci: {vimp_precompute_sbp.ci_}\np_value: {vimp_precompute_sbp.p_value_}\nhyp_test: {vimp_precompute_sbp.hyp_test_}')

#TODO: make function to do the spvim on all features and to get a table with the results.
#TODO: why is hypothesis test=False??
