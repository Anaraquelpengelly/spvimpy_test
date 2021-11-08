import xgboost as xgb
from xgboost.sklearn import XGBClassifier  
from sklearn.metrics import accuracy_score, roc_auc_score , roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import get, rcParams
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
col_sample_bytree=0.75,#prop of obs to be used for each tree. (TODO:this will be depriated!) 
use_label_encoder=False,
reg_alpha= 15, 
objective='binary:logistic',
nthread=4, #used for parallel processing! enter number of cores in the system.
scale_pos_weight=1, # for high class imbalance!helps in fater convergence..
seed=27
)
#%%
# modelfit(xgb_new, X_train, y_train, X_test, y_test)
#of note, should also do a grid search for the learning rate. 0.1 was better than 0.01 and that 0.3
