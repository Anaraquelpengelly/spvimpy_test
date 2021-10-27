# This is a test on simulated data
#%%
import numpy as np
import vimpy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
#%%
##----------
## Problem setup
##----------
## define a function for the conditional mean of Y given X

def cond_mean(x =None):
    f1=np.where(np.logical_and(-2 <=x[:, 0], x[:, 0] < 2), np.floor(x[:, 0]), 0)
    f2=np.where(x[:, 1] <= 0,1,0)
    f3=np.where(x[:, 2] > 0,1,0)
    f6=np.absolute(x[:,5]/4) ** 3
    f7=np.absolute(x[:, 6]/4) ** 5
    f11 = (7./3)*np.cos(x[:, 10]/2)
    ret = f1+f2+f3+f6+f7+f11
    return ret
## create data

np.random.seed(4747)
n = 100
p = 15
s = 1# feature you are interested in
x = np.zeros((n, p))
for i in range(0, x.shape[1]) :
    x[:, i] = np.random.normal(0, 2, n)

y= cond_mean(x) +np.random.normal(0, 1, n)

#%%
## ------
## preliminary step: get regression estimators
## -----
## use grid search to et optimal number of trees and learning rate
ntrees = np.arange(100, 500, 100)
lr = np.arange(0.01, 0.1, 0.05)
param_grid = [{'n_estimators': ntrees, 'learning_rate': lr}]

#set up cv objects
cv_full = GridSearchCV(GradientBoostingRegressor(loss='ls', max_depth= 1), param_grid=param_grid, cv = 5)
cv_small = GridSearchCV(GradientBoostingRegressor(loss='ls', max_depth = 1), param_grid=param_grid, cv= 5)

#%% 
## ------
## Get variable importance estimates
## -----

np.random.seed(12345)
#set up vimp object
vimp= vimpy.vim(y=y, x=x, s=1, pred_func = cv_full, 
measure_type='r_squared')
#%% get the point estimate of variable importance
vimp.get_point_est()
#?? does not give anything apart from warnings!
#%% get influence function estimate
vimp.get_influence_function()
#%% get standard error
vimp.get_se()
#%% get a confidence interval
vimp.get_ci()
#%% do a hypothesis test, compute p-value
vimp.hypothesis_test(alpha=0.05, delta=0)
#%% display the estimates
vimp.vimp_
vimp.se_
vimp.ci_
vimp.p_value_
vimp.hyp_test_

### Not quite sure I understand this.
### First, the CV object, what is it? is it the best performing model from the CV gid search ? but it wasn't fitted!??
### Second, the vimp, is the feature importance value, but for only one feature?? so in this case there is one important feature over the 15??
### third, just a comment: it would be really cool to see an example with xgboost... although I do remember that the whole point of the paper was to get the "same" feature importance, regardless of the model... so might do it with several different models.
#%% 
## ----
## example using precomputed estimates using cross validation
## This is what I need
## ---
np.random.seed(12345)
folds_outer=np.random.choice(a=np.arange(2), size= n, replace=True, p = np.array([0.5, 0.5]))
# randomly sample 0 or 1 100 times with replacement probability 0.5
#%%
#fit the full regression
cv_full.fit(x[folds_outer == 1, :], y[folds_outer ==1])
full_fit =cv_full.best_estimator_.predict(x[folds_outer ==1, :])
#%%fit reduced regression
x_small = np.delete(x[folds_outer == 0, :], s, 1)#delete the features in s
cv_small.fit(x_small, y[folds_outer==0])
small_fit = cv_small.best_estimator_.predict(x_small)
#%% get variable importance estimates
np.random.seed(12345)
vimp_precompute = vimpy.vim(y=y, x=x, s=1, f=full_fit, r=small_fit,
measure_type="r_squared", folds=folds_outer)
#s= feature group !!! need to double check what that is!
## get point estimate of variable importance
vimp_precompute.get_point_est()
## get the influence function estimate ( what was that?)
vimp_precompute.get_influence_function()
## get standard error:
vimp_precompute.get_se()
## get confidence interval
vimp_precompute.get_ci()
## do a hypothesis test, compute p-value
vimp_precompute.hypothesis_test()
#%%##display the estimates
print(f'vimp: {vimp_precompute.vimp_}\nse: {vimp_precompute.se_}\nci: {vimp_precompute.ci_}\np_value: {vimp_precompute.p_value_}\nhyp_test: {vimp_precompute.hyp_test_}')


