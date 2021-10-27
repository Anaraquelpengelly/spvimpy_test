#%% real life example  taken from : https://bdwilliamson.github.io/vimp/articles/introduction_to_vimp.html 
#get sourth african heart disease data:
import pandas as pd
import numpy as np
data = pd.read_csv("http://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data")
#%% data cleaning:
heart = data.iloc[0:data.shape[0] , 1:data.shape[1]]
#%%
heart['famhist'] = np.where(heart['famhist'] == 'Present',1, 0 )
#%%folds (not quite sure what this means, but I think it is the subsamples of the data)
# np.random.seed(12345)
# heart_folds = np.random.choice(a=np.arange(2), size= heart.shape[0], replace=True, p=np.array([0.5, 0.5]))
#TODO:DONE this is not working!!!it does not give equal umbers of 1 and 0... so I tried this:
import random
random.seed(12345)
percent=int(0.5*len(heart))
nums=percent*[1]+(len(heart)-percent)*[0]
random.shuffle(nums)
heart_folds=np.array(nums)
#%%
# In addition to the indicator of myocardial infarction chd, the outcome of interest, there are measurements on two groups of variables. 
# First are behavioral features: cumulative tobacco consumption, current alcohol consumption, and type A behavior (a behavioral pattern linked to stress). 
# Second are biological features: systolic blood pressure, low-density lipoprotein (LDL) cholesterol, adiposity (similar to body mass index), family history of heart disease, obesity, and age.

# Since there are nine features and two groups, it is of interest to determine variable importance both for the nine individual 
# features separately and for the two groups of features.

#%% linear regression
import statsmodels.formula.api as smf
import statsmodels.api as sm
#lm = smf.ols('chd ~ ') 
X= heart.drop(columns='chd')
X = X.to_numpy()
y=np.array(heart.chd)
#%%
#full_model = sm.OLS(heart['chd'], X).fit()
#attempt with sklearn linear regression
from sklearn.linear_model import LogisticRegressionCV
full_model= LogisticRegressionCV(cv=5, random_state=0)


#%%
full_fit = full_model.predict(X)
#TODO: The above is a bit strange, Why would you predict using the same data?? no train/test.. maybe that is the statistical way of doing it!
#%% estimate the reduced conditional means for each of the individual variables:
#by removing each feature individually:
# remove the outcome for the predictor matrix:
#new_X = heart.drop(columns=['chd']).iloc[heart_folds==2, :]
X=X.iloc[heart_folds ==0, :] #Here in the example it is heart folds ==2 , but it is a bit weird
#
#%%
red_mob_sbp = sm.OLS(full_fit, X.drop(columns=['sbp'])).fit()
red_fit_sbp =red_mob_sbp.predict(X.drop(columns=['sbp']))

#
red_mob_tob = sm.OLS(full_fit, X.drop(columns=['tobacco'])).fit()
red_fit_tob =red_mob_tob.predict(X.drop(columns=['tobacco']))
#
red_mob_ldl = sm.OLS(full_fit, X.drop(columns=['ldl'])).fit()
red_fit_ldl =red_mob_ldl.predict(X.drop(columns=['ldl']))
#
red_mob_adi = sm.OLS(full_fit, X.drop(columns=['adiposity'])).fit()
red_fit_adi =red_mob_adi.predict(X.drop(columns=['adiposity']))
#
red_mob_fam = sm.OLS(full_fit, X.drop(columns=['famhist'])).fit()
red_fit_fam =red_mob_fam.predict(X.drop(columns=['famhist']))
#
red_mob_tpa = sm.OLS(full_fit, X.drop(columns=['typea'])).fit()
red_fit_tpa =red_mob_tpa.predict(X.drop(columns=['typea']))
#
red_mob_obe = sm.OLS(full_fit, X.drop(columns=['obesity'])).fit()
red_fit_obe =red_mob_obe.predict(X.drop(columns=['obesity']))
#
red_mob_alc = sm.OLS(full_fit, X.drop(columns=['alcohol'])).fit()
red_fit_alc =red_mob_alc.predict(X.drop(columns=['alcohol']))
#
red_mob_age = sm.OLS(full_fit, X.drop(columns=['age'])).fit()
red_fit_age =red_mob_age.predict(X.drop(columns=['age']))
#%% import libraries
import vimpy
#%% Plug into vim:
#first inspect the function arguments in python:
import inspect
print(inspect.getargspec(vimpy.vim))
#%%
## plug these into vim
lm_vim_sbp = vimpy.vim(y = heart.chd, x = X , s = 0, 
pred_func=full_fit, measure_type = "r_squared")
#
lm_vim_tob = vimpy.vim(y = heart.chd, x = X , s = 1, 
pred_func=full_fit, measure_type = "r_squared")
#
lm_vim_ldl = vimpy.vim(y = heart.chd, x = X , s = 2, 
pred_func=full_fit, measure_type = "r_squared")
#
lm_vim_adi = vimpy.vim(y = heart.chd, x = X , s = 3, 
pred_func=full_fit, measure_type = "r_squared")
#
lm_vim_fam = vimpy.vim(y = heart.chd, x = X , s = 4, 
pred_func=full_fit, measure_type = "r_squared")
#
lm_vim_tpa = vimpy.vim(y = heart.chd, x = X , s = 5, 
pred_func=full_fit, measure_type = "r_squared")
#
lm_vim_obe= vimpy.vim(y = heart.chd, x = X , s = 6, 
pred_func=full_fit, measure_type = "r_squared")
#
lm_vim_alc = vimpy.vim(y = heart.chd, x = X , s = 7, 
pred_func=full_fit, measure_type = "r_squared")
#%%
lm_vim_age = vimpy.vim(y = y, x = X , s = 8, 
pred_func=full_model, measure_type = "r_squared")

#%% Make table with results:

lm_vim_age.get_point_est()
lm_vim_age.get_influence_function()
lm_vim_age.get_se()
lm_vim_age.get_ci()
#do a hypothesis test, compute p-value
lm_vim_age.hypothesis_test(alpha=0.05, delta=0)
#%%how to display the estimates:
lm_vim_age.vimp_
#%%
lm_vim_age.se_
#%%
lm_vim_age.ci_
#%%
lm_vim_age.p_value_

#%%
