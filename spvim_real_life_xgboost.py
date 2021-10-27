#%% real life example  taken from : https://bdwilliamson.github.io/vimp/articles/introduction_to_vimp.html 
#get sourth african heart disease data:
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
#Here might want to try the gradientboosting classifer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

ntrees = np.arange(100, 500, 100)
lr = np.arange(0.01, 0.1, 0.05)
param_grid = [{'n_estimators': ntrees, 'learning_rate': lr}]

#set up cv objects
cv_full = GridSearchCV(GradientBoostingClassifier(loss='deviance', max_depth= 1), param_grid=param_grid, cv = 5)
cv_small = GridSearchCV(GradientBoostingClassifier(loss='ls', max_depth = 1), param_grid=param_grid, cv= 5)

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
#%%Make forest plot
import altair as alt


