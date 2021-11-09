#%% real life example  taken from : https://bdwilliamson.github.io/vimp/articles/introduction_to_vimp.html 
#get sourth african heart disease data:
#Important note: this code runs on a conda virtual environment where all packages were installed using conda except vimpy, which was installed with pip
from altair.vegalite.v4.schema.core import InlineData
from numpy.core.numeric import outer
import pandas as pd
import numpy as np
data = pd.read_csv("http://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data")
np.random.seed(12345)
#%% data cleaning:
heart = data.iloc[0:data.shape[0] , 1:data.shape[1]]
#%%
heart['famhist'] = np.where(heart['famhist'] == 'Present',1, 0 )

#%%
X= heart.drop(columns='chd')
X_=X.to_numpy()
y=heart.chd
y_=np.array(y)
#%%
heart.drop(columns=['chd'], inplace=True)
#%% Now test wit han xgboost classifier that I fine tunned on the data with xgboost_fine_tunning.py
import xgboost as xgb
from xgboost.sklearn import XGBClassifier  
from sklearn.metrics import accuracy_score, roc_auc_score , roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
#now test
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

#%% Now we import vimpy (here is the link to the R vignette: https://bdwilliamson.github.io/vimp/articles/introduction_to_vimp.html)
import vimpy
import warnings
warnings.filterwarnings('ignore')
#note: it is very interesting that that when I change the measure_type which corresponds to the type of importance to compute, from r_squared to auc, 
# the feature importance (vimp) goes down, at least in the case of the first feature). 
# TODO: I wonder if the order of the feature importance changes?

#%% define new, required functions: 
def create_vimpy(alg, X, y, s, measure='r_squared'):
    '''creates a vimpy object'''
    vim_object = vimpy.vim(y=y, x=X, s=s, pred_func=alg, 
    measure_type=measure)
    return vim_object

def get_output(vimpy_object):
    '''argument+ vimp object, gets the outputs of the vimp object'''
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

def get_results(vim_obj):
    '''this function gets the results of a vim object on which the outputs have been called
    it returns, vimp_, se_, ci_, p_value_'''
    return [vim_obj.vimp_, vim_obj.se_, vim_obj.ci_, vim_obj.p_value_]

#final function:
def get_vimpy(X, y, alg):
    ''' give it a pandas dataframe with all the features you need to know the importance of (X),
    the outcome (y) and the algorithm to be used and get a dataframe with feature importances
    requires the following functions:
    - creaty_vimpy
    - get_output
    - get_results'''
    #get a numpy array version of X and y:
    X_=X.to_numpy()
    y_=np.array(y)
    #create a vimpy object for all features in X:
    ls_obj=[]
    for i in range(X_.shape[1]):
        ls_obj.append(create_vimpy(alg, X_, y_, i))
    #now use get_output on the items of the generated list:
    for i in range(len(ls_obj)):
        get_output(ls_obj[i])
    #now get results in a dictionary:
    dic={'feature':[], 'vimp':[], 'se':[], 
    'ci':[], 'p-value':[] }
    for i in range(len(ls_obj)):
        dic['feature'].append(list(X.columns)[i])
        dic['vimp'].append(get_results(ls_obj[i])[0])
        dic['se'].append(get_results(ls_obj[i])[1])
        dic['ci'].append(get_results(ls_obj[i])[2])
        dic['p-value'].append(get_results(ls_obj[i])[3])
    #Store the results in a dataframe:
    df=pd.DataFrame(dic)
    df['2.5%'] = [df['ci'].iloc[r][0][0] for r in range(len(df))]
    df['97.5%'] = [df['ci'].iloc[r][0][1] for r in range(len(df))]
    df.drop(columns=['ci'], inplace=True)
    df=df.sort_values(by='vimp', ascending=False, key=abs)#sort by absolute value

    return df

#%% get vimpy for all the features :
df= get_vimpy(X, y, xgb_new)
df

#%% Compare with the old model!:
from sklearn.ensemble import GradientBoostingClassifier

ntrees = np.arange(100, 500, 100)
lr = np.arange(0.01, 0.1, 0.05)
param_grid = [{'n_estimators': ntrees, 'learning_rate': lr}]

cv_full = GridSearchCV(GradientBoostingClassifier(loss='deviance', max_depth= 1), param_grid=param_grid, cv = 5)
cv_small = GridSearchCV(GradientBoostingClassifier(loss='deviance', max_depth= 1), param_grid=param_grid, cv = 5)
#%% 
df_gb=get_vimpy(X, y, cv_full)
df_gb

#%% make forest plot to compare:
import altair as alt

def make_forest(source, name):
    '''Takes source df and converts into forest plot 
    with dignificant feature importances in red.
    name is a string!'''
    #preprocessing:
    ##remove spaces from column names:
    source.columns=[col.replace(" ", "") for col in source.columns]
    source['feature'] = source['feature'].astype('string')
    source[['vimp', 'p-value','2.5%', '97.5%']]=source[['vimp', 'p-value','2.5%', '97.5%']].astype('float64')

    #points
    points = alt.Chart(source).mark_point(filled=True).encode(
    x=alt.X('vimp:Q', title='VIMP'),
    y=alt.Y(
        'feature:N', axis=alt.Axis(grid=True)), color=alt.condition('datum.p-value<0.05', alt.ColorValue('red'), alt.ColorValue('black'))
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
    chart=points+errorbars+line
    chart.save(f'{name}.png')
    return chart
#%%
#TODO: very different results! humm, I have to check the original code to see why!
make_forest(df_gb, 'auto')
#%%Make forest plot




#TODO: need to make the order of the Y axis be the order of the df!!!
#TODO: save the plot as png or svg? 

#%%
make_forest(df, 'auto_xgboost' )


#%%
############### ----
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
print(f'vimp: {vimp_precompute_sbp.vimp_}\nse: {vimp_precompute_sbp.se_}\nci: {vimp_precompute_sbp.ci_}\np-value: {vimp_precompute_sbp.p_value_}\nhyp_test: {vimp_precompute_sbp.hyp_test_}')

#TODO: make function to do the spvim on all features and to get a table with the results.
#TODO: why is hypothesis test=False??
