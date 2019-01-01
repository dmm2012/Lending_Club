
# coding: utf-8

# In[1]:


# Set working directory

get_ipython().run_line_magic('cd', '"C:/Users/devan/Documents/Analytics/Analytics Applications/Python/Lending Club HW"')


# In[3]:


# Importing dependencies

import math 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

import itertools
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Importing the approved loan data from Lending Club

lending = pd.read_csv("data_YearsAdded_OutliersDropped.csv", sep = ",")
lending.head()


# In[19]:


# Creating new emp_length (time loan applicant has been at current job) column

lending["emp_length_new"] = np.where(lending['emp_length']=="10+ years", "10+ years",
                                     np.where(np.isin(lending['emp_length'], ["1 year",'2 years','3 years','4 years','5 years']), "1-5 years",
                                              np.where(np.isin(lending['emp_length'], ['6 years','7 years','8 years','9 years']), "6-9 years",
                                                       np.where(np.isin(lending['emp_length'], ["< 1 year"]), "<1 year", "n/a"))))

# Creating a flag for A and B loans 

lending["depAB"] = np.where(np.isin(lending['grade'], ["A", "B"]), 1 , 0)


# In[20]:


# Split dataframe into 2 dataframes by years

lending_14 = lending[lending['Year']== 2014.0]
lending_16_17 = lending[lending['Year']!= 2014.0]

# Making sure the years were split properly

print(lending.shape)
print(lending_14.shape)
print(lending_16_17.shape)


# ### Creating 2014 model

# In[21]:


# Feature selection - 2014

X = lending_14.loc[:, ['acc_open_past_24mths_x', 'dti_x', 'inq_last_6mths_x', 'mort_acc_x', 'num_accts_ever_120_pd_x', 'num_actv_bc_tl_x', 'num_bc_sats_x', 'open_acc_x', 'pub_rec_bankruptcies_x', 'pub_rec_x']]
y = lending_14['depAB']


# In[95]:


# Splitting training and test 

# X_train, y_train = X, y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# lasso with SGD Classifier

lasso_14 = linear_model.SGDClassifier(alpha= 0.1, l1_ratio= 1, penalty= 'l1', verbose= 10)
lasso_14.fit(X_train, y_train) # Training the model

# Ridge with SGD Classifier

ridge_14 = linear_model.SGDClassifier(penalty='l2', alpha=0.1, l1_ratio=0.0, verbose = 0)
ridge_14.fit(X_train, y_train) # Training the model

# # Elastic Net

elastic_net_14 =linear_model.SGDClassifier(penalty='l1', alpha=.1, l1_ratio=0.5, fit_intercept=True)
elastic_net_14.fit(X_train, y_train)


# ### Creating 2016 model

# In[23]:


# Feature selection - 2016/2017

X2 = lending_16_17.loc[:, ['acc_open_past_24mths_x', 'dti_x', 'inq_last_6mths_x', 'mort_acc_x', 'num_accts_ever_120_pd_x', 'num_actv_bc_tl_x', 'num_bc_sats_x', 'open_acc_x', 'pub_rec_bankruptcies_x', 'pub_rec_x']]
y2 = lending_16_17['depAB']


# In[84]:


# Splitting training and test 

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

# lasso with SGD Classifier

lasso_16 = linear_model.SGDClassifier(penalty='l1', alpha=0.1, l1_ratio=1, verbose=10)
lasso_16.fit(X2_train, y2_train) # Training the model

# Ridge with SGD Classifier

ridge_16 = linear_model.SGDClassifier(penalty='l2', alpha=0.1, l1_ratio=0.0, verbose = 0)
ridge_16.fit(X2_train, y2_train) # Training the model

# Elastic Net

elastic_net_16 =linear_model.SGDClassifier(penalty='l1', alpha=.1, l1_ratio=0.5)
elastic_net_16.fit(X2_train, y2_train)


# ### Testing pre model with pre IPO data

# In[96]:


# Printing accuracy of 2014 model with 2014 data

y_pred_lasso2 = lasso_14.predict(X_test)
print('Accuracy of lasso regression (pre IPO) classifier on test set: {:.2f}'.format(lasso_pre.score(X_test, y_test)))

y_pred_ridge2 = ridge_14.predict(X_test)
print('Accuracy of ridge regression (pre IPO) classifier on test set: {:.2f}'.format(ridge_pre.score(X_test, y_test)))

y_pred_EN2 = elastic_net_14.predict(X_test)
print('Accuracy of Elastic Net (pre IPO) classifier on test set: {:.2f}'.format(elastic_net_pre.score(X_test, y_test)))


# ### Testing pre and post models with post IPO data

# In[91]:


# Printing accuracy for 2014 model with 2016/2017 data

y_pred_lasso = lasso_14.predict(X2_test)
print('Accuracy of lasso regression (pre IPO) classifier on test set: {:.2f}'.format(lasso_14.score(X2_test, y2_test)))

y_pred_ridge = ridge_14.predict(X2_test)
print('Accuracy of ridge regression (pre IPO) classifier on test set: {:.2f}'.format(ridge_14.score(X2_test, y2_test)))

y_pred_EN = elastic_net_14.predict(X2_test)
print('Accuracy of Elastic Net (pre IPO) classifier on test set: {:.2f}'.format(elastic_net_14.score(X2_test, y2_test)))


# In[92]:


# Printing classification reports for 2014 model with 2016/2017 data

print(classification_report(y2_test, y_pred_lasso))
print(classification_report(y2_test, y_pred_ridge))
print(classification_report(y2_test, y_pred_EN))


# In[93]:


# Printing accuracy of 2016 models with 2016/2017 data

y2_pred_lasso = lasso_16.predict(X2_test)
print('Accuracy of lasso regression (post IPO) classifier on test set: {:.2f}'.format(lasso_16.score(X2_test, y2_test)))

y2_pred_ridge = ridge_16.predict(X2_test)
print('Accuracy of ridge regression (post IPO) classifier on test set: {:.2f}'.format(ridge_16.score(X2_test, y2_test)))

y2_pred_EN = elastic_net_16.predict(X2_test)
print('Accuracy of Elastic Net (post IPO) classifier on test set: {:.2f}'.format(elastic_net_16.score(X2_test, y2_test)))


# In[94]:


# Printing classification reports for 2016 models with 2016/2017 data

print(classification_report(y2_test, y2_pred_lasso))
print(classification_report(y2_test, y2_pred_ridge))
print(classification_report(y2_test, y2_pred_EN))


# ### Running grid search to select the best parameters (for all models)

# In[ ]:


# # Grid Search for Lasso (2014)

# lsso = linear_model.SGDClassifier(random_state=42, n_jobs=-1)
# param_grid = { 
#     'penalty': ['l1'],
#     'alpha': [0.1, 0.3, 0.5, 0.7],
#     'l1_ratio' :  [.1, .5, .7, 1.0],
#     'verbose' :[0, 2, 4, 6, 10]
# }
# CV_lsso = GridSearchCV(estimator=lsso, param_grid=param_grid, cv= 5)
# CV_lsso.fit(X_train, y_train)


# #print out the parameters
# CV_lsso.best_params_


# Grid Search for Lasso (2016)

# lsso_post = linear_model.SGDClassifier(random_state=42, n_jobs=-1)
# param_grid = { 
#     'penalty': ['l1'],
#     'alpha': [0.1, 0.3, 0.5, 0.7],
#     'l1_ratio' : [.1, .5, .7, 1.0],
#     'verbose' :[0, 2, 4, 6, 10]
# }
# CV_lsso_post = GridSearchCV(estimator=lsso_post, param_grid=param_grid, cv= 5)
# CV_lsso_post.fit(X2_train, y2_train)


# #print out the parameters
# CV_lsso_post.best_params_


# In[1]:


# # Grid Search for Ridge (2014)

# rdg = linear_model.SGDClassifier(random_state=42, n_jobs=-1)
# param_grid = { 
#     'penalty': ['l2'],
#     'alpha': [0.1, 0.3, 0.5, 0.7],
#     'l1_ratio' : [0, .1, .5, .7, 1.0],
#     'verbose' :[0, 2, 4, 6, 10]
# }
# CV_rdg = GridSearchCV(estimator=rdg, param_grid=param_grid, cv= 5)
# CV_rdg.fit(X_train, y_train)


# #print out the parameters
# CV_rdg.best_params_

# # Grid Search for Ridge (2016)

# rdg_post = linear_model.SGDClassifier(random_state=42, n_jobs=-1)
# param_grid = { 
#     'penalty': ['l2'],
#     'alpha': [0.1, 0.3, 0.5, 0.7],
#     'l1_ratio' : [0, .1, .5, .7, 1.0],
#     'verbose' :[0, 2, 4, 6, 10]
# }
# CV_rdg_post = GridSearchCV(estimator=rdg_post, param_grid=param_grid, cv= 5)
# CV_rdg_post.fit(X2_train, y2_train)


# #print out the parameters
# CV_rdg_post.best_params_


# In[ ]:


# # Grid Search for Elastic Net (2014)

# EN = linear_model.SGDClassifier(random_state=42, n_jobs=-1)
# param_grid = { 
#     'penalty': ['l1'],
#     'alpha': [0.1, 0.3, 0.5, 0.7],
#     'l1_ratio' : [0.5],
#     'verbose' :[0, 2, 4, 6, 10]
# }
# CV_EN = GridSearchCV(estimator=EN, param_grid=param_grid, cv= 5)
# CV_EN.fit(X_train, y_train)


# #print out the parameters
# CV_EN.best_params_


# # Grid Search for Elastic Net (2016)

# EN_post = linear_model.SGDClassifier(random_state=42, n_jobs=-1)
# param_grid = { 
#     'penalty': ['l1'],
#     'alpha': [0.1, 0.3, 0.5, 0.7],
#     'l1_ratio' : [0.5],
#     'verbose' :[0, 2, 4, 6, 10]
# }
# CV_EN_post = GridSearchCV(estimator=EN_post, param_grid=param_grid, cv= 5)
# CV_EN_post.fit(X2_train, y2_train)


# #print out the parameters
# CV_EN_post.best_params_

