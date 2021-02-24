#!/usr/bin/env python
# coding: utf-8

# In[1]:


# UROP Phase2_1 Model selection and evaluation - Ridge Regression
# preprocessing: standardized all the features
# model selected: ridge regressor
# hyperparameter tuning: ridge regressor regularization strength
# evaluation metrics: r2 score and MSE in ten-fold crossvalidation
# other metrics: learning curve and prediction graph

# loading data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
d = np.load("heusler_mechanical.npz",allow_pickle = True)


# In[2]:


# splitting data
x = pd.DataFrame(d["X"],columns = d["X_column_label"]) 
y = pd.DataFrame(d["y"])
y = np.ravel(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)


# In[3]:


# pipeline and model training
# hyperparameter tuning using gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

pipe_rg_kpca = make_pipeline(StandardScaler(),
                       Ridge(random_state = 1))

param_grid = [{"ridge__alpha":[0,0.1,1,10,12,14,16,18,20]}]
gs_rg_kpca = GridSearchCV(estimator = pipe_rg_kpca,
                    param_grid = param_grid,
                    scoring = "r2",
                     cv = 10,
                     refit = True,
                     n_jobs = -1
                    )
gs_rg_kpca = gs_rg_kpca.fit(x_train,y_train)
print(gs_rg_kpca.best_score_)
print(gs_rg_kpca.best_params_)
sgd = gs_rg_kpca.best_estimator_


# In[4]:


# evaluate the best estimator by plotting the learning curve
# this will show if the model is over-fitting or under-fitting
# and will show if more data is necessary to improve this model

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores =                learning_curve(estimator = sgd,
                              X = x_train,
                              y = y_train,
                              train_sizes = np.linspace(
                              0.1,1.0,10),
                              cv = 10,
                              scoring = "r2",
                              n_jobs = -1)

train_mean = np.mean(train_scores,axis = 1)
train_std = np.std(train_scores,axis = 1)
test_mean = np.mean(test_scores,axis = 1)
test_std = np.std(test_scores,axis = 1)

plt.plot(train_sizes,train_mean,
        color = "blue", marker = "o",
        markersize = 5, label = "Training r2 scores")

plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha = 0.15,color = "blue")

plt.plot(train_sizes,test_mean,
        color = "green", linestyle = "--",
        marker = "s", markersize = 5,
        label = "Validation r2 scores")

plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha = 0.15,color = "green")
plt.grid()
plt.ylabel("r2 score")
plt.legend(loc = "lower right")
plt.ylim([0.3,1.03])
plt.show()


# In[5]:


# evaluation using crossvalidation

from sklearn.model_selection import KFold, cross_val_score

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(sgd, x, y, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(sgd, x, y, scoring='r2', cv=crossvalidation, n_jobs=1)

print('Cross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(scores), np.mean(np.abs(r2_scores))))
print('Folds: %i, mean RMSE: %.3f' % (len(scores), np.mean(np.abs(rmse_scores))))


# In[6]:


for i in range(0,10):
    print(test_mean[i] - train_mean[i])


# In[7]:


# evaluation using prediction graph
# optimally all points should lie on the line
# this will show how much the prediction deviates from the labelled value

from matminer.figrecipes.plot import PlotlyFig
from sklearn.model_selection import cross_val_predict
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
pf = PlotlyFig(x_title='bulk modulus(GPa)',
               y_title='Predicted bulk modulus(GPa)',
               title='Ridge regression',
               mode='notebook',
               filename="Ridge_regression.html")
pf.xy(xy_pairs=[(y, cross_val_predict(sgd, x, y, cv=crossvalidation)), ([40, 300], [40, 300])],  
      modes=['markers', 'lines'],
      lines=[{}, {'color': 'black', 'dash': 'dash'}], 
      showlegends=False
     )


# In[ ]:




