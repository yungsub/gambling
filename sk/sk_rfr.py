#!/usr/bin/env python
# coding: utf-8

# In[170]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
import pandas as pd
from time import time as time
import sqlite3
import numpy as np


# In[72]:


database = sqlite3.connect("gg_shit.db")

def _execute(command, params=(), commit=True):
    cursor = database.cursor()
    cursor.execute(command, params)
    data = [el for el in cursor]
    if commit: database.commit()
    return data

coefs_db = [obj[0] for obj in _execute("SELECT crash FROM history")]


# In[73]:


coefs_db = (coefs_db[:149964])


# In[74]:


coefs = np.array(coefs_db).reshape(24994,6)


# In[76]:


y = coefs[:, 5]


# In[77]:


X = coefs[:, :5]


# In[79]:


clf = GradientBoostingRegressor(random_seed = 4)


# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42);


# In[162]:


def score_func(y, y_pred):
    score = 0
    counter = 0
    for i in range(len(y)):
        if y_pred[i] - y[i] <= 0:
            score+=1
        counter +=1
    return score/counter


# In[163]:


best_score = make_scorer(score_func)


# In[186]:


param = np.arange(1,number,10)
param_grid2 = {"n_estimators": np.arange(2, 20, 2),
              "max_depth": np.arange(1, 28, 1),
              "min_samples_split": np.arange(2,15,1),
              "min_samples_leaf": np.arange(1,30,1),
              "max_leaf_nodes": np.arange(2,30,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}
grid_search = GridSearchCV(clf, param_grid2, scoring= best_score, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_score_

