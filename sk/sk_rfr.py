
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from time import time as time
import sqlite3
import numpy as np


database = sqlite3.connect("gg_shit.db")

def _execute(command, params=(), commit=True):
    cursor = database.cursor()
    cursor.execute(command, params)
    data = [el for el in cursor]
    if commit: database.commit()
    return data

coefs_db = [obj[0] for obj in _execute("SELECT crash FROM history")]
coefs_db = (coefs_db[:149964])
coefs = np.array(coefs_db).reshape(24994,6)
y = coefs[:, 5]
X = coefs[:, :5]

clf = GradientBoostingRegressor(random_seed = 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42);

def score_func(y, y_pred):
    score = 0
    counter = 0
    for i in range(len(y)):
        if y_pred[i] - y[i] <= 0:
            score+=1
        counter +=1
    return score/counter

best_score = make_scorer(score_func)

param = np.arange(1,number,10)
param_grid2 = {"n_estimators": [2,4,6,10,14,20],
              "max_depth": [2,3,5,9,14,20,25, 28],
              "min_samples_split": [2,3,4,9,13,15],
              "min_samples_leaf": [2,3,4,9,13,17,20,25,30],
              "max_leaf_nodes": [2,4,6,10,15,20,25,30],
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}
grid_search = GridSearchCV(clf, param_grid2, scoring= best_score, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

grid_search.best_score_
