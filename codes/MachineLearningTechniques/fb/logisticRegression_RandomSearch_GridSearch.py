# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:20:45 2016

@author: tasos
"""

print(__doc__)
from sklearn import linear_model
import numpy as np
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import pandas as pd
from sklearn import cross_validation as cval

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


def report2(grid_scores):
    
    print( grid_scores)
    
# Utility function to report best scores
def report(y_pred, grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=lambda x: (abs(x[1])), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        #If None, the score method of the estimator is used
        #Returns the coefficient of determination R^2 of the prediction.
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
    
        print("Fold cross scor: ", cval.cross_val_score(clf, X_train, y_train, cv=10).mean())

        print("Explained variance regression score: ", explained_variance_score(y_test, y_pred))
        print("Explained variance regression score, raw_values : ", explained_variance_score(y_test, y_pred, multioutput='raw_values'))
        print("Explained variance regression score, uniform_average : ", explained_variance_score(y_test, y_pred, multioutput='uniform_average'))
        print("Explained variance regression score, variance_weighted : ", explained_variance_score(y_test, y_pred, multioutput='variance_weighted'))
    
        print("---------------------------------------")  
        print("Mean absolute error : ", mean_absolute_error(y_test, y_pred)    )
        print("Mean absolute error, multioutput=raw_values : ", mean_absolute_error(y_test, y_pred, multioutput='raw_values'))
        print("Mean absolute error, multioutput=uniform_average : ", mean_absolute_error(y_test, y_pred, multioutput='uniform_average'))    
        
        #the best value is 0.0
        print("---------------------------------------")      
        print("Mean squared error : ", mean_squared_error(y_test, y_pred)  )
    
        #the best value is 0.0
        print("---------------------------------------")      
        print("Median absolute error : ", median_absolute_error(y_test, y_pred))
    
        #the best value is 1.0, can be negative 
        print("---------------------------------------")      
        print("R2 score : ", r2_score(y_test, y_pred) )
        print("R2 score, variance weighted : ", r2_score(y_test, y_pred, multioutput='variance_weighted') )
        print("R2 score, uniform average : ", r2_score(y_test, y_pred, multioutput='uniform_average') )
        print("R2 score, raw values : ", r2_score(y_test, y_pred, multioutput='raw_values') )
    
        print("Parameters: {0}".format(score.parameters))
        print("")

# read data
X = pd.read_csv('Datasets/thesis_facebook_data_v23.csv',header=None)

X.columns = ['status_id', 'status_message', 'link_name', 'status_type', 'status_link', 'status_published', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score', 'computer_score', 'computer_score_imdb', 'computer_score_pos_neg', 'computer_score_senti_word_net', 'computer_score_subjective', 'computer_score_amazon_tripadvisor', 'computer_score_goodreads', 'computer_score_opentable', 'computer_score_wnscore_inquirer' ]
X.status_type = X.status_type.map({'link':1, 'photo':2, 'status':3, 'video':4 })

# drop possible missings 
X = X.dropna()
#print (X.head(5))
#y = X['computer_score']
#y = X['computer_score_imdb']
#y = X['computer_score_pos_neg']
#y = X['computer_score_senti_word_net']
#y = X['computer_score_subjective']
#y = X['computer_score_amazon_tripadvisor']
#y = X['computer_score_goodreads']
#y = X['computer_score_opentable']
y = X['computer_score_wnscore_inquirer']

#print y
#raw_input()

#print (y.head(5))
#y = y*10
y = y.round()
y = y.astype(int)
#print y

# drop y from X
X = X.drop('computer_score',1)
X = X.drop('computer_score_imdb',1)
X = X.drop('computer_score_pos_neg',1)
X = X.drop('computer_score_senti_word_net',1)
X = X.drop('computer_score_subjective',1)
X = X.drop('computer_score_amazon_tripadvisor',1)
X = X.drop('computer_score_goodreads',1)
X = X.drop('computer_score_opentable',1)
X = X.drop('computer_score_wnscore_inquirer',1)

# drop categorical data
X = X.drop(['status_id', 'status_message', 'link_name', 'status_link', 'status_published'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

# build a classifier
clf = linear_model.LogisticRegression()

y_pred_regression_ordinary = clf.fit(X_train, y_train).predict(X_test)

# specify parameters and distributions to sample from
param_dist = {
               "C":  [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5 ],     
#               "fit_intercept":  ['True', 'False'],
               "solver":  ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
             }


# run randomized search
n_iter_search = 7
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
      
report(y_pred_regression_ordinary, random_search.grid_scores_)


print("---------------------------------------------")
print("---------------------------------------------")

# use a full grid over all parameters
param_grid = {
               "C":  [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5 ],     
 #              "fit_intercept":  ['True', 'False'],
               "solver":  ['newton-cg', 'lbfgs', 'liblinear', 'sag'] ,
             }

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(y_pred_regression_ordinary, grid_search.grid_scores_)

#report2(grid_search.grid_scores_)