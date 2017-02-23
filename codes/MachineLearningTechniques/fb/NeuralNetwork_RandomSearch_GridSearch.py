# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 17:13:20 2016

@author: tasos

Random Forest run Random Search and Grid Search
"""

print(__doc__)

import numpy as np
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import zero_one_loss
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
import random


# Utility function to report best scores
def report(y_pred, grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
              
        print("---------------------------------------")    
        print(classification_report(y_test, y_pred))
    
        print("---------------------------------------")   
        print("f1 score, weighted : ",f1_score(y_test, y_pred, average='weighted'))
        print("f1 score, macro : ",f1_score(y_test, y_pred, average='macro'))
        print("f1 score, micro : ",f1_score(y_test, y_pred, average='micro'))
    
        print("---------------------------------------")   
        print("recall score, weighted : ",recall_score(y_test, y_pred, average='weighted'))
        print("recall score, macro : ",recall_score(y_test, y_pred, average='macro'))
        print("recall score, micro : ",recall_score(y_test, y_pred, average='micro'))    
    
        #parameter beta balances between precision and recall
        print("---------------------------------------")       
        print("fbeta score, micro : ", fbeta_score(y_test, y_pred, average='micro', beta=0.5))
        print("fbeta score, macro : ", fbeta_score(y_test, y_pred, average='macro', beta=0.5))
        print("fbeta score, weighted : ", fbeta_score(y_test, y_pred, average='weighted', beta=0.5))
            
        print("---------------------------------------")       
        print("hamming loss : ", hamming_loss(y_test, y_pred))
    
        print("---------------------------------------")       
        print("jaccard similarity, normalized: ", jaccard_similarity_score(y_test, y_pred))
    
        print("---------------------------------------")       
        print("zero one loss: ", zero_one_loss(y_test, y_pred))
              
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


# build a classifier
clf = MLPClassifier()


#generate random tuples for the field hidden_layer_sizes
mylist = [ ( random.randint(0, 100), random.randint(0, 100) ) for k in range(100) ]

# specify parameters and distributions to sample from
param_dist = {
                  "hidden_layer_sizes" : mylist,
                  "activation" : ["identity", "logistic", "tanh", "relu"],
                  "solver" : ["lbfgs", "sgd", "adam"],
                  "alpha" : [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1 ],
                  "batch_size" : ["auto", 100, 125, 150, 200, 250, 300, 350],
                  "learning_rate" : ["constant", "invscaling", "adaptive"],
                  "max_iter" : [100, 150, 200, 250, 300],
                  "shuffle" : [True, False],
                  "tol" : [0.00001, 0.0001, 0.001, 0.01, 0.1 ],
                  "learning_rate_init" : [0.0001, 0.001, 0.01, 0.1 ],
                  "power_t" : [0.25, 0.5, 0.75, 1, 1.25, 1.5],
                  "momentum" : [0.1, 0.3, 0.5, 0.7, 0.9 ],
                  "nesterovs_momentum": [True, False],
                  "early_stopping" : [True, False],
                  "validation_fraction" : [0.1, 0.3, 0.5, 0.7, 0.9 ],
                  "beta_1" : [0.1, 0.3, 0.5, 0.7, 0.9 ],
                  "beta_2" : [0.1, 0.3, 0.5, 0.7, 0.9, 0.999 ],
                  "epsilon" : [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
              }

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
y_pred_dt = random_search.fit(X_train, y_train).predict(X_test)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(y_pred_dt, random_search.grid_scores_)

print("---------------------------------------------")
print("---------------------------------------------")
print("---------------------------------------------")

# run randomized search
n_iter_search = 2000
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
y_pred_dt = random_search.fit(X_train, y_train).predict(X_test)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(y_pred_dt, random_search.grid_scores_)

'''
print("---------------------------------------------")
# use a full grid over all parameters
param_grid = {

                  "hidden_layer_sizes" : mylist,
                  "activation" : ["identity", "logistic", "tanh", "relu"],
                  "solver" : ["lbfgs", "sgd", "adam"],
                  "alpha" : [0.0000001, 0.00001, 0.0001, 0.001, 0.01, 0.1 ],
                  "batch_size" : ["auto", 100, 125, 150, 200, 250, 300, 350],
                  "learning_rate" : ["constant", "invscaling", "adaptive"],
                  "max_iter" : [100, 150, 200, 250, 300],
                  "shuffle" : [True, False],
                  "tol" : [0.00001, 0.0001, 0.001, 0.01, 0.1 ],
                  "learning_rate_init" : [0.0001, 0.001, 0.01, 0.1 ],
                  "power_t" : [0.25, 0.5, 0.75, 1, 1.25, 1.5],
                  "momentum" : [0.1, 0.3, 0.5, 0.7, 0.9 ],
                  "nesterovs_momentum": [True, False],
                  "early_stopping" : [True, False],
                  "validation_fraction" : [0.1, 0.3, 0.5, 0.7, 0.9 ],
                  "beta_1" : [0.1, 0.3, 0.5, 0.7, 0.9 ],
                  "beta_2" : [0.1, 0.3, 0.5, 0.7, 0.9, 0.999 ],
                  "epsilon" : [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

              
              }

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)
y_pred_dt = grid_search.fit(X_train, y_train).predict(X_test)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(y_pred_dt, grid_search.grid_scores_)
'''