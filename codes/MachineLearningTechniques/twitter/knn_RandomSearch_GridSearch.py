# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 19:28:13 2016

@author: tasos

K Neirest Neighbors run Random Search and Grid Search
"""

print(__doc__)

import numpy as np
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score



# Utility function to report best scores
def report(y_pred, grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))

        #include all possible scores for classification
        print("---------------------------------------")    
        
        print("precision score, micro : ",precision_score(y_test, y_pred, average='micro'))
        print("precision score, macro : ",precision_score(y_test, y_pred, average='macro'))
        print("precision score, weighted : ",precision_score(y_test, y_pred, average='weighted'))
    
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

        print ("---------------------------------------")       
        print ("mean accuracy: ", accuracy_score(y_test, y_pred))		
              
        print("Parameters: {0}".format(score.parameters))
        print("")



# get some data
X = pd.read_csv('Datasets/thesis_twitter_data_v20.csv',header=None)

#print X.head(22)
X.columns = ['created_at', 'tweet_text', 'retweet_count', 'favorite_count', 'tweet_id', 'num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score', 'computer_score', 'computer_score_imdb', 'computer_score_pos_neg', 'computer_score_senti_word_net', 'computer_score_subjectivity', 'computer_score_amazon_tripadvisor', 'computer_score_goodreads', 'computer_score_opentable', 'computer_score_inquirer']

#print X.head(12)

X = X.dropna()

#y = X['computer_score']
#y = X['computer_score_imdb']
#y = X['computer_score_pos_neg']
#y = X['computer_score_senti_word_net']
#y = X['computer_score_subjectivity']
#y = X['computer_score_amazon_tripadvisor']
#y = X['computer_score_goodreads']
#y = X['computer_score_opentable']
y = X['computer_score_inquirer']

#print sorted(y.unique())
y = y.round()
y = y.astype(int)

X = X.drop('computer_score',1)
X = X.drop('computer_score_imdb',1)
X = X.drop('computer_score_pos_neg',1)
X = X.drop('computer_score_senti_word_net',1)
X = X.drop('computer_score_subjectivity',1)
X = X.drop('computer_score_amazon_tripadvisor',1)
X = X.drop('computer_score_goodreads',1)
X = X.drop('computer_score_opentable',1)
X = X.drop('computer_score_inquirer',1)

#drop categorical data
X = X.drop(X.columns[[0, 1, 4]], axis=1)

#print '-----------------'
#print X.head(2)
X = X.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)


# build a classifier
clf = KNeighborsClassifier()


print("---------------------------------------------")
print("metrics: euclidean, manhatan, chebyshev")
# specify parameters and distributions to sample from
param_dist = {
              "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],            
              "weights": ["uniform", "distance"],
              "n_neighbors" : sp_randint(1, 15),
              "leaf_size" :  sp_randint(15, 60),
              "metric": ["euclidean","manhattan", "chebyshev"]
              }

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
y_pred_knn = random_search.fit(X_train, y_train).predict(X_test)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(y_pred_knn, random_search.grid_scores_)



print("---------------------------------------------")
print("metrics: euclidean, manhatan, chebyshev")
# use a full grid over all parameters
param_grid = {
              "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],
              "weights": ["uniform", "distance"],
              "n_neighbors": np.arange( 1, 30, 1 ).tolist(),
              "leaf_size": np.arange( 5, 30, 1 ).tolist(),
              "metric": ["euclidean","manhattan", "chebyshev"]
             }
             
clf.get_params().keys()
# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)
y_pred_knn = grid_search.fit(X_train, y_train).predict(X_test)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(y_pred_knn, grid_search.grid_scores_)



print("------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------")
print("metric is minkowski")
# use a full grid for the metrics that have arguments 
param_grid2 = {
              "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],
              "weights": ["uniform", "distance"],
              "n_neighbors": np.arange( 1, 30, 1 ).tolist(),
              "leaf_size": np.arange( 5, 30, 1 ).tolist(),
              "metric": ["minkowski"],
              'metric_params':[                                
                                {'p':1.5},
                                {'p':2.5},
                                {'p':3.5},
                                {'p':3.0}
                               ] 
             }
             
# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid2)
start = time()
grid_search.fit(X, y)
y_pred_knn = grid_search.fit(X_train, y_train).predict(X_test)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(y_pred_knn, grid_search.grid_scores_)
