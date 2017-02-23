# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:30:56 2016

@author: tasos
"""
import pandas as pd
import time
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import cross_validation as cval
from sklearn import linear_model
from sklearn import grid_search
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import fbeta_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import average_precision_score
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
#from sklearn.neural_network import MLPClassifier

iterations = 1
debug_variable = True

def benchmark(model, cm, y_pred, wintitle='Figure 1'):
    print '\n\n' + wintitle + ' Results'
    s = time.time()
    for i in range(iterations):
        model.fit(X_train, y_train) 
    print "{0} Iterations Training Time: ".format(iterations), time.time() - s

    s = time.time()
    for i in range(iterations):
       score =model.score(X_test, y_test)
        
    print "{0} Iterations Scoring Time: ".format(iterations), time.time() - s
    print "Score is:", score
    fold_cross_scor = cval.cross_val_score(model, X_train, y_train, cv=10).mean()
    print "Fold cross scor: ", fold_cross_scor
    print "High-Dimensionality Score: ", round((score*100), 3)
    print("---------------------------------------")    
    '''
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
    '''
    '''
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    '''

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(y))
    plt.xticks(tick_marks, y, rotation=45)
    plt.yticks(tick_marks, y)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# get some data
X = pd.read_csv('Datasets/thesis_twitter_data_v21.csv',header=None)

#print X.head(22)
X.columns = ['created_at', 'tweet_text', 'retweet_count', 'favorite_count', 'tweet_id', 'num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score', 'computer_score', 'computer_score_imdb', 'computer_score_pos_neg', 'computer_score_senti_word_net', 'computer_score_subjectivity', 'computer_score_amazon_tripadvisor', 'computer_score_goodreads', 'computer_score_opentable', 'computer_score_inquirer']
#X.columns = ['created_at', 'tweet_text', 'retweet_count', 'favorite_count', 'tweet_id', 'num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score', 'computer_score', 'computer_score_imdb', 'computer_score_pos_neg', 'computer_score_senti_word_net']
#print X.head(12)

X = X.dropna()

X = X.dropna()

if debug_variable == False: 
	print ('Please insert number for lexicon you want to use: ')
	print ('1 - AFINN ')
	print ('2 - imdb ')
	print ('3 - Opinion Observer ')
	print ('4 - SentiWordNet ')
	print ('5 - Subjectivity: ')
	print ('6 - Amazon/TripAdvisor ')
	print ('7 - Goodreads ')
	print ('8 - Opentable ')
	print ('9 - inquirer ')
	
	user_choice = raw_input('')
	
	if user_choice == '1':
		print ('You chosed AFINN')
		y = X['computer_score']
	
	if user_choice == '2':
		print ('You chosed imdb')
		y = X['computer_score_imdb']
	
	if user_choice == '3':
		print ('You chosed Opinion Observer')
		y = X['computer_score_pos_neg']
	
	if user_choice == '4':
		print ('You chosed SentiWordNet')
		y = X['computer_score_senti_word_net']
	
	if user_choice == '5':
		print ('You chosed Subjectivity')
		y = X['computer_score_subjective']
	
	if user_choice == '6':
		print ('You chosed Amazon/TripAdvisor')
		y = X['computer_score_amazon_tripadvisor']

	if user_choice == '7':
		print ('You chosed Goodreads')
		y = X['computer_score_goodreads']
	
	if user_choice == '8':
		print ('You chosed Opentable')
		y = X['computer_score_opentable']

	if user_choice == '9':
		print ('You chosed inquirer')
		y = X['computer_score_wnscore_inquirer']


if debug_variable == True: 
	#y = X['computer_score_imdb']
	#y = X['computer_score_pos_neg']
	y = X['computer_score_senti_word_net']
	#y = X['computer_score_subjective']
	#y = X['computer_score_amazon_tripadvisor']
	#y = X['computer_score_goodreads']
	#y = X['computer_score_opentable']
	#y = X['computer_score_wnscore_inquirer']
	

#print sorted(y.unique())
y = y.round()
y = y.astype(int)
#y = y*10

X = X.drop('computer_score',1)
X = X.drop('computer_score_imdb',1)
X = X.drop('computer_score_pos_neg',1)
X = X.drop('computer_score_senti_word_net',1)
X = X.drop('computer_score_subjectivity',1)
X = X.drop('computer_score_amazon_tripadvisor',1)
X = X.drop('computer_score_goodreads',1)
X = X.drop('computer_score_opentable',1)
X = X.drop('computer_score_inquirer',1)

#print list(X.columns.values)
#print list(X.columns.values)

#drop categorical data
X = X.drop(X.columns[[0, 1, 4]], axis=1)


if debug_variable == False: 	
	print('')
	print ('Do you want to drop POS? Insert Y or N')
	user_choice_drop = raw_input('')
	if user_choice_drop == 'y' or user_choice_drop == 'Y':
		X = X.drop(['num_cc', 'num_cd', 'nun_ex', 'num_fw', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nns', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score'], axis=1)
		X = X.drop(['num_dt','num_in', 'num_jj', 'num_nn','num_nnp', 'num_vb', 'num_prp'], axis=1)
		print 'dropped..'
	
# drop all possible combinations 
#X = X.drop(['num_cc', 'num_to', 'num_vbp', 'num_vbg', 'num_vbn', 'num_vbd', 'num_md', 'num_rp', 'num_wp', 'num_jjr', 'num_fw', 'num_jjs', 'num_sym', 'num_rbr', 'num_wdt', 'num_pos', 'num_rbs', 'num_nnps', 'nun_ex', 'num_pdt', 'num_ls', 'num_wp6', 'num_prp6', 'num_uh'], axis=1)												
#X = X.drop(['num_nns', 'num_cd', 'num_rb', 'num_vbz'], axis=1)
#X = X.drop([ 'num_cc', 'num_cd', 'nun_ex', 'num_fw', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nns', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb'], axis=1)
#X = X.drop(['human_score'])
#X = X.drop(['num_dt','num_in', 'num_jj', 'num_nn','num_nnp', 'num_vb', 'num_prp', 'num_wrb'], axis=1)
#X = X.drop([ 'retweet_count', 'favorite_count'], axis=1)
print X.head(2)

#print X.head(2)
#y.plot.hist(alpha=0.5, bins=12)
#print X[pd.isnull(X).any(axis=1)]


if debug_variable == False: 
	print('')
	print ('How to preprocess?')
	print ('1 - Do NOT preprocess')
	print ('2 - Normalize ')
	print ('3 - MaxAbsScaler ')
	print ('4 - MinMaxScaler ')
	print ('5 - StandardScaler ')
	print ('6 - RobustScaler ')
	
	user_choice_pre = raw_input('')

	if user_choice_pre == '1':
		print ('You chosed NOT to preprocess')
		
	if user_choice_pre == '2':
		print ('You chosed normalize')
		X = preprocessing.normalize(X)
	
	if user_choice_pre == '3':
		print ('You chosed MaxAbsScaler')
		X = preprocessing.MaxAbsScaler().fit_transform(X)
	
	if user_choice_pre == '4':
		print ('You chosed MinMaxScaler')
		X = preprocessing.MinMaxScaler().fit_transform(X)
	
	if user_choice_pre == '5':
		print ('You chosed StandardScaler')
		X = preprocessing.StandardScaler().fit_transform(X)
	
	if user_choice_pre == '6':
		print ('You chosed RobustScaler')
		X = preprocessing.RobustScaler().fit_transform(X)

		
#X = preprocessing.normalize(X)
#X = preprocessing.MaxAbsScaler().fit_transform(X)
#X = preprocessing.MinMaxScaler().fit_transform(X)
#X = preprocessing.StandardScaler().fit_transform(X)
#X = preprocessing.RobustScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)



if debug_variable == False: 
	print('')
	print ('For defaut parameters press D, for grid search parameters press G?')
	user_choice_parameters = raw_input('')

	if user_choice_parameters == 'd' or user_choice_parameters == 'D':
		knn = KNeighborsClassifier()
		tree_decision = tree.DecisionTreeClassifier()
		random_forest = RandomForestClassifier()
		log_regr = linear_model.LogisticRegression()
		svc = SVC()
		linear_svm = svm.LinearSVC()
		#clf = MLPClassifier()
	
	else:
		if user_choice == '1' :
			print ('You chosed AFINN')
			knn = KNeighborsClassifier(n_neighbors=12, metric = 'manhattan', weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')	
			tree_decision = tree.DecisionTreeClassifier(presort =  True, splitter= 'best', max_leaf_nodes= 3, min_samples_leaf = 1, max_features = 'sqrt', criterion = 'entropy', min_samples_split = 8, max_depth = 3 )
			random_forest = RandomForestClassifier(oob_score = True, max_leaf_nodes = None,  min_samples_leaf = 1, n_estimators = 15,  min_samples_split = 2, criterion = 'gini',  max_features = 4,  max_depth = None,  class_weight = 'balanced')								  	
			log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
			svc = SVC(kernel='rbf', C= 1.8, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'crammer_singer', C = 0.1, loss = 'squared_hinge')				         			
			#clf = MLPClassifier(beta_1 = 0.7, beta_2 = 0.7, shuffle = True, nesterovs_momentum = True, solver = 'adam', validation_fraction = 0.1, learning_rate = 'invscaling', max_iter = 150, batch_size = 250, power_t = 0.75, learning_rate_init = 0.01, tol = 0.01, epsilon = 1e-06, alpha = 0.1, early_stopping = False, activation = 'tanh', momentum = 0.5, hidden_layer_sizes = (5, 22))																											 																																																									

		if user_choice == '2':
			print ('You chosed imdb')
			knn = KNeighborsClassifier(n_neighbors=7, metric = 'euclidean', weights = 'uniform',leaf_size = 5, algorithm = 'ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort = False, splitter= 'random', max_leaf_nodes= None, min_samples_leaf = 1, max_features = 'sqrt', criterion = 'entropy', min_samples_split = 38, max_depth = None)
			random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 7,  min_samples_leaf = 2, n_estimators = 5,  min_samples_split = 2, criterion = 'gini',  max_features = 3,  max_depth = 3,  class_weight = None)
			log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
			svc = SVC(kernel='rbf', C= 2.1, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 0.1, loss = 'hinge')	
			#clf = MLPClassifier(beta_1 = 0.3, beta_2 = 0.1, shuffle = True, nesterovs_momentum = True, solver = 'lbfgs', validation_fraction = 0.9, learning_rate = 'constant', max_iter = 200, batch_size = 'auto', power_t = 1, learning_rate_init = 0.001, tol = 0.01, epsilon = 0.01, alpha = 1e-05, early_stopping = False, activation = 'tanh', momentum = 0.1, hidden_layer_sizes = (72, 14))																											 																																																									
																																					
		if user_choice == '3':
			print ('You chosed Opinion Observer')
			knn = KNeighborsClassifier(n_neighbors=21, metric = 'manhattan', weights = 'distance',leaf_size = 10, algorithm = 'ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'random', max_leaf_nodes= None, min_samples_leaf = 10, max_features = None, criterion = 'entropy', min_samples_split = 2, max_depth = None)                                 
			random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 7,  min_samples_leaf = 5, n_estimators = 7,  min_samples_split = 2, criterion = 'gini',  max_features = 6,  max_depth = 3)
			log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
			svc = SVC(kernel='rbf', C= 0.8, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'crammer_singer', C = 0.1, loss = 'hinge')
			#clf = MLPClassifier(beta_1 = 0.3, beta_2 = 0.1, shuffle = False, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.3, learning_rate = 'constant', max_iter = 150, batch_size = 'auto', power_t = 1.5, learning_rate_init = 0.001, tol = 0.0001, epsilon = 0.01, alpha = 0.01, early_stopping = True, activation = 'logistic', momentum = 0.9, hidden_layer_sizes = (16, 60))																											 																																																									
																	  			
		if user_choice == '4':
			print ('You chosed SentiWordNet')
			knn = KNeighborsClassifier(n_neighbors=16, metric = 'euclidean', weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort = False, splitter= 'best', max_leaf_nodes= None, min_samples_leaf = 1, max_features = 'sqrt', criterion = 'entropy', min_samples_split = 15, max_depth = 3)
			random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 10,  min_samples_leaf = 5, n_estimators = 4,  min_samples_split = 4, criterion = 'entropy',  max_features = 5,  max_depth = None, class_weight = None )
			log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
			svc = SVC(kernel='rbf', C= 0.7, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'crammer_singer', C = 0.1, loss = 'hinge')
			#clf = MLPClassifier(beta_1 = 0.1, beta_2 = 0.9, shuffle = False, nesterovs_momentum = False, solver = 'adam', validation_fraction = 0.5, learning_rate = 'invscaling', max_iter = 250, batch_size = 125, power_t = 0.75, learning_rate_init = 0.001, tol = 0.01, epsilon = 0.1, alpha = 0.1, early_stopping = False, activation = 'relu', momentum = 0.9, hidden_layer_sizes = (49, 53))																											 																																																									

 		if user_choice == '5':
			print ('You chosed Subjectivity')
			knn = KNeighborsClassifier(n_neighbors=26, metric = 'minkowski', p=1.5, weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort = False, splitter= 'random', max_leaf_nodes = 7, min_samples_leaf = 10, max_features = None, criterion = 'entropy', min_samples_split = 4, max_depth = 3)
			random_forest = RandomForestClassifier(oob_score = True, max_leaf_nodes = 10,  min_samples_leaf = 1, n_estimators = 14,  min_samples_split = 3, criterion = 'entropy',  max_features = 'log2',  max_depth = None, class_weight = None )
			log_regr = linear_model.LogisticRegression(C=0.5, solver='newton-cg')	
			svc = SVC(kernel='rbf', C= 1.8, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 0.6, loss = 'squared_hinge')
 			#clf = MLPClassifier(beta_1 = 0.7, beta_2 = 0.999, shuffle = True, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.2, learning_rate = 'invscaling', max_iter = 300, batch_size = 200, power_t = 1.25, learning_rate_init = 0.01, tol = 0.01, epsilon = 1e-05, alpha = 1e-05, early_stopping = True, activation = 'relu', momentum = 0.9, hidden_layer_sizes = (24, 5))																											 																																																													
				
		if user_choice == '6':
			print ('You chosed Amazon/TripAdvisor')
			knn = KNeighborsClassifier(n_neighbors=17, metric = 'manhattan', weights = 'distance',leaf_size = 11, algorithm = 'ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes = None, min_samples_leaf = 1, max_features = None, criterion = 'gini', min_samples_split = 3, max_depth = None)
			random_forest = RandomForestClassifier(oob_score = True, max_leaf_nodes = None,  min_samples_leaf = 1, n_estimators = 13,  min_samples_split = 4, criterion = 'entropy',  max_features = 4,  max_depth = None, class_weight = None )
			log_regr = linear_model.LogisticRegression(C=0.5, solver='newton-cg')
			svc = SVC(kernel='rbf', C= 1.2, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 1.2, loss = 'squared_hinge')
			#clf = MLPClassifier(beta_1 = 0.5, beta_2 = 0.9, shuffle = True, nesterovs_momentum = True, solver = 'lbfgs', validation_fraction = 0.3, learning_rate = 'invscaling', max_iter = 150, batch_size = 125, power_t = 0.25, learning_rate_init = 0.001, tol = 1e-05, epsilon = 1e-07, alpha = 0.1, early_stopping = False, activation = 'tanh', momentum = 0.9, hidden_layer_sizes = (61, 36))																											 																																																									
				
		if user_choice == '7':
			print ('You chosed Goodreads')
			knn = KNeighborsClassifier(n_neighbors=13, metric = 'manhattan', weights = 'distance',leaf_size = 10, algorithm = 'ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes = None, min_samples_leaf = 1, max_features = 'auto', criterion = 'entropy', min_samples_split = 2, max_depth = None)
			log_regr = linear_model.LogisticRegression(C=0.3, solver='newton-cg')
			random_forest = RandomForestClassifier(oob_score = True, max_leaf_nodes = None,  min_samples_leaf = 2, n_estimators = 18,  min_samples_split = 2, criterion = 'entropy',  max_features = 5,  max_depth = None)		
			svc = SVC(kernel='rbf', C= 1.5, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 1.5, loss = 'squared_hinge')
			#clf = MLPClassifier(beta_1 = 0.3, beta_2 = 0.7, shuffle = False, nesterovs_momentum = False, solver = 'adam', validation_fraction = 0.7, learning_rate = 'invscaling', max_iter = 300, batch_size = 350, power_t = 1.25, learning_rate_init = 0.001, tol = 0.001, epsilon = 0.001, alpha = 1e-05, early_stopping = False, activation = 'tanh', momentum = 0.5, hidden_layer_sizes = (65, 9))																											 																																																									
  							
		if user_choice == '8':
			print ('You chosed Opentable')
			knn = KNeighborsClassifier(n_neighbors=9, metric = 'manhattan', weights = 'distance',leaf_size = 10, algorithm = 'kd_tree')
			tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes = None, min_samples_leaf = 1, max_features = None, criterion = 'entropy', min_samples_split = 2, max_depth = None)	
			random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = None,  min_samples_leaf = 3, n_estimators = 17,  min_samples_split = 2, criterion = 'gini',  max_features = 6,  max_depth = None, class_weight = None )	
			log_regr = linear_model.LogisticRegression(C=1.0, solver='newton-cg')	
			svc = SVC(kernel='rbf', C= 1.2, shrinking=True, probability= False)
			linear_svm = svm.LinearSVC(multi_class = 'crammer_singer', C = 0.6, loss = 'hinge')
 			#clf = MLPClassifier(beta_1 = 0.1, beta_2 = 0.3, shuffle = False, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.5, learning_rate = 'constant', max_iter = 300, batch_size = 125, power_t = 0.25, learning_rate_init = 0.01, tol = 0.001, epsilon = 1e-07, alpha = 0.001, early_stopping = True, activation = 'tanh', momentum = 0.5, hidden_layer_sizes = (65, 54))																											 																																																									
  			
		if user_choice == '9':
			print ('You chosed inquirer')
			knn = KNeighborsClassifier(n_neighbors=29, metric = 'euclidean', weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes = 10, min_samples_leaf = 10, max_features = 'sqrt', criterion = 'gini', min_samples_split = 23, max_depth = 3)                                                                                                                   															
			random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = None,  min_samples_leaf = 3, n_estimators = 13,  min_samples_split = 2, criterion = 'gini',  max_features = 3,  max_depth = None, class_weight = 'balanced_subsample' )				
			log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
			svc = SVC(kernel='rbf', C= 2.1, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 0.1, loss = 'hinge')
			#clf = MLPClassifier(beta_1 = 0.5, beta_2 = 0.1, shuffle = False, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.3, learning_rate = 'invscaling', max_iter = 200, batch_size = 300, power_t = 0.75, learning_rate_init =0.0001, tol = 0.0001, epsilon = 1e-06, alpha = 1e-06, early_stopping = False, activation = 'tanh', momentum = 0.3, hidden_layer_sizes = (76, 85))																											 																																																									
			
if debug_variable == True:
	
	#call the 4 classifiers
	# random search
	#knn = KNeighborsClassifier(n_neighbors=14, metric = 'manhattan', weights = 'distance',leaf_size = 34, algorithm = 'auto')
	# grid search
	#knn = KNeighborsClassifier(n_neighbors=12, metric = 'manhattan', weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')	
	knn = KNeighborsClassifier()
	# imdb grid search parameters
	#knn = KNeighborsClassifier(n_neighbors=7, metric = 'euclidean', weights = 'uniform',leaf_size = 5, algorithm = 'ball_tree')
	# positive/negative grid search
	#knn = KNeighborsClassifier(n_neighbors=21, metric = 'manhattan', weights = 'distance',leaf_size = 10, algorithm = 'ball_tree')
	# SentiWordNet Lexicon grid search
	#knn = KNeighborsClassifier(n_neighbors=16, metric = 'euclidean', weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')
	# Subjectivity Lexicon grid search
	#knn = KNeighborsClassifier(n_neighbors=26, metric = 'minkowski', p=1.5, weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')
	# Amazon-TripadvisorLexicon grid search
	#knn = KNeighborsClassifier(n_neighbors=17, metric = 'manhattan', weights = 'distance',leaf_size = 11, algorithm = 'ball_tree')
	# Goodreads lexicon random search			
	#knn = KNeighborsClassifier(n_neighbors=13, metric = 'manhattan', weights = 'distance',leaf_size = 10, algorithm = 'ball_tree')
	# Opentable lexicon random search			
	#knn = KNeighborsClassifier(n_neighbors=9, metric = 'manhattan', weights = 'distance',leaf_size = 10, algorithm = 'kd_tree')
	#knn = KNeighborsClassifier(n_neighbors=22, metric = 'minkowski', weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')	
	# wnscore_inquirer Grid search
	#knn = KNeighborsClassifier(n_neighbors=29, metric = 'euclidean', weights = 'distance',leaf_size = 5, algorithm = 'ball_tree')
	#knn = KNeighborsClassifier(n_neighbors=29, metric = 'minkowski', weights = 'distance',leaf_size = 5, algorithm = 'ball_tree', p = 2.5)						 
	tree_decision = tree.DecisionTreeClassifier()
	# imdb grid search parameters
	#tree_decision = tree.DecisionTreeClassifier(presort = False, splitter= 'random', max_leaf_nodes= None, min_samples_leaf = 1, max_features = 'sqrt', criterion = 'entropy', min_samples_split = 38, max_depth = None)
	# positive/negative grid search
	#tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'random', max_leaf_nodes= None, min_samples_leaf = 10, max_features = None, criterion = 'entropy', min_samples_split = 2, max_depth = None)                                 
	# SentiWordNet grid search
	#tree_decision = tree.DecisionTreeClassifier(presort = False, splitter= 'best', max_leaf_nodes= None, min_samples_leaf = 1, max_features = 'sqrt', criterion = 'entropy', min_samples_split = 15, max_depth = 3)
	# Subjectivity Lexixon grid search
	#tree_decision = tree.DecisionTreeClassifier(presort = False, splitter= 'random', max_leaf_nodes = 7, min_samples_leaf = 10, max_features = None, criterion = 'entropy', min_samples_split = 4, max_depth = 3)
	# Amazon-Tripadvisor Lexixon grid search
	#tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes = None, min_samples_leaf = 1, max_features = None, criterion = 'gini', min_samples_split = 3, max_depth = None)
	# Goodreads Lexixon grid search
	#tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes = None, min_samples_leaf = 1, max_features = 'auto', criterion = 'entropy', min_samples_split = 2, max_depth = None)
	# Opentable Lexixon grid search
	#tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes = None, min_samples_leaf = 1, max_features = None, criterion = 'entropy', min_samples_split = 2, max_depth = None)	
	# wnscore_inquirer Lexixon grid search
	#tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes = 10, min_samples_leaf = 10, max_features = 'sqrt', criterion = 'gini', min_samples_split = 23, max_depth = 3)                                                                                                                   															
	# Random Search
	#tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'best', max_leaf_nodes= None, min_samples_leaf = 1, max_features = 'sqrt', criterion = 'gini', min_samples_split = 4, max_depth = 3)
	# Grid Search
	#tree_decision = tree.DecisionTreeClassifier(presort = True, splitter= 'random', max_leaf_nodes= None, min_samples_leaf = 3, max_features = 'log2', criterion = 'gini', min_samples_split = 10, max_depth = None)					                
	#tree_decision = tree.DecisionTreeClassifier(presort =  True, splitter= 'best', max_leaf_nodes= 3, min_samples_leaf = 1, max_features = 'sqrt', criterion = 'entropy', min_samples_split = 8, max_depth = 3 )
	#random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 5,  min_samples_leaf = 2, n_estimators = 13,  min_samples_split = 5, criterion = 'gini',  max_features = 3,  max_depth = None,  class_weight = None)
	random_forest = RandomForestClassifier()
	# imdb lexicon grid search 
	#random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 7,  min_samples_leaf = 2, n_estimators = 5,  min_samples_split = 2, criterion = 'gini',  max_features = 3,  max_depth = 3,  class_weight = None)
	# positive/negative grid search
	#random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 7,  min_samples_leaf = 5, n_estimators = 7,  min_samples_split = 2, criterion = 'gini',  max_features = 6,  max_depth = 3)
	# SentiWordNet Lexicon grid search
	#random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 10,  min_samples_leaf = 5, n_estimators = 4,  min_samples_split = 4, criterion = 'entropy',  max_features = 5,  max_depth = None, class_weight = None )
	# Subjectivity Lexicon grid search
	#random_forest = RandomForestClassifier(oob_score = True, max_leaf_nodes = 10,  min_samples_leaf = 1, n_estimators = 14,  min_samples_split = 3, criterion = 'entropy',  max_features = 'log2',  max_depth = None, class_weight = None )
	# Amazon-Tripadvisor Lexicon grid search
	#random_forest = RandomForestClassifier(oob_score = True, max_leaf_nodes = None,  min_samples_leaf = 1, n_estimators = 13,  min_samples_split = 4, criterion = 'entropy',  max_features = 4,  max_depth = None, class_weight = None )
	# Opentable Lexicon grid search
	#random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = None,  min_samples_leaf = 3, n_estimators = 17,  min_samples_split = 2, criterion = 'gini',  max_features = 6,  max_depth = None, class_weight = None )	
	# wnscore_inquirer grid search
	#random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = None,  min_samples_leaf = 3, n_estimators = 13,  min_samples_split = 2, criterion = 'gini',  max_features = 3,  max_depth = None, class_weight = 'balanced_subsample' )				
	# Random Search
	#random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 5,  min_samples_leaf = 2, n_estimators = 17,  min_samples_split = 5, criterion = 'gini',  max_features = 5,  max_depth = 3,  class_weight = None)																				
	# Grid Search
	#random_forest = RandomForestClassifier(oob_score = False, max_leaf_nodes = 7,  min_samples_leaf = 3, n_estimators = 2,  min_samples_split = 3, criterion = 'entropy',  max_features = 2,  max_depth = None,  class_weight = None)
	# Grid Search 2
	#random_forest = RandomForestClassifier(oob_score = True, max_leaf_nodes = None,  min_samples_leaf = 1, n_estimators = 15,  min_samples_split = 2, criterion = 'gini',  max_features = 4,  max_depth = None,  class_weight = 'balanced')								  	
	log_regr = linear_model.LogisticRegression()
	# imdb grid search
	#log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
	# positive/negative grid search
	#log_regr = linear_model.LogisticRegression(C=0.3, solver='liblinear')
	# SentiWordNet Lexicon grid search
	#log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
	# Subjectivity Lexicon grid search
	#log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
	# Subjectivity Lexicon grid search
	#log_regr = linear_model.LogisticRegression(C=0.5, solver='newton-cg')
	# Goodreads Lexicon grid search
	#log_regr = linear_model.LogisticRegression(C=0.3, solver='newton-cg')
	# Opentable Lexicon grid search
	#log_regr = linear_model.LogisticRegression(C=1.0, solver='newton-cg')	
	# wnscore_inquirer Lexicon grid search
	#log_regr = linear_model.LogisticRegression(C=0.3, solver='sag')
	svc = SVC()
	linear_svm = svm.LinearSVC()

#call the 4 predicts 
y_pred_knn = knn.fit(X_train, y_train).predict(X_test)
y_pred_tree_decision = tree_decision.fit(X_train, y_train).predict(X_test)
y_pred_random_forest = random_forest.fit(X_train, y_train).predict(X_test)
y_pred_log_regr = log_regr.fit(X_train, y_train).predict(X_test)
y_pred_svc = svc.fit(X_train, y_train).predict(X_test)
y_pred_LinearSVC = linear_svm.fit(X_train, y_train).predict(X_test)
#y_pred_neural_network = clf.fit(X_train, y_train).predict(X_test)

#create the confusion matrixes
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm__tree_decision = confusion_matrix(y_test, y_pred_tree_decision)
cm_random_forest = confusion_matrix(y_test, y_pred_random_forest)
cm_log_regr = confusion_matrix(y_test, y_pred_log_regr)
cm_svc = confusion_matrix(y_test, y_pred_svc)
cm_LinearSVC = confusion_matrix(y_test, y_pred_LinearSVC)
#cm_neural_network = confusion_matrix(y_test, y_pred_neural_network)
'''         '''

benchmark(knn, cm_knn, y_pred_knn, 'KNeighbors')
benchmark(tree_decision, cm__tree_decision, y_pred_tree_decision, 'tree_decision')
benchmark(random_forest, cm_random_forest, y_pred_random_forest,  'random forest')
benchmark(log_regr, cm_log_regr, y_pred_log_regr,  'logistic regression')
benchmark(svc, cm_svc, y_pred_svc, 'SVC')
benchmark(linear_svm, cm_LinearSVC, y_pred_LinearSVC, 'linear_svm')
#benchmark(clf, cm_neural_network, y_pred_neural_network,  'neural network')