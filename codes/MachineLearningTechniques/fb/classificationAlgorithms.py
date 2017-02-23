# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:30:56 2016

@author: tasos
"""
import pandas as pd
import time
import sklearn
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
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn import svm
#from sklearn.neural_network import MLPClassifier

iterations = 1
debug_variable = False

def benchmark(model, cm, y_pred, wintitle='Figure 1'):
    print '\n\n' + wintitle + ' Results'
    s = time.time()
    for i in range(iterations):
        model.fit(X_train, y_train) 
    print "{0} Iterations Training Time: ".format(iterations), time.time() - s

    s = time.time()
    for i in range(iterations):
       score =model.score(X_test, y_test)
        
    print ("{0} Iterations Scoring Time: ".format(iterations), time.time() - s)
    print ("Score is:", score)
    fold_cross_scor = cval.cross_val_score(model, X_train, y_train, cv=10).mean()
    print ("Fold cross scor: ", fold_cross_scor)
    print ("High-Dimensionality Score: ", round((score*100), 3))
    print ("---------------------------------------")    
    
    '''				
    print ("precision score, micro : ",precision_score(y_test, y_pred, average='micro'))
    print ("precision score, macro : ",precision_score(y_test, y_pred, average='macro'))
    print ("precision score, weighted : ",precision_score(y_test, y_pred, average='weighted'))
    #print ("precision score, binary : ",precision_score(y_test, y_pred, average='binary',pos_label=0 ))
				
    print ("---------------------------------------")   
    print ("recall score, weighted : ",recall_score(y_test, y_pred, average='weighted'))
    print ("recall score, macro : ",recall_score(y_test, y_pred, average='macro'))
    print ("recall score, micro : ",recall_score(y_test, y_pred, average='micro'))   
    #print ("recall score, binary : ",precision_score(y_test, y_pred, average='binary',pos_label=0 ))
				
    print ("---------------------------------------")    
    print (classification_report(y_test, y_pred))

    print ("---------------------------------------")   
    print ("f1 score, weighted : ",f1_score(y_test, y_pred, average='weighted'))
    print ("f1 score, macro : ",f1_score(y_test, y_pred, average='macro'))
    print ("f1 score, micro : ",f1_score(y_test, y_pred, average='micro'))
    #print ("f1 score, binary : ",precision_score(y_test, y_pred, average='binary',pos_label=0 ))

    #parameter beta balances between precision and recall
    print ("---------------------------------------")       
    print ("fbeta score, micro : ", fbeta_score(y_test, y_pred, average='micro', beta=0.5))
    print ("fbeta score, macro : ", fbeta_score(y_test, y_pred, average='macro', beta=0.5))
    print ("fbeta score, weighted : ", fbeta_score(y_test, y_pred, average='weighted', beta=0.5))
    #print ("fbeta score, binary : ",precision_score(y_test, y_pred, average='binary',pos_label=0 ))
        
    print ("---------------------------------------")       
    print ("hamming loss : ", hamming_loss(y_test, y_pred))

    print ("---------------------------------------")       
    print ("jaccard similarity, normalized: ", jaccard_similarity_score(y_test, y_pred))

    print ("---------------------------------------")       
    print ("zero one loss: ", zero_one_loss(y_test, y_pred))
				
				
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


# read data
X = pd.read_csv('Datasets/thesis_facebook_data_v24.csv',header=None)
#X = pd.read_csv('Datasets/thesis_facebook_data_senti1.csv',header=None)


X.columns = ['status_id', 'status_message', 'link_name', 'status_type', 'status_link', 'status_published', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score', 'computer_score', 'computer_score_imdb', 'computer_score_pos_neg', 'computer_score_senti_word_net', 'computer_score_subjective', 'computer_score_amazon_tripadvisor', 'computer_score_goodreads', 'computer_score_opentable', 'computer_score_wnscore_inquirer' ]
#X.columns = ['status_id', 'status_message', 'link_name', 'status_type', 'status_link', 'status_published', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score', 'computer_score', 'computer_score_imdb', 'computer_score_pos_neg', 'computer_score_senti_word_net' ]
X.status_type = X.status_type.map({'link':1, 'photo':2, 'status':3, 'video':4 })

#print X['computer_score_senti_word_net'].head(2)
# drop possible missings 
X = X.dropna()
if debug_variable == True:
	print (X.head(5))

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


if debug_variable == True:
	print (y.head(5))


#y = y*10
#y = y.round()
y = y.astype(int)

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

if debug_variable == False: 	
	print('')
	print ('Do you want to drop POS? Insert Y or N')
	user_choice_drop = raw_input('')
	if user_choice_drop == 'y' or user_choice_drop == 'Y':
		X = X.drop(['num_cc', 'num_cd', 'nun_ex', 'num_fw', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nns', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score'], axis=1)
		X = X.drop(['num_dt','num_in', 'num_jj', 'num_nn','num_nnp', 'num_vb', 'num_prp'], axis=1)
		print 'dropped..'
	
# drop all possible combinations 
#X = X.drop(['num_cc', 'nun_ex', 'num_fw', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp6', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score'], axis=1)
#X = X.drop(['num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'human_score'], axis=1)
#X = X.drop(['status_type'], axis=1)
#X = X.drop(['num_cd', 'num_nns', 'num_rb', 'num_vbz'], axis=1)
#X = X.drop(['num_dt', 'num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb'], axis=1)
#X = X.drop(['num_reactions', 'num_comments', 'num_shares'], axis=1)

#print X.head(2)
#print X.columns.values.tolist()

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

# split the data to test and train 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

#print y_train
#raw_input()

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
			knn = KNeighborsClassifier(n_neighbors=15, metric='euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'best', max_leaf_nodes= None, min_samples_leaf= 10, min_samples_split = 4, criterion= 'entropy', max_features='sqrt', max_depth= 3)
			random_forest = RandomForestClassifier(n_estimators = 12, criterion = "entropy", max_features = 6, max_depth = 3, min_samples_split = 8, min_samples_leaf = 3, max_leaf_nodes = 7, oob_score = False, class_weight = None)			
			log_regr = linear_model.LogisticRegression(C=0.3, solver = 'sag')
			svc = SVC(kernel='rbf', C= 0.3, shrinking=True, probability= True)
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 1, loss = 'hinge')			
			#clf = MLPClassifier()
			
		if user_choice == '2':
			print ('You chosed imdb')
			knn = KNeighborsClassifier(n_neighbors=9, metric='chebyshev', weights='uniform', leaf_size=48, algorithm='kd_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'best', max_leaf_nodes= 7, min_samples_leaf= 1, min_samples_split= 15, criterion= 'entropy', max_features=None, max_depth= 3)
			random_forest = RandomForestClassifier(n_estimators = 12, criterion = "entropy", max_features = 6, max_depth = None, min_samples_split = 2, min_samples_leaf = 3, max_leaf_nodes = None, oob_score = True, class_weight = None )			
			log_regr = linear_model.LogisticRegression(C=0.3, solver = 'newton-cg')
			svc = SVC(kernel='rbf', C= 0.1, shrinking=True, probability= True)		
			linear_svm = svm.LinearSVC(multi_class = 'crammer_singer', C = 0.6, loss = 'squared_hinge')			
			#clf = MLPClassifier(beta_1 = 0.1, beta_2 = 0.999, shuffle = True, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.3, learning_rate = 'constant', max_iter = 200, batch_size = 'auto', power_t = 1.5, learning_rate_init = 0.1, tol = 1e-05, epsilon = 1e-05, alpha = 1e-05, early_stopping = True, activation = 'logistic', momentum = 0.3, hidden_layer_sizes = (13, 65))			
					
		if user_choice == '3':
			print ('You chosed Opinion Observer')
			knn = KNeighborsClassifier(n_neighbors=20, metric='chebyshev', weights='uniform', leaf_size=9, algorithm='ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'random', max_leaf_nodes= 3, min_samples_leaf= 10, min_samples_split= 15, criterion= 'entropy', max_features='log2', max_depth= None)
			random_forest = RandomForestClassifier(n_estimators = 12, criterion = "entropy", max_features = 6, max_depth = None, min_samples_split = 2, min_samples_leaf = 3, max_leaf_nodes = None, oob_score = True, class_weight = None )			
			log_regr = linear_model.LogisticRegression(C=0.3, solver = 'newton-cg')
			svc = SVC(kernel='rbf', C= 0.1, shrinking=True, probability= True)			
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 0.4, loss = 'squared_hinge')
			#clf = MLPClassifier(beta_1 = 0.9, beta_2 = 0.3, shuffle = False, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.9, learning_rate = 'invascaling', max_iter = 150, batch_size = 350, power_t = 1.5, learning_rate_init = 0.01, tol = 0.0001, epsilon = 0.1, alpha = 0.0001, early_stopping = False, activation = 'logistic', momentum = 0.9, hidden_layer_sizes = (92, 98))			
					
		if user_choice == '4':
			print ('You chosed SentiWordNet')
			knn = KNeighborsClassifier(n_neighbors=16, metric= 'euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'best', max_leaf_nodes= 10, min_samples_leaf= 1, min_samples_split= 15, criterion= 'entropy', max_features='log2', max_depth= None)			
			random_forest = RandomForestClassifier(n_estimators = 23, criterion = "gini", max_features = 'auto', max_depth = 5, min_samples_split = 13, min_samples_leaf = 7, max_leaf_nodes = 9, oob_score = True, class_weight = None )		
			log_regr = linear_model.LogisticRegression(C=0.3, solver = 'sag')
			svc = SVC(kernel='rbf', C= 0.1, shrinking=True, probability= True)		
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 1, loss = 'squared_hinge')			
			#clf = MLPClassifier(beta_1 = 0.3, beta_2 = 0.3, shuffle = False, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.3, learning_rate = 'invascaling', max_iter = 100, batch_size = 250, power_t = 0.5, learning_rate_init = 0.001, tol = 0.1, epsilon = 0.1, alpha = 0.1, early_stopping = False, activation = 'identity', momentum = 0.1, hidden_layer_sizes = (84, 22))			

		if user_choice == '5':
			print ('You chosed Subjectivity')
			knn = KNeighborsClassifier(n_neighbors=25, metric= 'euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'best', max_leaf_nodes= 7, min_samples_leaf= 1, min_samples_split= 4, criterion= 'gini', max_features='log2', max_depth= 3)			
			random_forest = RandomForestClassifier(n_estimators = 6, criterion = "gini", max_features = 4, max_depth = 3, min_samples_split = 7, min_samples_leaf = 8, max_leaf_nodes = None, oob_score = False, class_weight = None )		
			log_regr = linear_model.LogisticRegression(C=0.3, solver = 'sag')
			svc = SVC(kernel='rbf', C= 1.2, shrinking=True, probability= True)	
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 0.1, loss = 'squared_hinge')
			#clf = MLPClassifier(beta_1 = 0.9, beta_2 = 0.1, shuffle = True, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.5, learning_rate = 'invascaling', max_iter = 200, batch_size = 200, power_t = 1.5, learning_rate_init = 0.0001, tol = 0.0001, epsilon = 0.01, alpha = 0.1, early_stopping = False, activation = 'identity', momentum = 0.3, hidden_layer_sizes = (98, 74))			
																									
		if user_choice == '6':
			print ('You chosed Amazon/TripAdvisor')
			knn = KNeighborsClassifier(n_neighbors=27, metric= 'manhattan', weights='uniform', leaf_size=5, algorithm='ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'random', max_leaf_nodes= 5, min_samples_leaf= 10, min_samples_split= 23, criterion= 'entropy', max_features= None, max_depth= 3)
			random_forest = RandomForestClassifier(n_estimators = 7, criterion = "gini", max_features = 6, max_depth = None, min_samples_split = 7, min_samples_leaf = 5, max_leaf_nodes = 5, oob_score = False, class_weight = None )
			log_regr = linear_model.LogisticRegression(C=0.3, solver = 'sag')
			svc = SVC(kernel='rbf', C= 1, shrinking=True, probability= True)			
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 0.1, loss = 'squared_hinge')
			#clf = MLPClassifier(beta_1 = 0.9, beta_2 = 0.999, shuffle = True, nesterovs_momentum = True, solver = 'sgd', validation_fraction = 0.9, learning_rate = 'adaptive', max_iter = 200, batch_size = 300, power_t = 0.75, learning_rate_init = 0.01, tol = 0.001, epsilon = 1e-06, alpha = 1e-05, early_stopping = False, activation = 'identity', momentum = 0.1, hidden_layer_sizes = (82, 30))							
				
		if user_choice == '7':
			print ('You chosed Goodreads')
			knn = KNeighborsClassifier(n_neighbors=25, metric= 'euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'random', max_leaf_nodes= 7, min_samples_leaf= 10, min_samples_split= 2, criterion= 'entropy', max_features= 'sqrt', max_depth= 3)
			random_forest = RandomForestClassifier(n_estimators = 7, criterion = "gini", max_features = 6, max_depth = 3, min_samples_split = 6, min_samples_leaf = 8, max_leaf_nodes = None, oob_score = True, class_weight = None )
			log_regr = linear_model.LogisticRegression(C=1.5, solver = 'newton-cg')
			svc = SVC(kernel='rbf', C= 1.2, shrinking=True, probability= True)	
			linear_svm = svm.LinearSVC(multi_class = 'crammer_singer', C = 0.6, loss = 'hinge')	
			#clf = MLPClassifier(beta_1 = 0.5, beta_2 = 0.7, shuffle = True, nesterovs_momentum = False, solver = 'lbfgs', validation_fraction = 0.3, learning_rate = 'invscaling', max_iter = 100, batch_size = 300, power_t = 1.25, learning_rate_init = 0.0001, tol = 0.01, epsilon = 1e-08, alpha = 1e-05, early_stopping = True, activation = 'tanh', momentum = 0.1, hidden_layer_sizes = (95, 5))																											 																																																						
					
		if user_choice == '8':
			print ('You chosed Opentable')
			knn = KNeighborsClassifier(n_neighbors=13, metric= 'manhattan', weights='uniform', leaf_size=5, algorithm='ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'random', max_leaf_nodes= 7, min_samples_leaf= 1, min_samples_split= 4, criterion= 'gini', max_features= 'auto', max_depth= 3)
			random_forest = RandomForestClassifier(n_estimators = 10, criterion = "entropy", max_features = 2, max_depth = 3, min_samples_split = 8, min_samples_leaf = 2, max_leaf_nodes = 10, oob_score = False, class_weight = None)
			log_regr = linear_model.LogisticRegression(C=1.0, solver = 'liblinear')
			svc = SVC(kernel='rbf', C= 0.1, shrinking=True, probability= True)	
			linear_svm = svm.LinearSVC(multi_class = 'crammer_singer', C = 0.4, loss = 'hinge')			
			#clf = MLPClassifier(beta_1 = 0.7, beta_2 = 0.7, shuffle = False, nesterovs_momentum = True, solver = 'lbfgs', validation_fraction = 0.7, learning_rate = 'adaptive', max_iter = 300, batch_size = 100, power_t = 1.25, learning_rate_init = 0.1, tol = 0.01, epsilon = 0.01, alpha = 1e-07, early_stopping = True, activation = 'tanh', momentum = 0.7, hidden_layer_sizes = (5, 55))																											 																																																						
				
		if user_choice == '9':
			print ('You chosed inquirer')
			knn = KNeighborsClassifier(n_neighbors=11, metric= 'euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
			tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'random', max_leaf_nodes= 10, min_samples_leaf= 3, min_samples_split= 10, criterion= 'entropy', max_features= None, max_depth= None)
			random_forest = RandomForestClassifier(n_estimators = 6, criterion = "entropy", max_features = 4, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_leaf_nodes = None, oob_score = False, class_weight = "balanced_subsample")
			log_regr = linear_model.LogisticRegression(C=0.3, solver = 'newton-cg')
			svc = SVC(kernel='rbf', C= 0.1, shrinking=True, probability= True)			
			linear_svm = svm.LinearSVC(multi_class = 'ovr', C = 1.2, loss = 'hinge')									
			#clf = MLPClassifier(beta_1 = 0.5, beta_2 = 0.7, shuffle = False, nesterovs_momentum = False, solver = 'adam', validation_fraction = 0.5, learning_rate = 'invscaling', max_iter = 250, batch_size = 200, power_t = 1.5, learning_rate_init = 0.01, tol = 0.001, epsilon = 0.001, alpha = 1e-07, early_stopping = False, activation = 'relu', momentum = 0.9, hidden_layer_sizes = (32, 45))																											 																																																									
			
if debug_variable == True:	
	# call the 4 classifiers
	#knn = KNeighborsClassifier(n_neighbors=11, metric='chebyshev', weights='uniform', leaf_size=22, algorithm='kd_tree')
	#knn = KNeighborsClassifier(n_neighbors=29, metric='manhattan', weights='distance', leaf_size=5, algorithm='ball_tree')
	#knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski', weights='uniform', leaf_size=5, algorithm='ball_tree', p=1.5)
	#knn = KNeighborsClassifier(n_neighbors=15, metric='euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
	# imdb grid search parameters
	#knn = KNeighborsClassifier(n_neighbors=9, metric='chebyshev', weights='uniform', leaf_size=48, algorithm='kd_tree')
	knn = KNeighborsClassifier()
	# positive/negative grid search parameters
	#knn = KNeighborsClassifier(n_neighbors=20, metric='chebyshev', weights='uniform', leaf_size=9, algorithm='ball_tree')
	# SentiWordNet Lexixon grid search parameters
	#knn = KNeighborsClassifier(n_neighbors=16, metric= 'euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
	# Subjective Lexixon grid search parameters
	#knn = KNeighborsClassifier(n_neighbors=25, metric= 'euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
	# Amazon-Tripadvisor Lexixon grid search parameters
	#knn = KNeighborsClassifier(n_neighbors=27, metric= 'manhattan', weights='uniform', leaf_size=5, algorithm='ball_tree')
	#knn = KNeighborsClassifier(n_neighbors=28, metric= 'minkowski', p=1.5, weights='uniform', leaf_size=5, algorithm='ball_tree')
	# Goodreads Lexixon grid search parameters
	#knn = KNeighborsClassifier(n_neighbors=25, metric= 'euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
	#knn = KNeighborsClassifier(n_neighbors=25, metric= 'minkowski', weights='uniform', leaf_size=5, algorithm='ball_tree', p=2.5)
	# Opentable Grid Search parameters
	#knn = KNeighborsClassifier(n_neighbors=13, metric= 'manhattan', weights='uniform', leaf_size=5, algorithm='ball_tree')
	# wnscore_inquirer Grid Search parameters
	#knn = KNeighborsClassifier(n_neighbors=11, metric= 'euclidean', weights='uniform', leaf_size=5, algorithm='ball_tree')
	#knn = KNeighborsClassifier(n_neighbors=11, metric= 'minkowski', weights='uniform', leaf_size=5, algorithm='ball_tree', p=1.5)
	# imdb grid search parameters
	'''
	tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'best', max_leaf_nodes= 7, min_samples_leaf= 1, min_samples_split= 15, criterion= 'entropy', max_features=None, max_depth= 3)
	'''
	tree_decision = tree.DecisionTreeClassifier()
	# positive/negative grid search parameters
	'''
	tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'random', max_leaf_nodes= 3, min_samples_leaf= 10, min_samples_split= 15, criterion= 'entropy', max_features='log2', max_depth= None)
	'''	
	'''
	# SentiWordNet Lexicon
	tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'best', max_leaf_nodes= 10, min_samples_leaf= 1, min_samples_split= 15, criterion= 'entropy', max_features='log2', max_depth= None)
	'''														
	'''
	# Subjective Lexicon
	tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'best', max_leaf_nodes= 7, min_samples_leaf= 1, min_samples_split= 4, criterion= 'gini', max_features='log2', max_depth= 3)
	'''
	'''
	# Amazon-Tripadvisor Lexicon
	tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'random', max_leaf_nodes= 5, min_samples_leaf= 10, min_samples_split= 23, criterion= 'entropy', max_features= None, max_depth= 3)
	'''
	'''
	# Goodreads Lexicon
	tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'random', max_leaf_nodes= 7, min_samples_leaf= 10, min_samples_split= 2, criterion= 'entropy', max_features= 'sqrt', max_depth= 3)
	'''
	'''
	# Opentable Lexicon
	tree_decision = tree.DecisionTreeClassifier(presort=True, splitter= 'random', max_leaf_nodes= 7, min_samples_leaf= 1, min_samples_split= 4, criterion= 'gini', max_features= 'auto', max_depth= 3)
	'''  
	'''
	# wnscore_inquirer Lexicon
	tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'random', max_leaf_nodes= 10, min_samples_leaf= 3, min_samples_split= 10, criterion= 'entropy', max_features= None, max_depth= None)
	'''
	# imdb grid search parameters
	'''
	random_forest = RandomForestClassifier(n_estimators = 12, criterion = "entropy", max_features = 6, max_depth = None, min_samples_split = 2, min_samples_leaf = 3, max_leaf_nodes = None, oob_score = True, class_weight = None )
	'''
	'''
	tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'best', max_leaf_nodes= 3, min_samples_leaf= 3, min_samples_split= 55, criterion= 'entropy', max_features='log2', max_depth= 3)
	'''     
	# positive/negative grid search parameters
	'''
	random_forest = RandomForestClassifier(n_estimators = 4, criterion = "gini", max_features = 6, max_depth = 3, min_samples_split = 10, min_samples_leaf = 8, max_leaf_nodes = 7, oob_score = False, class_weight = None )
	'''
	'''
	# Amazon-Tripadvisor grid search parameters
	random_forest = RandomForestClassifier(n_estimators = 7, criterion = "gini", max_features = 6, max_depth = None, min_samples_split = 7, min_samples_leaf = 5, max_leaf_nodes = 5, oob_score = False, class_weight = None )
	'''
	'''
	# Goodreads grid search parameters
	random_forest = RandomForestClassifier(n_estimators = 7, criterion = "gini", max_features = 6, max_depth = 3, min_samples_split = 6, min_samples_leaf = 8, max_leaf_nodes = None, oob_score = True, class_weight = None )
	'''
	# imdb grid search parameters
	#log_regr = linear_model.LogisticRegression(C=0.3, solver = 'newton-cg')
	log_regr = linear_model.LogisticRegression()
	# positive/negative grid search parameters
	#log_regr = linear_model.LogisticRegression(C=0.3, solver = 'newton-cg')
	#log_regr = linear_model.LogisticRegression(C=0.5, solver = 'sag')
	# SentiWordNetLexicon and Subjective Lexicon and Amazon-Tripadvisor Grid Search Parameters
	#log_regr = linear_model.LogisticRegression(C=0.3, solver = 'sag')
	# Goodreads Grid Search Parameters
	#log_regr = linear_model.LogisticRegression(C=1.5, solver = 'newton-cg')=
	# Opentable Grid Search Parameters
	#log_regr = linear_model.LogisticRegression(C=1.0, solver = 'liblinear')
	# wnscore_inquirer Grid Search Parameters
	#log_regr = linear_model.LogisticRegression(C=0.3, solver = 'newton-cg')
	'''
	tree_decision = tree.DecisionTreeClassifier(presort=False, splitter= 'best', max_leaf_nodes= None, min_samples_leaf= 10, min_samples_split= 4, criterion= 'gini', max_features='sqrt', max_depth= 3)
	'''
	# 20 combinations 12,22 secs
	'''
	random_forest = RandomForestClassifier(n_estimators = 17, criterion = "entropy", max_features = 2, max_depth = None, min_samples_split = 6, min_samples_leaf = 7, max_leaf_nodes = 7, oob_score = True, class_weight = None)
	# 400 combinations 292.34 secs
	random_forest = RandomForestClassifier(n_estimators = 5, criterion = "gini", max_features = 3, max_depth = 3, min_samples_split = 3, min_samples_leaf = 3, max_leaf_nodes = 7, oob_score = True, class_weight = None)
	# 1000 combinations 743.17 secs
	random_forest = RandomForestClassifier(n_estimators = 8, criterion = "gini", max_features = 6, max_depth = 3, min_samples_split = 5, min_samples_leaf = 9, max_leaf_nodes = 5, oob_score = False, class_weight = None)
	'''
	'''
	# 2000 combinations 1526.73 secs
	random_forest = RandomForestClassifier(n_estimators = 12, criterion = "entropy", max_features = 6, max_depth = 3, min_samples_split = 8, min_samples_leaf = 3, max_leaf_nodes = 7, oob_score = False, class_weight = None)
	'''
	'''
	# SentiWordNet Lexicon
	random_forest = RandomForestClassifier(n_estimators = 23, criterion = "gini", max_features = 'auto', max_depth = 5, min_samples_split = 13, min_samples_leaf = 7, max_leaf_nodes = 9, oob_score = True, class_weight = None)
	'''
	'''
	random_forest = RandomForestClassifier(n_estimators = 6, criterion = "gini", max_features = 4, max_depth = 3, min_samples_split = 7, min_samples_leaf = 8, max_leaf_nodes = None, oob_score = False, class_weight = None )
	'''
	'''
	# Opentable Lexicon
	random_forest = RandomForestClassifier(n_estimators = 10, criterion = "entropy", max_features = 2, max_depth = 3, min_samples_split = 8, min_samples_leaf = 2, max_leaf_nodes = 10, oob_score = False, class_weight = None)
	'''
	'''
	# wnscore_inquirer Lexicon
	random_forest = RandomForestClassifier(n_estimators = 6, criterion = "entropy", max_features = 4, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_leaf_nodes = None, oob_score = False, class_weight = "balanced_subsample")
	'''
	random_forest = RandomForestClassifier()
	svc = SVC()
	linear_svm = svm.LinearSVC()
	#clf = MLPClassifier()																											 																																																									
			
# call the 4 predicts 
y_pred_knn = knn.fit(X_train, y_train).predict(X_test)
y_pred_tree_decision = tree_decision.fit(X_train, y_train).predict(X_test)
y_pred_random_forest = random_forest.fit(X_train, y_train).predict(X_test)
y_pred_log_regr = log_regr.fit(X_train, y_train).predict(X_test)
y_pred_svc = svc.fit(X_train, y_train).predict(X_test)
y_pred_LinearSVC = linear_svm.fit(X_train, y_train).predict(X_test)
#y_pred_neural_network = clf.fit(X_train, y_train).predict(X_test)

# create the confusion matrixes
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm__tree_decision = confusion_matrix(y_test, y_pred_tree_decision)
cm_random_forest = confusion_matrix(y_test, y_pred_random_forest)
cm_log_regr = confusion_matrix(y_test, y_pred_log_regr)
cm_svc = confusion_matrix(y_test, y_pred_svc)
cm_LinearSVC = confusion_matrix(y_test, y_pred_LinearSVC)
#cm_neural_network = confusion_matrix(y_test, y_pred_neural_network)

# get the benchmark scores 
benchmark(knn, cm_knn, y_pred_knn, 'KNeighbors')
benchmark(tree_decision, cm__tree_decision, y_pred_tree_decision, 'tree_decision')
benchmark(random_forest, cm_random_forest, y_pred_random_forest,  'random forest')
benchmark(log_regr, cm_log_regr, y_pred_log_regr,  'logistic regression')
benchmark(svc, cm_svc, y_pred_svc, 'SVC')
benchmark(linear_svm, cm_LinearSVC, y_pred_LinearSVC, 'linear_svm')
#benchmark(clf, cm_neural_network, y_pred_neural_network,  'neural network')