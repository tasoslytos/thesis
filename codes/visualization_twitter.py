# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:15:24 2016

@author: tasos
"""
import pandas as pd
import matplotlib
import numpy as np 
#matplotlib.style.use('ggplot') # Look Pretty
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import pylab as pl
from sklearn import manifold
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves


# get some data
X = pd.read_csv('MachineLearningTechniques/twitter/Datasets/thesis_twitter_data_v20.csv',header=None)

#print X.head(22)
X.columns = ['created_at', 'tweet_text', 'retweet_count', 'favorite_count', 'tweet_id', 'num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score', 'computer_score', 'computer_score_imdb', 'computer_score_pos_neg', 'computer_score_senti_word_net', 'computer_score_subjectivity', 'computer_score_amazon_tripadvisor', 'computer_score_goodreads', 'computer_score_opentable', 'computer_score_inquirer']

#print X.head(12)

X = X.dropna()

#y = X['computer_score']
y = X['computer_score_imdb']
#y = X['computer_score_pos_neg']
#y = X['computer_score_senti_word_net']
#y = X['computer_score_subjectivity']
#y = X['computer_score_amazon_tripadvisor']
#y = X['computer_score_goodreads']
#y = X['computer_score_opentable']
#y = X['computer_score_inquirer']

#print list(X.columns.values)
#print sorted(y.unique())
#print(y.value_counts())
#print(np.mean(y))
#print('----------------------------------')

#y=y*10
y = y.round()
y = y.astype(int)
#print(y.value_counts())



#X = X.drop('computer_score',1)

#drop categorical data
X = X.drop(X.columns[[0, 1, 4]], axis=1)

X = X.dropna()

y.plot.hist(alpha=0.5, bins=12, title='Frequency Machine Score per Tweet')
plt.xlabel('Machine Score')
plt.ylabel('Frequency Apearance')
plt.grid()

#y.plot.hist(alpha=0.5, bins=16, title='Frequency Machine Score per Facebook Status')

#print X
#print(np.mean(X, axis=0).sort_values(ascending=False))


part_of_speech = X[['num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb']] 
temp = part_of_speech.mean()
#print temp.sum()

#print part_of_speech
reactions = X[['retweet_count', 'favorite_count']]
X['total_reactions'] = X['retweet_count'] + X['favorite_count'] 
#print(part_of_speech)
#print(np.mean(part_of_speech))
#print('----------------------------------')





#print(reactions)
#print(np.mean(reactions))
#print('----------------------------------')
#print X
#reactions.plot.hist(alpha=0.5, bins=6)
#reactions.plot.hist(alpha=0.5, bins=59)

'''
X[['num_nn', 'num_nnp', 'num_vb']] .plot.hist(alpha=0.3, bins=14, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')
plt.show()

X[['num_nnp']] .plot.hist(alpha=0.3, bins=10, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')
plt.show()

X[['num_nn']] .plot.hist(alpha=0.3, bins=10, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')
plt.show()

X[['num_vb']] .plot.hist(alpha=0.3, bins=10, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')
plt.show()
'''

'''
X[['retweet_count']].plot.hist(alpha=0.5, bins=21)
pl.xlim([0,400])
pl.ylim([0,300])
plt.xlabel('Retweet Counts')
plt.ylabel('Frequency Apearance')

X[['favorite_count']].plot.hist(alpha=0.5, bins=59)
pl.xlim([0,400])
pl.ylim([0,180])
plt.xlabel('Favorite Counts')
plt.ylabel('Frequency Apearance')
plt.show()
'''


#part_of_speech.plot.hist(alpha=0.5, bins=16)
#reactions.plot.hist(alpha=0.5, bins=36)
#ret_cou = X[['retweet_count']]
#fav_cou = X[['favorite_count']]
#ret_cou.plot.hist(alpha=0.5, bins=100)
#fav_cou.plot.hist(alpha=0.5, bins=100)
#X.plot.scatter(x='retweet_count', y='computer_score', title = 'Computer Score on Retweets')
#X.plot.scatter(x='favorite_count', y='computer_score', title = 'Computer Score on Favorites')
#plt.show()

'''
X.plot.scatter(x='num_vb', y='computer_score', title='Computer Score on Verb, Base Form')
plt.xlabel('vb, Verb, Base Form')
plt.ylabel('computer_score')
plt.grid()
plt.show()
X.plot.scatter(x='num_vb', y='total_reactions', title='Total Reactions on Verb, Base Form')
plt.xlabel('vb, Verb, Base Form')
plt.ylabel('total_reactions')
plt.grid()
plt.show()
X[['num_vb']].plot.hist(alpha=0.5, bins=7, title = 'Frequency Verb, Base Form')
plt.xlabel('Verb, Base Form')
plt.grid()
plt.show()
'''
'''
X.plot.scatter(x='num_nnp', y='computer_score', title='Computer Score on Proper Noun, Singular')
plt.xlabel('nnp, Proper Noun, Singular')
plt.ylabel('computer_score')
plt.grid()
plt.show()
X.plot.scatter(x='num_nnp', y='total_reactions', title='Total Reactions on Proper Noun, Singular')
plt.xlabel('nnp, Proper Noun, Singular')
plt.ylabel('total_reactions')
plt.grid()
plt.show()
X[['num_nnp']].plot.hist(alpha=0.5, bins=11, title = 'Frequency Proper Noun, Singular per Tweet')
plt.xlabel('Proper Noun, Singular')
plt.grid()
plt.show()
'''
'''
X.plot.scatter(x='num_nn', y='computer_score', title='Computer Score on Noun, Singular or Mass')
plt.xlabel('nn, Noun, Singular or Mass')
plt.ylabel('computer_score')
plt.grid()
plt.show()
X.plot.scatter(x='num_nn', y='total_reactions', title='Total Reactions on Noun, Singular or Mass')
plt.xlabel('nn, Noun, Singular or Mass')
plt.ylabel('total_reactions')
plt.grid()
plt.show()
X[['num_nn']].plot.hist(alpha=0.5, bins=13, title = 'Frequency Proper Noun, Singular per Tweet')
plt.xlabel('nn, Noun, Singular')
plt.ylabel('total_reactions')
plt.grid()
plt.show()
'''
'''
X.plot.scatter(x='num_dt', y='computer_score', title='Computer Score on Determiner')
plt.xlabel('dt, Determiner')
plt.ylabel('computer_score')
plt.grid()
plt.show()
X.plot.scatter(x='num_dt', y='total_reactions', title='Total Reactions on Determiner')
plt.xlabel('dt, Determiner')
plt.ylabel('total_reactions')
plt.grid()
plt.show()
X[['num_dt']].plot.hist(alpha=0.5, bins=5, title = 'Frequency Determiner')
plt.xlabel('dt, Determiner')
plt.ylabel('total_reactions')
plt.grid()
plt.show()
'''
'''
X.plot.scatter(x='num_prp', y='computer_score', title='Computer Score on Personal pronoun')
plt.xlabel('Personal pronoun')
plt.ylabel('computer_score')
plt.grid()

X.plot.scatter(x='num_in', y='computer_score', title='Computer Score on Preposition or Subordinating Conjunction')
plt.xlabel('Preposition or Subordinating Conjunction')
plt.ylabel('computer_score')
plt.grid()
'''
'''
X.plot.scatter(x='num_jj', y='computer_score', title='Computer Score on Adjective')
plt.xlabel('Adjective')
plt.ylabel('computer_score')
plt.grid()
plt.show()
X.plot.scatter(x='num_jj', y='total_reactions', title='Total Reactions on Adjective')
plt.xlabel('jj, Adjective')
plt.ylabel('total_reactions')
plt.grid()
plt.show()
X[['num_jj']].plot.hist(alpha=0.5, bins=5, title = 'Frequency Determiner')
plt.xlabel('jj, Adjective')
plt.ylabel('total_reactions')
plt.grid()
plt.show()
'''
#X.plot.scatter(x='num_in', y='computer_score')
#X.plot.scatter(x='num_nn', y='computer_score')
#X.plot.scatter(x='num_nnp', y='computer_score')
#X.plot.scatter(x='num_vb', y='computer_score')
#X.plot.scatter(x='num_jj', y='computer_score')
#X.plot.scatter(x='human_score', y='computer_score')

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlabel('Retweet')
#ax.set_ylabel('Computer Score')
#ax.set_zlabel('Num of nouns')
#ax.scatter(X.retweet_count, X.computer_score, X.num_nn, c='r', marker='.')
#plt.show()
#ax.scatter(X.favorite_count, X.computer_score, X['num_jj'], c='r', marker='.')
#plt.show()

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Reactions')
ax.set_ylabel('Computer Score')
ax.set_zlabel('Num of determiners')
ax.scatter(X.total_reactions, X.computer_score, X['num_dt'], c='r', marker='.')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Reactions')
ax.set_ylabel('Computer Score')
ax.set_zlabel('Num of Noun, Singular or Mass')
ax.scatter(X.total_reactions, X.computer_score, X['num_nn'], c='r', marker='.')
plt.show()
'''

#y = X['computer_score']
#y = X['computer_score_imdb']
#y = X['computer_score_pos_neg']
#y = X['computer_score_senti_word_net']
#y = X['computer_score_subjectivity']
#y = X['computer_score_amazon_tripadvisor']
#y = X['computer_score_goodreads']
#y = X['computer_score_opentable']
#y = X['computer_score_inquirer']

#X.plot.scatter(x='total_reactions', y='computer_score_inquirer', title='Computer Score on Reactions')
#X.plot.scatter(x='retweet_count', y='computer_score_imdb', title='Computer Score on Retweet')
#X.plot.scatter(x='favorite_count', y='computer_score_imdb', title='Computer Score on Favorite')
#plt.show()


fig, ax = plt.subplots(1,2, figsize=(20,8))
#ax[0].hist(y, normed=True, alpha=0.5)
hist, bins = np.histogram(y)
ax[1].bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=0.5)
#ax[0].set_title('normed=True')
ax[1].set_title('Score Appearance Probability')
plt.xlabel('Scores')
plt.ylabel('Probability of Apearance')
plt.grid()





'''
pca = PCA(n_components=3)
pca.fit(X)
T = pca.transform(X)
print T.shape
#print T
pl.plot(np.array(T[:,0]),np.array(T[:,1]), np.array(T[:,2]),'x-',label = 'PCA')
print('----------------------------------')
'''
'''
iso = manifold.Isomap(n_neighbors=4 ,n_components=3 )
iso.fit(X)
manifold = iso.transform(X)
print X.shape
print manifold.shape
#pl.plot(np.array(manifold[:,0]),np.array(manifold[:,1]),'x-',label = 'PCA')
pl.plot(np.array(manifold[:,0]),np.array(manifold[:,1]),np.array(manifold[:,2]) , 'x-',label = 'PCA')
'''

X2 = part_of_speech
X3 = reactions
#print X2.head(2)
#print X3.head(2)
string_reactions = ['-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '9']
string_reactions = ['0', '1', '2', '3', '4']

part_of_speech_7_b = X[['num_dt', 'num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb']] 
part_of_speech_4_b = X[['num_jj', 'num_nn', 'num_nnp', 'num_vb']] 
new_computer_score =[]

'''	
x_computer_score = X.total_reactions.tolist()
new_computer_score =[]
for i in range(0,len(x_computer_score)):
	if x_computer_score[i] < 100:
		new_computer_score.append(0)
	elif 	 x_computer_score[i] < 200:
		new_computer_score.append(1)
	elif 	 x_computer_score[i] < 500:
		new_computer_score.append(2)
	#elif 	 x_computer_score[i] < 8:
	#	new_computer_score.append(3)
	else :	
		new_computer_score.append(3)

'''
x_computer_score = X.computer_score.tolist()
for i in range(0,len(x_computer_score)):
	if x_computer_score[i] < 1:
		new_computer_score.append(0)
	elif 	 x_computer_score[i] < 3:
		new_computer_score.append(1)
	elif 	 x_computer_score[i] < 4:
		new_computer_score.append(2)
	#elif 	 x_computer_score[i] < 8:
	#	new_computer_score.append(3)
	else :	
		new_computer_score.append(3)	

#X['target_names'] = [X.computer_score[i] for i in X.status_type]
#X2['target_names'] = [string_reactions[i] for i in X.computer_score]
#X3['target_names'] = [string_reactions[i] for i in X.computer_score]
#X3['target_names'] = [string_reactions[i] for i in new_computer_score]
part_of_speech_7_b['target_names'] = [string_reactions[i] for i in new_computer_score]
#part_of_speech_7_b['target_names'] = [string_reactions[i] for i in X.computer_score]
#part_of_speech_4_b['target_names'] = [string_reactions[i] for i in new_computer_score]
#print X.status_type

'''
# Parallel Coordinates Start Here:
plt.figure()
parallel_coordinates(part_of_speech_7_b, 'target_names')
plt.xlabel('Part Of Speech')
plt.ylabel('Machine Score')
plt.show()
'''

'''
plt.figure()
parallel_coordinates(X3, 'target_names')
plt.show()

X3['target_names'] = [string_reactions[i] for i in new_computer_score]
part_of_speech_4_b['target_names'] = [string_reactions[i] for i in new_computer_score]

# Andrews Curves Start Here:
plt.figure()
andrews_curves(part_of_speech_4_b, 'target_names')
plt.show()

# Andrews Curves Start Here:
plt.figure()
andrews_curves(X3, 'target_names')
plt.show()
'''