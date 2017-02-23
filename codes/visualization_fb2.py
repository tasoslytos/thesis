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
from scipy.stats import gaussian_kde
from collections import defaultdict
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
			
				
X = pd.read_csv('MachineLearningTechniques/fb/Datasets/thesis_facebook_data_v24.csv',header=None)

#print X.head(1)
X.columns = ['status_id', 'status_message', 'link_name', 'status_type', 'status_link', 'status_published', 'num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys', 'num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb', 'human_score', 'computer_score', 'computer_score_imdb', 'computer_score_pos_neg', 'computer_score_senti_word_net', 'computer_score_subjective', 'computer_score_amazon_tripadvisor', 'computer_score_goodreads', 'computer_score_opentable', 'computer_score_wnscore_inquirer' ]
#print X
#y = X['computer_score']
#y = X['computer_score_imdb']
y = X['computer_score_pos_neg']
#y = X['computer_score_senti_word_net']
#y = X['computer_score_subjective']
#y = X['computer_score_amazon_tripadvisor']
#y = X['computer_score_goodreads']
#y = X['computer_score_opentable']
#y = X['computer_score_wnscore_inquirer']

#print sorted(y.unique())
#print(y.value_counts())
print 'mean value: '
print(np.mean(y))

y = y.round()
y = y.astype(int)
print(y.value_counts())
#y=y*10
#print('----------------------------------')

#X = X.drop('computer_score',1)

#X = pd.get_dummies(X,columns=['status_type'])
X.status_type = X.status_type.map({'link':1, 'photo':2, 'status':3, 'video':4 })

'''
X.status_type.plot.hist(alpha=0.3, title='Status Type per Facebook Status')
plt.xlabel('Status Type')
plt.ylabel('Frequency Apearance')
plt.grid()
'''

#X = pd.get_dummies(X['status_type']).astype(bool)
#print X

#print X.status_type

#drop categorical data
X = X.drop(X.columns[[0, 1, 2, 4, 5]], axis=1)

X = X.dropna()


y.plot.hist(alpha=0.5, bins=16, title='Frequency Machine Score per Facebook Status')
plt.xlabel('Machine Score')
plt.ylabel('Frequency Apearance')
plt.grid()

#print(np.mean(X, axis=0))


part_of_speech = X[['num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb']] 

temp = part_of_speech.mean()
print temp.sum()

part_of_speech_7 = X[['num_dt', 'num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb']] 
part_of_speech_11 = X[['num_dt', 'num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb', 'num_nns', 'num_cd', 'num_rb', 'num_vbz']] 

#print part_of_speech


reactions = X[['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']]
reactions_minus = X[['num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']]

#print(part_of_speech)
#print sorted(np.mean(part_of_speech))
#print('----------------------------------')
#print np.mean(part_of_speech)


most_used_part_of_speech = X[['num_dt', 'num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb']] 
#print(np.mean(most_used_part_of_speech))
#print('----------------------------------')

'''
#print(reactions)
print(np.mean(reactions))
plt.figure()
X[['num_reactions']].plot.hist(alpha=0.5, bins=5)
X[['num_jj' ]].plot.hist(alpha=0.5, bins=7)
X[['num_nn']].plot.hist(alpha=0.5, bins=9)
X[['num_nnp']].plot.hist(alpha=0.5, bins=14)
X[['num_prp']].plot.hist(alpha=0.5, bins=6)
X[['num_vb']].plot.hist(alpha=0.5, bins=13)
plt.show()

plt.figure()
print('----------------------------------')
'''

# plot reactions 
'''
X[['num_reactions']].plot.hist(alpha=0.5, )
X[['num_comments' ]].plot.hist(alpha=0.5)
X[['num_shares']].plot.hist(alpha=0.5)
X[['num_likes']].plot.hist(alpha=0.5)
X[['num_loves']].plot.hist(alpha=0.5)
X[['num_wows']].plot.hist(alpha=0.5)
X[['num_hahas']].plot.hist(alpha=0.5)
X[['num_sads']].plot.hist(alpha=0.5)
X[['num_angrys']].plot.hist(alpha=0.5)
'''
#only_minor_reactions.plot.hist(alpha=0.5)
#part_of_speech.plot.hist(alpha=0.5, bins=16)

'''
X[['num_reactions']].plot.hist(alpha=0.5, title='Frequency Reactions per Facebook Status')
plt.xlabel('Reactions')
plt.ylabel('Frequency Apearance')
plt.grid()

X[['num_comments']].plot.hist(alpha=0.5, title='Frequency Comments per Facebook Status')
plt.xlabel('Comments')
plt.ylabel('Frequency Apearance')
plt.grid()


X[['num_shares']].plot.hist(alpha=0.5, title='Frequency Shares per Facebook Status')
plt.xlabel('Shares')
plt.ylabel('Frequency Apearance')
plt.grid()

part_of_speech.plot.hist(alpha=0.3, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')

part_of_speech_11.plot.hist(alpha=0.3, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')

part_of_speech_7.plot.hist(alpha=0.3, bins=14, title='Part of Speech per Facebook Status')
plt.xlabel('Part Of Speech')
plt.ylabel('Frequency Apearance')

X[['num_nn', 'num_nnp', 'num_vb']] .plot.hist(alpha=0.3, bins=14, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')

X[['num_nn']] .plot.hist(alpha=0.3, bins=9, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')
plt.grid()

X[['num_nnp']] .plot.hist(alpha=0.3,bins=14, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')

X[['num_vb']] .plot.hist(alpha=0.3, bins=13, title='Part of Speech per Facebook Status')
plt.xlabel('Part of Speech')
plt.ylabel('Frequency Apearance')

'''
#most_used_part_of_speech.plot.hist()
#X[['num_dt', 'num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb']] .plot.hist()
#X[['num_dt', ]] .plot.hist(alpha=0.5, bins=5)
#X[['num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb']] .plot.hist()

'''
X[['num_reactions']].plot.hist(alpha=0.5, bins=16, title='Frequency Reactions per Facebook Status')
plt.xlabel('Reactions')
plt.ylabel('Frequency Apearance')

X[['num_comments']].plot.hist(alpha=0.5, bins=16, title='Frequency Comments per Facebook Status')
plt.xlabel('Comments')
plt.ylabel('Frequency Apearance')

X[['num_shares']].plot.hist(alpha=0.5, bins=16, title='Frequency Shares per Facebook Status')
plt.xlabel('Shares')
plt.ylabel('Frequency Apearance')

X[['num_likes']].plot.hist(alpha=0.5, bins=16, title='Frequency Likes per Facebook Status')
plt.xlabel('Likes')
plt.ylabel('Frequency Apearance')

X[['num_loves']].plot.hist(alpha=0.5, bins=16, title='Frequency Loves per Facebook Status')
plt.xlabel('Loves')
plt.ylabel('Frequency Apearance')

X[['num_wows']].plot.hist(alpha=0.5, bins=16, title='Frequency Wows per Facebook Status')
plt.xlabel('Wows')
plt.ylabel('Frequency Apearance')

X[['num_hahas']].plot.hist(alpha=0.5, bins=16, title='Frequency Hahas per Facebook Status')
plt.xlabel('Hahas')
plt.ylabel('Frequency Apearance')

X[['num_sads']].plot.hist(alpha=0.5, bins=16, title='Frequency Sads per Facebook Status')
plt.xlabel('Sads')
plt.ylabel('Frequency Apearance')

X[['num_angrys']].plot.hist(alpha=0.5, bins=16, title='Frequency Angrys per Facebook Status')
plt.xlabel('Angrys')
plt.ylabel('Frequency Apearance')
'''

'''
plt.figure()
X[['num_dt']].plot.hist(alpha=0.5, bins=5)
X[['num_jj' ]].plot.hist(alpha=0.5, bins=7)
X[['num_nn']].plot.hist(alpha=0.5, bins=9)
X[['num_nnp']].plot.hist(alpha=0.5, bins=14)
X[['num_prp']].plot.hist(alpha=0.5, bins=6)
X[['num_vb']].plot.hist(alpha=0.5, bins=13)
plt.show()

plt.figure()

X[['num_dt']].hist(alpha=0.3)
X[['num_jj']].hist(alpha=0.3)
X[['num_nn']].hist(alpha=0.3)
X[['num_dt']].hist(alpha=0.3)
X[['num_nnp']].hist(alpha=0.3)
X[['num_prp']].hist(alpha=0.3)
X[['num_vb']].hist(alpha=0.3)

X[['num_dt', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb']] .plot.hist()
plt.show()


'''
#X.plot.scatter(x='status_type', y='computer_score', title='Computer Score on Status Type')
#X.plot.scatter(x='status_type', y='num_reactions', title='Reactions on Status Type')
#X.plot.scatter(x='status_type', y='computer_score', title='Computer Score on Status Type')

'''
X.plot.scatter(x='status_type', y='num_likes', title='Reactions on Status Type')
X.plot.scatter(x='status_type', y='num_loves', title='Reactions on Status Type')
X.plot.scatter(x='status_type', y='num_wows', title='Reactions on Status Type')
X.plot.scatter(x='status_type', y='num_hahas', title='Reactions on Status Type')
X.plot.scatter(x='status_type', y='num_sads', title='Reactions on Status Type')
X.plot.scatter(x='status_type', y='num_angrys', title='Reactions on Status Type')
'''

#X.plot.scatter(x='status_type', y='num_comments', title='Comments on Status Type')
#X.plot.scatter(x='status_type', y='num_shares', title='Shares on Status Type')
#'num_reactions', 'num_comments', 'num_shares',


#y = X['computer_score']
#y = X['computer_score_imdb']
#y = X['computer_score_pos_neg']
#y = X['computer_score_senti_word_net']
#y = X['computer_score_subjective']
#y = X['computer_score_amazon_tripadvisor']
#y = X['computer_score_goodreads']
#y = X['computer_score_opentable']
#y = X['computer_score_wnscore_inquirer']


#X.plot.scatter(x='num_reactions', y='computer_score_wnscore_inquirer', title='Computer Score on Reactions')
#X.plot.scatter(x='num_comments', y='computer_score_pos_neg', title='Computer Score on Comments')
#X.plot.scatter(x='num_shares', y='computer_score_senti_word_net', title='Computer Score on Shares')
X.plot.scatter(x='num_wp', y='computer_score_imdb', title='Computer Score on Shares')




#x = y

fig, ax = plt.subplots(1,2, figsize=(20,8))
#ax[0].hist(y, normed=True, alpha=0.5)
hist, bins = np.histogram(y)
ax[1].bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=0.5)
#ax[0].set_title('normed=True')
ax[1].set_title('Score Appearance Probability')
plt.xlabel('Scores')
plt.ylabel('Probability of Apearance')
plt.grid()





#X.plot.scatter(x='num_likes', y='computer_score')

#most used PoS


'''
X.plot.scatter(x='num_vb', y='computer_score', title='Computer Score on Verb, Base Form')
plt.grid()
X.plot.scatter(x='num_vb', y='num_reactions', title='Reactions on VB')
plt.grid()
X[['num_vb']].plot.hist(alpha=0.5, bins=13, title='Frequency VB per Facebook Status')
plt.xlabel('vb, Verb, Base Form')
plt.ylabel('Frequency Apearance')
plt.grid()

X.plot.scatter(x='num_nnp', y='computer_score', title='Computer Score on Proper Noun, Singular')
plt.grid()
X.plot.scatter(x='num_nnp', y='num_reactions', title='Reactions on NNP')
plt.grid()
X[['num_nnp']].plot.hist(alpha=0.5, bins=14, title='Frequency NNP per Facebook Status')
plt.xlabel('nnp, Proper Noun, Singular')
plt.ylabel('Frequency Apearance')
plt.grid()

X.plot.scatter(x='num_nn', y='computer_score', title='Computer Score on Noun, Singular or Mass')
plt.grid()
X.plot.scatter(x='num_nn', y='num_reactions', title='Reactions on NN')
plt.grid()
X[['num_nnp']].plot.hist(alpha=0.5, bins=14, title='Frequency NN per Facebook Status')
plt.xlabel('nn, Noun, Singular or Mass')
plt.ylabel('Frequency Apearance')
plt.grid()

X.plot.scatter(x='num_dt', y='computer_score', title='Computer Score on DT')
X.plot.scatter(x='num_dt', y='num_reactions', title='Reactions on DT')
X[['num_dt']].plot.hist(alpha=0.5, bins=5, title='Frequency DT per Facebook Status')
plt.xlabel('dt, base form')
plt.ylabel('Frequency Apearance')

X.plot.scatter(x='num_prp', y='computer_score', title='Computer Score on Preposition or Subordinating Conjunction')
plt.grid()
X.plot.scatter(x='num_prp', y='num_reactions', title='Reactions on Preposition or Subordinating Conjunction')
plt.grid()
X[['num_prp']].plot.hist(alpha=0.5, bins=6, title='Frequency PRP per Facebook Status')
plt.xlabel('Preposition or Subordinating Conjunction')
plt.ylabel('Frequency Apearance')
plt.grid()

X.plot.scatter(x='num_in', y='computer_score', title='Computer Score on Preposition or Subordinating Conjunction')
X.plot.scatter(x='num_in', y='num_reactions', title='Reactions on Preposition or Subordinating Conjunction')
X[['num_in']].plot.hist(alpha=0.5, bins=5, title='Frequency PRP per Facebook Status')
plt.xlabel('Preposition or Subordinating Conjunction')
plt.ylabel('Frequency Apearance')

X.plot.scatter(x='num_jj', y='computer_score', title='Computer Score on Adjective')
plt.grid()
X.plot.scatter(x='num_jj', y='num_reactions', title='Reactions on Adjective')
plt.grid()
X[['num_jj']].plot.hist(alpha=0.5, bins=7, title='Frequency JJ per Facebook Status')
plt.xlabel('Adjective')
plt.ylabel('Frequency Apearance')
plt.grid()
'''
'''
freq = [0,0,0,0,0,0,0,0,0]
X[['num_jj' ]].plot(alpha=0.5)
#print X[['num_jj' ]]
num_jj = X[['num_jj' ]]
#print num_jj
for i in range(0, len(num_jj)):
	if i not in X.num_jj:
		X.num_jj[i] = 1
	for j in range(0,len(freq)):
		if X.num_jj[i] == j:
			freq[j] = freq[j] + 1 
	#print X.num_jj[i]

print freq[0]
print freq[8]
plt.plot(freq)



freq_prp = [0,0,0,0,0,0,0,0,0]
X[['num_prp' ]].plot(alpha=0.5)
#print X[['num_jj' ]]
num_prp = X[['num_prp' ]]
#print num_jj
for i in range(0, len(num_prp)):
	if i not in X.num_prp:
		X.num_prp[i] = 1
	for j in range(0,len(freq_prp)):
		if X.num_prp[i] == j:
			freq_prp[j] = freq_prp[j] + 1 
	#print X.num_jj[i]

print freq_prp[0]
print freq_prp[6]
plt.plot(freq_prp)
'''

'''
X.plot.scatter(x='num_nn', y='computer_score', title='Computer Score on Noun, Singular or Mass')
X.plot.scatter(x='num_vb', y='computer_score', title='Computer Score on Verb, Base Form')
X.plot.scatter(x='num_dt', y='computer_score', title='Computer Score on Determiner')
X.plot.scatter(x='num_in', y='computer_score', title='Computer Score on Preposition or Subordinating Conjunction')
X.plot.scatter(x='num_jj', y='computer_score', title='Computer Score on Adjective')
X.plot.scatter(x='num_prp', y='computer_score', title='Computer Score on Personal Pronoun')
'''

density = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#print density

x_nnp = X.num_nnp
#print x_nnp
#x_nnp = defaultdict(0)
density_300 = []
for i in range(0,len(X.num_nnp)):	
	if i not in X.num_nnp:
		X.num_nnp[i] = 1
	for j in range(0, 14):
		if X.num_nnp[i] == j:
			density[j] = density[j] + 1

for i in range(0,len(X.num_nnp)):	
	for j in range(0, 14):
		if X.num_nnp[i] == j:
			density_300.append(density[j])
			
#x_nnp
#y
x = X.num_nnp
z = density_300


#print density
#print density_300

'''
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()
'''





#all possible PoS
'''
X.plot.scatter(x='num_cc', y='computer_score', title='Computer Score on Proper Noun, Singular')
X.plot.scatter(x='num_cd', y='computer_score', title='Computer Score on Noun, Singular or Mass')
X.plot.scatter(x='nun_ex', y='computer_score', title='Computer Score on Verb, Base Form')
X.plot.scatter(x='num_fw', y='computer_score', title='Computer Score on Determiner')
X.plot.scatter(x='num_jjr', y='computer_score', title='Computer Score on Preposition or Subordinating Conjunction')
X.plot.scatter(x='num_jjs', y='computer_score', title='Computer Score on Adjective')
X.plot.scatter(x='num_ls', y='computer_score', title='Computer Score on Personal Pronoun')
X.plot.scatter(x='num_md', y='computer_score', title='Computer Score on Proper Noun, Singular')
X.plot.scatter(x='num_nns', y='computer_score', title='Computer Score on Noun, Singular or Mass')
X.plot.scatter(x='num_nnps', y='computer_score', title='Computer Score on Verb, Base Form')
X.plot.scatter(x='num_pdt', y='computer_score', title='Computer Score on Determiner')
X.plot.scatter(x='num_pos', y='computer_score', title='Computer Score on Preposition or Subordinating Conjunction')
X.plot.scatter(x='num_jjs', y='computer_score', title='Computer Score on Adjective')
X.plot.scatter(x='num_prp6', y='computer_score', title='Computer Score on Personal Pronoun')
X.plot.scatter(x='num_rb', y='computer_score', title='Computer Score on Proper Noun, Singular')
X.plot.scatter(x='num_rbr', y='computer_score', title='Computer Score on Noun, Singular or Mass')
X.plot.scatter(x='num_rbs', y='computer_score', title='Computer Score on Verb, Base Form')
X.plot.scatter(x='num_rp', y='computer_score', title='Computer Score on Determiner')
X.plot.scatter(x='num_sym', y='computer_score', title='Computer Score on Preposition or Subordinating Conjunction')
X.plot.scatter(x='num_to', y='computer_score', title='Computer Score on Adjective')
X.plot.scatter(x='num_uh', y='computer_score', title='Computer Score on Personal Pronoun')
X.plot.scatter(x='num_vbd', y='computer_score', title='Computer Score on Proper Noun, Singular')
X.plot.scatter(x='num_vbg', y='computer_score', title='Computer Score on Noun, Singular or Mass')
X.plot.scatter(x='num_vbn', y='computer_score', title='Computer Score on Verb, Base Form')
X.plot.scatter(x='num_vbp', y='computer_score', title='Computer Score on Determiner')
X.plot.scatter(x='num_vbz', y='computer_score', title='Computer Score on Preposition or Subordinating Conjunction')
X.plot.scatter(x='num_wdt', y='computer_score', title='Computer Score on Adjective')
X.plot.scatter(x='num_wp', y='computer_score', title='Computer Score on Personal Pronoun')
X.plot.scatter(x='num_wp6', y='computer_score', title='Computer Score on Adjective')
X.plot.scatter(x='num_wrb', y='computer_score', title='Computer Score on Personal Pronoun')
'''
#X.plot.scatter(x='num_reactions', y='computer_score', title='Computer Score on Reactions')

#X.plot.scatter(x='human_score', y='computer_score')

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Reactions')
ax.set_ylabel('Computer Score')
ax.set_zlabel('Num of adjectives')

ax.scatter(X.num_reactions, X.computer_score, X['num_jj'], c='r', marker='.')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Reactions')
ax.set_ylabel('Computer Score')
ax.set_zlabel('Num of Preposition or Subordinating Conjunction')

ax.scatter(X.num_reactions, X.computer_score, X['num_prp'], c='r', marker='.')
plt.show()
'''


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
iso = manifold.Isomap(n_neighbors=6, n_components=3 )
iso.fit(X)
manifold = iso.transform(X)
print X.shape
print manifold.shape
#pl.plot(np.array(manifold[:,0]),np.array(manifold[:,1]),'x-',label = 'PCA')
pl.plot(np.array(manifold[:,0]),np.array(manifold[:,1]),np.array(manifold[:,2]) , 'x-',label = 'PCA')
'''


part_of_speech_b = X[['num_cc', 'num_cd', 'num_dt', 'nun_ex', 'num_fw', 'num_in', 'num_jj', 'num_jjr', 'num_jjs', 'num_ls', 'num_md', 'num_nn', 'num_nns', 'num_nnp', 'num_nnps', 'num_pdt', 'num_pos', 'num_prp', 'num_prp6', 'num_rb', 'num_rbr', 'num_rbs', 'num_rp', 'num_sym', 'num_to', 'num_uh', 'num_vb', 'num_vbd', 'num_vbg', 'num_vbn', 'num_vbp', 'num_vbz', 'num_wdt', 'num_wp', 'num_wp6', 'num_wrb']] 
part_of_speech_7_b = X[['num_dt', 'num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb']] 
part_of_speech_11_b = X[['num_dt', 'num_in', 'num_jj', 'num_nn', 'num_nnp', 'num_prp', 'num_vb', 'num_nns', 'num_cd', 'num_rb', 'num_vbz']] 
#print part_of_speech
reactions_b = X[['num_reactions', 'num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']]
reactions_minus = X[['num_comments', 'num_shares', 'num_likes', 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']]
reactions_c = X[[ 'num_comments', 'num_shares']]
reactions_d = X[[ 'num_loves', 'num_wows', 'num_hahas', 'num_sads', 'num_angrys']]

#X2 = X.drop(X.columns[[ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 ]], axis=1)
#X3 = X.drop(X.columns[[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 21, 22  ]], axis=1)
#print X3

'''
#string_reactions = ['-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '9', '10', '12', '13']
string_reactions = ['0', '1', '2', '3', '4']
x_computer_score = X.computer_score.tolist()

#print x_computer_score[0]
new_computer_score =[]
#print len(x_computer_score)

for i in range(0,len(x_computer_score)):
	if x_computer_score[i] < -1:
		new_computer_score.append(0)
	elif 	 x_computer_score[i] < 2:
		new_computer_score.append(1)
	elif 	 x_computer_score[i] < 5:
		new_computer_score.append(2)
	#elif 	 x_computer_score[i] < 8:
	#	new_computer_score.append(3)
	else :	
		new_computer_score.append(3)
'''		
#X['target_names'] = [X.computer_score[i] for i in X.status_type]
#part_of_speech_4_b = X[['num_jj', 'num_nn', 'num_nnp', 'num_vb']] 
#part_of_speech_4_b['target_names'] = [string_reactions[i] for i in new_computer_score]
#part_of_speech_7_b['target_names'] = [string_reactions[i] for i in X.computer_score]
#part_of_speech_7_b['target_names'] = [string_reactions[i] for i in new_computer_score]
#reactions_b['target_names'] = [string_reactions[i] for i in X.computer_score]
#reactions_b['target_names'] = [string_reactions[i] for i in new_computer_score]
#reactions_c['target_names'] = [string_reactions[i] for i in new_computer_score]
#reactions_d['target_names'] = [string_reactions[i] for i in new_computer_score]
####################
####################
####################
#reactions_b['target_names'] = [string_reactions[i] for i in X.computer_score]
#reactions_c['target_names'] = [string_reactions[i] for i in X.computer_score]
#reactions_d['target_names'] = [string_reactions[i] for i in X.computer_score]
####################
####################
####################


string_reactions = ['-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '9', '10', '12', '13']
string_reactions = ['0', '1', '2', '3', '4']
string_reactions = ['0', '1', '2', '3']
x_computer_score = X.num_reactions.tolist()
print x_computer_score[0]
new_computer_score =[]
print len(x_computer_score)

for i in range(0,len(x_computer_score)):
	if x_computer_score[i] < 10000:
		new_computer_score.append(0)
	elif 	 x_computer_score[i] <100000:
		new_computer_score.append(1)
	elif 	 x_computer_score[i] < 160000:
		new_computer_score.append(2)
	#elif 	 x_computer_score[i] < 8:
	#	new_computer_score.append(3)
	else :	
		new_computer_score.append(3)
		
#X['target_names'] = [X.computer_score[i] for i in X.status_type]
#part_of_speech_4_b = X[['num_jj', 'num_nn', 'num_nnp', 'num_vb']] 
#part_of_speech_4_b['target_names'] = [string_reactions[i] for i in new_computer_score]
#part_of_speech_7_b['target_names'] = [string_reactions[i] for i in X.computer_score]
#part_of_speech_7_b['target_names'] = [string_reactions[i] for i in new_computer_score]
#reactions_b['target_names'] = [string_reactions[i] for i in X.computer_score]
#reactions_b['target_names'] = [string_reactions[i] for i in new_computer_score]
reactions_c['target_names'] = [string_reactions[i] for i in new_computer_score]
#reactions_d['target_names'] = [string_reactions[i] for i in new_computer_score]

#print X.status_type
'''
# Parallel Coordinates Start Here:
plt.figure()
parallel_coordinates(reactions_c, 'target_names')
plt.xlabel('Reactions')
plt.ylabel('Total Reactions')
plt.show()
'''

'''
plt.figure()
parallel_coordinates(reactions_d, 'target_names')
plt.show()
plt.figure()
parallel_coordinates(reactions_b, 'target_names')
plt.show()
plt.figure()
parallel_coordinates(reactions_c, 'target_names')
plt.show()
'''

# Andrews Curves Start Here:
#plt.figure()
#andrews_curves(part_of_speech_4_b, 'target_names')
#plt.show()
#plt.figure()
#andrews_curves(reactions_c, 'target_names')
#plt.show()
#plt.figure()
#andrews_curves(reactions_d, 'target_names')
#plt.show()