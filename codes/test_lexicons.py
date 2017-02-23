# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 23:17:06 2017

@author: TasosLytos
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick

# read data
#X = pd.read_csv('lexicons/AFINN.csv')
X = pd.read_csv('lexicons/wnscores_inquirer.csv')
#X = pd.read_csv('lexicons/final_version_of_imdb_lexicon.csv')
#X = pd.read_csv('lexicons/final_version_of_amazon-tripadvisor_lexicon_without_type_without_duplicates_avg.csv')
#X = pd.read_csv('lexicons/final_version_of_goodreads_lexicon_without_type_without_duplicates_avg.csv')
#X = pd.read_csv('lexicons/opinionObserver.csv')
#X = pd.read_csv('lexicons/SubjectiveLexicon2.csv')


print X.head(2)
print '---------------------------------------------'

print  'total words : ', len(X)
print 'mean value is : ' , X['Score'].mean()
print 'the variation is : ' , X['Score'].var()
print 'unique is : ' , X['Score'].unique()
print 'total number of unique values  : ' , len(X['Score'].unique())
print 'value counts is : ' 
print X['Score'].value_counts()


#X_sorted = X.sort(['Score'])
#print X_sorted
#df = X['ExpectedRate']


#my_var = X['ExpectedRate'].value_counts()
#my_var = 100*my_var/len(X)
#print my_var


#my_var = X['ExpectedRate']
#print my_var

'''
my_var = X['avg']
my_var.plot.hist(alpha=0.5, title='Frequency of the scores', bins=55)
#my_var.plot.hist(alpha=0.5, title='Frequency of the scores', bins=10)
plt.xlabel('Scores')
plt.ylabel('Frequency of Apearance')
plt.grid()
'''


'''
x = X['avg']

fig, ax = plt.subplots(1,2, figsize=(20,8))

#ax[0].hist(x, normed=True, alpha=0.5)

hist, bins = np.histogram(x)
ax[1].bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]), alpha=0.5)

#ax[0].set_title('normed=True')
ax[1].set_title('Score Appearance Probability')


plt.xlabel('Scores')
plt.ylabel('Probability of Apearance')
plt.grid()
'''





'''
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


x = X['score'].value_counts()

# Make a normed histogram. It'll be multiplied by 100 later.
plt.hist(x, bins=6, normed=True, alpha=0.5)

# Create the formatter using the function to_percent. This multiplies all the
# default labels by 100, making them all percentages
formatter = FuncFormatter(to_percent)

# Set the formatter
plt.gca().yaxis.set_major_formatter(formatter)

plt.xlabel('Scores')
plt.title('Score Appearance Percentage %')
plt.ylabel('Frequency of Apearance')
plt.grid()

plt.show()

'''





'''
my_var = X['avg'].value_counts()
df =my_var.to_frame()
df.reset_index(inplace=True)
df.columns = ['score','frequency']
#print df
df.plot(kind='bar',x='score',y='frequency')
plt.xlabel('Scores')
plt.ylabel('Frequency of Apearance')
plt.grid()
'''


'''
my_var = X['avg'].value_counts()
my_var = my_var/8699
print my_var
plt.scatter(my_var.index, my_var)
plt.xlabel('Scores')
plt.title('Probability of Appearance')
plt.ylabel('Frequency of Apearance')
plt.grid()
'''