# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 00:10:55 2017

@author: TasosLytos
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# read data

X1 = pd.read_csv('lexicons/negative-words.csv')
X2 = pd.read_csv('lexicons/positive-words.csv')

X1['score'] = -1
X2['score'] = 1

X_unify = X1.append(X2, ignore_index=True)
#bigdata = data1.append(data2, ignore_index=True)

#print X1.head(2)
#print X2.head(2)
print X_unify.head(2)

X_unify.to_csv('opinionObserver.csv')