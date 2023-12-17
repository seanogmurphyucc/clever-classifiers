import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.naive_bayes import ComplementNB
from sklearn import neighbors, datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans

from sklearn.datasets import make_blobs

from yellowbrick.cluster import InterclusterDistance

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
import glob


import sys

np.set_printoptions(threshold=sys.maxsize)

nolocs = False


#read in data
input_file = "./datasets/bank-full.csv"
df = pd.read_csv(input_file, header = 0, delimiter = ";")


df.insert(17, "Trust", 10000, True)

train, test = train_test_split(df, test_size=0.20, shuffle=True)

poison, clean = train_test_split(train, test_size=0.20, shuffle=True)




print(train.describe())
print(poison.describe())
print(test.describe())

print(df.iat[0,0])

poison.replace("housemaid", "admin.", inplace=True)
#poison.replace("entrepreneur", "ent1", inplace=True)
#poison.replace("self-employed", "entrepreneur", inplace=True)
#poison.replace("ent1", "self-employed", inplace=True)

poison["Trust"] = poison["Trust"].replace(10000,1)




poison.to_csv("poisonbal8.csv", sep=';')
train.to_csv("cleanbal8.csv", sep=";");
test.to_csv("testbal8.csv", sep=";")



poisoned = pd.concat([train,poison],axis=0)
#for index, row in poisoned.iterrows():           
  #  print(row[16])

poisoned.to_csv("poisonedbal8.csv", sep=";", index=False)
					



#construct training and testing sets, 80/20
#train, test = train_test_split(df, test_size=0.2, shuffle=True)
train = df;
#print(train.describe())
#print(test.describe())

#save our santised data to file for manual inspection
#df.to_csv(path_or_buf="./ClareFiltered.csv",index=False)

#choose our classification target; predictors should contain the remaining features

#make X and Y axes for training 



#X is made up of only the columns listed in predictors
X = train.copy()




