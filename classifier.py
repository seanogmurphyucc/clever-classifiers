import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

from sklearn import preprocessing


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


import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans


import sys

np.set_printoptions(threshold=sys.maxsize)


#read in data
input_file = "./datasets/bottle.csv"
df = pd.read_csv(input_file, header = 0, delimiter = ",")




#clean up data - remove spaces and slashes
#df.columns = [c.replace('/', '') for c in df.columns]
#df.columns = [c.replace(' ', '_') for c in df.columns]

#clean up data - fill in empty spaces
#df.Time_started.fillna("Unknown", inplace=True)
#df.Time_finished.fillna("Unknown", inplace=True)
#df.Elapsed_Duration.fillna("Unknown",inplace=True)

#clean up data - remove rows without start/finish/duraction data

#df.dropna(
 #   axis='T_degC',
   # how='any',
  #  thresh=None,
  #  subset=None,
 #   inplace=True
#)



##clean up data - convert 0 and 1 to booleans
#df.replace(0,False,inplace=True)
#df.replace(1,True,inplace=True)

print (df.shape)




#use LabelEncoder to exchange strings with integers
#le = preprocessing.LabelEncoder()

#hardcoding the times so as to enforce ordering of integers
#le.fit(df)
#list(le.classes_)




allAxes = ['T_degC', 'Salnty','O2Sat','ChlorA']
df = df.loc[:,allAxes]
df = df.dropna()


#default Trust value is 4
df['Trust'] = 1000


train, test = train_test_split(df, test_size=0.2, shuffle=True)

#set up the columns for our csv
print("Prop,SqrErr, R2, MaxErr, MeanGammaDev")

#The following code we want to run 10 times
#each time we take a 0,10,20,30% of df and poison its T_degC - 25 % of the poisoned data is adjusted upwards by 25%
#the "poisoned" data has a Trust value of 1 regardless of whether it was altered or not

#loop starts here

for proportion in range(10):  
  if proportion > 0:

   
    clean,poison = train_test_split(train, test_size=0.1*proportion, shuffle=True)

    #TODO poison the O2 and T_degX here

   
    
    #remains, poisoned = train_test_split(poison, test_size=0.25, shuffle=True)

    #poisoned O2 is higher by a factor of 0-25%
    poison.T_degC = poison.T_degC * ((random.random()/10) +1)
    poison.O2Sat = poison.O2Sat * ((random.random()/10) +1)

    #frames = remains,poisoned
   # poison = pd.concat(frames)

    #alter the Trust weight to 1
    poison['Trust'] =1
    clean['Trust'] = 100



    frames = [clean,poison]
   # df = clean
    train = pd.concat(frames)
    #train = clean
  #  df = clean
  #  train = df
    
 


#construct training and testing sets, 80/20

#print(train.describe())
#print(test.describe())

#choose our classification target; predictors should contain the remaining features
  predictors = ['T_degC','O2Sat','Salnty',]
  #predictors = ['T_degC']
  target = 'ChlorA'

 #we use the weights from the train set
  weights = train['Trust']

#then we remove the weights for training with (instead we pass it as its own array later)
  allAxes = ['T_degC', 'Salnty','O2Sat','ChlorA']
  train = train.loc[:,allAxes]
#make X and Y axes for training 

#X is made up of only the columns listed in predictors
  X = train.loc[:,predictors]
#y is the class to be predicted
  y = train['ChlorA']

  kmeans = MiniBatchKMeans(n_clusters=8,
                         n_init=3,  
                         random_state=0,
                         max_iter=10).fit(X)
  #print(kmeans.cluster_centers_)

  linear = LinearRegression()
  linear.fit(X, y, sample_weight=weights)
  
 # y_pred = kmeans.predict(test.loc[:,predictors])

  y_pred = linear.predict(test.loc[:,predictors])

  cs = metrics.r2_score(test[target], y_pred)

  #print(cs,"," ps)


  

  #pred = pd.DataFrame(y_pred)
#print(pred.to_csv())

#print(test[target].to_csv())
  per = proportion*10
  
 # print("{}%,".format(per),metrics.mean_squared_error(test[target], y_pred) , "," , metrics.r2_score(test[target], y_pred),",",metrics.max_error(test[target], y_pred),",",metrics.mean_gamma_deviance(test[target], y_pred))
  

#loop ends here





