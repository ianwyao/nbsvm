# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:04:18 2020

@author: Ian
"""
import nltk
from sklearn import preprocessing
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
#from mpl_toolkits.mplot3d import Axes3D 
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz 
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
from IPython.display import Image  
## conda install pydotplus
#import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
#from nltk.stem import WordNetLemmatizer 
#LEMMER = WordNetLemmatizer() 
from nltk.stem.porter import PorterStemmer
import string
import numpy as np



STEMMER=PorterStemmer()
def MY_STEMMER(str_input):
    words = re.sub(r"[^A-Za-z\']", " ", str_input).lower().split()
    words = [STEMMER.stem(word) for word in words]
    return words



MyVect_STEM=CountVectorizer(input='filename',
                        analyzer = 'word',
                        stop_words='english',
                        ##stop_words=["and", "or", "but"],
                        token_pattern='(?u)[a-zA-Z]+',
                        #token_pattern=pattern,
                        tokenizer=MY_STEMMER,
                        #strip_accents = 'unicode', 
                        lowercase = True
                        )



#We will be creating new data frames
FinalDF_STEM=pd.DataFrame()


## This code assumes that it is in the same folder/location
## as the folders. It will loop through the files in
## these folders and will build the list needed to use
## CounterVectorizer. 

for name in ['hip','pop','rock']: ## Charlie Puth and Alan Walker

    builder=name+"DF"
    #print(builder)
    path = name
    
    FileList=[]
    for item in os.listdir(path):
        #print(path+ "\\" + item)
        next=path+ "\\" + item
        FileList.append(next)  
        print("full list...")
        #print(FileList)
        X1=MyVect_STEM.fit_transform(FileList)
        
        ColumnNames1=MyVect_STEM.get_feature_names()
        NumFeatures1=len(ColumnNames1)

        #print("Column names: ", ColumnNames2)
        #Create a name
        
    builderS=pd.DataFrame(X1.toarray(),columns=ColumnNames1)
    ## Add column
    #print("Adding new column....")
    builderS["LABEL"]=name

    #print(builderS)
    FinalDF_STEM= FinalDF_STEM.append(builderS)

FinalDF=FinalDF_STEM.fillna(0)   




TrainDF, TestDF = train_test_split(FinalDF, test_size=0.2)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
#print(TestLabels)
## remove labels
## Make a copy of TestDF
CopyTestDF=TestDF.copy()
TestDF = TestDF.drop(["LABEL"], axis=1)
print(TestDF)

## DF seperate TRAIN SET from the labels
TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
#print(TrainDF_nolabels)
TrainLabels=TrainDF["LABEL"]
#print(TrainLabels)




####################################################################
########################### Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB

MyModelNB= MultinomialNB()

MyModelNB.fit(TrainDF_nolabels, TrainLabels)
Prediction = MyModelNB.predict(TestDF)
print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabels)


## confusion matrix
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
### prediction probabilities
## columns are the labels in alphabetical order

print(np.round(MyModelNB.predict_proba(TestDF),2))

## VIS
from sklearn import metrics 
print(metrics.classification_report(TestLabels, Prediction))
print(metrics.confusion_matrix(TestLabels, Prediction))

import seaborn as sns
sns.heatmap(SVM_matrix, square=True, annot=True, fmt='d', 
            xticklabels=["Hip Hop", "Pop",'Rock'], yticklabels=["Hip Hop", "Pop",'Rock'])
plt.xlabel('True Label')
plt.ylabel('Predicted Label')


#############################################
###########  SVM ############################
#############################################
from sklearn.svm import LinearSVC
SVM_Model=LinearSVC(C=.01)
SVM_Model.fit(TrainDF_nolabels, TrainLabels)

print("SVM prediction:\n", SVM_Model.predict(TestDF))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model.predict(TestDF))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")


##################Record Data
dfr = pd.read_csv('diff_genres.csv')
dfr = dfr.drop(['filename'],axis = 1)
dfr = dfr.iloc[:,[0,1,28]]
n = 0
while (n < len(dfr['label'])):
    if (dfr['label'][n] == 'blues'):
        dfr['label'][n] = 1
    if (dfr['label'][n] == 'classical'):
        dfr['label'][n] = 2
    if (dfr['label'][n] == 'country'):
        dfr['label'][n] = 3
    if (dfr['label'][n] == 'disco'):
        dfr['label'][n] = 4
    if (dfr['label'][n] == 'hiphop'):
        dfr['label'][n] = 5
    if (dfr['label'][n] == 'jazz'):
        dfr['label'][n] = 6
    if (dfr['label'][n] == 'metal'):
        dfr['label'][n] = 7
    if (dfr['label'][n] == 'pop'):
        dfr['label'][n] = 8
    if (dfr['label'][n] == 'reggae'):
        dfr['label'][n] = 9
    if (dfr['label'][n] == 'rock'):
        dfr['label'][n] = 10
    n = n + 1
dfr['label'] = dfr['label'].astype(int)
from sklearn.naive_bayes import GaussianNB
TrainDF, TestDF = train_test_split(dfr, test_size=0.3)
TrainLabels = TrainDF['label']
TestLabels = TestDF['label']
TrainDF_nolabels = TrainDF.drop(['label'],axis = 1)
TestDF_nolabels = TestDF.drop(['label'],axis = 1)
MyModelNB= MultinomialNB()

MyModelNB.fit(TrainDF_nolabels, TrainLabels)
Prediction = MyModelNB.predict(TestDF_nolabels)
print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabels)

## Check that all data are numeric
print(dfr['label'].dtypes)

## confusion matrix
from sklearn.metrics import confusion_matrix

cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
### prediction probabilities
## columns are the labels in alphabetical order

print(np.round(MyModelNB.predict_proba(TestDF_nolabels),2))

## VIS
from sklearn import metrics 
print(metrics.classification_report(TestLabels, Prediction))
print(metrics.confusion_matrix(TestLabels, Prediction))

import seaborn as sns
sns.heatmap(cnf_matrix.T, square=True, annot=True, fmt='d', 
            xticklabels=["blues", "classical",'country','disco','hiphop','jazz','metal','pop','reggae','rock'], yticklabels=["blues", "classical",'country','disco','hiphop','jazz','metal','pop','reggae','rock'])
plt.xlabel('True Label')
plt.ylabel('Predicted Label')


SVM_Model1=LinearSVC(C=1)
SVM_Model1.fit(TrainDF_nolabels, TrainLabels)

print("SVM prediction:\n", SVM_Model1.predict(TestDF_nolabels))
print("Actual:")
print(TestLabels)

SVM_matrix = confusion_matrix(TestLabels, SVM_Model1.predict(TestDF_nolabels))
print("\nThe confusion matrix is:")
print(SVM_matrix)
print("\n\n")
#PLOT
from sklearn import metrics 
import seaborn as sns
sns.heatmap(SVM_matrix, square=True, annot=True, fmt='d', 
            xticklabels=["blues", "classical",'country','disco','hiphop','jazz','metal','pop','reggae','rock'], yticklabels=["blues", "classical",'country','disco','hiphop','jazz','metal','pop','reggae','rock'])
plt.xlabel('True Label')
plt.ylabel('Predicted Label')












########################################################
##
##  SVM Visualization Options
##
#################################################################
import numpy as np
import matplotlib.pyplot as plt


## To view the SVM - we will need to create a new dataset
## With only two decisions:  1  for Accept and 0 for Not Accept 
## Not Accept includes both Waitlist and Decline.

## Our data right now is
print(TrainDF_nolabels)
print(TrainLabels)

## We need to fix the StudentTrainLabels
## to be 0 or 1. 
StudentTrainLabelsBIN=TrainLabels.copy()  ## Make a copy!
print(StudentTrainLabelsBIN)

StudentTrainLabelsBIN[StudentTrainLabelsBIN == 1] = 0
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 2] = 0
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 3] = 0
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 4] = 0
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 6] = 0
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 8] = 0
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 10] = 1
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 9] = 1
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 7] = 1
StudentTrainLabelsBIN[StudentTrainLabelsBIN == 5] = 1

print(StudentTrainLabelsBIN)
y=StudentTrainLabelsBIN.copy()
X=TrainDF_nolabels.copy()


## Fix the data type of y

y=y.astype(str).astype(int)
X=X.astype(float)
print(y.dtype)
print(X.dtypes)

######################################################
## See what we have:
print(X.head())
## We need X to be 2D so we can see it. 

######################
## Fit the model
import matplotlib.pyplot as plt
## Figure number
fignum = 1

MyModel=sklearn.svm.SVC(C=10, kernel='linear', gamma="scale")
MyModel.fit(X, y)

# get the separating hyperplane
w = MyModel.coef_[0]
print(w)
## This is the slope of the sep line
slope = -w[0] / w[1]
print(slope)

## Create a range of the x-axis and call it xx
xx = np.linspace(-30, 30)
## Create the equation of the line
print(MyModel.intercept_[0])
yy = (slope * xx) - ((MyModel.intercept_[0]) / w[1])

# plot the parallels - dotted lines - 
##  to the separating hyperplane that pass through the
    # support vectors (margin away from hyperplane in direction
    # perpendicular to hyperplane). 
    ## This is sqrt(1+slope^2)
margin = 1 / np.sqrt(np.sum(MyModel.coef_ ** 2))
yy_lower = yy - np.sqrt(1 + slope ** 2) * margin
yy_upper = yy + np.sqrt(1 + slope ** 2) * margin

# plot the line, the points, and the nearest vectors to the plane
plt.figure(fignum)
plt.clf()
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_lower, 'k--')
plt.plot(xx, yy_upper, 'k--')

#print(MyModel.support_vectors_[:,1])
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.scatter.html
plt.scatter(MyModel.support_vectors_[:, 0], 
            MyModel.support_vectors_[:, 1], 
            s=200, #size of points
            #marker="X",
            facecolors='none', 
            zorder=10, 
            edgecolors='k')

#print(X.iloc[:,0])
plt.scatter(X.iloc[:, 0], X.iloc[:, 1],c = y,
            s=200, ## size of points
            zorder=10, 
            cmap=plt.cm.Paired,
            edgecolors='k')
plt.axis('tight')
x_min = 0
x_max = 1000
y_min = 0
y_max = 1000

XX, YY = np.mgrid[x_min:x_max:10j, y_min:y_max:5j]
Z = MyModel.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(XX.shape)
plt.figure(fignum, figsize=(10, 10))
plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())
#fignum = fignum + 1

plt.show()












from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    #xy = np.vstack([X.ravel(), Y.ravel()]).T
   # P = model.decision_function(xy).reshape(X.shape)
    Xpred = np.array([X.ravel(), Y.ravel()] + [np.repeat(0, X.ravel().size) for _ in range(5)]).T
    
    pred = model.predict(Xpred).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, pred, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);