# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 00:02:10 2020

@author: lenovo
"""

StudentDF=pd.read_csv(".csv")
print(StudentDF.head())

from sklearn.model_selection import train_test_split
StudentTrainDF, StudentTestDF = train_test_split(StudentDF, test_size=0.3)



##-----------------------------------------------------------------
##
## Now we have a training set and a testing set. 
#print("\nThe training set is:")
print(StudentTrainDF)
#print("\nThe testing set is:")
print(StudentTestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
StudentTestLabels=StudentTestDF["Decision"]
#print(StudentTestLabels)
## remove labels
StudentTestDF = StudentTestDF.drop(["Decision"], axis=1)
print(StudentTestDF)

## Set up the training data so the models get what they expect
StudentTrainDF_nolabels=StudentTrainDF.drop(["Decision"], axis=1)
print(StudentTrainDF_nolabels)

StudentTrainLabels=StudentTrainDF["Decision"]
print(StudentTrainLabels)

#------------------------
## Some models do not run on qualitative data.....
## So, we will need to remove the variables: Gender and State

StudentTrainDF_nolabels_quant=StudentTrainDF_nolabels.drop(["Gender"], axis=1)
StudentTrainDF_nolabels_quant=StudentTrainDF_nolabels_quant.drop(["State"], axis=1)
StudentTestDF_quant=StudentTestDF.drop(["Gender"], axis=1)
StudentTestDF_quant=StudentTestDF_quant.drop(["State"], axis=1)
#------------------------------
print(StudentTestDF_quant)

## Check that all data are numeric
print(StudentTestDF_quant.dtypes)

####################################################################
########################### Naive Bayes ############################
####################################################################
#from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
MyModelNB= MultinomialNB()
## When you look up this model, you learn that it wants the 
## DF seperate from the labels

MyModelNB.fit(StudentTrainDF_nolabels_quant, StudentTrainLabels)
Prediction = MyModelNB.predict(StudentTestDF_quant)
print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(StudentTestLabels)
## confusion matrix
#from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
cnf_matrix = confusion_matrix(StudentTestLabels, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(MyModelNB.predict_proba(StudentTestDF_quant),2))

########### Metrics -------------------------
from sklearn import metrics

print(metrics.classification_report(StudentTestLabels, Prediction))
print(metrics.confusion_matrix(StudentTestLabels, Prediction))






