################ Ian   Weiham
library(tm)
library(stringr)
library(wordcloud)
library(slam)
library(quanteda)
library(SnowballC)
library(arules)
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) 
library(plyr) 
library(ggplot2)
library(factoextra) 
library(mclust) 
library(naivebayes)
library(tidyverse)
library(ggplot2)
library(caret)
library(caretEnsemble)
library(psych)
library(Amelia)
library(mice)
library(GGally)
library(e1071)



df = read.csv('diff_genres.csv')
df = df[,c(10,11,12,13,14,30)]
df$label = as.factor(df$label)
MusicDF = df

## Record Data
## MAKE test and train data
(Size <- (as.integer(nrow(MusicDF)/3)))  
## Test will be 1/3 of the data
(SAMPLE <- sample(nrow(MusicDF), Size))

(DF_Test_Music<-MusicDF[SAMPLE, ])
(DF_Train_Music<-MusicDF[-SAMPLE, ])
##
## REMOVE the labels and KEEP THEM
##   
str(DF_Test_Music$label)  ## Notice that the label is called "label" and
## is correctly set to type FACTOR.
str(DF_Train_Music$label)
##Check balance of test dataset
table(DF_Test_Music$label)
##################################### REMOVE AND SAVE LABELS...
## Copy the Labels
Actual_Award
<- DF_Test_Music$label
## Remove the labels
DF_Test_Music_NL<-DF_Test_Music[ , -which(names(DF_Test_Music) %in% c("label"))]
(DF_Test_Music_NL[1:3,])
## Check size
(ncol(DF_Test_Music_NL))
#(DF_Test_Student_NL)
## Train...--------------------------------
## Copy the Labels
(DF_Train_Music_Labels <- DF_Train_Music$label)
## Remove the labels
DF_Train_Music_NL<-DF_Train_Music[ , -which(names(DF_Train_Music) %in% c("label"))]
(DF_Train_Music_NL[1:5, 1:5])
## Check size
(ncol(DF_Train_Music_NL))
#################  Set up the SVM -----------------
## Soft svm with cost as the penalty for points
## being in the wrong location of the margin
## boundaries
## There are many kernel options...

###################################################
## Polynomial Kernel...
SVM_fit_P <- svm(label~., data=DF_Train_Music, 
                 kernel="linear", cost=1, 
                 scale=FALSE)
print(SVM_fit_P)

(pred_L <- predict(SVM_fit_P, DF_Test_Music_NL, type="class"))
(L_table<-table(pred_L, Actual_Award))
library(gridExtra)
library(grid)
grid.table(L_table)

plot(SVM_fit_P, data=DF_Train_Music, mfcc1~mfcc2 
    )
