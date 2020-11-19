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

## Load the data and convert every variable into factor
MusicDataFile="NB_artists.csv"
MusicDF<-read.csv(MusicDataFile)
MusicDF$country <- as.factor(MusicDF$country)
MusicDF$genre <- as.factor(MusicDF$genre)
MusicDF$Max_key <- as.factor(MusicDF$Max_key)
MusicDF$award <- as.factor(MusicDF$award)

## remove all of the artist names
MusicDF = MusicDF[,3:8]



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
str(DF_Test_Music$award)  ## Notice that the label is called "award" and
## is correctly set to type FACTOR.
str(DF_Train_Music$award)
##Check balance of test dataset
table(DF_Test_Music$award)
##################################### REMOVE AND SAVE LABELS...
## Copy the Labels
Actual_Award <- DF_Test_Music$award
## Remove the labels
DF_Test_Music_NL<-DF_Test_Music[ , -which(names(DF_Test_Music) %in% c("award"))]
(DF_Test_Music_NL[1:3,])
## Check size
(ncol(DF_Test_Music_NL))
#(DF_Test_Student_NL)
## Train...--------------------------------
## Copy the Labels
(DF_Train_Music_Labels <- DF_Train_Music$award)
## Remove the labels
DF_Train_Music_NL<-DF_Train_Music[ , -which(names(DF_Train_Music) %in% c("award"))]
(DF_Train_Music_NL[1:5, 1:5])
## Check size
(ncol(DF_Train_Music_NL))


##########################################################
## Record Data----------------------------
#######################################################
(NB_e1071_Music<-naiveBayes(DF_Train_Music_NL, DF_Train_Music_Labels, laplace = 1))
Predict_Award <- predict(NB_e1071_Music, DF_Test_Music_NL)
table(Predict_Award,Actual_Award)
(NB_e1071_Pred_Music)
##Visualize
plot(NB_e1071_Pred_Music)


library(gridExtra)
library(grid)
tt3 <- ttheme_minimal(
  core=list(bg_params = list(fill = blues9[1:2], col=NA),
            fg_params=list(fontface=2)),
  colhead=list(fg_params=list(col="navyblue", fontface=2L)),
  rowhead=list(fg_params=list(col="orange", fontface=2L)))


grid.table(NB_e1071_Music$tables$genre, theme = tt3)


Actual_Award <- factor(c('No', 'No', 'Yes', 'Yes'))
Predict_Award <- factor(c('No', 'Yes', 'No', 'Yes'))
Y      <- c(8,5,6,14)
df <- data.frame(Actual_Award, Predict_Award, Y)

library(ggplot2)
ggplot(data =  df, mapping = aes(x = Actual_Award, y = Predict_Award)) +
  geom_tile(aes(fill = Y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1,size = 10) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_bw() + theme(legend.position = "none")

NB_e1071_Music$tables$Max_key



#################  Set up the SVM -----------------
## Soft svm with cost as the penalty for points
## being in the wrong location of the margin
## boundaries
## There are many kernel options...

###################################################
## Polynomial Kernel...
SVM_fit_P <- svm(award~., data=DF_Train_Music, 
                 kernel="polynomial", cost=.1, 
                 scale=FALSE)
print(SVM_fit_P)

