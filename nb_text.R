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
library(tm)
#install.packages("tm")
library(stringr)
library(wordcloud)
# ONCE: install.packages("Snowball")
## NOTE Snowball is not yet available for R v 3.5.x
## So I cannot use it  - yet...
##library("Snowball")
##set working directory
## ONCE: install.packages("slam")
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
library(SnowballC)
library(arules)
##ONCE: install.packages('proxy')
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(ggplot2)
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering
library(naivebayes)
#Loading required packages
#install.packages('tidyverse')
library(tidyverse)
#install.packages('ggplot2')
library(ggplot2)
#install.packages('caret')
library(caret)
#install.packages('caretEnsemble')
library(caretEnsemble)
#install.packages('psych')
library(psych)
#install.packages('Amelia')
library(Amelia)
#install.packages('mice')
library(mice)
#install.packages('GGally')
library(GGally)
library(e1071)


## Next, load in the documents (the corpus)
NovelsCorpus <- Corpus(DirSource("Novels_Corpus1"))
(ndocs<-length(NovelsCorpus))
##------------
## OK - now we have both datasets read in. 
## For the text corpus, we need to do a few things.....
##------------
##The following will show you that you read in all the documents
(summary(NovelsCorpus))  ## This will list the docs in the corpus

###################################################################
#######       Change the COrpus into a DTM, a DF, and  Matrix
#######
####################################################################

Novels_dtm <- DocumentTermMatrix(NovelsCorpus,
                                 control = list(
                                   stopwords = TRUE, ## remove normal stopwords
                                   wordLengths=c(4, 15), ## get rid of words of len 3 or smaller or larger than 15
                                   removePunctuation = TRUE,
                                   removeNumbers = TRUE,
                                   tolower=TRUE,
                                   #stemming = TRUE,
                                   remove_separators = TRUE
                                   #stopwords = MyStopwords,
                                   
                                   #removeWords(MyStopwords),
                                   #bounds = list(global = c(minTermFreq, maxTermFreq))
                                 ))
########################################################
################### Have a look #######################
################## and create formats #################
########################################################
#(inspect(Novels_dtm))  ## This takes a look at a subset - a peak
DTM_mat <- as.matrix(Novels_dtm)
(DTM_mat[1:13,1:10])

#########################################################
######### OK - Pause - now the data is vectorized ######
## Its current formats are:
## (1) Novels_dtm is a DocumentTermMatrix R object
## (2) DTM_mat is a matrix
#########################################################

#Novels_dtm <- weightTfIdf(Novels_dtm, normalize = TRUE)
#Novels_dtm <- weightTfIdf(Novels_dtm, normalize = FALSE)

## Look at word freuqncies out of interest
(WordFreq <- colSums(as.matrix(Novels_dtm)))

(head(WordFreq))
(length(WordFreq))
ord <- order(WordFreq)
(WordFreq[head(ord)])
(WordFreq[tail(ord)])
## Row Sums
(Row_Sum_Per_doc <- rowSums((as.matrix(Novels_dtm))))

## I want to divide each element in each row by the sum of the elements
## in that row. I will test this on a small matrix first to make 
## sure that it is doing what I want. YOU should always test ideas
## on small cases.
#############################################################
########### Creating and testing a small function ###########
##           
##                  Practice and test first!
##
#############################################################
## Create a small pretend matrix
## Using 1 in apply does rows, using a 2 does columns
(mymat = matrix(1:12,3,4))
freqs2 <- apply(mymat, 1, function(i) i/sum(i))  ## this normalizes
## Oddly, this re-organizes the matrix - so I need to transpose back
(t(freqs2))
## OK - so this works. Now I can use this to control the normalization of
## my matrix...
#############################################################

## Copy of a matrix format of the data
Novels_M <- as.matrix(Novels_dtm)
(Novels_M[1:13,1:5])

## Normalized Matrix of the data
Novels_M_N1 <- apply(Novels_M, 1, function(i) round(i/sum(i),5))
(Novels_M_N1[1:13,1:5])
## NOTICE: Applying this function flips the data...see above.
## So, we need to TRANSPOSE IT (flip it back)  The "t" means transpose
Novels_Matrix_Norm <- t(Novels_M_N1)
(Novels_Matrix_Norm[1:13,1:10])

############## Always look at what you have created ##################
## Have a look at the original and the norm to make sure
(Novels_M[1:13,1:10])
(Novels_Matrix_Norm[1:13,1:10])

######################### NOTE #####################################
## WHen you make calculations - always check your work....
## Sometimes it is better to normalize your own matrix so that
## YOU have control over the normalization. For example
## scale used directly may not work - why?

##################################################################
###############   Convert to dataframe     #######################
##################################################################

## It is important to be able to convert between format.
## Different models require or use different formats.
## First - you can convert a DTM object into a DF...

Novels_DF <- as.data.frame(as.matrix(Novels_dtm))
(Novels_DF[1:4, 1:4])
#(head(Novels_DF))
str(Novels_DF)
(Novels_DF$aunt)
(nrow(Novels_DF))  ## Each row is a novel
## Fox DF format

######### Next - you can convert a matrix (or normalized matrix) to a DF
Novels_DF_From_Matrix_N<-as.data.frame(Novels_Matrix_Norm)
(Novels_DF_From_Matrix_N[1:4, 1:4])

#######################################################################
#############   Making Word Clouds ####################################
#######################################################################
## This requires a matrix - I will use Novels_M from above. 
## It is NOT mornalized as I want the frequency counts!
## Let's look at the matrix first
(Novels_M[,c(1:1000)])
wordcloud(colnames(Novels_M), Novels_M, max.words = 500)

############### Look at most frequent words by sorting ###############
(head(sort(Novels_M[13,], decreasing = TRUE), n=20))



############## -------------- ############################
##
## NAIVE BAYES
## 
#################################################

## We have a dataframe of text data called Novels_DF_From_Matrix_N
(Novels_DF[1:5, 1:5])

## Next, because our file names, such as Austen_Emma, etc are the labels, 
## we can use these to create a dataframe of labels and then bind that
## to our Novels_DF_From_Matrix_N

## Get the row names
(DF_Row_Names <- row.names(Novels_DF))

## For Naive Bayes, we want all the authors to be labeled - so all Austen books to
## Austen, all Cbronte books to be CBronte, etc. 


## New and empty list for the labels
MyNamesList <- c()
for(next_item in DF_Row_Names){
  #print(next_item)
  ## 
  Names <- strsplit(next_item, "_")
  Next_Name<- Names[[1]][1]
  if(Next_Name == "EBronte"){
    Next_Name<-"CBronte"
  }
  MyNamesList<-c(MyNamesList,Next_Name)
}

### Use the list of labels to bind together with your DF to created labeled data.
print(MyNamesList)

Labeled_DF_Novels <- cbind(MyNamesList, Novels_DF)
(Labeled_DF_Novels[1:5, 1:5])

## Then, create a DF with no row names
rownames(Labeled_DF_Novels) <- c()

## Check both
(Labeled_DF_Novels[1:5, 1:5])

############ !!!!!!!!!!!!!!!!!!! ##
## Notice that the NAME OF THE LABEL
## is "MyNamesList"
########################################

##########################################################
##
##  Create the Testing and Training Sets         
##
########################################################
#####################################################
##        Grabbing Every X value  ##################
####################################################
## 
## This method works whether the data is in order or not.
X = 3   ## This will create a 1/3, 2/3 split. 
## Of course, X can be any number.
(every_X_index<-seq(1,nrow(Labeled_DF_Novels),X))

## Use these X indices to make the Testing and then
## Training sets:

DF_Test<-Labeled_DF_Novels[every_X_index, ]
DF_Train<-Labeled_DF_Novels[-every_X_index, ]
## View the created Test and Train sets
(DF_Test[1:5, 1:5])
(DF_Train[1:5, 1:5])


## Use tables to check the balance...
## WARNING - do not do this is you have high D data !!!
## DO NOT RUN THESE ON HIGH D DATA :)
##sapply(DF_Test, table)  ## Looks good!
##sapply(DF_Train, table)  ## Looks good!

################  OK - now we have a Training Set
###############        and a Testing Set.

###############################################
#######    RANDOM SAMPLE OPTION       #########
###############################################
#set.seed(1234)  ## The number inside can be anything
#(YourSampleSize <- (as.integer(nrow(Labeled_DF_Novels)/3)))
#(My_SAMPLE <- sample(nrow(Labeled_DF_Novels), YourSampleSize))

#DF_Test_R<-Labeled_DF_Novels[My_SAMPLE, ]
#DF_Train_R<-Labeled_DF_Novels[-My_SAMPLE, ]
## View the created Test and Train sets
#(DF_Test[, 1:5])
#(DF_Train[, 1:5])

## Remove and KEEP the label from the test & train sets.
## Make sure label is factor type
str(DF_Test$MyNamesList)  ## Notice that the label is called "MyNamesList" and
## is correctly set to type FACTOR. This is IMPORTANT!!
str(DF_Train$MyNamesList)  ## GOOD! also type FACTOR

## Copy the Labels
(DF_Test_Labels <- DF_Test$MyNamesList)
str(DF_Test_Labels)
## Remove the labels
DF_Test_NL<-DF_Test[ , -which(names(DF_Test) %in% c("MyNamesList"))]
(DF_Test_NL[1:5, 1:5])

## REMOVE THE LABEL FROM THE TRAINING SET...
DF_Train_Labels<-DF_Train$MyNamesList
DF_Train_NL<-DF_Train[ , -which(names(DF_Train) %in% c("MyNamesList"))]
(DF_Train_NL[1:5, 1:5])

###########################################################
##
##          Prepare the record data for 
##          analysis. Create testing and
##          training sets.
##
##          REMEMBER - we are assuming here that
##          the record dataset is already 100% clean
################################################################

###############################################################
##
##                         NAIVE BAYES
##
###############################################################

############
## What do we have?
## ------------------------------------
## For Text Data, we have:
## ---------------------------------
## DF_Test_NL    ## Text test set
## DF_Train_NL   ## Text Training set
## The label is called "MyNamesList" 
## DF_Test_Labels  ## Testset labels
## DF_Train_Labels  ## training labels
##
## ----------------------------
## For Record data, we have:
##-----------------------------
## DF_Test_Student_NL  ## Testset
## DF_Train_Student_NL  ## Training set
## Label name is "Decision"
## Test labels:
## DF_Test_Student_Labels
## DF_Train_Student_Labels

  #################################################################
##
####### RUN Naive Bayes --------------------------------------------
##
######################################################################

## TEXT DATA
## https://www.rdocumentation.org/packages/e1071/versions/1.7-3/topics/naiveBayes
DF_Train_Labels = as.factor(DF_Train_Labels)
NB_e1071_2<-naiveBayes(DF_Train_NL, DF_Train_Labels, laplace = 1)
NB_e1071_Pred <- predict(NB_e1071_2, DF_Test_NL)
#NB_e1071_2
table(NB_e1071_Pred,DF_Test_Labels)
(NB_e1071_Pred)

##Visualize
plot(NB_e1071_Pred)



#########################################
## Variable importance and other NB option
###############################################
## The following may only run if you convert from DF to matrix...
## I will not do that here - but FYI...
# memory.limit()
# memory.limit(1e+09)
library(mlbench)
library(caret)
x = DF_Train_NL
y = DF_Train_Labels
control <- trainControl(method="repeatedcv", number=5, repeats=2)
# # train the model
model <- train(y~., data=x, method="nb", trControl=control)
# # estimate variable importance
importance <- varImp(model, scale=FALSE)
# # summarize importance
print(importance)
# # plot importance
plot(importance)
#####################################################

 Actual_Genre <- factor(c('electronic','electronic','electronic', 'electronic','electronic',
                          'folk','folk','folk','folk','folk',
                          'metal','metal','metal','metal','metal',
                          'pop','pop','pop','pop','pop',
                          'rap','rap','rap','rap','rap'))
 Predict_Genre <- factor(c('electronic','folk','metal','pop','rap',
                           'electronic','folk','metal','pop','rap',
                           'electronic','folk','metal','pop','rap',
                           'electronic','folk','metal','pop','rap',
                           'electronic','folk','metal','pop','rap'))
 Y      <- c(0,4,0,0,0,0,4,0,0,0,0,3,0,0,0,0,3,0,0,0,0,4,0,0,0)
 df <- data.frame(Actual_Genre, Predict_Genre, Y)
 
 library(ggplot2)
 ggplot(data =  df, mapping = aes(x = Actual_Genre, y = Predict_Genre)) +
   geom_tile(aes(fill = Y), colour = "white") +
   geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1,size = 10) +
   scale_fill_gradient(low = "blue", high = "red") +
   theme_bw() + theme(legend.position = "none")
 
 
 
 #################  Set up the SVM -----------------
 ## Soft svm with cost as the penalty for points
 ## being in the wrong location of the margin
 ## boundaries
 ## There are many kernel options...
 
 ###################################################
 ## Polynomial Kernel...
 DF_Train$MyNamesList = as.factor(DF_Train$MyNamesList)
 SVM_fit_L <- svm(MyNamesList~., data=DF_Train, 
                  kernel="polynomial", cost=1, 
                  scale=FALSE)
 print(SVM_fit_L)
 
 DF_Test_Labels = as.factor(DF_Test_Labels)
 (pred_L <- predict(SVM_fit_L, DF_Test_NL, type="class"))
 (L_table<-table(pred_L,DF_Test_Labels ))
 library(gridExtra)
 library(grid)
 grid.table(L_table)
 

 plot(pred_L)
 