
library(knitr)
library(kableExtra)
library(dplyr)
library(DataExplorer)
library(funModeling)
library(tidyverse)
library(class)
library(rpart)
library(rpart.plot)
library(e1071)
library(corrplot)
library(caTools)
library(party)
library(ISLR)
library(readxl)
library(pROC)
library(lattice)
library(e1071) 
library(ggplot2)
library(multiROC)
library(MLeval)
library(AppliedPredictiveModeling)
library(Hmisc)
library(quantmod) 
library(nnet)
library(caret)
library(NeuralNetTools)


bank_main_data.raw <- read.csv(file = "/Users/nselvarajan/Desktop/R/Assignment3/bank-additional-full.csv", header = T, sep = ";",stringsAsFactors = T)
bank_main_data.raw <- data.frame(bank_main_data.raw, stringsAsFactors = FALSE)


# Plot missing values of all the features in the dataset.

plot_missing(bank_main_data.raw)


# Ploting histograms for numerical variables.

plot_num(bank_main_data.raw)

# Get  metric table with many indicators for all numerical variables, automatically skipping the non-numerical variables.

profiling_num(bank_main_data.raw)


# Plot variable importance with  several metrics such as entropy (en), mutual information(mi), information gain (ig) and gain ratio (gr).


var_imp <- var_rank_info(bank_main_data.raw, "y")
# Plotting 
ggplot(var_imp, 
       aes(x = reorder(var, gr), 
           y = gr, fill = var)
) + 
  geom_bar(stat = "identity") + 
  coord_flip() + 
  theme_bw() + 
  xlab("") + 
  ylab("Variable Importance 
       (based on Information Gain)"
  ) + 
  guides(fill = FALSE)


# Bivariate analysis crosss plot showing relationship of each and every variable with respect to target variable 

cross_plot(data=bank_main_data.raw, target="y")



# Select variables relevant to customers:Based on the variable importance, we will use pdays, poutcome,previous, duration, cons.price.idx,cons.conf.idx,contact feature for # further analysis. 

subsets <- data.frame(   as.factor(bank_main_data.raw$y),
                         as.numeric((bank_main_data.raw$pdays)),
                         as.numeric(as.factor(bank_main_data.raw$poutcome)),
                         as.numeric((bank_main_data.raw$previous)),
                         as.numeric((bank_main_data.raw$duration)),
                         as.numeric(as.factor(bank_main_data.raw$contact)), 
                         as.numeric((bank_main_data.raw$cons.conf.idx)),
                         as.numeric((bank_main_data.raw$cons.price.idx)))
colnames(subsets) <- c("Term_Deposit", 
                       "NumberOfDaysPassedAfterLastContact",
                       "PreviousMarketingOutCome", 
                       "NoOfContactsPerformed", 
                       "LastContactDuration", 
                       "ContactCommunicationType", 
                       "ConsumerPriceIndex", 
                       "ConsumerConfidenceIndex")

set.seed(212)
trainIndex <- createDataPartition(subsets$Term_Deposit, p = 0.8, list=FALSE, times=3)
subTrain <- subsets[trainIndex,]
subTest <- subsets[-trainIndex,]
TrainingParameters <- trainControl(method = "repeatedcv", number = 10, repeats=3,classProbs = TRUE)

nnetGrid <-  expand.grid(size = seq(from = 1, to = 5, by = 1)
                         ,decay = seq(from = 0.1, to = 0.2, by = 0.1)
)
nn_model <- train(Term_Deposit ~ ., subTrain,
                  method = "nnet",  algorithm = 'backprop',     
                  trControl= TrainingParameters,
                  preProcess=c("scale","center"),
                  na.action = na.omit,
                  #metric = "ROC",
                  tuneGrid = nnetGrid,
                  trace=FALSE,
                  verbose=FALSE)      

nn_model$results   
plot(nn_model)


prediction <- predict(nn_model, subTest[-1])                           
table(prediction, subTest$Term_Deposit)  

accuracy <- sum(prediction == (subTest$Term_Deposit))/length(subTest$Term_Deposit)
print(accuracy)

confusionNN <-confusionMatrix(as.factor(prediction),as.factor(subTest$Term_Deposit))
print(confusionNN)


library(NeuralNetTools)
varImp_nn<-varImp(nn_model)
print(varImp_nn)
ggplot(varImp_nn)
plot(varImp_nn)

library(NeuralNetTools)
plotnet(nn_model, y_names = "Term DEPOSIT")
title("Graphical Representation of our Neural Network")



# Machine Learning: Classification using SVM

library(knitr)
library(kableExtra)
library(dplyr)
library(ggplot2)
library(DataExplorer)
library(ggplot2)
library(funModeling)
library(tidyverse)
library(class)
library(rpart)
library(rpart.plot)
library(e1071)
library(caret)
library(corrplot)
library(caTools)
library(party)
library(DataExplorer)
library(ggplot2)
library(funModeling)



bank_main_data.svm <- read.csv(file = "/Users/nselvarajan/Desktop/R/Assignment3/bank-additional.csv", header = T, sep = ";",stringsAsFactors = T)
bank_main_data.svm <- data.frame(bank_main_data.svm, stringsAsFactors = FALSE)
subsets_svm <- data.frame(   as.factor(bank_main_data.svm$y),
                             as.numeric((bank_main_data.svm$pdays)),
                             as.numeric(as.factor(bank_main_data.svm$poutcome)),
                             as.numeric((bank_main_data.svm$previous)),
                             as.numeric((bank_main_data.svm$duration)),
                             as.numeric(as.factor(bank_main_data.svm$contact)), 
                             as.numeric((bank_main_data.svm$cons.conf.idx)),
                             as.numeric((bank_main_data.svm$cons.price.idx)))
colnames(subsets_svm) <- c("Term_Deposit", 
                           "NumberOfDaysPassedAfterLastContact",
                           "PreviousMarketingOutCome", 
                           "NoOfContactsPerformed", 
                           "LastContactDuration", 
                           "ContactCommunicationType", 
                           "ConsumerPriceIndex", 
                           "ConsumerConfidenceIndex")


set.seed(212)
trainIndexSVM <- createDataPartition(subsets_svm$Term_Deposit, p = 0.8, list=FALSE, times=3)
subTrainSVM <- subsets_svm[trainIndexSVM,]
subTestSVM <- subsets_svm[-trainIndexSVM,]

# SVM Classifier using Linear Kernel


trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
set.seed(323)
grid <- expand.grid(C = c( 0.25, 0.5, 1))
svm_Linear_Grid <- train(Term_Deposit ~ ., data = subTrainSVM, method = "svmLinear", trControl=trctrl, preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
svm_Linear_Grid


plot(svm_Linear_Grid)

predictionsvm <- predict(svm_Linear_Grid, subTestSVM[-1]) 
table(predictionsvm, subTestSVM$Term_Deposit)   

accuracysvm <- sum(predictionsvm == (subTestSVM$Term_Deposit))/length(subTestSVM$Term_Deposit)
print(accuracysvm)

#confusionNNSvm <-confusionMatrix(as.factor(predictionsvm),as.factor(subTestSVM$Term_Deposit))
#print(confusionNNSvm)


# SVM Classifier using Non-Linear Kernel

set.seed(323) 
grid_radial <- expand.grid(sigma = c(0.25, 0.5,0.9),
                           C = c(0.25, 0.5,1))
svm_Radial <- train(Term_Deposit ~ ., data = subTrainSVM, method = "svmRadial",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),tuneGrid = grid_radial,
                    tuneLength = 10)

svm_Radial

predictionnonlinearsvm <- predict(svm_Radial, subTestSVM[-14])                          
accuracynonlinearsvm <- sum(predictionnonlinearsvm == (subTestSVM$Term_Deposit))/length(subTestSVM$Term_Deposit)
print(accuracynonlinearsvm)



algo_results <- resamples(list(SVM_RADIAL=svm_Radial, SVM_LINEAR=svm_Linear_Grid))

summary(algo_results)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(algo_results, scales=scales)

splom(algo_results)
