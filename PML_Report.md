# Practical Machine Learning - Project Report
Emil Traicu  
March 18, 2017  


```r
knitr::opts_chunk$set(echo = TRUE)
```

## Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of this project is to build a machine learning algorithm to predict activity quality (classe) from activity monitors.


## Getting and loading the data

Load libraries:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.3.2
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.3.2
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.3.2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.3.2
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.3.2
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.3.3
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```r
library(knitr)
```
Load the data:


```r
set.seed(12345)
trainURL  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL   <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#Read the data and replace empty values by NA
trainDS   <- read.csv(url(trainURL), na.strings=c("NA","#DIV/0!",""))
testDS    <- read.csv(url(testURL ), na.strings=c("NA","#DIV/0!",""))
dim(trainDS)
```

```
## [1] 19622   160
```

```r
dim( testDS)
```

```
## [1]  20 160
```

## Cleaning and preprocessing the data

Visually inspect data
Remove columns that are obviously not predictors:


```r
train_Sub <- trainDS[,8:length(trainDS)]
trainDS   <- train_Sub
dim(trainDS)
```

```
## [1] 19622   153
```

```r
rm(train_Sub)
```

Remove columns with missing value:


```r
trainDS <- trainDS[,(colSums(is.na(trainDS)) == 0)]
dim(trainDS)
```

```
## [1] 19622    53
```

Remove variables with most NA-s (use threshold of >60%)


```r
train_Sub <- trainDS
for (i in 1:length(trainDS)) {
  if (sum(is.na(trainDS[ , i])) / nrow(trainDS) >= .60) {
    for (j in 1:length(train_Sub)) {
      if (length(grep(names(trainDS[i]), names(train_Sub)[j]))==1) {
        train_Sub <- train_Sub[ , -j]
      }
    }
  }
}
trainDS <- train_Sub 
dim(trainDS)
```

```
## [1] 19622    53
```

```r
rm(train_Sub)
```
Remove the variables with values near zero


```r
nzv       <- nearZeroVar(trainDS,saveMetrics=TRUE)
ztrainDS  <- trainDS[,nzv$nzv==FALSE]
dim(ztrainDS)
```

```
## [1] 19622    53
```
Preprocess the data:


```r
#numericIdx       <- which(lapply(ztrainDS, class) %in% "numeric")
#preprocessModel  <- preProcess(ztrainDS[,numericIdx],method=c('knnImpute', 'center', 'scale'))
#pztrainDS        <- predict(preprocessModel, ztrainDS[,numericIdx])
#pztrainDS$classe <- ztrainDS$classe
#numericIdx       <- which(lapply(testDS, class) %in% "numeric")
#preprocessModel  <- preProcess(testDS[,numericIdx],method=c('knnImpute', 'center', 'scale'))
#ptestDS          <- predict(preprocessModel,testDS[,numericIdx])
```
## Preparing the data sets


```r
set.seed(12345)
trainIdx  <- createDataPartition(ztrainDS$classe, p=3/4, list=FALSE)
sztrainDS <- ztrainDS[trainIdx, ]
szvalidDS <- ztrainDS[-trainIdx, ]
dim(sztrainDS)  
```

```
## [1] 14718    53
```

```r
dim(szvalidDS)
```

```
## [1] 4904   53
```
Remove the 'classe' column


```r
colNames1 <- colnames(trainDS)
colNames2 <- colnames(trainDS[,-53])
```

Eliminate all the filtered out variables, from train dataset, from test and validation data set


```r
sztestDS   <- testDS[names(testDS) %in% colNames1]         
cszvalidDS <- szvalidDS[colNames2] 
#dim(sztrainDS)
#dim(sztestDS)
#dim(szvalidDS)
#dim(cszvalidDS)
```

Coerce the data into the same type


```r
csztestDS <- sztestDS
for (i in 1:length(csztestDS) ) {
        for(j in 1:length(sztrainDS)) {
        if( length( grep(names(sztrainDS[i]), names(csztestDS)[j]) ) ==1)  {
            class(csztestDS[j]) <- class(sztrainDS[i])
        }      
    }      
}
#And to make sure Coertion really worked:
csztestDS <- rbind(sztrainDS[2, -53] , csztestDS) #note row 2 does not mean anything, this will be removed right.. now:
csztestDS <- csztestDS[-1,]
```

## Model building

Three algorithms will be used for buiding 3 models and then look to see which one produces the best out-of-sample accuracty. The three algorithms are:

1. Decision trees with CART (rpart)
2. Random forest decision trees (rf)
3. Gradient Boosting (gbm)

### Cross validation

Cross validation is done for each model with K = 5.


```r
fitControl <- trainControl(method='cv', number = 5)
```


```r
modFitD3 <- train(
  classe ~ ., 
  data=sztrainDS,
  trControl=fitControl,
  method='rpart'
)
predictD3 <- predict(modFitD3, cszvalidDS)
cfmxD3    <- confusionMatrix(predictD3, szvalidDS$classe)
cfmxD3
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 870 162  29  46  16
##          B 159 530  36 136 221
##          C 273 214 674 429 224
##          D  88  43 116 193  51
##          E   5   0   0   0 389
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5416          
##                  95% CI : (0.5275, 0.5556)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4245          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6237   0.5585   0.7883  0.24005  0.43174
## Specificity            0.9279   0.8604   0.7184  0.92732  0.99875
## Pos Pred Value         0.7747   0.4898   0.3716  0.39308  0.98731
## Neg Pred Value         0.8611   0.8904   0.9414  0.86155  0.88647
## Prevalence             0.2845   0.1935   0.1743  0.16395  0.18373
## Detection Rate         0.1774   0.1081   0.1374  0.03936  0.07932
## Detection Prevalence   0.2290   0.2206   0.3699  0.10012  0.08034
## Balanced Accuracy      0.7758   0.7095   0.7534  0.58368  0.71525
```

```r
plot(cfmxD3$table, col = cfmxD3$byClass, main = paste("Decision Tree Confusion Matrix: Accuracy =", round(cfmxD3$overall['Accuracy'], 4)))
```

![](PML_Report_files/figure-html/D3-1.png)<!-- -->

```r
#fancyRpartPlot(modFitD3)
```

### Random Forest

```r
modFitRF <- train(
  classe ~ ., 
  data=sztrainDS,
  trControl=fitControl,
  method='rf',
  ntree=250
)
predictRF <- predict(modFitRF, cszvalidDS, type = "raw")
cfmxRF    <- confusionMatrix(predictRF, szvalidDS$classe)
cfmxRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1392    6    0    0    0
##          B    3  939    3    0    0
##          C    0    4  848   12    2
##          D    0    0    4  792    5
##          E    0    0    0    0  894
## 
## Overall Statistics
##                                           
##                Accuracy : 0.992           
##                  95% CI : (0.9891, 0.9943)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9899          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9895   0.9918   0.9851   0.9922
## Specificity            0.9983   0.9985   0.9956   0.9978   1.0000
## Pos Pred Value         0.9957   0.9937   0.9792   0.9888   1.0000
## Neg Pred Value         0.9991   0.9975   0.9983   0.9971   0.9983
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2838   0.1915   0.1729   0.1615   0.1823
## Detection Prevalence   0.2851   0.1927   0.1766   0.1633   0.1823
## Balanced Accuracy      0.9981   0.9940   0.9937   0.9914   0.9961
```

```r
#plot(modFitRF)
plot(cfmxRF$table, col = cfmxRF$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cfmxRF$overall['Accuracy'], 4)))
```

![](PML_Report_files/figure-html/RF-1.png)<!-- -->

### GBM

```r
modFitGBM1 <- train(
  classe ~ ., 
  data=sztrainDS,
  trControl=fitControl,
  method='gbm',
  verbose = FALSE
)
```

```
## Loading required package: plyr
```

```r
PredictGBM <- predict(modFitGBM1, newdata=cszvalidDS)
cfmxGBM    <- confusionMatrix(PredictGBM, szvalidDS$classe)
cfmxGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1377   38    0    0    3
##          B   11  877   23    2    8
##          C    7   34  820   22    9
##          D    0    0   11  773   13
##          E    0    0    1    7  868
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9615          
##                  95% CI : (0.9557, 0.9667)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9512          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9871   0.9241   0.9591   0.9614   0.9634
## Specificity            0.9883   0.9889   0.9822   0.9941   0.9980
## Pos Pred Value         0.9711   0.9522   0.9193   0.9699   0.9909
## Neg Pred Value         0.9948   0.9819   0.9913   0.9925   0.9918
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2808   0.1788   0.1672   0.1576   0.1770
## Detection Prevalence   0.2892   0.1878   0.1819   0.1625   0.1786
## Balanced Accuracy      0.9877   0.9565   0.9706   0.9778   0.9807
```

```r
plot(modFitGBM1)
```

![](PML_Report_files/figure-html/GBM-1.png)<!-- -->

```r
plot(cfmxGBM$table, col = cfmxGBM$byClass, main = paste("Gradient Boosted Confusion Matrix: Accuracy =", round(cfmxGBM$overall['Accuracy'], 4)))
```

![](PML_Report_files/figure-html/GBM-2.png)<!-- -->
## Predicting Results on the Test Data

Random Forests provided an accuracy in the validation dataset of 99.20%, which was superior to the one provided by Decision Trees or GBM models. 
The expected out-of-sample error is 100-99.20 = 0.80%.


```r
predictFin <- predict(modFitRF, sztestDS, type = "raw")
predictFin
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```







