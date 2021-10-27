Lifestyle Predictive Models
================
Alex Prevatte & Chennade Brown
10/31/2021

-   [Introduction](#introduction)
-   [Data](#data)
-   [Summarizations](#summarizations)
-   [Modeling](#modeling)
    -   [Linear Regression Models](#linear-regression-models)
    -   [Random Forest Model](#random-forest-model)
    -   [Boosted Tree Model](#boosted-tree-model)
-   [Comparison](#comparison)

## Introduction

Our project involves creating predictive models and automating Markdown
reports for each article topic (Lifestyle, Social Media, Business,
World, Entertainment, and Technology) using the [Online News Popularity
Data
Set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity).
This dataset contains quantitative information including the number of
words, images, and links, as well as qualitative information such as
article topic and day published. The goal of this report is to predict
the number of shares for each article topic using linear regression,
random forest, and boosted tree models. Our team used 55 prediction
attributes for ensemble methods and a combination of the following
attributes for linear regression to predict article shares:

`n_tokens_title` : Number of words in the title `n_tokens_content` :
Number of words in the content `rate_positive_words` : Rate of positive
words among non-neutral tokens `num_hrefs` : Number of links `num_imgs`
: Number of images `num_keywords` : Number of keywords in the metadata
`global_subjectivity` : Text subjectivity `max_positive_polarity` : Max.
polarity of positive words `num_imgs` : Number of images `kw_avg_avg` :
Avg. keyword (avg. shares) `kw_min_avg` : Avg. keyword (min. shares)
`kw_max_avg` : Avg. keyword (max. shares)

``` r
library(tidyverse)
library(dplyr)
library(knitr)
library(GGally)
library(corrplot)
library(caret)
library(randomForest)
library(doParallel)
```

## Data

``` r
# Import the Data.
 onlineNews <- read_csv("OnlineNewsPopularity.csv")
```

``` r
# Create articleTopic variable from data_channel_is_* variables

onlineNewsLifestyle <- onlineNews %>% 
  mutate(articleTopic = if_else(`data_channel_is_lifestyle` == 1, "Lifestyle",
                          if_else(`data_channel_is_bus` == 1, "Business",
                                  if_else(`data_channel_is_entertainment` == 1, "Entertainment", 
                                          if_else(`data_channel_is_socmed` == 1, "Social Media",
                                                  if_else(`data_channel_is_tech` == 1,"Tech",
                                                          if_else(`data_channel_is_world` == 1, "World", "NULL")))))))

# Filtered rows based on articleTopic. Removed data_channel_is_* variables, and  
# non-predictive variables (URL, Timedelta, articleTopic).

onlineNewsLifestyle <- filter(onlineNewsLifestyle, articleTopic == params$topic) %>% 
  select(- c(url, `timedelta`, articleTopic, `data_channel_is_lifestyle`,
             `data_channel_is_bus`, `data_channel_is_entertainment`,
             `data_channel_is_socmed`, `data_channel_is_tech`, `data_channel_is_world`))

# Remove space in front of each column name
names(onlineNewsLifestyle) <- sub(" ", "", names(onlineNewsLifestyle))

# Convert to dataframe for plotting
onlineNewsLifestyle <- data.frame(onlineNewsLifestyle)
```

## Summarizations

``` r
# Summary statistics for shares
sum <- summary(onlineNewsLifestyle$shares)
std <- sd(onlineNewsLifestyle$shares)
# Median and mean to be used for histogram plot
median <- sum[3]
mean <- sum[4]
print(sum)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      28    1100    1700    3682    3250  208300

``` r
# Summary statistics for Number of Words in Title
sum <- summary(onlineNewsLifestyle$n_tokens_title)
std <- sd(onlineNewsLifestyle$n_tokens_title)
print(sum)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   3.000   8.000  10.000   9.766  11.000  18.000

``` r
# Summary statistics for Number of Words in Content
sum <- summary(onlineNewsLifestyle$n_tokens_content)
std <- sd(onlineNewsLifestyle$n_tokens_content)
print(sum)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     0.0   308.5   502.0   621.3   795.0  8474.0

``` r
# Create Day variable to combine days for contingency table
dayShares <- onlineNewsLifestyle %>% 
  mutate(Day = if_else(weekday_is_monday == 1, "M",
                          if_else(weekday_is_tuesday == 1, "T",
                                  if_else(weekday_is_wednesday == 1, "W", 
                                          if_else(weekday_is_thursday == 1, "R",
                                                  if_else(weekday_is_friday == 1,"F",
                                                          if_else(weekday_is_saturday == 1,"Sat",
                                                                  if_else(weekday_is_sunday == 1,"Sun", "NULL"))))))))

# Add Popular variable for rows where shares is greater than the median
dayShares <- dayShares %>% mutate(Popular = 
                                    ifelse(shares > median, "Yes", "No")) %>% select(Popular, everything())

# Contingency table for day of the week and shares greater than the median
table(dayShares$Popular, factor(dayShares$Day, levels = c("M", "T", "W", 
                                                          "R", "F", "Sat", "Sun")))
```

    ##      
    ##         M   T   W   R   F Sat Sun
    ##   No  175 196 215 201 167  71  80
    ##   Yes 147 138 173 157 138 111 130

``` r
# Table of shares for each day
tableTotal <- aggregate(dayShares$shares, by=list(Day=dayShares$Day), FUN=sum)
colnames(tableTotal) <- c("Day", "Total_Shares")
tableTotal$Day <- factor(tableTotal$Day, levels = c("M", "T", "W", 
                                                          "R", "F", "Sat", "Sun"))
tableTotal
```

    ##   Day Total_Shares
    ## 1   F       922890
    ## 2   M      1399319
    ## 3   R      1253096
    ## 4 Sat       739366
    ## 5 Sun       795979
    ## 6   T      1386933
    ## 7   W      1231194

``` r
# Scatterplot of Number of Words in Title vs. Number of Shares
titleCorrelation <- cor(onlineNewsLifestyle$shares, onlineNewsLifestyle$n_tokens_title)

ggplot(onlineNewsLifestyle, aes(n_tokens_title, shares)) + 
  labs(y = "Number of Shares", x = "Number of Words in Title") +
  ggtitle("Number of Words in Title vs. Number of Shares") +
  geom_point() + 
  geom_text(x = 15, y = 200000, size = 5, 
            label = paste0("Correlation =", round(titleCorrelation, 3)))
```

![](Reports/Lifestyle_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
# Histogram of Total Shares. Mean and Median vertical lines have been included.
ggplot(onlineNewsLifestyle, aes(x=shares)) +
  labs(y = "Frequency", x = "Number of Shares") +
  ggtitle("Frequency of Total Shares") +
  geom_histogram(binwidth = 500, color="darkblue", fill="lightblue") +
  xlim(0,30000) +
  geom_vline(aes(xintercept = median), colour="black") + 
  geom_vline(aes(xintercept = mean), colour="red")
```

![](Reports/Lifestyle_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

``` r
# Barplot of Total Shares vs. Day of the Week 
ggplot(tableTotal, aes(x=Day, y = Total_Shares)) +
  geom_bar(stat="identity", fill = "steelblue") +
  labs(y = "Number of Shares", x = "Day of the Week") +
  ggtitle("Total Shares by Day of the Week") 
```

![](Reports/Lifestyle_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->

``` r
# Summary statistics for the number of words in the content.
sumWords <- summary(onlineNewsLifestyle$n_tokens_content)
sumWords
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     0.0   308.5   502.0   621.3   795.0  8474.0

``` r
# Summary statistics for the number of links in the content. 
sumLinks <- summary(onlineNewsLifestyle$num_hrefs)
sumLinks
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##    0.00    6.00   10.00   13.42   18.00  145.00

``` r
# Summary statistics for the number of images in the content.
sumImages <- summary(onlineNewsLifestyle$num_imgs)
sumImages
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   0.000   1.000   1.000   4.905   8.000 111.000

``` r
# Add Word variable for words to include low, average, and high word count and combine with dayshares dataframe for contingency table.
wordLinkShares <- dayShares %>% mutate(Words = if_else(n_tokens_content <= 500, "Low", if_else(n_tokens_content <= 625, "Average", "High")))
```

``` r
# Add Link variable for the number of links in the content to include low, average, and high count.
wordLinkShares <- wordLinkShares %>% mutate(Links = if_else(num_hrefs <= 9, "Low", if_else(num_hrefs <= 14, "Average","High")))

# Remove extra columns added to onLineNewsLifestyle.
onlineNewsLifestyle$WordCount <- NULL
onlineNewsLifestyle$Popular <- NULL
onlineNewsLifestyle$Words <- NULL
onlineNewsLifestyle$Links <- NULL
onlineNewsLifestyle$Day <- NULL
```

``` r
# Contingency table for the number of words in the content based on grouping the word count into categories of low, average, and high word count and grouping the shares based on popularity (shares greater than the median).
table(wordLinkShares$Popular, factor(wordLinkShares$Words, levels = c("Low", "Average", "High")))
```

    ##      
    ##       Low Average High
    ##   No  579     142  384
    ##   Yes 468     136  390

``` r
# Contingency table for the number of links in the content based on grouping the link count into categories of low, average,and high and shares based on popularity (shares greater than the median)
table(wordLinkShares$Popular, factor(wordLinkShares$Links, levels = c("Low", "Average", "High")))
```

    ##      
    ##       Low Average High
    ##   No  544     246  315
    ##   Yes 435     191  368

``` r
# Select predictors to view in GGPairs plot.
xpred <- onlineNewsLifestyle %>% select(kw_avg_avg, n_tokens_content, num_imgs, kw_min_avg, kw_max_avg, num_hrefs, shares)

xpred2 <- onlineNewsLifestyle %>% select(num_keywords, global_subjectivity, rate_positive_words, n_tokens_title, max_positive_polarity, shares)

# GGPairs plot to view correlation among the predictors.  Correlation greater than 75% indicates the predictors are highly correlated.
ggpairs(xpred, title = "Correlogram with ggpairs")
```

![](Reports/Lifestyle_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
ggpairs(xpred2, title = "Correlogram with ggpairs")
```

![](Reports/Lifestyle_files/figure-gfm/unnamed-chunk-14-2.png)<!-- -->

``` r
# The following scatterplot shows the trend of shares as a function of the number of links in the content.  An upward trend in the points indicates that articles with more links are shared more often.  A downward trend would indicate that articles with more links are shared less often.  If there is neither an upward or downward trend this indicates that the number of links in the article has no effect on whether the article will be shared.
correlation <- cor(onlineNewsLifestyle$shares, onlineNewsLifestyle$num_hrefs)  
g <- ggplot(onlineNewsLifestyle, aes(x = num_hrefs, y = shares)) + labs(y ="Number of Shares", x = "Number of Links")
g + geom_point(col = "purple") + ggtitle("Number of Links vs. Shares") + geom_text(x = 125, y = 30000, size = 5, label = paste0("Correlation = ", round(correlation, 2)))
```

![](Reports/Lifestyle_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
# The following scatterplot shows the relationship between the rate of positive words in the articles and the number of shares.  If the plots are on an upward trajectory then articles with more positive words are shared the most.  If the plots are on a downward trend then the articles with the most positive words are shared the least.
correlationTwo <- cor(onlineNewsLifestyle$shares, onlineNewsLifestyle$rate_positive_words)  
g <- ggplot(onlineNewsLifestyle, aes(x = rate_positive_words, y = shares)) + labs(y ="Number of Shares", x = "Rate of Positive Words")
g + geom_point(col = "blue") + ggtitle("Rate of Positive Words vs. Shares") + geom_text(x = 0.75, y = 100000, size = 5, label = paste0("Correlation = ", round(correlationTwo, 2)))
```

![](Reports/Lifestyle_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

## Modeling

``` r
# Split the data into a training (70% of the data) and test set (30% of the data).
set.seed(90)
trainIndex <- createDataPartition(onlineNewsLifestyle$shares, p = 0.7, list = FALSE)
onlineNewsTrain <- onlineNewsLifestyle[trainIndex, ]
onlineNewsTest <- onlineNewsLifestyle[-trainIndex, ]
dim(onlineNewsTrain)
```

    ## [1] 1472   53

``` r
dim(onlineNewsTest)
```

    ## [1] 627  53

### Linear Regression Models

A linear regression model is a model used to determine the relationship
between two or more variables by fitting a linear equation to the data.
The linear equation is in the form of: Y = Bo + B1X + E where Y is the
response variable, X is the explanatory variable, Bo is the y intercept,
and B1 is the slope. The model can be used to predict the value of the
response variable based on the values of the explanatory variables. For
example a linear regression model can be used to model the relationship
between years of experience and salary.

``` r
# Use parallel computing for our models
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

# Store model one in a formula.
modelOne <- as.formula("shares ~ n_tokens_content + n_tokens_title + rate_positive_words + num_keywords + global_subjectivity + max_positive_polarity + num_imgs")

# Model fit on the training data.
fit1 <- train(modelOne, data = onlineNewsTrain,
              method = "lm",
              preProcess = c("center", "scale"),
              trControl = trainControl(method = "cv", number = 10))
fit1
```

    ## Linear Regression 
    ## 
    ## 1472 samples
    ##    7 predictor
    ## 
    ## Pre-processing: centered (7), scaled (7) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1324, 1325, 1324, 1325, 1324, 1325, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE     
    ##   8388.512  0.01659042  3365.702
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

``` r
# Backward selection chosen for additional linear model
modelTwo <- as.formula("shares ~ kw_avg_avg + kw_min_avg + kw_max_avg + num_hrefs + n_tokens_content")

# Model fit on training data
fit2 <- train(modelTwo, data = onlineNewsTrain,
              method = "lm",
              preProcess = c("center", "scale"),
              trControl = trainControl(method = "cv", number = 10))
fit2
```

    ## Linear Regression 
    ## 
    ## 1472 samples
    ##    5 predictor
    ## 
    ## Pre-processing: centered (5), scaled (5) 
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 1325, 1325, 1326, 1325, 1324, 1325, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE     
    ##   8651.633  0.02847098  3375.921
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

### Random Forest Model

Random Forest models are an extension of the tree based method bagging.
The random forest algorithm creates multiple trees from bootstrapped
samples, averages those results, and uses a random subset of predictors
for each bootstrap sample/tree fit. Random forests can be used for
classification and regression problems.

``` r
rfFit <- train(shares ~., data = onlineNewsTrain, 
               method = "rf", 
               trainControl = trainControl(method = "cv",
                                           number = 5),
               tuneGrid = data.frame(mtry = 1:10))
rfFit
```

    ## Random Forest 
    ## 
    ## 1472 samples
    ##   52 predictor
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 1472, 1472, 1472, 1472, 1472, 1472, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE       Rsquared     MAE     
    ##    1     9853.586  0.006748714  3486.656
    ##    2     9918.826  0.005737714  3577.999
    ##    3     9991.710  0.004592254  3636.907
    ##    4    10055.343  0.004414908  3677.926
    ##    5    10103.831  0.004652757  3696.081
    ##    6    10178.273  0.003842776  3719.815
    ##    7    10238.087  0.003876617  3751.219
    ##    8    10280.960  0.003604932  3762.539
    ##    9    10342.428  0.003671853  3779.418
    ##   10    10410.033  0.003317771  3788.638
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 1.

### Boosted Tree Model

Boosting is an ensemble tree-based method used for regression and
classification. Boosting seeks to improve prediction by training on
decision trees which are grown sequentially. Each subsequent tree is
grown on a modified version of the original data. The predictions are
then updated as new trees are grown, minimizing the error of the
previous tree. The parameters used in boosting are the number of trees,
interaction depth (number of splits on a tree), shrinkage, and minimum
number of observations in trees’ terminal nodes.

``` r
# Boosted tree. After examining ranges of values for number of trees and interaction depth, 25 trees with an interaction depth of 1 had the lowest test RMSE.
boostedGrid <- expand.grid(
  interaction.depth=c(1:4), 
  n.trees=c(25,50,100,150,200),
  shrinkage=0.1,
  n.minobsinnode=10
)

boostedTree <- train(shares ~., 
                     data = onlineNewsTrain,
             method = "gbm",
             preProcess = c("center", "scale"),
             trControl = trainControl(method = "repeatedcv", number = 5,
                                      repeats = 3),
             tuneGrid = boostedGrid)

# End parallel computing
stopCluster(cl)
```

## Comparison

``` r
# Test error computed for the first linear model.
pred <- predict(fit1, newdata = onlineNewsTest)
m <- postResample(pred, obs = onlineNewsTest$shares)

# Test error computed for the backwards selection linear model.
pred2 <- predict(fit2, newdata = onlineNewsTest)
p <- postResample(pred2, obs = onlineNewsTest$shares)

# Test error computed for the random forest model.
predForest <- predict(rfFit, newdata = onlineNewsTest)
n <- postResample(predForest, onlineNewsTest$shares)

# Test error computed for the boosted tree model.
predBoost <- predict(boostedTree, newdata = onlineNewsTest)
p2 <- postResample(predBoost, obs = onlineNewsTest$shares)


# Produce a table for all comparisons
comparisons <- t(rbind(p[1],p2[1],m[1],n[1]))
colnames(comparisons) <- c("Linear Model Backwards Selection", "Boosted Tree", "Linear Model One", "Random Forest")
comparisons
```

    ##      Linear Model Backwards Selection Boosted Tree Linear Model One Random Forest
    ## RMSE                         5880.118     5834.382         5942.093      5724.291

``` r
# Compare RMSE values and store in a data frame..
LmOneRmse <- sqrt(mean((pred-onlineNewsTest$shares)^2))
LmBackSel <- sqrt(mean((pred2-onlineNewsTest$shares)^2))
rForestRmse <- sqrt(mean((predForest-onlineNewsTest$shares)^2))
tree <- sqrt(mean((predBoost-onlineNewsTest$shares)^2))
modelRmse <- c(LinearModelOne = LmOneRmse, LinearModelBckSel = LmBackSel, rForest = rForestRmse, boostedTree = tree)

# Store results in a data frame.
df <- as.data.frame(modelRmse)

# Name the column in the data frame.
attributes(df)$names[1] <- "RMSE"
df
```

    ##                       RMSE
    ## LinearModelOne    5942.093
    ## LinearModelBckSel 5880.118
    ## rForest           5724.291
    ## boostedTree       5834.382

``` r
# Use the slice_min function to return the row with the lowest RMSE.
BestModel <- df %>% slice_min(RMSE)
BestModel
```

    ##             RMSE
    ## rForest 5724.291

``` r
# Use the apply function to return the final winner.
apply(X = select(df, RMSE), MARGIN = 2,
      FUN = function(x){
        winner <- df %>% slice_min(RMSE)
        names(winner) <- c("Winner")
        winner
      })
```

    ## $RMSE
    ##           Winner
    ## rForest 5724.291