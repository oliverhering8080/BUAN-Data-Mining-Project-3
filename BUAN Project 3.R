# BUAN 4310 Data Mining and Big Data
# Tinh-An Le, Zack Carey, Oliver Hering
# Project 3

# Brief Problem Description ----------------------------------------------------
# Queenie has decided to launch a real estate business and wants to understand 
# the market. We were provided with a data set including home prices in King 
# County. Utilizing the subset of the real estate market data that our group was 
# assigned, we were tasked with exploring the data and building a suitable model
# to build a suitable model to help determine what predicts home prices in King
# County.

# Objective --------------------------------------------------------------------
# Our objective was to build an appropriate model for the company from the King
# County home prices data subset and predict the price for new houses given in
# a separate test data set.

# Description of the Data ------------------------------------------------------
# The data subset included a multitude of different variables, including dates 
# and times of sales, aspects of the house, and location. The variables were the
# ID of sale, year, month, day and day of the week of the sale, price, number of
# bedrooms and bathrooms, square footage of home and lot, number of floor 
# levels, whether its a waterfront view or not, and the number of times it has
# been viewed. It also included the condition, grade, square footage of house
# apart from the basement and the basement, year built, year renovated, zipcode,
# latitude, longitude, and information regarding renovations. Our data included
# both numerical and ordinal variable types. we wanted to predict is the price 
# of the home. 

# Set up -----------------------------------------------------------------------

library(rpart)
library(rpart.plot)
library(forecast)
library(caret)


houses <- read.csv('house_3.csv', header = TRUE)

head(houses)
str(houses)
t(t(names(houses)))

# Data Transformations ---------------------------------------------------------
# For our data transformation, we removed some variables that we deemed not 
# necessary for our model. We removed all variables except bedrooms, bathrooms,
# square footage of the home and house, number of floors, whether it was a 
# waterfront property or not, the number of times it was viewed, condition, 
# grade based on King County gradign system, square footage of the house apart 
# from the basement, square footage of the basement, the year built, the year
# renovated, and the price. 

# Removing Unneeded Variables
houses <- houses[, c(8:20, 7)]
t(t(names(houses)))


# Training Validation split
set.seed(666)

train_index <- sample(1:nrow(houses), 0.6 * nrow(houses))
valid_index <- setdiff(1:nrow(houses), train_index)

train_df <- houses[train_index, ]
valid_df <- houses[valid_index, ]

nrow(train_df)
nrow(valid_df)

str(train_df)
str(valid_df)

# Details on Model -------------------------------------------------------------
# For our model, we decided to conduct a regression tree due to the numeric 
# manner of the target variable, price. We first split the home data subset into
# a training and validation set to ensure we could evaluate the accuracy of the
# model. We also decided to also do a shallower tree so that we were able to
# compare the accuracy between the shallow and full-length tree. 

# Regression Tree
tree <- rpart(price ~ ., data = train_df, method = 'anova')
prp(tree)


printcp(tree)
plotcp(tree)


# Predicting train_df and valid_df
predict_train <- predict(tree, train_df)
accuracy(predict_train, train_df$price)

predict_valid <- predict(tree, valid_df)
accuracy(predict_valid, valid_df$price)


# Trying a shallower tree
shallowtree <- rpart(price ~ ., data = train_df, method = 'anova', maxdepth = 3)
prp(shallowtree)

predict_train_shallow <- predict(shallowtree, train_df)
accuracy(predict_train_shallow, train_df$price)

predict_valid <- predict(tree, valid_df)
accuracy(predict_valid, valid_df$price)


# Predicting new records - full tree
houses_test <- read.csv('house_test_3.csv', header = TRUE)
t(t(names(houses_test)))
head(houses_test)

houses_test <- houses_test[, c(7:19)]
t(t(names(houses_test)))

houses_test_prediction <- predict(tree, newdata = houses_test)
houses_test_prediction


# Predicting new records - shallow tree
houses_test_prediction_shallow <- predict(shallowtree, newdata = houses_test)
houses_test_prediction_shallow


# for these three specific records, the shallow tree returns the same predictions
# as the full length tree.
# RMSE is lower for the full length tree indicating less error.

# Discussion of Final Model ----------------------------------------------------

# Predictions for New Houses ---------------------------------------------------
