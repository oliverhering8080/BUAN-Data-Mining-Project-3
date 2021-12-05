# BUAN 4310 Data Mining and Big Data
# Tinh-an Le, Zack Carey, Oliver Hering
# Project 3

library(rpart)
library(rpart.plot)
library(forecast)
library(caret)


houses <- read.csv('house_3.csv', header = TRUE)

head(houses)
str(houses)
t(t(names(houses)))

# removing unneeded variables
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


