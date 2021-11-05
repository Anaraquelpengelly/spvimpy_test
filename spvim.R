# tests
rm(list = ls())

library("RCurl")

heart_data <- read.csv(text = getURL("http://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data"), header = TRUE, stringsAsFactors = FALSE)
## minor data cleaning
heart <- heart_data[, 2:dim(heart_data)[2]]
heart$famhist <- ifelse(heart$famhist == "Present", 1, 0)
## folds
heart_folds <- sample(rep(seq_len(2), length = dim(heart)[1]))

## estimate the full conditional mean using linear regression
full_mod <- lm(chd ~ ., data = subset(heart, heart_folds == 1))
full_fit <- predict(full_mod)

## estimate the reduced conditional means for each of the individual variables
X <- as.matrix(heart[, -dim(heart)[2]])[heart_folds == 2, ] # remove the outcome for the predictor matrix
red_mod_sbp <- lm(full_fit ~ X[,-1])
red_fit_sbp <- predict(red_mod_sbp)

# install vimp
install.packages("vimp")
library("vimp")
library("dplyr")
?vim
