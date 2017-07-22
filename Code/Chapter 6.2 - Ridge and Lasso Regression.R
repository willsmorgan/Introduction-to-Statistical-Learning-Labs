#This script corresponds with Chapter 6 of Inroduction to Statistical Learning
#Author : William Morgan

#6.6 Lab 2: Ridge Regression and the Lasso
library(glmnet) #glmnet() function for ridge/lasso 
library(ISLR) #Hitters data

##### SECTION 1: Ridge Regression #####
#Prepare the Hitters data as done in the previous lab
hitters <- Hitters
hitters <- na.omit(hitters)
attach(hitters)
colnames(hitters) <- tolower(colnames(hitters))

#Before continuing, take a look at the glmnet() vignette to get an idea of what we need beforehand
  #x - input matrix containing rows of observations
  #y - response variable matrix
  #lambda - a user created decreasing sequence of lambda values

#Lambda will be a vector of length 100, ranging from 10^10 to 10^-2
x <- model.matrix(salary~., hitters)[,-1]
y <- salary
grid <-10^seq(10,-2,length=100) 
ridge.mod <- glmnet(x,y, alpha=0, lambda=grid)

### Pause for Analysis ###
#Because we supplied a 100 lambdas, we've got a coefficient matrix that is 20x100
#We expect that the coefficients of models with larger lambdas will be much smaller than those with smaller lambdas
  #Remember, the sequence is decreasing so the further along the sequence the larger the coefficients

#Find the 50th and 100th lambdas, along with the coefficients associated with that model and their l_2 norm (excluding the intercept)
ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50])^2)

ridge.mod$lambda[100]
coef(ridge.mod)[,100]
sqrt(sum(coef(ridge.mod)[-1,100])^2)

#Just for fun, let's practice writing a function to extract this information for any lambda
shrinkage_coef <- function(glmnet_mod, ld) {
  coef_names <- names(coef(glmnet_mod)[,50])
  return(list(print(paste("The Lambda value is:",glmnet_mod$lambda[ld])),
         print(paste("The coefficient for",coef_names,"is:", coef(glmnet_mod)[,ld])),
         print(paste("The l_2 norm of these coefficients is:", sqrt(sum(coef(glmnet_mod)[-1,ld])^2)))
         ))
}

#Now that we've got a practice run down, let's split our sample to do some testing
set.seed(1)
train <- sample(1:nrow(x), nrow(x)/2)
test <- (-train)
y.test <- y[test]

### Important Note ###
#There are two common ways to randomly split a data set:
  #You can produce a random logical vector (TRUE/FALSE) and select observations corresponding to TRUE for the training data
  #Alternatively, randomlychoose a subset of numbers between 1 and n, which can then be used as the indices for the training data
#We do (and have been doing) the former in previous labs; this lab makes use of the latter

#Fit a ridge regression on the training set and evaluate its MSE on the test, using lambda = 4
ridge_mod <- glmnet(x[train,], y[train], alpha=0, lambda=grid, thresh=1e-12)
ridge_pred <- predict(ridge_mod, s=4, newx=x[test,]) #s option sets the lambda value, newx specifies new observations used to make predicitons
mean((ridge_pred - y.test)^2) #MSE

#Compare to the test MSE when lambda is extremely large (coefficients are approximately 0)
ridge_pred <- predict(ridge_mod, s=1e10, newx=x[test,])
mean((ridge.pred - y.test)^2)

#Finally, let's check if the ridge regression gives us better results than the least sqaures option (lambda = 0)
ridge_pred <- predict(ridge_mod, s=0, newx=x[test,], exact = T) #the exact option allow us to specify that lambda is exactly 0, instead of searching for the smalles value of lambda in "grid"
mean((ridge_pred - y.test)^2)

#Instead of arbitrarily choosing a lambda value a priori, it is better to use cross-validation to find the best lambda
  #This can be done using cv.glmnet(), which conducts 10-fold CV and can be increased to n-folds with the option nfolds
set.seed(1)
cv_out <- cv.glmnet(x[train,], y[train], alpha=0)
plot(cv_out)

best_lam <- cv_out$lambda.min

#What is the test MSE associated with this lambda?
ridge_pred <- predict(ridge_mod, s=best_lam, newx=x[test,])
mean((ridge_pred - y.test)^2)

#FINALLY we can run ridge regression on the entire data set, now that we have found the best value for our tuning parameter
out <- glmnet(x,y,alpha=0)
predict(out, type = "coefficients", s=best_lam)[1:20,]

##### End of 6.2.1 #####

##### SECTION 2: The Lasso #####
#Fit the lasso model and observe how some of the coefficients are exactly 0
lasso_mod <- glmnet(x[train,], y[train], alpha = 1, lambda = grid)
plot(lasso_mod)

#Perform CV and compute test errors
set.seed(1)
cv_out <- cv.glmnet(x[train,], y[train],alpha=1)
plot(cv_out)

best_lam <- cv_out$lambda.min
lasso_pred <- predict(lasso_mod, s=best_lam, newx=x[test,])
mean((lasso_pred - y.test)^2)

#Fit the lasso over the entire data set
out <- glmnet(x,y,alpha=1, lambda = grid)
lasso_coef <- predict(out, type = "coefficients", s=best_lam)[1:20,]
lasso_coef
lasso_coef[lasso_coef!=0]

### Pause for analysis ###
#The test MSE for the lasso is very similar to the ridge regression, but it does have a minor advantage:
  #12 of the 19 coefficients in the lasso model are exactly 0

##### End of 6.2.2 #####