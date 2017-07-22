#This script corresponds with Chapter 5 of Introduction to Statistical Learning
#Author : William Morgan

#5.3 Lab - Cross-Validation and the Bootstrap
set.seed(1)
library("ISLR") #Auto, Portfolio data sets
library("boot") #cv.glm()
###### 5.3.1 - The Validation Set Approach ######
#Objective - estimate test error rates that result from fitting various linear models on the Auto data set

#We randomly generate a vector of integers to tell us how to divide up our data
attach(Auto)
train <- sample(392,196)

#Fit the model with the training set, and find the calculate the mean squared erroro
lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train)
mean((mpg - predict(lm.fit, Auto))[-train]^2)

#Refit the model using higher order terms
lm.fit2 <- lm(mpg ~ poly(horsepower,2), data = Auto, subset = train)
mean((mpg - predict(lm.fit2, Auto))[-train]^2)

lm.fit3 <- lm(mpg ~ poly(horsepower,3), data = Auto, subset = train)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)

#If we want to find a different training set, simply change the seed and repeat this process
set.seed(2)
train <- sample(392,196)

lm.fit <- lm(mpg ~ horsepower, data = Auto, subset = train)
mean((mpg - predict(lm.fit, Auto))[-train]^2)

lm.fit2 <- lm(mpg ~ poly(horsepower,2), data = Auto, subset = train)
mean((mpg - predict(lm.fit2, Auto))[-train]^2)

lm.fit3 <- lm(mpg ~ poly(horsepower,3), data = Auto, subset = train)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)

### Pause for Analysis ###
#The error terms are consistent across training sets and fits, we can also see that the quadratic term results in the least MSE for each sample

###### End of 5.3.1 ######

###### 5.3.2 - Leave-One-Out Cross-Validation ######
#Objective - Conduct LOOCV on the Auto data and observe error rates 

#The LOOCV estimate can be automatically estimated using the cv.glm() function, which is found in the "boot" library
glm.fit <- glm(mpg ~ horsepower, data = Auto)
cv.err <- cv.glm(Auto, glm.fit)
cv.err$delta

### Pause for analysis ###
#cv.glm() provides us a list with several components:
  #The delta vector contains the CV results - the average of the n test error rates

#Let's repeat this process for increasingly large polynomial terms and see how $delta changes
cv.error <- rep(0,5)
for (i in 1:5){
  glm.fit <- glm(mpg~poly(horsepower, i), data = Auto)
  cv.error[i] <- cv.glm(Auto, glm.fit)$delta[1]
}

cv.error

### Coding Tip ###
#Notice how we create the cv.error vector to accomodate the loop:
  #Before the loop begins, we create a empty vector that has the same length as the number of iterations of the loop (5)
  #With each iteration of the loop, a new model is created and a new MSE estimate is found
  #The indexing variable, i, tells us which iteration we are on and allows us to throw the newest estimate into the correct position in the cv.error vector

### Pause for analysis ###
#We can see that the average MSE drops off when we change to a quadratic term, the it doesn't really improve with higher degrees

###### End of 5.3.2 #######

###### 5.3.3 k-Fold Cross-Validation ######
#Objective - Use k-fold CV to generate estimates for the MSE

#The cv.glm() function we used earlier allows for k-fold CV, so we can repeat pretty much the exact same process
set.seed(17)
cv.error.10 <- rep(0,10)
for (i in 1:10){
  glm.fit <- glm(mpg~poly(horsepower, i), data = Auto)
  cv.error.10[i] <- cv.glm(Auto, glm.fit, K = 10)$delta[1]
}
cv.error.10

###### End of 5.3.3 ######

###### 5.3.4 - The Bootstrap ######
#Objective - Illustrate a use of bootstrapping with the Portfolio and Auto data sets (ISLR)

#Bootstrapping in R requires the computation of a statistic of interest
  #In this example, we write a function alpha.fn() to find the proportion of our money to invest in X (alpha) that minimizes variance below:
    #Var(alpha*X + (1-alpha)Y)
alpha.fn <- function(data, index){
  X <- data$X[index]
  Y <- data$Y[index]
  return((var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)))
}

#Now, we randomly select 100 observations (with replacement) from the portfolio to find an bootstrapped estimate of alpha
set.seed(1)
alpha.fn(Portfolio, sample(100,100, replace=T))

#To do conduct a bootstrap analysis we'd want to repeat this process, record the estimates for alpha, and find its standard deviation
  #Luckily, boot() does this for us
boot(Portfolio, alpha.fn, R=1000)

#Bootstrapping can be used to assess the variablility of coefficient estimates and predictions from a specific learning method:
  #Here, we use it to assess the variablility of the estimates of the coefficients in a linear regression model that predicts mpg using horsepower
  #We first define a function that takes in the Auto data, runs a regression and returns the estimated coefficients
  #Then, we can randomly generate bootstrapped samples and iterate as many times as we'd like
boot.fn <- function(data, index){
  return(coef(lm(mpg~horsepower, data=data, subset=index)))
}

set.seed(1)
boot.fn(Auto, 1:392) #Return the coefficients of the model that uses all observations
boot.fn(Auto, sample(392,392,replace=T)) #Return the coefficients of the model that samples 392 observations (w/replacement) from the original sample

boot(Auto, boot.fn, 1000)
summary(lm(mpg~horsepower, data=Auto))$coef #show this output so we can analyse differences in SEs 

### Pause for Analysis ###
#From this output we can see that the standard error for estimated intercept and slope terms are .8655 and .0076 respectively (in the bootstrap)
#Furthermore, why might there be a difference in the bootstrapped SEs and the single linear model?
  #It is because the estimate for variance is biased upwards

#Let's repeat what we just did above using a quadratic term 
boot.fn <- function(data, index){
  return(coef(lm(mpg~horsepower + I(horsepower^2), data=data, subset=index)))
}

set.seed(1)
boot(Auto, boot.fn, 1000)
summary(lm(mpg~horsepower+I(horsepower^2), data=Auto))$coef

###### End of 5.3.4 ######