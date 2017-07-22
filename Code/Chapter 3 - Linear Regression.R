#This script corresponds with Chapter 3 of Introduction to Statistical Learning
#Author : William Morgan

#3.6 Lab - Linear Regression
library("MASS") #collection of data sets and functions
library("ISLR") #data sets associated with the text
library("car") #vif()
###### 3.6.2 - Simple Linear Regression ######
#Objective - use Boston data set to predict median house value in neighborhoods around Boston

#Load our data set, and take a quick glance and it's structure
fix(Boston)
names(Boston)

#Start with a basic regression using the lm(y ~ x, data) function
lm.fit <- lm(medv ~ lstat, data = Boston)

#Alternatively, you can attach the data set so that R knows to look for medv and lstat in the lm() function
attach(Boston)
lm.fit <- lm(medv ~ lstat)

#To get the detailed results of the regression, we use summary() on the lm object
summary(lm.fit)

#The lm object contains quite a bit of information, so we use names() to parse it out and find what we want
names(lm.fit)

#An obvious result of interest is the coefficient values; we can call on them by name (lm.fit$coefficients) or use coef()
coef(lm.fit)

#For confidence intervals around these estimates, use the confint() function
confint(lm.fit)

#predict() is used to produce confidence intervals and prediction intervals for the prediction of medv for a given value of lstat
predict(lm.fit, data.frame(lstat=c(5,10,15)), interval = "confidence")

#Now let's visualize this least squares regression line
plot(lstat, medv)
abline(lm.fit)

#abline() allows us to draw any line, so play around with the options lwd (length), col (color), and pch (plotting symbol)
abline(lm.fit, lwd = 3)
abline(lm.fit, lwd=3, col= "red")
plot(lstat, medv, col="red")
plot(lstat, medv, pch=20)
plot(lstat, medv, pch="+")
plot(1:20, 1:20, pch=1:20)

#The lm() function automatically produces diagnostic plots, which we can see using plot()
plot(lm.fit)

#Instead of viewing the 4 graphs individually, we can tell R to split the display into 4 plots
par(mfrow=c(2,2))
plot(lm.fit)

#Residuals can be plotted individually with residuals() or rstudent()
plot(predict(lm.fit), residuals(lm.fit)) 
plot(predict(lm.fit), rstudent(lm.fit))

#Leverage statistics can be computed for our predictors with hatvalues(); to find the observation with the largest leverage which.max() is used
plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))

###### End of 3.6.2 ######

###### 3.6.3 - Multiple Linear Regression ######
#Objective - Extend the current example using multiple predictors

#In order to add more predictors to our model, we use "+" in the lm syntax
lm.fit <- lm(medv ~ lstat + age, data=Boston)
summary(lm.fit)

#It is tedious to type out every one of our predictors, so we can use "." on the right hand side of the regression equation
lm.fit <- lm(medv ~ ., data=Boston)
summary(lm.fit)

#Check out the R^2 and Residual Squared Error of this model
summary(lm.fit)$r.sq
summary(lm.fit)$sigma

#Variance Inflation Factor (vif) is found with vif() in the "car" package
vif(lm.fit)

#If we want to exclude one or more of the predictors from our model, use -varname on the right hand side of the equation
lm.fit1 <- lm(medv ~ . -age, data=Boston)
summary(lm.fit1)
lm.fit1 <- update(lm.fit, ~.-age)

#Finally, we can include interaction terms in the model by multiplying the two variables of interest within the lm() function
summary(lm(medv ~ lstat*age, data=Boston))

###### End of 3.6.3 ######

#Note - 3.6.4 was included in the previous section because it was so short

###### 3.6.5 Non-linear Transformations of the Predictors ######

#lm() can accomodate for transformations of the predicors using I(varname)
lm.fit2 <- lm(medv ~ lstat + I(lstat^2))
summary(lm.fit2)

#To quantify how much better the quadratic terms fits the model, use anova()
lm.fit <- lm(medv ~ lstat)
anova(lm.fit, lm.fit2)

#For more evidence, let's visualize the residuals
par(mfrow=c(2,2))
plot(lm.fit2)

#We can include higher order polynomials by using poly(var, degree) within the lm function
lm.fit5 <- lm(medv ~ poly(lstat, 5))
summary(lm.fit5)

#Last but not least, check out a log transformation of our predictor "rm"
summary(lm(medv ~ log(rm), data=Boston))

###### End of 3.6.5 ######

###### 3.6.6 - Qualitative Predictors ######
#Objective - use the Carseats data set in the ISLR library to predict sales based on a number of predictors

fix(Carseats)
names(Carseats)

#Fortunately, R is able to recognize categorical variables and will automatically generate dummies when we run a regression
lm.fit <- lm(Sales ~ . + Income:Advertising + Price:Age, data=Carseats) #using ":" is another way to include interactions
summary(lm.fit)

#The contrasts() function returns the coding that R uses for the dummy variables
attach(Carseats)
contrasts(ShelveLoc)

###### End of 3.6.6 ######

###### 3.6.7 - Writing Functions (optional, but very useful) ######
#Objective - write a custom function to load the libraries we use over and over

loadlibraries <- function() {
  library("ISLR")
  library("MASS")
  print("The libraries have been loaded")
}

loadlibraries()

###### End of 3.6.7 ######