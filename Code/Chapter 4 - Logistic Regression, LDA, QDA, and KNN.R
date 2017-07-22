#This script corresponds with Chapter 4 of Introduction to Statistical Learning
#Author : William Morgan

#4.6 Lab - Logistic Regression, LDA, QDA, and KNN
library("MASS") #lda()
library("ISLR") #location of Smarket data
library("class") #knn()

#Note - 4.6.1 is integrated into 4.6.2 because of length

###### 4.6.2 - Logistic Regression ######
#Objective - Fit a logistic regression to predict the direction of stock market data (Smarket)

#Begin with a cursory inspection of the data structure
colnames(Smarket) <- tolower(colnames(Smarket)) 
attach(Smarket)

### Writing Tip ###
#Sometimes variable names are capitalized inconsistently and can be difficult to remember;
#Simply change all the names of the variables (column names) to lowercase so you can be sure how they are written

names(Smarket) #Lag1 - Lag5 indicate the percentage returns for each of the five previous trading days
dim(Smarket)
summary(Smarket)
cor(Smarket[,-9]) #The 9th column variable, Direction, is qualitative so it should be excluded from our correlation matrix

plot(volume)

#We now fit a logit model in order to predict Direction using Lag1-Lag5 and Volume with the glm() function
glm.fit <- glm(direction ~ lag1 + lag2 + lag3 + lag4 + lag5 + volume, 
               data=Smarket, family="binomial") #The family="binomial" argument tells R we are running a logistic regression
summary(glm.fit)
summary(glm.fit)$coef

### Writing Tip ###
#When a function has many arguments, the line of code can get annoyingly long to read. To make it easier, 
  #start a new indented line after an argument (like this)

### Pause for Analysis ###
#Take a look at the coefficient for lag1 - it is negative, which implies that positive returns yesterday mean less chance of stock increase today
  #However, the p-value of lag1 (0.15) is kind of large, so there's not clear evidence of this relation being true

#Because our output is in log-odds, we need to use the option type="response" when making predictions if we want P(Y=1|X)
glm.probs <- predict(glm.fit, type="response")
glm.probs[1:10]

#To make sure that these probabilities are for the market going Up, check how R codes direction - we need to know that Y=1 implies up
contrasts(direction)

#Instead of looking at these probabilties one by one, we can set a threshold to tell us what our observations predict about the market
glm.pred <- rep("Down", 1250)
glm.pred[glm.probs>.5] = "Up"

#Now that we've got predicted values and true values of direction, we can create a confusion matrix to assess accuracy
table(glm.pred, direction)
mean(glm.pred==direction)

### Pause for Analysis ###
#The output of the mean() function tells us the fraction of days that our model predicted correctly - 52.16% of the time
  #That means that our model has a 47.84% training error rate - hardly better than flipping a coin
  #Furthermore, the training error rate generally underestimates the test error rate, so the actual accuracy may be worse

#The training data for the model is the entire data set, leaving nothing for testing. Let's split up the data to create a test set
train <- (year < 2005) #Create training set indicator for the years 2001-2004
Smarket.2005 <- Smarket[!train, ] #Subset the original data to keep data with train == 0
direction.2005 <- Smarket.2005[,"direction"] #true values of the test set

### Coding Tip ###
#Instead of splitting the original data (Smarket) into two new sets, training and testing, we simply use a new indicator variable for training obs.
#This is useful when working with extremely large data sets and will keep RAM usage down

#Fit a new logistic regression using only the subset of observations for which train == 1, and then find how that model predicts the test set
glm.fit <- glm(direction ~ lag1 + lag2 + lag3 + lag4 + lag5 + volume, data = Smarket,
               family = "binomial", subset = train)
glm.probs <- predict(glm.fit, Smarket.2005, type = "response")

#Create Yes/No predictions just as we did with the previous model
glm.pred = rep("Down", length(glm.probs))
glm.pred[glm.probs>.5] = "Up"

### Coding Tip ###
#For better reproducibility of your code, it's usually a good idea to refrain from using explicit numbers, like we did in line 48 with glm.pred
  #Instead, try to find ways to generalize what you're trying to do so that it can be done again without many edits to the original code
  #For instance, when we create the "Down" vector,  instead of using the number 252 in the second argument we just say length(glm.probs)
  #Both are equivalent lines of code, but the second allows us to easily change the size of the test set and not worry about running into errors

table(glm.pred, direction.2005)
mean(glm.pred == direction.2005)

### Pause for Analysis ###
#The results aren't that great - the test error rate is 52%
#We noticed earlier that the p-values were pretty lackluster, so it might be worth investigating a similar model that excludes the terms with the highest p-values

#Let's fit a new model, only using lag1 and lag2 as predictors
glm.fit = glm(direction ~ lag1 + lag2, data = Smarket, family = "binomial", subset = train)
glm.probs = predict(glm.fit, Smarket.2005, type = "response")

glm.pred <- rep("Down", length(glm.probs))
glm.pred[glm.probs>.5] = "Up"

table(glm.pred, direction.2005)
mean(glm.pred == direction.2005)

### Pause for Analysis ###
#Results are a little better - 55.95% of our observations were predicted correctly by the model
#Further still, it has a 58% rate of correctly predicting an upward movement in the market (True Positives / (True Positives + False Positives))

### End of 4.6.2 ###

### 4.6.3 - Linear Discriminant Analysis ###
#Objective - Perform LDA on the Smarket data to predict direction

#We use the lda() function on the training data that we previously created; notice that the syntax is identical to lm() and glm()
lda.fit <- lda(direction ~ lag1 + lag2, data = Smarket, subset = train)
lda.fit

### Pause for Analysis ###
#The output first tells us that 49.19% of our training data corresponds to days where the market went down, and 50.8% went up
#The group means is a cross-tabulation of the means of the predictors for each level of the dependent variable, direction
#Lastly, the coefficients of linear discriminants gives us the linear combination of lag1 and lag2 that create the decision boundary
  #If the expression -.64*lag1 - .5135*lag2 is large, the LDA classifier will predict a market increase
  #Likewise, if the above expression is small, the LDA classifier will predict a market decrease

#The plot() function produces plots of the linear discriminants, obtained by plugging in values of lag1 and lag2 for each of the training obs.
plot(lda.fit)

### Pause for Analysis ###
#Observe the distributions of the plots for Up and Down - it is centered mostly around 0
  #What does this tell us? 
    #Because there are very few extreme values in these distributions, the model has a tough time predicting market movement in either class

#Now we make some predictions, and see how our results compare to those of the logistic regression
lda.pred = predict(lda.fit, Smarket.2005)
names(lda.pred) 
  #Notice the output of lda.pred:
    #"class" tells us the prediction of the LDA
    #"posterior" is a matrix of probailities of the observations belonging to that class
    #"x" contains the linear discrminants

lda.class = lda.pred$class
table(lda.class, direction.2005)
mean(lda.class==direction.2005) #The mean is pretty much identical to that of the logistic regression!

### End of 4.6.3 ###

### 4.6.4 - Quadratic Discriminant Analysis ###
#Objective - Fit a QDA model to the Smarket data and observe any difference from the previous methods

#The syntax for the qda() function is identical to lda(), so we can quickly replicate what we just did
qda.fit <- qda(direction ~ lag1 + lag2, data = Smarket, subset = train)
qda.fit # Notice that the qda() function does not report the coefficients of the linear discriminants

#Make the same usual predictions
qda.class <- predict(qda.fit, Smarket.2005)$class
table(qda.class, direction.2005)
mean(qda.class==direction.2005) #Our mean has increased to about 60%, which suggests the quadratic form assumed by QDA better fits the relationship

### End of 4.6.4 ###

### 4.6.5 K-Nearest Neighbors ###
#Objective - perform KNN using the knn() function, which is part of the class library

#The syntax for knn() is different from the commands that we have used in the past;
  #knn(train, test, cl, k)
    #train - the matrix or data frame of training set cases
    #test - the matrix or data frame of test set cases
    #cl - a vector containing the class labels for the training observations
    #k - the number of neighbors considered

library("class")

#Create the training and testing sets
train.X <- cbind(lag1, lag2)[train,]
test.X <- cbind(lag1, lag2)[!train,]
train.direction <- direction[train]

#Now use knn() to predict the market's movement  for the dates in 2005 (test set)
set.seed(1)
knn.pred <- knn(train.X, test.X, train.direction, k=1)
table(knn.pred, direction.2005)
mean(knn.pred==direction.2005)

#We only used one nearest neighbor, so obviously the predictive power isn't great. Let's try it with k=3
knn.pred <- knn(train.X, test.X, train.direction, k=3)
table(knn.pred, direction.2005)
mean(knn.pred==direction.2005)

###### End of 4.6.5 ######

###### 4.6.6 - An Application to Caravan Insurance Data ######
#Objective - Use KNN to make predictions as to whether or not car insurance will be bought
  #We will be using the Caravan data set found in the ISLR library

#Load the data, attach it, and inspect
dim(Caravan)
colnames(Caravan) <- tolower(colnames(Caravan))
attach(Caravan)
sapply(Caravan, class) #check the class of each variable to see which are categorical and which are continuous

summary(purchase)

### Pause for quick note ###
#The KNN classifier relies on identifying observations that are near one another, so the scale of the predictors is very important
  #Any variables that are on a large scale will have a much larger effect on the distance between the observations
    #ex: a difference of $1000 in salary has a much larger effect on the classifier than say a difference in 50 years of age
  #As a result, classifiers tend to overstate the importance of variables with larger scales and understate smaller ones

#All variables excluding purchase are numeric variables, so when we standardize the d.f. we ignore Purchase
standardized.x <- scale(Caravan[,-86])
var(Caravan[,1])
var(Caravan[,2])

var(standardized.x[,1])
var(standardized.x[,2])

#Now we split the observations into a test set, containing the first 1000 obs, and a training set, containing the remaining ones
test <- 1:1000
train.x <- standardized.x[-test,]
test.x <- standardized.x[test,]

train.y <- purchase[-test]
test.y <- purchase[test]

set.seed(1)
knn.pred <- knn(train.x, test.x, train.y, k=1)
mean(test.y!= knn.pred)
mean(test.y!="No")

### Pause for analysis ###
#It seems like we've got a solid classifier - the test error rate is only 11.8%!
#However, only 6% of people actually bought insurance, so we could have an error rate down to 6% if we changed the classifier to always predict No

table(knn.pred, test.y)
mean(knn.pred == test.y)

#Let's repeat the process using more neighbors to see if we get a more accurate classifier
knn.pred <- knn(train.x, test.x, train.y, k=3)
table(knn.pred, test.y)
mean(knn.pred == test.y)

knn.pred <- knn(train.x, test.x, train.y, k=5)
table(knn.pred, test.y)
mean(knn.pred == test.y)

#Now that we have our knn predictions, let's compare it to a logistic regression model using .5 as a cutoff
glm.fit <- glm(purchase ~., data = Caravan, family = binomial, subset = -test)

glm.probs <- predict(glm.fit, Caravan[test,], type = "response")
glm.pred <- rep("No", 1000)
glm.pred[glm.probs>.5] <- "Yes"

table(glm.pred, test.y)
mean(glm.pred == test.y)

#We actually did a terrible job predicting people buying the insurance, they're all wrong! Change to cutoff probability to see if we get better accuracy
glm.pred <- rep("No", 1000)
glm.pred[glm.probs>.25] <- "Yes"

table(glm.pred, test.y)
mean(glm.pred == test.y)

###### End of 4.6.6 ######