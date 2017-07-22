#This script corresponds with Chapter 6 of Inroduction to Statistical Learning
#Author : William Morgan

#6.5 Lab 1: Subset Selection Methods
library(ISLR)
library(leaps)

##### SECTION 1: Best Subset Selection #####
#Load the "Hitters" data and do a quick inspection of it
fix(Hitters)
colnames(Hitters) <- tolower(colnames(Hitters))
names(Hitters)
dim(Hitters)

#Salary is missing for some observations, so find out how many and drop them
sum(is.na(Hitters$salary))
Hitters <- na.omit(Hitters)

#Using the regsubsets() function, perform best subset selection
regfit.full <- regsubsets(salary~., Hitters)
summary(regfit.full)

### Pause for Analysis ###
#An asterisk indicates that a given variable is included in the corresponding model
  #For instance, the best model containing 3 variables will have Hits, CRBI, and PutOuts

#By default, we only see the first 8 models, but with the nvmax option the function can return results with as many variables as desired
regfit.fuller <- regsubsets(salary~., Hitters, nvmax=19)
reg.summary <- summary(regfit.fuller)

#summary() will also return R^2, RSS, adj R^2, C_p, and BIC of the model; let's check those out
names(reg.summary)
reg.summary$rsq #Observe that monotonic relationship between R^2 and the number of included variables

#Plot the RSS, adj R^2, C_p, and BIC for all of the models at once to do some visual judgement
par(mfrow=c(2,2))

plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")

plot(reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted Rsq", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col="red", cex=2, pch=20) #Grab the point at which adjr2 is maximized, and the value associated with it (x,y)
abline(v = which.max(reg.summary$adjr2), col = "red", lty=2) #drop a dashed line down to the x-axis (lty=2 for dashed)

plot(reg.summary$cp, xlab = "NumVar", ylab = "Cp", type= "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col ="red", cex=2, pch=20)
abline(v = which.min(reg.summary$cp), col = "red", lty=2)

plot(reg.summary$bic, xlab="NumVar", ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col ="red", cex=2, pch=20)
abline(v = which.min(reg.summary$bic), col = "red", lty=2)

#The regsubsets() function has a built-in plot option for doing just this, let's try it out
  #The top row of each plot contains a black square for each variable selected from the optimal model associated with that statistic
plot(regfit.fuller, scale = "r2") 
plot(regfit.fuller, scale = "adjr2")
plot(regfit.fuller, scale = "Cp")
plot(regfit.fuller, scale = "bic")

#Let's observe the coefficients from the six-variable model
coef(regfit.fuller, 6)


##### End of 6.1.1 #####

##### SECTION 2: Forward and Backward Stepwise Selection #####
#The regsubsets() function can also perform stepwise selection using the method= "" option
regfit.fwd <- regsubsets(salary~., data = Hitters, nvmax=19, method = "forward")
summary(regfit.fwd)

regfit.bwd <- regsubsets(salary~., data=Hitters, nvmax=19, method = "backward")
summary(regfit.bwd)

### Pause for Analysis ###
#Check out the differences in the 1,2, and 3-variable models:
  #in the forward selection output, CRBI is the best variable in the single variable model and hits is added along to the best model with two variables
  #in the backward selectin output, CRuns is the best variable in the single model, and CRuns/hits are the best variables for model with two vars

##### End of 6.1.2 #####

##### SECTION 3: Choosing among models using Cross-Valdation #####
#Create a logical vector that's the same length as num_obs in Hitters; this will determine how we split the into training/testing sets 
set.seed(1)
train <- sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)
test <- (!train)

#Perform best subset selection on the training set
regfit.best <- regsubsets(salary~., data=Hitters[train,], nvmax=19)

#Goal now is to compute validation set error for each model that was created; we must first make a model matrix from the test data
  #This creates a matrix that contains all the data that will be included when the model is run (builds bold "X" matrix)
test.mat <- model.matrix(salary~., data=Hitters[test,])

#Create an emtpy vector that is 19-long, and loop through each model in the regfit.best object to extract coefficients, find predictions, and calculate MSE
val.errors <- rep(NA, 19)
for(i in 1:19) {
  coefi <- coef(regfit.best, id = i)
  pred <- test.mat[,names(coefi)]%*%coefi
  val.errors[i] <- mean((Hitters$salary[test]-pred)^2)
}

#Which model has the lowest MSE and what are its coefficients?
which.min(val.errors)
coef(regfit.best, which.min(val.errors))

#Note that we had to create our own predictions, as regsubsets() does not contain an argument for it; let's write a function that will generalize this process
predict_regsubsets <- function(regfit_object, testing_set, id, ...){
  form <- as.formula(regfit_object$call[[2]]) #return the formula used in the regsubsets() command, the second element grabs the form of the model
  mat <- model.matrix(form, testing_set) #create the model matrix using the training_set and the formula from the previous line
  coefi <- coef(regfit_object, id)
  xvars <- names(coefi)
  mat[,xvars]%*%coefi
}

#We saw that the ten-variable model performed best, so let's rerun the regsubsets() using the testing data and select the ten-variable model
regfit.bester <- regsubsets(salary~., data= Hitters, nvmax=19)
coef(regfit.bester, 10)

#Finally, let's do some k-fold cross-validation
k <- 10
set.seed(1)
folds <- sample(1:k, nrow(Hitters), replace = TRUE)

#In the jth fold, the elements of Hitters belonging to j are considered the test set, while everything != j is considered training
cv_errors <- matrix(NA, k, 19, dimnames = list(NULL, paste(1:19)))
for (j in 1:k){
  best.fit <- regsubsets(salary~., data=Hitters[folds!=j,], nvmax=19)
  max_vars <- best.fit$call[[4]]
  for(i in 1:19){
    pred <- predict_regsubsets(best.fit, Hitters[folds==j,], id = i) #MLEM
    cv_errors[j,i] <- mean((Hitters$salary[folds==j]-pred)^2)
  }
}

#Use the apply function to average over the columns of this matrix to find the average MSE for each model
mean_cv_errors <- apply(cv_errors, 2, mean)
mean_cv_errors

#Plot the means to visually find the best performing model
par = mfrow(c(1,1))
plot(mean_cv_errors, type = "b")

#It should be obvious that the model with the lowest test MSE is the model with the 11 variables, so run the 11 variable model on the entire data set
reg.best <- regsubsets(salary~., data=Hitters, nvmax=19)
coef(reg.best, 11)

##### End of 6.1.3 #####