#This script corresponds with Chapter 2 of Inroduction to Statistical Learning
#Author : William Morgan
#2.3 Lab: Introduction to R
library(ISLR)

##### SECTION 1: Getting Started with R ######
#We can create a vector x of any length by using the c() command
x <- c(1,3,2,5)

#Note that instead of using <- to create an object we can simply use = 
x = c(1,6,2)
y = c(1,4,3)

#We can apply standard mathematical operations to numeric vectors of similar lengths:
x+y
x*y

#To find the length of a vector, simply use the command length()
length(x)
length(y)

#To see all of the objects in our environment, we can use the command ls()
ls()

#We can remove objects from our environment using rm()
rm(x,y)

#It is possible to remove all objects at once:
rm(list=ls())

#The matrix() function can be used to create a matrix of numeric values
x <- matrix(data=c(1,2,3,4), nrow=2, ncol=2)
x

#The byrow option will fill the matrix horizontally with the data
matrix(c(1,2,3,4),2,2, byrow=TRUE)

#As before, standard operations can be applied to matrices
x^2
sqrt(x)

#The rnorm() function generates a vector of random normal variables, and cor() finds the correlation between two variables
x <- rnorm(50)
y <- x + rnorm(50, mean=50, sd=1)
cor(x,y)

#The set.seed() function will allows us to reproduce our code more easily when generating random variables
set.seed(1303)
rnorm(50)

#mean(), var(), and sd() are more mathematical functions that are self-explanatory
set.seed(3)
y <- rnorm(100)
mean(y)
var(y)
sqrt(var(y))
sd(y)

##### END OF SECTION 1 ######

##### SECTION 2: Graphics ######
#The plot() function is the quickest way to plot data; plot(x,y) will give us a scatter plot between x and y
x <- rnorm(100)
y <- rnorm(100)
plot(x,y)
plot(x,y,xlab="this is the x-axis", ylab="this is the y-axis", main="x vs y")

#To save the image of the plots we create, there are several options
pdf("figure 1: basic ass scatterplot.pdf")
plot(x,y,col="green")
dev.off() #This function tells R that we are done creating the plot

#seq(a,b,length=.) is a function that allows us to generate a sequence of numbers between a and b with a specified length
x <- seq(1,10)
x
x <- 1:10 #a:b is shorthand for seq(1,10)
x <- seq(-pi, pi, length=50)

#Let's try to graph three-dimensional data using contour()
y <- x 

f <- outer(x,y,function(x,y) cos(y)/(1+x^2)) #We write a cosine function whose domain is x,y and range is f
contour(x,y,f)
contour(x,y,f,nlevels=45,add=T)

fa=(f-t(f))/2 #Transpose our range matrix and transform it
contour(x,y,fa,nleveles=15)

#image() works in the same way, except that it adds color (much like a heatmap)
image(x,y,fa)

#persp() will give us a three-dimensional plot, with options theta and phi to rotate the viewing angles
persp(x,y,fa)
persp(x,y,fa, theta=30)
persp(x,y,fa, theta=30, phi=20)
persp(x,y,fa, theta=30, phi=40)
persp(x,y,fa, theta=30, phi=70)

##### END OF SECTION 2 #####

##### SECTION 3: Loading and Indexing Data #####
A <- matrix(1:16, 4,4)

#Square brackets are used for indexing and selecting subsets of data
  #The first number in the bracket refers to the row, and the second number refers to the column
  #In order to grab an entire row or an entire column, use a comma in place of a number
  #A negative sign indicates to R that everything but the listed numbers will be grabbed

A[2,3]
A[c(1,3),c(2,4)]
A[1:3,2:4]
A[,2]
A[2,]
A[-c(1,3),]

#dim() outputs the number of rows followed by the number of columns
dim(A)

#We begin by loading the Auto dataset taken from the ISLR website; it is saved in csv format so we must use read.csv()
auto <- read.csv("Data\Auto.csv")
fix(auto) #quick visual inspection
dim(auto) #get an idea of how large this dataset is
names(auto) #check out the names of the variables in this data

#Deal with missing observations by omitting them
auto <- na.omit(auto)

#Let's begin visualizing some of this data
attach(auto) #attach() allows the user to call on the column names of auto directly
plot(cylinders, mpg)

cylinders <- as.factors(cylinders) #This variable is more appropriately interpreted as a categorical variable
plot(cylinders, mpg) #Now that cylinders is a factor, plot() will output a boxplot
plot(cylinders, mpg, col="red")
plot(cylinders, mpg, col="red", varwidth=T)

#hist() can be used to plot a histogram
hist(mpg)
hist(mpg, col=2) #note that col=2 is equivalent to col="red"
hist(mpg, col=2, breaks=15)

#pairs() creates a scatterplot matrix for every pair of variables in a given data set; this can also be for a subset of those variables
pairs(auto)
pairs(~ mpg + displacement+ horsepower + weight + acceleration, auto)

#The identify() function lets you manually select points on a plot to identify
plot(horsepower, mpg)
identify(horsepower, mpg, name)

#summary() gives a numerical summary of each variable in a data set
summary(auto)
