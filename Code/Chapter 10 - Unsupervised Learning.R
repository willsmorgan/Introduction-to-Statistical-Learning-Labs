#This script corresponds with Chapter 10 of Inroduction to Statistical Learning
  #Because the labs for Chapter 10 are relatively short, they will all be included here
#Author : William Morgan
#Sounds like a lot of HOOPLA

#10.4 Lab 1: Principal Components Analysis
library(ISLR)
library(leaps)

##### SECTION 1: Performing PCA ######
#Objective - Perform PCA on the USArrests data, which is contained in the base R package
states <- row.names(USArrests) #The rows contain the 50 states in alphabetical order
states
names(USArrests) #The columns of the data contain four variables: Murder, Assault, UrbanPop, and Rape

#Briefly examine the mean and variance of the four columns
apply(USArrests, 2, mean) 
apply(USArrests, 2, var)

### Pause for Analysis ###
#First, notice how the apply() function is used - we are applying the mean() and variance() functions to the columns (second argument; 2) of the USArrests data
#Second, observe the large difference in the means and variances of our variables:
  #If we did not standardize the variables, the PCA would mainly be driven by Assault

#Perform principal components analysis using the prcomp() function
pr.out <- prcomp(USArrests, scale = T) #prcomp() centers the variables to have mean zero by default, while scale = T scales the variables to have std. dev of 1
names(pr.out) #center and scale components correspond to means and std. devs of the variables before implementing PCA

pr.out$rotation #the rotation matrix provides the principal component loading vectors
dim(pr.out$x) #x contains the principal component score vectors
biplot(pr.out, scale = 0)

#Find the amount of variance explained by each PC
pr.var <- pr.out$sdev^2
pr.var

#To compute the proportion of variance explained by each PC, divide the variance explained of each PC by the total variance explained by all four PC
pve <- pr.var / sum(pr.var)
pve

#Plot the PVE of each component as well as the cumulative PVE
plot(pve, xlab  = "Principal Component",
     ylab = "Proportion of Variance Explained",
     ylim = c(0,1),
     type = "b")

lines(cumsum(pve),
      type = "b",
      col = "green")

##### End of Section 1 #####

##### SECTION 2: K-Means Clustering #####
#Objective - Find the clusters of simulated data using the kmeans() function

#Create a matrix containing two well-defined clusters
set.seed(2)
x <- matrix(rnorm(50*2), ncol = 2)
x[1:25, 1] <- x[1:25, 1] + 3
x[1:25, 2] <- x[1:25, 2] - 4

#Perform K-means clustering with K=2 and plot the results
km.out <- kmeans(x, 2, nstart=20)
km.out$cluster

plot(x, col = (km.out$cluster+1), main = "K-means Clustering with K=2", 
     xlab = "", ylab = "", pch = 20, cex = 2)

#We knew that this data would have two clusters beforehand, so let's observe what happens when we run it with K=3
set.seed(4)
km.out <- kmeans(x, 3, nstart=20)
km.out

plot(x, col = (km.out$cluster+1), main = "K-means Clustering with K=3", 
     xlab = "", ylab = "", pch = 20, cex = 2)

#To run the kmeans() function with multiple initial cluster assignments, use the nstart argument
set.seed(3)
km.out <- kmeans(x, 3, nstart = 1)
km.out$tot.withinss

km.out <- kmeans(x, 3, nstart = 20)
km.out$tot.withinss #Observe how this value is smaller than the previous result with only one initial set

##### End of Section 2 #####

##### SECTION 3: Hierarchical Clustering #####
#Objective - Use Euclidean distance as a dissimilarity measure to find clusters in the simulated data from the previous section

#Perform hierarchical clustering with complete, single, and average linkages
hc.complete <- hclust(dist(x), method = "complete")
hc.single <- hclust(dist(x), method = "single")
hc.average <- hclust(dist(x), method = "average")

#Plot the dendrograms from each clustering
par(mfrow=c(1,3))
plot(hc.complete, main = "Complete Linkage", cex = .9)
plot(hc.single, main = "Single Linkage", cex = .9)
plot(hc.average, main = "Average Linkage", cex = .9)

#Use cutree() to determine clusters associated with a given cut of a dendrogram tree
cutree(hc.complete, 2)
cutree(hc.single, 2) #There exists a point that belongs to its own cluster, it will probably be necessary to increase the number of clusters
cutree(hc.average, 2)

#Rerun hclust() with scaled variables
xsc <- scale(x)
plot(hclust(dist(xsc), method = "complete"), main = "Hierarchical Clustering with Scaled Features")

#Practice clustering usinga correlation-based distance measure
x <- matrix(rnorm(30*3), ncol = 3)
dd <- as.dist(1-cor(t(x)))
plot(hclust(dd, method = "complete"), main = "Complete Linkage with Correlation-Based Distance"))

###### End of Section 3 #####