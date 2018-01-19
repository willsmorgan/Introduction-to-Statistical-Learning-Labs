Chapter 9 Lab: Support Vector Machines
================

This lab primarily focuses on creating creating and tweaking a variety of Support Vector Classifiers using the `e1071` library. The text mentions the availability of the `LiblineaR` library for very large linear problems, but we do not use it in this lab. We conclude the lab with a discussion/implementation of ROC curves using the `ROCR` library. The topics that will be covered in this lab include:

-   Support Vector Classifiers

-   Support Vector Machines (SVM)

-   ROC Curves

-   SVMs with K &gt; 2 Classes

------------------------------------------------------------------------

9.6.1: Support Vector Classifier
--------------------------------

We begin the lab by building up a support vector classifier with the `svm()` function. Recall that this classifier works by defining a linear decision boundary to separate the two classes. With this in mind, we need to make sure that we set the `kernel` option in `svm()` to "linear"

``` r
# Set up toy data to work with
set.seed(1)
dat <- tibble(
  x1 = rnorm(500),
  x2 = rnorm(500),
  y = factor(c(rep('A',250), (rep('B',250))))
)

# Add a little of separability between the classes
dat$x1[dat$y == "A"] <-  dat$x1[dat$y == "A"] + 5

# Check to see if classes are linearly separable
ggplot(dat, aes(x1, x2, col = y)) + 
  geom_jitter()
```

![](Chapter_9_-_Support_Vector_Machines_files/figure-markdown_github/Create%20toy%20data-1.png)

Now that we've created our toy data and can clearly see they are not linearly separable, let's fit the support vector classifier. Note that we need to make sure that `y` is encoded as a factor variable in our formula. We set the `cost` argument to 10 in this example as well. (Default is 1)

``` r
svc <- svm(y ~ ., data = dat, kernel = 'linear',
           cost = 10, scale = FALSE)
plot(svc, dat)
```

![](Chapter_9_-_Support_Vector_Machines_files/figure-markdown_github/SVC-1.png)

``` r
summary(svc)
```

    ## 
    ## Call:
    ## svm(formula = y ~ ., data = dat, kernel = "linear", cost = 10, 
    ##     scale = FALSE)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  10 
    ##       gamma:  0.5 
    ## 
    ## Number of Support Vectors:  10
    ## 
    ##  ( 5 5 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  A B

Our support vector classifier works pretty well (as it should, the data was made to be very easily separated). Upon closer inspection, we can see that there are 10 support vectors, 5 from each class. You'll notice that these 10 support vectors are plotted as crosses on the above graph, and everything is left as an open circle

What should we expect to see if we set a smaller cost parameter? With a smaller cost paramater our decision boundary gets a little more rigid and will allow for fewer errors in the training set. In short - smaller margins, fewer support vectors

``` r
svc2 <- svm(y ~ ., data = dat, kernel = 'linear',
            cost = 1, scale = FALSE)

summary(svc2)
```

    ## 
    ## Call:
    ## svm(formula = y ~ ., data = dat, kernel = "linear", cost = 1, 
    ##     scale = FALSE)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  1 
    ##       gamma:  0.5 
    ## 
    ## Number of Support Vectors:  13
    ## 
    ##  ( 7 6 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  A B

How can we determine the best cost parameter to use? You should know by now that cross-validation is pretty much the answer to every parameter-tuning problem. Luckily, the `e1071` library contains the `tune()` function, which will allow us to perform 10-fold cross-validation over a range of cost parameters.

``` r
svc_cv <- tune(svm,y ~., data = dat, kernel = 'linear',
                   ranges = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
```

`tune()` returns an object of class `tune`, which contains results from the 10-fold CV, including the best performing parameter values and their respective errors (`best.parameters`, `best.performance`), a data frame of all parameter combinations and their corresponding results (`performances`)

We'll take a glance at these components and check the results of our best-performing cross-validated model

``` r
summary(svc_cv)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost
    ##     1
    ## 
    ## - best performance: 0.004 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.246 0.17231754
    ## 2 1e-02 0.006 0.01349897
    ## 3 1e-01 0.006 0.01349897
    ## 4 1e+00 0.004 0.01264911
    ## 5 5e+00 0.004 0.01264911
    ## 6 1e+01 0.004 0.01264911
    ## 7 1e+02 0.004 0.01264911

``` r
best <-  svc_cv$best.model
summary(best)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = y ~ ., data = dat, ranges = list(cost = c(0.001, 
    ##     0.01, 0.1, 1, 5, 10, 100)), kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  1 
    ##       gamma:  0.5 
    ## 
    ## Number of Support Vectors:  24
    ## 
    ##  ( 12 12 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  A B

Now that we have a working model, let's generate some test data (which should've been done beforehand but is fine for instructional purposes). Keep in mind that we're making our own data to have a pretty well-defined class boundary, so our test error should be practically nil (and it is, as you'll see)

``` r
# Change the seed so we don't get the exact same data
set.seed(2)
test <- tibble(
  x1 = rnorm(500),
  x2 = rnorm(500),
  y = factor(c(rep('A',250), (rep('B',250))))
)

# Add the same amount of separability between the classes
test$x1[test$y == "A"] <-  test$x1[test$y == "A"] + 5

# Make predictions
predictions <- predict(best, test)
table(predictions, truth = test$y)
```

    ##            truth
    ## predictions   A   B
    ##           A 249   1
    ##           B   1 249

------------------------------------------------------------------------

9.6.2 Support Vector Machines
-----------------------------

This section is mostly going to be an extension of the previous one, only now we are learning how to use different kernels. In particular, we will do one example with a radial kernel and another with a polynomial kernel. We begin by creating sample data

``` r
# Set up toy data to work with
set.seed(1)
dat <- tibble(
  x1 = rnorm(500),
  x2 = rnorm(500),
  y = factor(c(rep('A',250), (rep('B',250))))
)

# Add a little of separability between the classes
dat$x1[dat$y == "A"] <-  dat$x1[dat$y == "A"] + 3
```

We now fit the SVM using the `kernel = 'radial'` argument and setting the gamma parameter to one

``` r
train <- dat %>%
  sample_frac(.5)

test <-  setdiff(dat, train)

rad_kern <-  svm(y ~ ., data = train, kernel = 'radial', gamma = 1, cost = 1)
plot(rad_kern, train)
```

![](Chapter_9_-_Support_Vector_Machines_files/figure-markdown_github/radial%20kernel-1.png)

Instead of spending too much time on this preliminary result, let's just continue on by doing some cross-validation to determine the best gamma and cost parameter values. Then, we can check how well those parameter values perform on the training set.

``` r
rad_cv <-  tune(svm, y ~ ., data = train, kernel = 'radial',
                ranges = list(cost = c(.1, 1, 10, 100, 1000),
                              gamma = c(.5, 1, 2, 3, 4)))

summary(rad_cv)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost gamma
    ##     1   0.5
    ## 
    ## - best performance: 0.096 
    ## 
    ## - Detailed performance results:
    ##     cost gamma error dispersion
    ## 1  1e-01   0.5 0.108 0.07554248
    ## 2  1e+00   0.5 0.096 0.06850791
    ## 3  1e+01   0.5 0.096 0.08044322
    ## 4  1e+02   0.5 0.112 0.07728734
    ## 5  1e+03   0.5 0.128 0.08599742
    ## 6  1e-01   1.0 0.096 0.06310485
    ## 7  1e+00   1.0 0.096 0.07351493
    ## 8  1e+01   1.0 0.100 0.07601170
    ## 9  1e+02   1.0 0.124 0.08315982
    ## 10 1e+03   1.0 0.136 0.08682038
    ## 11 1e-01   2.0 0.096 0.05719363
    ## 12 1e+00   2.0 0.108 0.07315129
    ## 13 1e+01   2.0 0.112 0.08175845
    ## 14 1e+02   2.0 0.148 0.09624044
    ## 15 1e+03   2.0 0.188 0.07067924
    ## 16 1e-01   3.0 0.096 0.05719363
    ## 17 1e+00   3.0 0.112 0.08175845
    ## 18 1e+01   3.0 0.120 0.08640988
    ## 19 1e+02   3.0 0.140 0.08055364
    ## 20 1e+03   3.0 0.176 0.07820202
    ## 21 1e-01   4.0 0.100 0.06036923
    ## 22 1e+00   4.0 0.112 0.08175845
    ## 23 1e+01   4.0 0.144 0.09834181
    ## 24 1e+02   4.0 0.156 0.07876830
    ## 25 1e+03   4.0 0.164 0.06915361

``` r
table(prediction = predict(rad_cv$best.model),
      actual = train$y)
```

    ##           actual
    ## prediction   A   B
    ##          A 117  11
    ##          B  11 111

We'll just quickly run the same analysis with a polynomial kernel so we can compare the results

``` r
poly_cv <-  tune(svm, y ~ ., data = train, kernel = 'polynomial',
                 ranges = list(cost = c(.1, 1, 10, 100, 1000),
                               degree = c(2, 3, 4)))

table(prediction = predict(poly_cv$best.model),
      actual = train$y)
```

    ##           actual
    ## prediction   A   B
    ##          A 115   7
    ##          B  13 115

------------------------------------------------------------------------

9.6.3 ROC Curves
----------------

In order to produce ROC curves from SVM fits, we need to extract the fitted decision values by setting `decision.values = T` in `svm()`. Once we have that, we use the `ROCR` library to help us build the curves.

For this example, we create the ROC curves using the training data because it is slightly less effort

``` r
# Run first SVM and observe ROC performance
svm_roc <-  svm(y ~ ., data = train, kernel= 'radial',
                gamma = 2, cost = 1, decision.values = T)
fitted <-  svm_roc$decision.values

perform <-  prediction(fitted, train$y) %>%
  performance(., 'tpr', 'fpr')

original <-  tibble(x = unlist(perform@x.values),
              y = unlist(perform@y.values))

ggplot(original, aes(x,y)) + 
  geom_line()
```

![](Chapter_9_-_Support_Vector_Machines_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
# Rerun svm with more liberal gamma values
svm_roc2 <-  svm(y ~ ., data = train, kernel= 'radial',
                gamma = 50, cost = 1, decision.values = T)
fitted2 <-  svm_roc2$decision.values

perform2 <-  prediction(fitted2, train$y) %>%
  performance(., 'tpr', 'fpr')

high_gamma <-  tibble(x = unlist(perform2@x.values),
              y = unlist(perform2@y.values))

# Graphically compare the two models
bind_rows("original" = original, "adjusted gamma" = high_gamma, .id = "Model") %>%
  ggplot(., aes(x, y, col = Model)) + 
  geom_line()
```

![](Chapter_9_-_Support_Vector_Machines_files/figure-markdown_github/unnamed-chunk-1-2.png)

------------------------------------------------------------------------

9.6.4 SVM with K &gt; 2 Classes
-------------------------------

When our dependent variable is categorical with more than two classes, `svm()` will perform classification using the one-versus-one approach. Recall that this requires ${K} \\choose {2}$ SVMs for each possible comparison, so it might be kind of slow.

``` r
# Create sample data
set.seed(1)
dat <- tibble(
  x1 = rnorm(600),
  x2 = rnorm(600),
  y = factor(c(rep('A', 200), (rep('B', 200)), rep('C', 200)))
)

# Artificially add separability between the classes
dat$x1[dat$y == "A"] <- dat$x1[dat$y == "A"] + 5
dat$x1[dat$y == "B"] <- dat$x1[dat$y == "B"] - 5

# Fit the SVM
multi_class <-  svm(y ~., data = dat, kernel = 'radial', cost = 10, gamma = 1)
plot(multi_class, dat)
```

![](Chapter_9_-_Support_Vector_Machines_files/figure-markdown_github/SVM%20Multiclassification-1.png)
