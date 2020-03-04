# Lab 3: Bayesian Learning and Boosting

## Assignment 1
*Use the provided function, `genBlobs()`, that returns Gaussian distributed data points together with class labels, to generatesome test data. Compute the ML-estimates for the data and plot the 95%-confidence interval using the function 
`plotGaussians`.*

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass1.png "ass1")

## Assignment 3
*Run `testClassifier` for the datasets and take note of the accuracies. Use `plotBoundary` to plot the decision boundary of the 2D iris dataset.*

#### Iris dataset classified using Naive Bayesian classifier

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass3.png "ass3")

```
Trial: 0 Accuracy 84.4
Trial: 10 Accuracy 95.6
Trial: 20 Accuracy 93.3
Trial: 30 Accuracy 86.7
Trial: 40 Accuracy 88.9
Trial: 50 Accuracy 91.1
Trial: 60 Accuracy 86.7
Trial: 70 Accuracy 91.1
Trial: 80 Accuracy 86.7
Trial: 90 Accuracy 91.1
Final mean classification accuracy  89 with standard deviation 4.16
```

#### Vowels dataset classified using Naive Bayesian classifier

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass3vow.png "ass3vow")

```
Trial: 0 Accuracy 61
Trial: 10 Accuracy 66.2
Trial: 20 Accuracy 74
Trial: 30 Accuracy 66.9
Trial: 40 Accuracy 59.7
Trial: 50 Accuracy 64.3
Trial: 60 Accuracy 66.9
Trial: 70 Accuracy 63.6
Trial: 80 Accuracy 62.3
Trial: 90 Accuracy 70.8
Final mean classification accuracy  64.7 with standard deviation 4.03
```

*Answer the following questions:* 
1. *When can a feature independence assumption be reasonable and when not?*
2. *How does the decision boundary look for the Iris dataset? How could one improve the classification results for this scenario by changing classifier or, alternatively, manipulating the data?*

[//]: # (Hello)


1. A very simple example when we can assume independence is if we want to classify the exam performance of students at KTH and we use age and eye colour of a student as features. These are clearly not related and can therefore be considered independent. 
Let's say we do the same study but (for some reason) use height and shoe size instead. Then it is not reasonable to assume independence since these clearly depend on each other. 

2. We can see that the decision boundary intersect class 1 and 2 whereas the boundary between it shows a clear distinction between class 0 and 1. This is because class 0 is clearly separable from the rest of the of the data compared to class 1 and 2 which show some overlap. We could improve this by assigning weights to points that display a large classification error over several iteration. This way, we can focus on the points that are difficult to separate from the rest of the data. 

## Assignment 5

*Compute the classification accuracy of the boosted classifier on some data sets using testClassifier from labfuns.py and compare it with those of the basic classifier on the vowels and iris data sets (see Assignment 3):*

1. *Is there any improvement in classification accuracy? Why/why not?*
2. *Plot the decision boundary of the boosted classifier on iris and compare it with that of the basic. What differences do you notice? Is the boundary of the boosted version more complex?*
3. *Can we make up for not using a more advanced model in the basic classifier
(e.g. independent features) by using boosting?*

#### Iris dataset classified using boosted Naive Bayesian classifier

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass5iris.png "ass5iris")

```
Trial: 0 Accuracy 95.6
Trial: 10 Accuracy 100
Trial: 20 Accuracy 93.3
Trial: 30 Accuracy 91.1
Trial: 40 Accuracy 97.8
Trial: 50 Accuracy 93.3
Trial: 60 Accuracy 93.3
Trial: 70 Accuracy 97.8
Trial: 80 Accuracy 95.6
Trial: 90 Accuracy 93.3
Final mean classification accuracy  94.7 with standard deviation 2.82
```

#### Vowels dataset classified using boosted Naive Bayesian classifier

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass5vow.png "ass5vow")

```
Trial: 0 Accuracy 76.6
Trial: 10 Accuracy 86.4
Trial: 20 Accuracy 83.1
Trial: 30 Accuracy 80.5
Trial: 40 Accuracy 72.7
Trial: 50 Accuracy 76
Trial: 60 Accuracy 81.8
Trial: 70 Accuracy 82.5
Trial: 80 Accuracy 79.9
Trial: 90 Accuracy 83.1
Final mean classification accuracy  80.2 with standard deviation 3.52
```

1. There's a clear improvement in classification accuracy for both datasets, which is the purpose of boosting. This is because we have added weights to the data points which we update iteratively to focus on the points that are difficult to classify. 
2. The decision boundary for the iris dataset better fits to the data this time around and is more "tailored" to the data. For the vowel dataset we observe the opposite. 
3. For the iris set it seems to suffice with using boosting, but not fully with the vowel dataset. We some improvement, but it could probably be better given a more accurate method. 


## Assignment 6

*Answer questions 1-3 in assignment 5 for the decision tree.*

#### Iris dataset classified using decision tree classifier

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass6iris.png "ass6iris")


```
Trial: 0 Accuracy 95.6
Trial: 10 Accuracy 100
Trial: 20 Accuracy 91.1
Trial: 30 Accuracy 91.1
Trial: 40 Accuracy 93.3
Trial: 50 Accuracy 91.1
Trial: 60 Accuracy 88.9
Trial: 70 Accuracy 88.9
Trial: 80 Accuracy 93.3
Trial: 90 Accuracy 88.9
Final mean classification accuracy  92.4 with standard deviation 3.71
```


#### Iris dataset classified using boosted decision tree classifier

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass6irisboost.png "ass6irisboost")


```
Trial: 0 Accuracy 95.6
Trial: 10 Accuracy 100
Trial: 20 Accuracy 95.6
Trial: 30 Accuracy 93.3
Trial: 40 Accuracy 93.3
Trial: 50 Accuracy 95.6
Trial: 60 Accuracy 88.9
Trial: 70 Accuracy 93.3
Trial: 80 Accuracy 93.3
Trial: 90 Accuracy 93.3
Final mean classification accuracy  94.6 with standard deviation 3.65
```

1. The classification accuracy is marginally improved with the boosted tree compared to the boosted naive bayesian classifier. Both method seem to be equally good for this dataset. 
2. In this case the decision boundary is a piecewise continous straight line, which probably stems from the fact that the tree classfies each instance binary. With the non-bosted version its simply a straight line that cannot separate the data. The boosted version is able to tailor itself around the data so that the clusters can be separated. 
3. In this case boosting improves the accuracy, although it is rather good for the non-boosted verion as well. 

#### Vowels dataset classified using decision tree classifier

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass6vow.png "ass6vow")

```
Trial: 0 Accuracy 63.6
Trial: 10 Accuracy 68.8
Trial: 20 Accuracy 63.6
Trial: 30 Accuracy 66.9
Trial: 40 Accuracy 59.7
Trial: 50 Accuracy 63
Trial: 60 Accuracy 59.7
Trial: 70 Accuracy 68.8
Trial: 80 Accuracy 59.7
Trial: 90 Accuracy 68.2
Final mean classification accuracy  64.1 with standard deviation 4
```


#### Vowels dataset classified using boosted decision tree classifier

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab3/lab3py/ass6vow.png "ass6vowboost")

```
Trial: 0 Accuracy 85.7
Trial: 10 Accuracy 87.7
Trial: 20 Accuracy 87
Trial: 30 Accuracy 92.9
Trial: 40 Accuracy 85.1
Trial: 50 Accuracy 81.8
Trial: 60 Accuracy 89
Trial: 70 Accuracy 85.7
Trial: 80 Accuracy 87.7
Trial: 90 Accuracy 84.4
Final mean classification accuracy  86.8 with standard deviation 2.78
```

1. The non-boosted version performs similarity to the bayesian classifier and we see improvement with the boosted version. There is improvement in the boosted version because it is able to learn from previous instances of a similar tree. 
2. The decision boundary is very poor in this case and we don't see any improvement with the boosted one either. 
3. Boosting the decision tree classifier improves the accuracy but not the decsion boundary, which remains very poor even after boosting. 