## Answers to questions from Lab 1

### Assignment 0:
MONK-2 is the hardest to learn since it requires the most questions to verify the truth value. 

### Assignment 1:

| Dataset | Entropy        |
|---------|----------------|
| MONK-1  | 1.0            |
| MONK-2  | 0.957117428265 |
| MONK-3  | 0.999806132805 |

### Assignment 2: 

We have the simple definition of entropy as 

E = -plog(p) -qlog(q),

with 

q = 1 - p.

For a uniform distribution we have 

q = p,

and 

q ≠ p, 

for a non-uniform distribution. 

The entropy will be higher for a uniform distribution because there is more unpredicability about the data set, since both outcomes are equally likely (assuming binary). By contrast, the entropy is low for a non-uniform distribution because there is more clarity with regards to what will happen, since one outcome will be more probable than another (again, assuming binary outcomes). See the following examples:

#### Entropy for uniform distribution

p = q = 0.5
E = 1.

#### Entropy for non-uniform distribution

p = 0.2, q = 0.7,
E = 0.72.

### Assignment 3


| Dataset | a_1   | a_2   | a_3   | a_4   | a_5   | a_6   |
|---------|-------|-------|-------|-------|-------|-------|
| MONK-1  | 0.075 | 0.006 | 0.005 | 0.026 | 0.287 | 0.001 |
| MONK-2  | 0.004 | 0.002 | 0.001 | 0.016 | 0.017 | 0.006 |
| MONK-3  | 0.007 | 0.294 | 0.001 | 0.003 | 0.256 | 0.007 |

The optimal attribute to be used to split data at root node for the different datasets is

MONK-1: a_5
MONK-2: a_5
MONK-3: a_2

### Assignment 4

When maximizing the information gain, the entropy for the datasets decrease maximally since we then know as much as possible at that given stage. By using information gain, we can more easily determine when it is optimal to ask about which attributes to find out more about or dataset quicker. 


### Assignment 5

We start by splitting the tree based on attribute a_5 and compute the information gain for the new leafs that we have obtained, shown in the table below:

| Dataset | a_1   | a_2   | a_3   | a_4   | a_5 | a_6   |
|---------|-------|-------|-------|-------|-----|-------|
| a_5 = 1     | 0.0   | 0.0   | 0.0   | 0.0   | 0.0 | 0.0   |
| a_5 = 2    | 0.040 | 0.015 | 0.037 | **0.049** | 0.0 | 0.026 |
| a_5 = 3     | 0.033 | 0.002 | 0.018 | 0.019 | 0.0 | **0.045** |
| a_5 = 4     | **0.206** | 0.034 | 0.026 | 0.076 | 0.0 | 0.003 |

We start by noting that the dataset is zero for all attributes, which is because we have all information we need when a_5 = 1. For the other three nodes, the attribute with the largest information gain is typed in boldface and thus indicates which attribute we should check for these nodes. Splitting these nodes accordingly, we compute the majority class for each of them and obtain the following tree shown below.

![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab1/ass-5-tree.png "tree for assignment 5")

---
Computing the train and test errors we obtain the following results 

| Dataset | E_train | E_test |
|---------|---------|--------|
| MONK-1  | 1.0     | 0.829  |
| MONK-2  | 1.0     | 0.692  |
| MONK-3  | 1.0     | 0.944  |

We first note that the there is no error for the training data, since that data was used to generate the trees to begin with. 

### Assignment 6 

In essence: if do not prune our decision tree at all it can be prone to have high variance, i.e. overfitting. On the other hand, if we 

### Assignment 7 


![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab1/ass-7-monk1.png "monk-1 for assignment 7")
![alt text](https://github.com/lindahlf/machine-learning/blob/master/Lab1/ass-7-monk3.png "monk-3 for assignment 7")

