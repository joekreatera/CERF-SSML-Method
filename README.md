# CERF-SSML-Method
 Semisupervised multilabel method based on Random Forests


## Abstract 
Financial, Health and Education institutions produce Big Data. 
Labeling subjects, patients or students is expensive and complex
as databases are evolved to multi-label settings. To use the data, labeling Challenge
must be performed automatically to avoid manual procedures and expla-
nations must be provided as regulations apply. Random Forest methods Hypothesis
provide non-linear multi-labeling solutions discovering patterns based on
small data. These algorithms can work on unsupervised settings due to
clustering techniques on each node. We propose an Explainable Semi- Proposal
supervised Multi-Label Random Forest Method, based on Gower Dis-
tance and Clustering algorithms to complete missing label information
and inductively predict new observations. Through a 20 dataset study Experimental Study
with five missing data ratios, we provide evidence of state-of-the-art Mi-
cro F1 Score, AUPRC, AUROC and Label Rank Average Precision. The
model identifies the final clusters columns ranges with their associated
labels and provides performance metrics based on the test dataset


## Method

Our method utilizes the standard DT induction process. Nevertheless, instead 
of selecting only an attribute and a value, it discovers sets of features 
that best bi-partition the training instances, considering both attribute
and label spaces. It is known that SSML methods consider three semi-supervised
assumptions, and we incorporate them through random feature sets, bagging and
clustering techniques. When an instance needs a prediction, each node ́s selected
set of features is tested. The closest cluster to the unobserved example decides
the next child node according to the least distance instance. When the algorithm
reaches a leaf node, it predicts each label by applying a specific local model, built
during training. This final step is consistent with Binary Relevance frameworks
and is a fallback when the algorithm is unable to discover patterns through
clustering.


## How to use the method

This algorithm is built on python with dependencies listed on requirements.txt. Although the versions are set, it should work with the latest alternatives. 

After cloning this repo, go to the folder and install with
```
pip install -r /path/to/requirements.txt
```

To use the method, provided that the datasets and results folder is created (as it is originally on this repo), just do:
```
python ./test_basic.py 2
```
This will try 2 parameter combinations defined in the file. To change the dataset or the paramters, open the test_basic.py. The file already has a description on the train funcion to explain ways to do this. 

To use the explainability feature, apply the method with 1 parameter combination
```
python ./test_basic.py 1 explain
```