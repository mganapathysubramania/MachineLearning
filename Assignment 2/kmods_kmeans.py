#!/usr/bin/env python
# coding: utf-8
#ManishGanapathySubramanianBharatwaj
# In[371]:


import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import sklearn.cluster as cluster
import math
import sklearn.neighbors
from matplotlib import style
from numpy import linalg as LA
import math
from sklearn.neighbors import DistanceMetric as Dm
from random import seed, choice
from kmodes.kmodes import KModes


# # Question 1

# In[372]:


#Question 1
inputcsv_groceries_df=pd.read_csv("C:/Sem 2/ML/Assignment 2/Groceries.csv")#Reading the Input data


# In[373]:


items_groceries = inputcsv_groceries_df.groupby(['Customer'])['Item'].apply(list).values.tolist()


# ### Question1.A 
# #### Number of Unique Items histogram and 25th,50th and 75th percentile of histogram

# In[374]:



number_of_Purchase = inputcsv_groceries_df.groupby('Customer').size()
frequencytable = pd.Series.sort_index(pd.Series.value_counts(number_of_Purchase))

plt.bar(frequencytable.index.values, frequencytable, color='green', width=1)
plt.ylabel("Frequency")
plt.xlabel("No of Unique Items")
plt.title("No of Unique Items Histogram")
plt.show()
number_of_customers= inputcsv_groceries_df['Customer'].max()
sumofcustomerfrequencytable = frequencytable.cumsum()

percentiles25 = sumofcustomerfrequencytable[sumofcustomerfrequencytable >= number_of_customers/4].index[0]
print("25th percentile = " + str(percentiles25))
percentiles50 = sumofcustomerfrequencytable[sumofcustomerfrequencytable >= number_of_customers/2].index[0]
print("50th percentile = " + str(percentiles50))
percentiles75 = sumofcustomerfrequencytable[sumofcustomerfrequencytable > 3*number_of_customers/4].index[0]
print("75th percentile = " + str(percentiles75))


# ### Question 1.B
# #### K-itemssets(atleast 75 customers) and largest K value

# In[375]:


transencod = TransactionEncoder()
transencod_array = transencod.fit(items_groceries).transform(items_groceries)
ItemIndicatorformat = pd.DataFrame(transencod_array, columns=transencod.columns_)
minSup = 75/number_of_customers
K = 3
frequent_itemsets = apriori(ItemIndicatorformat, min_support = minSup, max_len = K, use_colnames = True)
print("Number of Itemsets found are " + str(frequent_itemsets.count()['itemsets'])+" for K value "+str(K))
K=4
frequent_itemsets = apriori(ItemIndicatorformat, min_support = minSup, max_len = K, use_colnames = True)
print("Number of Itemsets found are " + str(frequent_itemsets.count()['itemsets'])+" for K value "+str(K))
print("So the Largest value of K among the itemsets is " + str(K))


# ### Question 1.C 
# #### Association Rule whoese confidence metrices are grater than or equal to 0.01(1%)

# In[376]:


association_rules_freq_itemsets = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print(association_rules_freq_itemsets)
print()
print()
print()
print("Total number of Association rules for 1% " + str(association_rules_freq_itemsets.count()[0]))


# ### Question 1.D
# #### Confidene vs Support

# In[377]:


plt.figure(figsize=(6,4))
plt.scatter(association_rules_freq_itemsets['confidence'], association_rules_freq_itemsets['support'], s = association_rules_freq_itemsets['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")

plt.show()


# ### Question 1.E
# #### Confidence metrics are greater than or equal to 0.6 (60%)

# In[378]:


antecendetsarr=np.array(association_rules_freq_itemsets[association_rules_freq_itemsets.confidence>=0.6]['antecedents'])
consequentsarr=np.array(association_rules_freq_itemsets[association_rules_freq_itemsets.confidence>=0.6]['consequents'])
confidencearr=np.array(association_rules_freq_itemsets[association_rules_freq_itemsets.confidence>=0.6]['confidence'])
supportarr=np.array(association_rules_freq_itemsets[association_rules_freq_itemsets.confidence>=0.6]['support'])
liftarr=np.array(association_rules_freq_itemsets[association_rules_freq_itemsets.confidence>=0.6]['lift'])
data=np.column_stack((antecendetsarr,consequentsarr,confidencearr,supportarr,liftarr))
requiredData=pd.DataFrame(data,columns=['Antecents','Consequents','Confidence','Support','Lift'])
print("Confidence metrics are greater than or equal to 60% : \n",requiredData)


# # Question 2

# In[379]:


inputcsv_cars_df=pd.read_csv("C:/Sem 2/ML/Assignment 2/cars.csv")#Reading the Input data


# ### Question 2.A
# #### Frequencies of the categorical feature Type

# In[380]:


#Q2.a
inputcsv_cars_dftype=inputcsv_cars_df['Type'].value_counts()
print(inputcsv_cars_dftype)


# ### Question 2.B
# #### Frequencies of the categorical feature DriveTrain

# In[381]:


#Q2.b
inputcsv_cars_driveTrain = inputcsv_cars_df['DriveTrain'].value_counts()
print(inputcsv_cars_driveTrain)


# ### Question 2.C
# #### Distance metric between ‘Asia’ and ‘Europe’ for Origin

# In[382]:


inputcsv_cars_origin = inputcsv_cars_df['Origin'].value_counts()
Asia_origin = inputcsv_cars_origin['Asia']
Europe_origin = inputcsv_cars_origin['Europe']
distance_metric_origin=1/Asia_origin + 1/Europe_origin
print("Distance metric between asia and europe origin "+str(distance_metric_origin))


# ### Question 2.D
# #### Distance metric between Cylinders = 5 and Cylinders = Missing

# In[383]:


inputcsv_cars_Cylinders = inputcsv_cars_df['Cylinders'].value_counts()
inputcsv_cars_Cylinders


# In[384]:


count=0
x=float('nan')
inputcsv_cars_Cy=inputcsv_cars_df['Cylinders']
df = pd.DataFrame(inputcsv_cars_Cy)
for index, row in df.iterrows():
    #print(row[0])
    if math.isnan(row[0]):
        #print(index,row[0])
        count+=1
    #count+=1
cylinders_five=inputcsv_cars_Cylinders[5.0]
cylinders_missing=count
distance_metric_cylinder=1/cylinders_five + 1/cylinders_missing
print("Distance metric between cylinder 5 and missing cylinders "+str(distance_metric_cylinder))


# ### Question 2.E and F

# In[385]:


# random categorical data
km = KModes(n_clusters=3, init='Huang', n_init=6, verbose=1)

df=inputcsv_cars_df.dropna(axis='columns')
clusters = km.fit_predict(df)

# Print the cluster centroids
print()
print("Centroid of Clusters")
print(km.cluster_centroids_)


# # Question 3

# In[386]:


FourCircleCSV = pd.read_csv("C:/Sem 2/ML/Assignment 2/FourCircle.csv")


# ### Question 3.A
# #### Total Clusters after plotting x on horizontal axis and y on vertical axis

# In[387]:


nofObjects = FourCircleCSV.shape[0]
plt.scatter(np.array(FourCircleCSV['x']), np.array(FourCircleCSV['y']))
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
numberofclusters = 4
print("Number of clusters 4")


# ### Question 3.B
# #### K-means Algorithm

# In[388]:


trainData = FourCircleCSV[['x','y']]
kmeans = cluster.KMeans(n_clusters=numberofclusters, random_state=60616).fit(trainData)
FourCircleCSV['Cluster1'] = kmeans.labels_
plt.scatter(FourCircleCSV['x'], FourCircleCSV['y'], c = FourCircleCSV['Cluster1'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()


# ### Question 3.C
# ### nearest neighbor algorithm using the Euclidean distance and egien value plot
# ### number of neighbours tried between 1 to 15 
# 

# In[389]:


number_neighbours = 6
print("number of neighbours = "+str(number_neighbours))
kNNSpec = sklearn.neighbors.NearestNeighbors(number_neighbours, algorithm = 'brute', metric = 'euclidean')
neighbors = kNNSpec.fit(trainData)
dist_3, indexes = neighbors.kneighbors(trainData)
distObject = Dm.get_metric('euclidean')
distances = distObject.pairwise(trainData)
Adjacency = np.zeros((nofObjects, nofObjects))
Degree = np.zeros((nofObjects, nofObjects))
for i in range(nofObjects):

    for j in indexes[i]:

        if (i <= j):

            Adjacency[i,j] = math.exp(- distances[i][j])

            Adjacency[j,i] = Adjacency[i,j]
for i in range(nofObjects):

    sum = 0

    for j in range(nofObjects):

        sum += Adjacency[i,j]

    Degree[i,i] = sum
Laplacematrix = Degree - Adjacency

evals, evecs = LA.eigh(Laplacematrix)
plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.ylabel('Eigenvalue')
plt.xlabel('Sequence')
plt.grid(True)
plt.show()


# ### Question 3.D
# #### Adjacency matrix, the Degree matrix, and the Laplacian matrix
# #### Eigen Values which are zero

# In[390]:


print("Adjacency Matrix:")
print(Adjacency)
print("Degree Matrix:")
print(Degree)
print("Laplace Matrix: ")
print(Laplacematrix)
print()
print()
print()
print("As per my understanding Eigen values which are practically zero are 4 ")
print("Values of the “zero” eigenvalues in scientific notation")
val=[]
for i in range(0,9):
    if round(evals[i,],4)==0:
        print("Practical value "+str(abs(round(evals[i,],4))),"scientific notation "+str(evals[i,]))
        val.append(i)


# ### Question 3.E
# #### K-mean algorithm on the eigenvectors that correspond to your “practically” zero eigenvalues

# In[391]:


eigenvaluesnew=evecs[:,val]


# In[392]:


numof_newclusters=4
kmeans_zeroval = cluster.KMeans(numof_newclusters, random_state=60616).fit(eigenvaluesnew)
FourCircleCSV['Cluster2'] = kmeans_zeroval.labels_
plt.scatter(FourCircleCSV['x'], FourCircleCSV['y'],c=FourCircleCSV['Cluster2'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

