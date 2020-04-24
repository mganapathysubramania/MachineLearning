#!/usr/bin/env python
# coding: utf-8
#ManishGanapathySubramanianBharatwaj
# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from numpy import linalg as Li
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier as kNC



# In[2]:


#Reading the Normal Sample CSV
inputcsv_normalsample=pd.read_csv("NormalSample.csv")


# In[3]:


#Question1a
xaxisvalues = inputcsv_normalsample['x'].values.copy()
xLen= len(xaxisvalues)
#q1
q1 = np.percentile(xaxisvalues,25)
#q3
q3 = np.percentile(xaxisvalues,75)
print("first quartile ",q1,"third quartile ",q3) 
#Interquartile
Interquartile_Range = q3-q1 #Interquartile
print("InterQuarantile Range = ",Interquartile_Range)
bins=2*Interquartile_Range*(xLen**(-1/3))
print("Recommended bin-width for the histogram of x",round(bins,2))#3 decimal places


# In[4]:


#Question1b
print("Maxmimum Value of x",max(xaxisvalues))
print("Minimum Value of x",min(xaxisvalues))


# In[5]:


#Question1c
max_value_x=math.ceil(max(xaxisvalues))
min_value_x=math.floor(min(xaxisvalues))
print("Smallest Integer less than the Maxmimum Value of x",max_value_x)
print("Largest Integer less then the Minimum Value of x",min_value_x)


# In[6]:


def weight(u):
    if u<=0.5 and u>-0.5:
        return 1
    else:
        return 0


def DensityEstimation(min_value_x, max_value_x, xaxisvalues, h1):
    Mid_Point = []
    i = round(min_value_x + (h1/2),2)
    Mid_Point.append(i)

    while i+h1 < max_value_x:
        i = round(i + h1,2)
        Mid_Point.append(i)

    Density_Estimation = []
    for j in Mid_Point:
        wu=0
        for x in xaxisvalues:
            u = (x - j)/h1
            wu = wu + weight(u)
        de=wu/(len(xaxisvalues)*h1)
        Density_Estimation.append([j, round(de,2)])
        
    df = pd.DataFrame(Density_Estimation, columns =['MidPoint', 'Density Estimate']) 
    df=df.set_index('MidPoint')
    print(df)

    bins_graph = [round(x - (h1/2),2) for x in Mid_Point]
    bins_graph.append(max_value_x)

    plt.hist(xaxisvalues, bins_graph, normed=True)
    plt.title("Normalized Histogram Plot (h = " + str(h1) + ")")
    plt.show()


# In[7]:


#Question1d
DensityEstimation(min_value_x,max_value_x,xaxisvalues,0.25)


# In[8]:


#Question1e
DensityEstimation(min_value_x,max_value_x,xaxisvalues,0.5)


# In[9]:


#Question1f
DensityEstimation(min_value_x,max_value_x,xaxisvalues,1)


# In[10]:


#Question1g
DensityEstimation(min_value_x,max_value_x,xaxisvalues,2)


# In[11]:


#Question1h
"""
As per my observation, whenever the bin-width is lower the gragh contains more information, as you can see that the h=.25
has more information than that of h=2 or h=1, but that graph it is not easy to read and inspect. 
But the graph with h=0.5 provides much more readability and gives best insights into the shape. 

"""


# In[12]:


#Question2a
print(inputcsv_normalsample.x.describe())
min_whis = q1-(1.5*Interquartile_Range)

max_whis = q3+(1.5*Interquartile_Range)

print("The minimum whisker is " + str(min_whis) + " and the maximum whisker is "+ str(max_whis))


# In[13]:


#Question2b
for i in inputcsv_normalsample.group.unique():

    grp = inputcsv_normalsample['x'][inputcsv_normalsample['group']==i]

    q1 = np.percentile(grp,25)

    q3 = np.percentile(grp,75)

    Interquartile_Range = q3-q1

    print(grp.describe())

    min_whishker = q1-(1.5*Interquartile_Range)

    max_whishker = q3+(1.5*Interquartile_Range)

    print("The minimum whisker for {} is {} and the maximum whisker for {} is {}".format(i, min_whishker, i, max_whishker))


# In[14]:


#Question2c
boxplotvalues=plt.boxplot(xaxisvalues)
plt.title("Box Plot of x")
plt.show()


# In[15]:


Whishker = np.matrix([item.get_ydata() for item in boxplotvalues['whiskers']])[:,1]
print("Whiskers on Python's box plot " + str(Whishker))


# In[16]:


#No, Python’s boxplot does not display the 1.5 IQR whiskers correctly


# In[17]:


#Question2D

group0 = inputcsv_normalsample[inputcsv_normalsample['group']==0]['x'].values.copy()
group1 = inputcsv_normalsample[inputcsv_normalsample['group']==1]['x'].values.copy()

data=[xaxisvalues, group0, group1]
plt.boxplot(data)
plt.xticks([1, 2, 3], ["X", "X (group 0)", "X (group 1)"])
plt.title("Box plot of X, X(group 0), X(group 1)")
plt.show()


# In[18]:


#Question3a
Fraudcsv = pd.read_csv("Fraud.csv")
Frauds = Fraudcsv['FRAUD']

Fraud1 = Frauds.sum()

#Q3 Part a
FraudPercent = Fraud1*100/len(Frauds)
print("Percent of fraudulent investigations is " + str(round(FraudPercent,4)) + "%")


# In[19]:

#ManishGanapathySubramanianBharatwaj
#question3b
sns.boxplot(x=Fraudcsv.FRAUD,y=Fraudcsv.TOTAL_SPEND)
sns.boxplot(x="FRAUD", y="TOTAL_SPEND", data=Fraudcsv,order=["Non Fraudulent", "Fraudulent"])
plt.show()

# In[20]:


sns.boxplot(x=Fraudcsv.FRAUD,y=Fraudcsv.DOCTOR_VISITS)
sns.boxplot(x="FRAUD", y="DOCTOR_VISITS", data=Fraudcsv,order=["Non Fraudulent", "Fraudulent"])
plt.show()

# In[21]:#ManishGanapathySubramanianBharatwaj


sns.boxplot(x=Fraudcsv.FRAUD,y=Fraudcsv.NUM_CLAIMS)
sns.boxplot(x="FRAUD", y="NUM_CLAIMS", data=Fraudcsv,order=["Non Fraudulent", "Fraudulent"])
plt.show()

# In[22]:


sns.boxplot(x=Fraudcsv.FRAUD,y=Fraudcsv.MEMBER_DURATION)
sns.boxplot(x="FRAUD", y="MEMBER_DURATION", data=Fraudcsv,order=["Non Fraudulent", "Fraudulent"])

plt.show()
# In[23]:


sns.boxplot(x=Fraudcsv.FRAUD,y=Fraudcsv.OPTOM_PRESC)
sns.boxplot(x="FRAUD", y="OPTOM_PRESC", data=Fraudcsv,order=["Non Fraudulent", "Fraudulent"])
plt.show()

# In[24]:


sns.boxplot(x=Fraudcsv.FRAUD,y=Fraudcsv.NUM_MEMBERS)
sns.boxplot(x="FRAUD", y="NUM_MEMBERS", data=Fraudcsv,order=["Non Fraudulent", "Fraudulent"])
plt.show()

# In[25]:


#question3c
totalspend = Fraudcsv['TOTAL_SPEND']
doctorvistits = Fraudcsv['DOCTOR_VISITS']
numclaims = Fraudcsv['NUM_CLAIMS']
memberduriation = Fraudcsv['MEMBER_DURATION']
optompresc = Fraudcsv['OPTOM_PRESC']
nummembers = Fraudcsv['NUM_MEMBERS']

matrix = np.matrix([totalspend, doctorvistits, numclaims, memberduriation, optompresc, nummembers]).transpose()


matrixtranspose = matrix.transpose() * matrix
eigenvaluesvals, eigenvectors = Li.eigh(matrixtranspose)
filteer = []
for i in range(0,len(eigenvaluesvals)):
    if eigenvaluesvals[i] >= 1:
        filteer.append(i)

eigenvectorsfilter = eigenvectors[filteer].transpose()
eigenvaluesvalsfilter = eigenvaluesvals[filteer] 

print("Number of Dimesions " + str(len(eigenvaluesvalsfilter)))

transformationmatrix = eigenvectors * Li.inv(np.sqrt(np.diagflat(eigenvaluesvals)))
print("Transformation Matrix : ")
print(transformationmatrix)

transformationmatrix_x = matrix * transformationmatrix
print("The Transformed :")
print(transformationmatrix_x)


toproveorth = transformationmatrix_x.transpose() * transformationmatrix_x
print("Tranformationmatrix*transposeoftranformationmatrix")
print(toproveorth)


# In[26]:#ManishGanapathySubramanianBharatwaj


#Question3d

kNNS1 = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
Neighbors = kNNS1.fit(transformationmatrix_x)
kNclasifier = kNC(n_neighbors = 5)
NeighborsClassifier = kNclasifier.fit(transformationmatrix_x,np.array(Frauds.values))
score = kNclasifier.score(transformationmatrix_x, np.array(Frauds.values), sample_weight=None)
print("The result of score function is " + str(round(score,4)))


# In[27]:
#ManishGanapathySubramanianBharatwaj

#Question3e
#For the observation which has these input variable values: TOTAL_SPEND = 7500, 
#DOCTOR_VISITS = 15, NUM_CLAIMS = 3, MEMBER_DURATION = 127, OPTOM_PRESC = 2, and NUM_MEMBERS = 2.
#find its five neighbors.  Please list their input variable values and the target values. 
focal = [7500, 15, 3, 127, 2, 2]#hardcoded as prof given
print("focal observation =" + str(focal))
transformationmatrixFocalobervation = focal*transformationmatrix #using results form part c
print("Transformed focal observation =" + str(transformationmatrixFocalobervation))
myNeighbors_t = Neighbors.kneighbors(transformationmatrixFocalobervation, return_distance = False)
print("indices of the five neighbors are " + str(myNeighbors_t))

Neighbormatrix = matrix[myNeighbors_t]
print("The input and target values of the nearest neighbors are \n")
stack = np.hstack((myNeighbors_t.transpose(),Neighbormatrix[0,:,:]))
transpose_fraudval = np.matrix(Frauds.values).transpose()
targetVals = transpose_fraudval[myNeighbors_t]
stack = np.hstack((stack, targetVals[0,:,:]))
dataframe = pd.DataFrame(stack.tolist(), columns =["ID", "TOTAL_SPEND", "DOCTOR_VISITS", "NUM_CLAIMS", "MEMBER_DURATION", "OPTOM_PRESC", "NUM_MEMBERS", "Target Value"]) 
dataframe=dataframe.set_index('ID')
print(dataframe)


# In[28]:


#Question3f
Predicted_Prob_fraud = 1

if(Predicted_Prob_fraud > (FraudPercent/100)):

    print("Observation is fraudulent.")

else:

    print("Observation is not fraudulent.")


# In[ ]:




