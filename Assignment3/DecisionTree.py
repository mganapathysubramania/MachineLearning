#!/usr/bin/env python
# coding: utf-8

# In[29]:


from sklearn.model_selection import train_test_split

import math
from math import log
import pandas as pd
import numpy as np
import itertools
import sklearn.metrics as metrics
import sys

import matplotlib.pyplot as plt


# In[30]:


Claimhistory_csv = pd.read_csv("C:/Sem 2/ML/Assignment3/claim_history.csv")
Claimhistorysplit_train, Claimhistorysplit_test = train_test_split(Claimhistory_csv, test_size = 0.25, random_state = 60616, stratify = Claimhistory_csv['CAR_USE'])#split using sklearns train_test_split


# In[31]:


print("1. a")
print("--"*30)
print("counts and proportions of the target variable in the Training partition")
print(Claimhistorysplit_train.groupby('CAR_USE').size())
print()
print(Claimhistorysplit_train.groupby('CAR_USE').size() / Claimhistorysplit_train.shape[0])
print("--"*30)


# In[32]:


print("1. b")
print("--"*30)
print("counts and proportions of the target variable in the Test partition")
print(Claimhistorysplit_test.groupby('CAR_USE').size())
print()
print(Claimhistorysplit_test.groupby('CAR_USE').size() / Claimhistorysplit_test.shape[0])
print("--"*30)


# In[33]:


#Train
p_Commercial_train = (Claimhistorysplit_train.groupby('CAR_USE').size() / Claimhistorysplit_train.shape[0])[0]
p_private_train = (Claimhistorysplit_train.groupby('CAR_USE').size() / Claimhistorysplit_train.shape[0])[1]
#Test
p_Commercial_test = (Claimhistorysplit_test.groupby('CAR_USE').size() / Claimhistorysplit_test.shape[0])[0]
p_private_test = (Claimhistorysplit_test.groupby('CAR_USE').size() / Claimhistorysplit_test.shape[0])[1]


# In[34]:


print("1. c")
print("--"*30)
p_train= (p_Commercial_train * 0.75) / (p_Commercial_train*0.75 + p_Commercial_test*0.25)
print("probability that an observation is in the Training partition given that CAR_USE = Commercial ") 
print(p_train)
print("--"*30)


# In[35]:


print("1. d")
print("--"*30)
p_test = (p_private_test * 0.25) / (p_private_test*0.25 + p_private_train*0.75)
print("probability that an observation is in the Test partition given that CAR_USE = Private ")
print(p_test)
print("--"*30)


# In[36]:


#Question 2
print("2. a")
print("--"*30)
trainClaimhistory = Claimhistorysplit_train[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()
Y = Claimhistorysplit_train['CAR_USE']

rootnodeentropy = -1 * ((p_Commercial_train)*log(p_Commercial_train,2) + (p_private_train)*log(p_private_train,2))
print("entropy value of the root node " + str(rootnodeentropy))
print("--"*30)


# In[37]:


def calculateEntropy(Commercial, Private):
    total = Commercial + Private
    pCommercial = Commercial / total
    pPrivate = Private / total
    
    logpCommercial = log(pCommercial,2) if pCommercial != 0 else 0
    logpPrivate = log(pPrivate,2) if pPrivate != 0 else 0
    
    entropy = -1 * (pCommercial * logpCommercial + pPrivate * logpPrivate)
    
    return entropy


# In[38]:


def getEntropyPerPredictor(data, predictors, Type):
    #getting possible setts
    setpredictors=set(predictors)
    if Type == "Nominal":#finding sets if the type is nominal
        leng = len(setpredictors)
        if leng==2 or leng==3:
            possibleSets = set(itertools.combinations(setpredictors, 1))
        updateSet=set()
        if leng % 2 == 0:#even values
            numofvals=int(leng/2)
            for i in range(1,numofvals):
                updateSet.update(set(itertools.combinations(setpredictors, i)))
            nth_subset = set(itertools.combinations(setpredictors, numofvals))
            nth_subset = set(itertools.islice(nth_subset, int(len(nth_subset)/2)))
            updateSet.update(nth_subset)
        else:
            numofvals=math.floor(leng/2)#odd Values
            for i in range(1,numofvals+1):
                updateSet.update(set(itertools.combinations(setpredictors, i)))
        possibleSets = updateSet
    elif Type == "Ordinal":#finding sets if the type is ordinal
        uplist = []
        leng=len(setpredictors)
        for i in range(1,leng):
            uplist.append(set(itertools.islice(setpredictors, i)))
        updateSet = set(frozenset(i) for i in uplist)
        possibleSets = [list(x) for x in updateSet]
        
    commercialTotal = len(data[data['CAR_USE'] == 'Commercial'])
    privateTotal = len(data[data['CAR_USE'] == 'Private'])
    Total = commercialTotal + privateTotal
    entropyList=[]
    for possibleSet in possibleSets:
        filtData = data[predictors.isin(possibleSet)]
        totalinset = len(filtData)
        commercialinset = len(filtData[filtData['CAR_USE'] == 'Commercial'])
        privateinset = len(filtData[filtData['CAR_USE'] == 'Private'])
        entropy = calculateEntropy(commercialinset, privateinset)
        total2 = Total - totalinset
        commercial2 = commercialTotal - commercialinset
        private2 = privateTotal - privateinset
        entropy2 = calculateEntropy(commercial2, private2)
        splitEntropy = (totalinset/Total)*entropy + (total2/Total)*entropy2
        entropyList.append([possibleSet, splitEntropy])
    splitEntropys = np.array(entropyList)[:,1]#getting the entropy of the possible sets
    minSplitEntropy = min(splitEntropys)#getting the minimum entropy in the possible sets
    return entropyList[np.where(splitEntropys == minSplitEntropy)[0][0]]#returning the min split entropy value list



# In[39]:


def getSplitCondition(node):
    cartype_entropy_predictor = getEntropyPerPredictor(node, node['CAR_TYPE'], "Nominal")
    #print(cartype_entropy_predictor)
    occupation_entropy_predictor = getEntropyPerPredictor(node, node['OCCUPATION'], "Nominal")
    #print(occupation_entropy_predictor)
    education_entropy__predictor = getEntropyPerPredictor(node, node['EDUCATION'], "Ordinal")
    #print(edu_entropy)
    valuesof3types = [cartype_entropy_predictor, occupation_entropy_predictor, education_entropy__predictor]
    entropyof3types = np.array(valuesof3types)[:,1]
    splitCondition = valuesof3types[np.where(entropyof3types == min(entropyof3types))[0][0]]#splitting based on entropy as split critiron is entropy metric
    return splitCondition


# In[40]:


def countTargetVals(node):
    
    com = len(node[node['CAR_USE'] == 'Commercial'])
    pri = len(node[node['CAR_USE'] == 'Private']) 
    #print(node)
    print("Commercial: " + str(com))
    print("Private: " + str(pri))
    ent1=calculateEntropy(com,pri)
    print("Entropy: "+ str(ent1))
    return (com/(com+pri))


# In[41]:


def printentropy(node):
    
    com = len(node[node['CAR_USE'] == 'Commercial'])
    pri = len(node[node['CAR_USE'] == 'Private']) 
    ent1=calculateEntropy(com,pri)
    print("Entropy: "+ str(ent1))
    return 0


# In[42]:


print("2. b")
print("--"*30)
splitcriterion = getSplitCondition(trainClaimhistory)
print("Split criterion is " + str(splitcriterion[0]))
#for these questions as the The split criterion is the Entropy metric I have found the minimum Split Entropy 
#and replaced the the column name

Truenode = trainClaimhistory[ (trainClaimhistory['OCCUPATION'] == 'Blue Collar') | (trainClaimhistory['OCCUPATION'] == 'Student') | (trainClaimhistory['OCCUPATION'] == 'Unknown') ]

cartypeTrue = set(Truenode['CAR_TYPE'])
occupationTrue = set(Truenode['OCCUPATION'])
educationTrue = set(Truenode['EDUCATION'])

print("Split Criterion True:")
print("Car Type: " + str(cartypeTrue))
print("Occupation: " + str(occupationTrue))
print("Education: " + str(educationTrue))

Falsenode = trainClaimhistory[~trainClaimhistory.isin(Truenode)].dropna()
#Displaying content in node where condition is false
cartypeFalse = set(Falsenode['CAR_TYPE'])
occupationFalse = set(Falsenode['OCCUPATION'])
educationFalse = set(Falsenode['EDUCATION'])

print("Split Criterion True:")
print("Car Type: " + str(cartypeFalse))
print("Occupation: " + str(occupationFalse))
print("Education: " + str(educationFalse))

print("Layer 1 split criterion Column name Occupaion")
print("Leaf 1")
print(occupationTrue)
print("Count of Values in Split: ",len(Truenode))
problay1=countTargetVals(Truenode)


print("Leaf 2")
print(occupationFalse)
print("Count of Values in Split: ",len(Falsenode))
problay1=countTargetVals(Falsenode)

print("--"*30)


# In[43]:


print("2. c")
print("--"*30)
print("Entropy of first layer " + str(splitcriterion[1]))
print("--"*30)


# In[44]:


print("2. d")
print("--"*30)
maxdepth=2 #The maximum number of branches is two.  The maximum depth is two.
numberofleaves=2**maxdepth
print("Number of leaves is: ",numberofleaves)
print("--"*30)


# In[45]:


print("2. e")
print("--"*30)
print("Layer 2")
print("--"*30)
splitCriterionTrueNode =  getSplitCondition(Truenode)
print("Layer 1 Leaf 1")
#for these questions as the The split criterion is the Entropy metric I have found the minimum Split Entropy 
#and replaced the the column name
#print(splitCriterionTrueNode)
print("Entropy of Root Node: ", end =" " )
printentropy(Truenode)
TrueTruesplitCriterionTrueNode = Truenode[ (Truenode['EDUCATION'] == 'Below High School') ]
TrueFalsesplitCriterionTrueNode = Truenode[~Truenode.isin(TrueTruesplitCriterionTrueNode)].dropna()

print("split criterion Column name: Education")
print("--"*30)
print("Leaf 1")
print("Layer 1 is True and layer 2 is True \n")
print(TrueTruesplitCriterionTrueNode['EDUCATION'].unique())
print("Count of Values in Split: ",len(TrueFalsesplitCriterionTrueNode['EDUCATION']))
probTrueTrue = countTargetVals(TrueTruesplitCriterionTrueNode)
print("Prob: " + str(probTrueTrue) + "\n")
print("--"*30)
print("Leaf 2")
print("Layer 1 is True and Layer 2 is False \n")
print(TrueFalsesplitCriterionTrueNode['EDUCATION'].unique())
print("Count of Values in Split: ",len(TrueFalsesplitCriterionTrueNode['EDUCATION']))
probTrueFalse = countTargetVals(TrueFalsesplitCriterionTrueNode)
print("Prob: " + str(probTrueFalse) + "\n")
print("--"*30)
print()
print("--"*30)

splitCriterionFalseNode = getSplitCondition(Falsenode)
#print(splitCriterionFalseNode)
print("Layer 1 Leaf 2")
print("Entropy of Root Node: ", end =" " )
printentropy(Falsenode)

FalseTruesplitCriterionFalseNode = Falsenode[ (Falsenode['CAR_TYPE'] == 'Minivan') | (Falsenode['CAR_TYPE'] == 'SUV') | (Falsenode['CAR_TYPE'] == 'Sports Car')]
FalseFalsesplitCriterionFalseNode = Falsenode[~Falsenode.isin(FalseTruesplitCriterionFalseNode)].dropna()

print("split criterion Column name: Car Type")
print("--"*30)

print("Leaf 1")
print("Layer 1 is False and Layer 2 is True")
print(FalseTruesplitCriterionFalseNode['CAR_TYPE'].unique())
print("Count of Values in Split: ",len(FalseTruesplitCriterionFalseNode['CAR_TYPE']))
probFalseTrue = countTargetVals(FalseTruesplitCriterionFalseNode)
print("Prob: " + str(probFalseTrue) + "\n")
print("--"*30)
print("Leaf 2")
print("Layer 1 is False and Layer 2 is False")
print(FalseFalsesplitCriterionFalseNode['CAR_TYPE'].unique())
print("Count of Values in Split: ",len(FalseFalsesplitCriterionFalseNode['CAR_TYPE']))
probFalseFalse = countTargetVals(FalseFalsesplitCriterionFalseNode)
print("Prob: " + str(probFalseFalse) + "\n")
print("--"*30)


# In[46]:


#adding predicted prob based on the event as commercail
Claimhistorysplit_train['predictedprob'] = 0
TrueFalsesplitCriterionTrueNode['predictedprob']   = (Claimhistorysplit_train['CAR_USE'][TrueFalsesplitCriterionTrueNode.index.values].value_counts()/TrueFalsesplitCriterionTrueNode.shape[0])['Commercial']
TrueTruesplitCriterionTrueNode['predictedprob']   = (Claimhistorysplit_train['CAR_USE'][TrueTruesplitCriterionTrueNode.index.values].value_counts()/TrueTruesplitCriterionTrueNode.shape[0])['Commercial']
FalseFalsesplitCriterionFalseNode['predictedprob']   = (Claimhistorysplit_train['CAR_USE'][FalseFalsesplitCriterionFalseNode.index.values].value_counts()/FalseFalsesplitCriterionFalseNode.shape[0])['Commercial']
FalseTruesplitCriterionFalseNode['predictedprob']   = (Claimhistorysplit_train['CAR_USE'][FalseTruesplitCriterionFalseNode.index.values].value_counts()/FalseTruesplitCriterionFalseNode.shape[0])['Commercial']

Claimhistorysplit_train=pd.concat([TrueTruesplitCriterionTrueNode,TrueFalsesplitCriterionTrueNode,FalseTruesplitCriterionFalseNode,FalseFalsesplitCriterionFalseNode])


# In[47]:


def Kolmogorov_Smirnov(Claimhistorysplit_train):
    
    stats = pd.DataFrame(columns = ["Cut_off","Kolmogorov_Smirnov"])
    for threshold in np.arange(0.05,0.60,0.01):
        Claimhistorysplit_train.loc[Claimhistorysplit_train['predictedprob'] >= threshold, 'Predicted'] = "Commercial"
        Claimhistorysplit_train.loc[Claimhistorysplit_train['predictedprob'] < threshold, 'Predicted'] = "Private"
        TruePositive = Claimhistorysplit_train['CAR_USE'][Claimhistorysplit_train['Predicted'] == Claimhistorysplit_train['CAR_USE']].value_counts()['Commercial']
        TrueNegative = Claimhistorysplit_train['CAR_USE'][Claimhistorysplit_train['Predicted'] == Claimhistorysplit_train['CAR_USE']].value_counts()['Private']
        FalsePositive = Claimhistorysplit_train['CAR_USE'][Claimhistorysplit_train['Predicted'] != Claimhistorysplit_train['CAR_USE']].value_counts()['Commercial']
        FalseNegative = Claimhistorysplit_train['CAR_USE'][Claimhistorysplit_train['Predicted'] != Claimhistorysplit_train['CAR_USE']].value_counts()['Private']
        TruePositiveRate = TruePositive/(TruePositive+FalseNegative)
        TrueNegativeRate = TrueNegative/(TrueNegative+FalsePositive)
        FalsePositiveRate = 1-TrueNegativeRate
        ks=TruePositiveRate-FalsePositiveRate
        stats = stats.append({'Cut_off':threshold,'Kolmogorov_Smirnov':ks},ignore_index = True)
        
    return stats


# In[48]:


stats = Kolmogorov_Smirnov(Claimhistorysplit_train)
print("2. f")
print("--"*30)
Kolmogorov_Smirnov = stats[stats['Kolmogorov_Smirnov'] == max(stats['Kolmogorov_Smirnov'])].iloc[0,:][1]#getting max Kolmogorov-Smirnov statistic
eventprobcutoff =  stats[stats['Kolmogorov_Smirnov'] == max(stats['Kolmogorov_Smirnov'])].iloc[0,:][0]#getting max cutoff

print("Kolmogorov-Smirnov statistic value :"+ str(Kolmogorov_Smirnov))
print("Event probability cutoff value : " + str(eventprobcutoff))
print("--"*30)


# In[49]:


#applying the same above for test data
print("3. a")
print("--"*30)
testClaimhistory = Claimhistorysplit_test[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()
Truenode2 = testClaimhistory[ (testClaimhistory['OCCUPATION'] == 'Blue Collar') | (testClaimhistory['OCCUPATION'] == 'Student') | (testClaimhistory['OCCUPATION'] == 'Unknown') ]
Falsenode2 = testClaimhistory[~testClaimhistory.isin(Truenode2)].dropna()
TrueTruenodetest = Truenode2[ (Truenode2['EDUCATION'] == 'Below High School') ]
TrueFalsenodetest = Truenode2[~Truenode2.isin(TrueTruenodetest)].dropna() 
FalseTruenodetest = Falsenode2[ (Falsenode2['CAR_TYPE'] == 'Minivan') | (Falsenode2['CAR_TYPE'] == 'SUV') | (Falsenode2['CAR_TYPE'] == 'Sports Car') ]
FalseFalsenodetest = Falsenode2[~Falsenode2.isin(FalseTruenodetest)].dropna()

threshold = float((Claimhistorysplit_train.groupby('CAR_USE').size() / Claimhistorysplit_train.shape[0])['Commercial'])
print("Threshold : ",threshold)

testClaimhistory['predictedprob'] = 0
TrueFalsenodetest['predictedprob']   = (testClaimhistory['CAR_USE'][TrueFalsenodetest.index.values].value_counts()/TrueFalsenodetest.shape[0])['Commercial']
TrueTruenodetest['predictedprob']   = (testClaimhistory['CAR_USE'][TrueTruenodetest.index.values].value_counts()/TrueTruenodetest.shape[0])['Commercial']
FalseFalsenodetest['predictedprob']   = (testClaimhistory['CAR_USE'][FalseFalsenodetest.index.values].value_counts()/FalseFalsenodetest.shape[0])['Commercial']
FalseTruenodetest['predictedprob']   = (testClaimhistory['CAR_USE'][FalseTruenodetest.index.values].value_counts()/FalseTruenodetest.shape[0])['Commercial']

testClaimhistory=pd.concat([TrueTruenodetest,TrueFalsenodetest,FalseTruenodetest,FalseFalsenodetest])


testClaimhistory.loc[testClaimhistory['predictedprob'] >= threshold, 'Predicted'] = "Commercial"
testClaimhistory.loc[testClaimhistory['predictedprob'] < threshold, 'Predicted'] = "Private"

misClassRate = len(testClaimhistory[ testClaimhistory['CAR_USE'] != testClaimhistory['Predicted'] ]) / len(testClaimhistory)

print("Misclassification rate is: " + str(misClassRate*100)+" %")
print("--"*30)


# In[50]:


print("3. b")
print("--"*30)
print("Threshold based on Kolmogorov-Smirnov",eventprobcutoff)

testClaimhistory.loc[testClaimhistory['predictedprob'] >= eventprobcutoff, 'Predicted_KS'] = "Commercial"
testClaimhistory.loc[testClaimhistory['predictedprob'] < eventprobcutoff, 'Predicted_KS'] = "Private"

misClassRate2 = len(testClaimhistory[ testClaimhistory['CAR_USE'] != testClaimhistory['Predicted_KS'] ]) / len(testClaimhistory)
print("Misclassification Rate in the Test partition:"+str(misClassRate2*100)+" %")
print("--"*30)


# In[51]:


print("3. c")
print("--"*30)

numofcom = testClaimhistory[testClaimhistory['CAR_USE'] == "Commercial"][['CAR_USE','predictedprob']]
numofpri = testClaimhistory[testClaimhistory['CAR_USE'] == "Private"][['CAR_USE','predictedprob']]

AvgSquErr = ( np.sum(np.square(1-np.array(numofcom['predictedprob']))) 
+ np.sum(np.square(0-np.array(numofpri['predictedprob']))) ) / (len(numofcom) + len(numofpri))

RootAvgSquErr = math.sqrt(AvgSquErr)
print("Root average squared error: " + str(RootAvgSquErr))
print("--"*30)


# In[52]:


print("3. d")
print("--"*30)
y_true = (0.5 + 0.5) * np.isin(testClaimhistory['CAR_USE'], ['Commercial'])
pred_prob_y = np.array(testClaimhistory['predictedprob'].tolist())
AUC = metrics.roc_auc_score(y_true, pred_prob_y)
print("Area Under Curve " + str(AUC))
print("--"*30)


# In[53]:


print("3. e")
print("--"*30)
Gini=2*AUC-1
print("Gini Coefficient"+ str(Gini))
print("--"*30)


# In[56]:


print("3. f")
print("--"*30)
y_true = np.array(testClaimhistory['CAR_USE'])
predicted_probability = np.array(testClaimhistory['predictedprob'])
eventnoneve=pd.DataFrame({'x':y_true, 'y':predicted_probability})
    
non_event = []
event = []
for i in range(len(eventnoneve)):
    if eventnoneve['x'][i] == 'Commercial':
        event.append(eventnoneve['y'][i])
       
    else:
        non_event.append(eventnoneve['y'][i])

concordant = 0
discordant = 0
for i in range(len(event)):
    for j in range (len(non_event)):
        if event[i] > non_event[j]:
            concordant += 1
        elif event[i] < non_event[j]:
            discordant += 1
            
#Goodman-Kruskal Gamma= (# Concordant - # Discordant) / (#Concordant + #Discordant)
Gamma=(concordant - discordant)/(concordant + discordant)
print("Goodman-Kruskal Gamma statistic : ", Gamma)

print("--"*30)


# In[55]:


print("3. g")
print("--"*30)

Y = np.array(testClaimhistory['CAR_USE'].tolist())
# Generate the coordinates for the ROC curve

fpr, tpr, thresholds = metrics.roc_curve(Y, pred_prob_y, pos_label = 'Commercial')
# Add two dummy coordinates
OneMinusSpecificity = np.append([0], fpr)

Sensitivity = np.append([0], tpr)

OneMinusSpecificity = np.append(OneMinusSpecificity, [1])
Sensitivity = np.append(Sensitivity, [1])
# Draw the ROC curve

plt.figure(figsize=(6,6))
plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)#diagonal reference line
plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title("Receiver Operating Characteristic curve")
ax = plt.gca()
ax.set_aspect('equal')
plt.show()
print("--"*30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




