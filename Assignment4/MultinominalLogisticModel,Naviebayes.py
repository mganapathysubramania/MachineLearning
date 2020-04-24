#!/usr/bin/env python
# coding: utf-8
#Manish Ganapathy Subramanian Bharatwaj
# In[3]:
#A20449394


from sklearn import naive_bayes
import numpy as np
import scipy
import pandas as pd
import math
import sympy
import statsmodels.api as stats


# In[41]:


def build_mnlogit(fullX, y, debug = "N"):#referred from professor slides.
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == "Y"):
        print("\nColumn Numbers of the Non-redundant Columns :")
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method="newton", full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == "Y"):
        print(thisFit.summary())
        print("\nModel Parameter Estimates  :\n" + str(thisParameter))
        print("\nModel Log-Likelihood Value : "+ str(thisLLK))
        print( "Number of Free Parameters  : "+ str(thisDF))

    # Recreate the estimates of the full parameters
    workParams = pd.DataFrame(np.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pd.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = "0_x").fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams)


# In[4]:


purchase_likelihood_df = pd.read_csv("C:/Sem 2/ML/Assignment4/Purchase_Likelihood.csv")

insurance_df = purchase_likelihood_df["insurance"].astype("category")
group_size_df = pd.get_dummies(purchase_likelihood_df[["group_size"]].astype("category"))
homeowner_df = pd.get_dummies(purchase_likelihood_df[["homeowner"]].astype("category"))
married_couple_df = pd.get_dummies(purchase_likelihood_df[["married_couple"]].astype("category"))


# ### 1. C

# In[43]:


# Intercept only model

model = pd.DataFrame(insurance_df.where(insurance_df.isnull(), 1))
llk_model, df_model, fullparams_model= build_mnlogit(model, insurance_df, debug = "Y")


# In[44]:


# Intercept + group_size
model_1  = stats.add_constant(group_size_df, prepend = True)

llk_model_1, df_model_1, fullparams_model_1 = build_mnlogit(model_1, insurance_df, debug = "Y")

chi_model_1 = 2 * (llk_model_1 - llk_model)
degree_model_1 = df_model_1 - df_model
significance_model_1 = scipy.stats.chi2.sf(chi_model_1, degree_model_1)
print()
print("Model-1")
print()
print("---"*40)

print("\n Deviance Chi-Square Test \n")
print("---"*40)
print("Chi-Square Statistic " + str(chi_model_1))
print("Degree of freedom    " + str(degree_model_1))
print("Significance         " + str(significance_model_1))
print()
print("---"*40)


# In[45]:


# Intercept + group_size + homeowner

grp_1 = group_size_df
grp_1 = grp_1.join(homeowner_df)
model_2  = stats.add_constant(grp_1, prepend = True)

llk_model_2, df_model_2, fullparams_model_2 = build_mnlogit(model_2, insurance_df, debug = "Y")

chi_model_2 = 2 * (llk_model_2 - llk_model_1)
degree_model_2 = df_model_2 - df_model_1
significance_model_2 = scipy.stats.chi2.sf(chi_model_2, degree_model_2)
print()
print("\nModel-2:" + "\n")
print()
print("---"*40)

print("Deviance Chi-Square Test")
print("---"*40)
print("Chi-Square Statistic " + str(chi_model_2))
print("Degree of freedom    " + str(degree_model_2))
print("Significance         " + str(significance_model_2))
print()
print("---"*40)


# In[46]:


# Intercept + group_size + homeowner + married_couple

grp_2 = group_size_df
grp_2 = grp_2.join(homeowner_df)
grp_2 = grp_2.join(married_couple_df)
model_3  = stats.add_constant(grp_2, prepend = True)

llk_model_3, df_model_3, fullparams_model_3 = build_mnlogit(model_3, insurance_df, debug = "Y")

chi_model_3 = 2 * (llk_model_3 - llk_model_2)
degree_model_3 = df_model_3 - df_model_2
significance_model_3 = scipy.stats.chi2.sf(chi_model_3, degree_model_3)
print()
print("\nModel-3:" + "\n")
print()
print("---"*40)
print(" Deviance Chi-Square Test")
print("---"*40)
print("Chi-Square Statistic " + str(chi_model_3))
print("Degree of freedom    " + str(degree_model_3))
print("Significance         " + str(significance_model_3))
print()
print("---"*40)


# In[47]:


def create_interaction(df_1, df_2):
    columnname1 = df_1.columns
    columnname2 = df_2.columns
    out_df = pd.DataFrame()
    
    for c1 in columnname1:
        for c2 in columnname2:
            out_names = c1 + "*" + c2
            out_df[out_names] = df_1[c1] * df_2[c2]
    
    return (out_df)


# In[48]:


# Intercept + group_size + homeowner + married_couple + (group_size * homeowner)

comb_1 = create_interaction(group_size_df, homeowner_df)

grp_3 = group_size_df
grp_3 = grp_3.join(homeowner_df)
grp_3 = grp_3.join(married_couple_df)
grp_3 = grp_3.join(comb_1)
model_4  = stats.add_constant(grp_3, prepend = True)

llk_model_4, df_model_4, fullparams_model_4 = build_mnlogit(model_4, insurance_df, debug = "Y")

chi_model_4 = 2 * (llk_model_4 - llk_model_3)
degree_model_4 = df_model_4 - df_model_3
significance_model_4 = scipy.stats.chi2.sf(chi_model_4, degree_model_4)
print()
print("Model-4")
print()
print("---"*40)
print("\n Deviance Chi-Square Test")
print("---"*40)
print("Chi-Square Statistic " + str(chi_model_4))
print("Degree of freedom    " + str(degree_model_4))
print("Significance         " + str(significance_model_4))
print()
print("---"*40)


# In[49]:


# Intercept + group_size + homeowner + married_couple + (group_size * homeowner) + (group_size + married_couple)

comb_2 = create_interaction(group_size_df, married_couple_df)

grp_4 = group_size_df
grp_4 = grp_4.join(homeowner_df)
grp_4 = grp_4.join(married_couple_df)
grp_4 = grp_4.join(comb_1)
grp_4 = grp_4.join(comb_2)
model_5  = stats.add_constant(grp_4, prepend = True)

llk_model_5, df_model_5, fullparams_model_5 = build_mnlogit(model_5, insurance_df, debug = "Y")

chi_model_5 = 2 * (llk_model_5 - llk_model_4)
degree_model_5 = df_model_5 - df_model_4
significance_model_5 = scipy.stats.chi2.sf(chi_model_5, degree_model_5)
print()
print("Model-5")
print()
print("---"*40)
print(" Deviance Chi-Square Test")
print("---"*40)
print("Chi-Square Statistic " + str(chi_model_5))
print("Degree of freedom    " + str(degree_model_5))
print("Significance         " + str(significance_model_5))
print()
print("---"*40)


# In[50]:


# Intercept + group_size + homeowner + married_couple + (group_size * homeowner) + (group_size * married_couple) + (homeowner * married_couple)

comb_3 = create_interaction_predictors(homeowner_df, married_couple_df)

grp_5 = group_size_df
grp_5 = grp_5.join(homeowner_df)
grp_5 = grp_5.join(married_couple_df)
grp_5 = grp_5.join(comb_1)
grp_5 = grp_5.join(comb_2)
grp_5 = grp_5.join(comb_3)
model_6  = stats.add_constant(grp_5, prepend = True)

llk_model_6, df_model_6, fullparams_model_6 = build_mnlogit(model_6, insurance_df, debug = "Y")

chi_model_6 = 2 * (llk_model_6 - llk_model_5)
degree_model_6 = df_model_6 - df_model_5
significance_model_6 = scipy.stats.chi2.sf(chi_model_6, degree_model_6)
print()
print("Model-6")
print()
print("---"*40)
print("Deviance Chi-Square Test ")
print("---"*40)
print("Chi-Square Statistic " + str(chi_model_6))
print("Degree of freedom    " + str(degree_model_6))
print("Significance         " + str(significance_model_6))
print()
print("---"*40)


# ### 1.A

# In[52]:


model_matrix = grp_5
model_matrix = stats.add_constant(model_matrix, prepend = True)
reduced_form, inds = sympy.Matrix(model_matrix.values).rref()
print( "Aliased columns that you found in your model matrix" )
for i in range(0, len(model_matrix.columns)):
    if i not in inds:
        print(model_matrix.columns[i])


# ### 1. B

# In[53]:


category = insurance_df.cat.categories
df = len(inds) * (len(category) - 1)

print("The degree of freedom that the my model has " + str(df))
print()


# ### 1. D

# In[71]:


significance_values = [significance_model_1,significance_model_2, significance_model_3, significance_model_4, significance_model_5, significance_model_6]

for i in range(0, len(significance_values)):
    print("The feature importance index with significance value for model " + str((i + 1)) + " is : " + str(-np.log10(significance_values[i])))


# ### 2. A

# In[7]:


targetvariable = purchase_likelihood_df['insurance']
nominalPredictor1 = purchase_likelihood_df['group_size']
nominalPredictor2 = purchase_likelihood_df['homeowner']
nominalPredictor3 = purchase_likelihood_df['married_couple']


crosstabtable = pd.crosstab(index = [nominalPredictor1,nominalPredictor2,nominalPredictor3],columns = targetvariable, margins = True, dropna = True)

tot = crosstabtable.drop('All', 1)

predProbTable = crosstabtable.div(tot.sum(1), axis='index')

print("Predicted Probability table: ")
print("---"*40)
print(predProbTable)


# ### 2. B

# In[57]:


# part b
temp = []
for i in range(1, 5):#looping groupsize
    for j in range(0, 2):#looping homeowner
        for k in range(0, 2):#looping married_couples
            temp.append(round(predProbTable[1][i][j][k] / predProbTable[0][i][j][k], 5))
            
print("P(insurance = 1) / P(insurance = 0) is : ", max(temp))


# In[ ]:





# In[ ]:





# ### 3. A

# In[60]:


print("Frequency count of the target variable" )
print(purchase_likelihood_df.groupby("insurance").size())
print()
print("---"*40)
print("Class probabilities of the target variable" )
print()
print(purchase_likelihood_df.groupby("insurance").size() / len(purchase_likelihood_df))
print("---"*40)


# In[61]:


def RowWithColumn(rowVar, columnVar, show = "ROW"):#referred slides given by professor
    countTable = pd.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
    print("Frequency Table: \n", countTable)
    return


# ### 3. B

# In[80]:


print("---"*40)
RowWithColumn(purchase_likelihood_df['group_size'],purchase_likelihood_df['insurance'])
print("---"*40)


# ### 3. C

# In[81]:


print("---"*40)
RowWithColumn(purchase_likelihood_df['homeowner'],purchase_likelihood_df['insurance'])
print("---"*40)


# ### 3. D

# In[82]:


print("---"*40)
RowWithColumn(purchase_likelihood_df['married_couple'], purchase_likelihood_df['insurance'])
print("---"*40)


# In[83]:


def ChiSquareTest (xCat, yCat, debug = 'N'):#referred slides given by professor
    obsCount = pd.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
    cTotal = obsCount.sum(axis = 1)
    rTotal = obsCount.sum(axis = 0)
    nTotal = np.sum(rTotal)
    expCount = np.outer(cTotal, (rTotal / nTotal))
    
    chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()

    cramerV = chiSqStat / nTotal
    if (cTotal.size > rTotal.size):
        cramerV = cramerV / (rTotal.size - 1.0)
    else:
        cramerV = cramerV / (cTotal.size - 1.0)
    cramerV = np.sqrt(cramerV)

    return cramerV


# ### 3.E

# In[79]:


cramervstat_group_size = ChiSquareTest(purchase_likelihood_df['group_size'], purchase_likelihood_df['insurance'])
cramervstat_homeowner = ChiSquareTest(purchase_likelihood_df['homeowner'], purchase_likelihood_df['insurance'])
cramervstat_married_couple = ChiSquareTest(purchase_likelihood_df['married_couple'], purchase_likelihood_df['insurance'])
print("---"*40)
print("Cramer’s V statistics for group_size is " + str(cramervstat_group_size))
print("Cramer’s V statistics for homeowner is " + str(cramervstat_homeowner))
print("Cramer’s V statistics for married_couple is " + str(cramervstat_married_couple))
print("---"*40)


# In[85]:





# ### 3. F

# In[8]:




column = ['group_size', 'homeowner', 'married_couple']
Trainx = purchase_likelihood_df[column].astype('category')
Trainy = purchase_likelihood_df['insurance'].astype('category')
_objNB = naive_bayes.MultinomialNB(alpha = 1.0e-10)
thisFit = _objNB.fit(Trainx, Trainy)

gstest = pd.DataFrame(['1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4'], columns=['group_size'])
hotest = pd.DataFrame(['0','0','0','0','0','0','0','0','1','1','1','1','1','1','1','1'], columns=['homeowner'])
mctest = pd.DataFrame(['0','0','0','0','1','1','1','1','0','0','0','0','1','1','1','1'], columns=['married_couple'])

test_x = gstest
test_x = test_x.join(hotest)
test_x = test_x.join(mctest)

test_x = test_x[column].astype('category')
y_test_pred_prob = pd.DataFrame(_objNB.predict_proba(test_x), columns = ['0', '1','2'])
y_test_score = pd.concat([test_x, y_test_pred_prob], axis = 1)
print("---"*40)                                                                              
print(y_test_score)
print("---"*40)


# ### 3. G

# In[10]:


print("combination of group_size, homeowner, and married_couple will maximize the odds value Prob(insurance = 1) / Prob(insurance = 0)")
print("---"*40)
y_test_score['odd value(1/0)'] = y_test_score['1'] / y_test_score['0']
print(y_test_score.loc[y_test_score['odd value(1/0)'].idxmax()])
print("---"*40)


# In[ ]:




