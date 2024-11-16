#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns


# In[ ]:





# In[11]:


dataset=pd.read_csv("Concrete Compressive Strength.csv")


# In[12]:


dataset.shape


# In[13]:


dataset.head()


# In[14]:


###### Checking dublicate values 

for feature in dataset.columns:
    print(f"{feature} --- {dataset.duplicated(feature).sum()}")


# In[15]:


## checking missing values 

dataset.isnull().sum()


# In[16]:


dataset.describe()


# In[17]:


##### checking the correlation between the variable 

plt.figure(figsize=(20,15))
ax=sns.heatmap(dataset.corr(),cmap="RdYlGn",annot=True,linewidth=2)



# In[18]:


## separating the discrete variables 

discrete_feature=[feature for feature in dataset.columns if len(dataset[feature].unique())<20 ]
print("Discrete Variables Count:{}".format(len(discrete_feature)))


# In[19]:


discrete_feature


# In[20]:


for feature in dataset.columns:
    if feature=="Age (day)":
        pass
    else:
        data = dataset.copy() 
        bar = sns.histplot(data[feature], kde=True)  # KDE=True to show smooth density curve
        skewness = data[feature].skew()
        bar.legend(["skewness: {:.2f}".format(skewness)])
        plt.xlabel(feature)
        plt.title(f"Distribution of {feature}")
        plt.show()


# In[21]:


## checking outliers 
for feature in dataset.columns:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# In[22]:


## for statistical analysis 

import scipy.stats as stat
import pylab


# In[23]:


## leanear dataset can be the good approach so using log to dataset feature ## normal distribution QQ plot

for feature in dataset.columns:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    elif feature == "Age (day)":
        pass
    else:
        data[feature]=np.log(data[feature])
        plt.subplot(1,2,1)
        data[feature].hist()
        plt.subplot(1,2,2)
        stat.probplot(data[feature],dist='norm',plot=pylab)
        plt.xlabel(feature)
        plt.ylabel('concrete compressive strength(Mpa,megapascals)')
        plt.title(feature)
        plt.show()
        


# In[35]:


## is not in normal distribution 
for feature in dataset.columns:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    
    elif feature==("Age (day)"):
        pass
    else :
        data [feature]=np.log(data [feature])
        plt. subplot (1,2,1)
        data[ feature].hist()
        plt. subplot (1,2,2)
        stat. probplot(data[feature],dist='norm' ,plot=pylab)
        plt. xlabel(feature)
        plt.ylabel( 'Concrete compressive strength(MPa, megapascals) ')
        plt. title (feature)
        plt. show()
        


# In[36]:


dataset


# In[44]:


for feature in dataset.columns:
    if feature == "Water (component 4)(kg in a m^3 mixture)":
        pass
    if feature== "Age (day)":
        pass
    elif feature == "Coarse Aggregate (component 6)(kg in a m^3 mixture) " :
        pass
    elif feature == "Fine Aggregate (component 7) (kg in a m^3 mixture)":
        pass
    else:
        dataset [feature]=np.log1p(dataset[feature])


# In[47]:


Q1 = dataset. quantile (0.25)
Q3 = dataset. quantile (0.75)
IQR = Q3 - Q1
print(IQR)


# In[49]:


dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) | (dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
dataset.shape


# In[50]:


sns.pairplot(data , height=10 , size = 5 , markers="o")


# In[54]:


for feature in dataset.columns:
    plt.scatter(dataset[feature], dataset.iloc[:, -1])  
    plt.xlabel(feature)
    plt.ylabel('Concrete compressive strength (MPa, megapascals)')
    plt.show()


# In[55]:


x = dataset.iloc[:, : -1]
y = dataset.iloc[ : ,-1]


# In[67]:


##feature scaling 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)


# In[68]:


## Splitting the test and train set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# In[74]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score  
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"Value of R^2 is {score}")

# Calculate Adjusted R^2
n = len(y_train) 
k = X_train.shape[1]  
Adjusted_score = 1 - ((1 - score) * (n - 1) / (n - k - 1))
print(f"Value of adjusted R^2 is {Adjusted_score}")


# In[71]:


print('RMSE for Linear Regression < =',np.sqrt(mean_squared_error(y_test,y_pred)))


# In[72]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[75]:


#  KNN Regressor
knn = KNeighborsRegressor(n_neighbors=5) 


knn.fit(X_train, y_train)


y_pred_knn = knn.predict(X_test)


mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f"KNN Regression Mean Squared Error (MSE): {mse_knn}")
print(f"KNN Regression R^2 Score: {r2_knn}")


