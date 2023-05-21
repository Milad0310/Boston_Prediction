#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[49]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# # import dataset

# In[50]:


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]


# In[51]:


print(X)


# In[52]:


print(Y)


# #  Evaluating for Missing data

# In[53]:


dataset.isnull().sum()


# # spliting dataset into training & test set

# In[54]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .25, random_state = 0)


# # Multiple linear regression

# In[65]:


from sklearn.linear_model import LinearRegression
mul_reg = LinearRegression()
mul_reg.fit(X_train, Y_train)


# # Predicting the Test set results

# In[82]:


y_pred = mul_reg.predict(X_test)
print(y_pred)
#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))


# # polynomial regression

# In[75]:


from sklearn.preprocessing import PolynomialFeatures
pol_regr = PolynomialFeatures(degree = 4)
X_poly = pol_regr.fit_transform(X)
lin2_regr = LinearRegression()
lin2_regr.fit(X_poly, Y)


# # Predicting the Test set results

# In[83]:


lin2_regr.predict(X_test)


# # SVR

# # Feature Scaling

# In[86]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_sc = sc_X.fit_transform(X)
Y_sc = sc_Y.fit_transform(Y)


# In[87]:


from sklearn.svm import SVR
svr_regr = SVR()
svr_regr.fit(X_train, Y_train)


# In[88]:


svr_regr.predict(X_test)


# # Decision tree rehression

# In[35]:


from sklearn.tree import DecisionTreeRegressor
tree_regr = DecisionTreeRegressor()
tree_regr.fit(X_train, Y_train)
tree_regr.predict(X_test)


# # Randome forest regresion

# In[42]:


from sklearn.ensemble import RandomForestRegressor
rand_for_regr = RandomForestRegressor(n_estimators = 100)
rand_for_regr.fit(X_train, Y_train)


# In[43]:


rand_for_regr.predict(X_test)


# In[ ]:




