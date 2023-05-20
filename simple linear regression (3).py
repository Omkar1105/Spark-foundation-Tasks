#!/usr/bin/env python
# coding: utf-8

# # Grip sparks foundation

# Auther : Omkar dinesh patil
# Task: Predection using supervised Machine Learning

# In[7]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings("ignore")


# In[9]:


# data=pd.read_clipboard()
#data
data=pd.read_csv('F:\Data I.csv')
data


# In[10]:


d_1=data.head(10)
d_1


# In[23]:


# plotting the distribution of score
data.plot(x='hours',y='scores',style='o')
plt.title('hours vs percentage')
plt.xlable1('hours Studied')
plt.ylable1('percentage scores')
plt.show()

# Preparing the Data
# In[24]:


x= data.iloc[:,:-1].values


# In[25]:


y=data.iloc[:,1].values


# In[35]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2, random_state=0)


# # Training the Algorithm

# In[39]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

print ("Training complete")


# In[40]:


#plotting the regression line 
line= regressor.coef_*x+ regressor.intercept_


# In[41]:


plt.scatter(x,y)
plt.plot(x,line);
plt.show()

# Making Predictions
# In[42]:


print(x_test)
y_pred=regressor.predict(x_test)
y_pred


# In[43]:


y_test


# In[47]:


#df=pd.Dataframe({'Actual':Y_test,'Predicted':Y_pred})
#df

df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[48]:


regressor.intercept_


# In[49]:


regressor.get_params()


# In[52]:


# you can also test with your own data
Hr=np.array([9.25]).reshape(1,1)
print("No.of hours:",Hr)
print("Predicted Score:",regressor.predict(Hr))


# In[54]:


from sklearn.metrics import r2_score, mean_absolute_error
b=r2_score(y_test,y_pred)
b


# # Evaluating the model

# In[55]:


# mean absolute error
print ('Mean Absolute Error:',
       mean_absolute_error(y_test,y_pred))

