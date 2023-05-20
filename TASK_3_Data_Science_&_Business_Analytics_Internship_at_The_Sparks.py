#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - Data Science &Business Analytics Internship 
# 

# #### TASK 3: Perfom 'Exploratory Data Analysis' on dataset 'SampleSuperstore'
# #### In this task we will bw try to find out the weak areas where we can work to make more profit.
# #### steps are as follow:
# ####  * Importing the libraries
# ####  * Reading the dataset
# ####  * Data Preprocessing
# ####  * EDA
# ####  * Data Visualization 
# 
# ### Author - Omkar Dinesh Patil

# # importing libraries 

# In[1]:


# In this step we will import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


#To ignor the warnings
import warnings as wg
wg.filterwarnings('ignore')


# # Reading the dataset

# In[38]:


dataset=pd.read_csv("SampleSuperstore.csv")


# In[39]:


dataset.head()


# # Data preprocessing

# In[40]:


dataset.shape


# In[41]:


dataset.columns


# In[42]:


dataset.isnull().sum()


# In[43]:


dataset.info()


# In[44]:


dataset.describe()


# In[45]:


# cheking for duplicates
dataset.duplicated().sum()


# In[46]:


#dropping the duplicates
dataset.drop_duplicates()
dataset.head()


# In[47]:


#removing the unnecessary columns such as postal code
dataset=dataset.drop(['Postal Code'],axis=1)


# In[48]:


dataset.head()


# # Exploratory Data Analysis

# In[49]:


# visualizing the dataset as a whole using the pair plot
import seaborn as sns
sns.pairplot(dataset)


# In[50]:


# finding the pairwise correlations between the columns and visualising using heatmaps
dataset.corr()
plt.figure(figsize=(10,5))
sns.heatmap(dataset.corr(),annot=True)
plt.show()


# # Visualising the categories

# In[51]:


plt.figure(figsize=(6,6))
textprops={"fontsize":15}
plt.title('Category')
plt.pie(dataset['Category'].value_counts(),labels=dataset['Category'].value_counts().index,autopct='%1.1f%%',textprops=textprops)
plt.show()


# In[52]:


plt.figure(figsize=(10,16))
dataset.groupby('Category')['Profit','Sales'].agg(['sum']).plot.bar()
plt.ylabel('Profit')
plt.show()


# In[55]:


#Computing top categories in terms of sales from first 100 observations
top_category_s=dataset.groupby("Category").Sales.sum().nlargest(n=100)
#Computing top categories in terms of profit from first 100 observations
top_category_p=dataset.groupby("Category").Profit.sum().nlargest(n=100)

#plotting to see it visually
plt.style.use('seaborn')
top_category_s.plot(kind='bar',figsize=(10,5),fontsize=14)
top_category_p.plot(kind='bar',figsize=(10,5),fontsize=14,color='red')
plt.xlabel('Category',fontsize=15)
plt.ylabel('Total sales/profits',fontsize=15)
plt.title("Top Category sales vs Profits",fontsize=15)
plt.show()


# # visualising the sub categories

# In[56]:


#Computing top categories in terms of sales from first 100 observations
top_subcategory_s=dataset.groupby("Sub-Category").Sales.sum().nlargest(n=100)
#Computing top categories in terms of profit from first 100 observations
top_subcategory_p=dataset.groupby("Sub-Category").Profit.sum().nlargest(n=100)

#plotting to see it visually
plt.style.use('seaborn')
top_subcategory_s.plot(kind='bar',figsize=(10,5),fontsize=14)
top_subcategory_p.plot(kind='bar',figsize=(10,5),fontsize=14,color='red')
plt.xlabel('Sub-Category',fontsize=15)
plt.ylabel('Total sales/profits',fontsize=15)
plt.title("Top Sub-Category sales vs Profits",fontsize=15)
plt.show()


# In[59]:


# A more detailed view
plt.figure(figsize=(14,12))
statewise=dataset.groupby(['Sub-Category'])['Profit'].sum().nlargest(50)
statewise.plot.barh()  #h for horozontal


# ### The above graph clearly shows that Copies and phones have the hightest sales andprofit has negative profit .
# 
# # Visualising the discount

# In[60]:


plt.figure(figsize=(8,7))
sns.lineplot(dataset['Discount'],dataset['Profit'],data=dataset)


# # Visualising the Sales vs Profit in diffrent Regions

# In[61]:


plt.figure(figsize=(6,6))
plt.title('Region')
plt.pie(dataset['Region'].value_counts(),labels=dataset['Region'].value_counts().index,autopct='%1.1f%%')
plt.show()


# ### The graph shows that West and East have same Profit thought sales in the East are less as compared to West.
# 
# # Visualising the Sales vs Profits in different states

# In[62]:


#Computing top categories in terms of sales from first 100 observations
top_states_s=dataset.groupby("State").Sales.sum().nlargest(n=10)
#Computing top categories in terms of profit from first 100 observations
top_states_p=dataset.groupby("State").Profit.sum().nlargest(n=10)

#plotting to see it visually
plt.style.use('seaborn')
top_states_s.plot(kind='bar',figsize=(10,5),fontsize=14)
top_states_p.plot(kind='bar',figsize=(10,5),fontsize=14,color='red')
plt.xlabel('States',fontsize=15)
plt.ylabel('Total sales',fontsize=15)
plt.title("Top 10 states sales vs Profits",fontsize=15)
plt.show()


# # Cheking the interdependency of Sales ,Profit and Discount

# In[65]:


plt.style.use('seaborn')
dataset.plot(kind="scatter",figsize=(10,5),x= 'Sales',y= 'Profit',c= "Discount",s= 20,fontsize=16,colormap='viridis')
plt.ylabel('Total Profits',fontsize=16)
plt.title("Interdependency of Sales,Profits and Discounts",fontsize=16)
plt.show()


# ### The graph clearly shows that if we give more Discount on our products sales increases but profit decresses.

# # CONCLUSION:
# ## The weak areas where one can work to make more Profit are:
# ## 1. we should limit sales of furniture and increses that of technology and office suppliers as furniture has very less profit as compered to sales.
# ## 2. Considering the sub-categaries sales of tables should be minimized.
# ## 3. Incraese sales more in the east as profit is more.
# ## 4. We should concentrate on the states like 'New York' and 'California' to make profits.

# In[ ]:




