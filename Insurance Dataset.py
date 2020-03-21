#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('E:\\itsstudytym\\Python Project\\ML Notebook Sessions\\Insurance Prediction\\insurance_data.csv')
data.head()


# In[5]:


x = data.iloc[:,0]
y = data.iloc[:,1]


# In[7]:


plt.scatter(x,y,color='blue',marker='+')


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=0)


# In[29]:


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train,y_train)


# In[34]:


x_test


# In[42]:


#y_test = y_test.values.reshape(-1,1)
y_predict = reg.predict(y_test)
y_predict


# In[46]:


reg.predict_proba(x_test)


# In[50]:


import seaborn as sns
sns.regplot(x,y,logistic=True,marker='+',color='red')
plt.show()


# In[51]:


reg.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




