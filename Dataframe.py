#!/usr/bin/env python
# coding: utf-8

# In[27]:


pip install scipy


# In[28]:


import pandas as pd
import numpy as np
import scipy


# In[5]:


myData = pd.read_csv('./data/Travel Destination Rating.csv')
kaggleData = pd.read_csv('./data/google_review_ratings.csv')


# In[6]:


myData.head()
kaggleData.head()


# In[7]:


myData = myData.drop(columns = 'Timestamp')
kaggleData = kaggleData.drop(columns = 'User')


# In[8]:


myData.head()

kaggleData = kaggleData.drop(columns = 'Unnamed: 25')
kaggleData.head()


# In[9]:


columnsContent = ['Sex', 'Age', 'Location', 'With']
columnsData = ['Resort', 'Beaches', 'Park', 'Museum', 'Malls', 'Zoo', 'Bar', 'Art Gallerly', 'NightClub', 'Beauty', 'Cafe', 'Mountain', 'Monument' ]


# In[10]:


# Attribute 2 : Average ratings on churches 
Attribute 3 : Average ratings on resorts 
Attribute 4 : Average ratings on beaches 
Attribute 5 : Average ratings on parks 
# Attribute 6 : Average ratings on theatres 
Attribute 7 : Average ratings on museums 
Attribute 8 : Average ratings on malls 
Attribute 9 : Average ratings on zoo 
# Attribute 10 : Average ratings on restaurants 
Attribute 11 : Average ratings on pubs/bars 
# Attribute 12 : Average ratings on local services 
# Attribute 13 : Average ratings on burger/pizza shops 
# Attribute 14 : Average ratings on hotels/other lodgings 
# Attribute 15 : Average ratings on juice bars 
Attribute 16 : Average ratings on art galleries 
Attribute 17 : Average ratings on dance clubs 
# Attribute 18 : Average ratings on swimming pools 
# Attribute 19 : Average ratings on gyms 
# Attribute 20 : Average ratings on bakeries 
Attribute 21 : Average ratings on beauty & spas 
Attribute 22 : Average ratings on cafes 
Attribute 23 : Average ratings on view points 
Attribute 24 : Average ratings on monuments 
# Attribute 25 : Average ratings on gardens


# In[11]:


useColumns = ['Category 2',
              'Category 3',
              'Category 4',
              'Category 6',
              'Category 7',
              'Category 8',
              'Category 10',
              'Category 15',
              'Category 16',
              'Category 20',
              'Category 21',
              'Category 22',
              'Category 23']
              
kaggleData = kaggleData[useColumns]
kaggleData.head()


# In[12]:


content = myData.iloc[:,:4]
content.columns = columnsContent
content.head()

data = myData.iloc[:, 4:]
data.columns = columnsData
kaggleData.columns = columnsData


# In[13]:


data.head()
content.head()


# In[14]:


kaggleData.head()


# In[15]:


# kaggleData.replace('NaN', 'NAN')


# In[16]:


kaggleData.head()


# In[17]:


data.replace(0, 1)


# In[124]:


finalData = pd.concat([data, kaggleData], axis=0)
data.shape


# In[125]:


finalData.head()
finalData.shape
finalData.to_csv('final_data.csv', encoding='utf-8')


# In[126]:


finalData


# In[127]:


NaNData = finalData[(finalData == 0.00).any(axis = 1)]
NaNData.to_csv('test_data.csv', encoding='utf-8')
NaNData


# In[128]:


fullData = finalData[~(finalData == 0.00).any(axis = 1)]
fullData.to_csv('full_data.csv', encoding='utf-8')
fullData


# In[36]:


def normalization(df):
    df_mean = df.mean(axis = 1)
    return df.subtract(df_mean, axis = 'rows')


# In[60]:


finalData = normalization(finalData)
finalData


# In[52]:


finalData = np.nan_to_num(finalData)
finalData = pd.DataFrame(finalData)


# In[33]:



def similarity_pearson(x, y):
    import scipy.stats
    return scipy.stats.pearsonr(x, y)[0]


# In[53]:


similarity_matrix = np.array([similarity_pearson(finalData.iloc[i, :], finalData.iloc[j, :]) for i in range(0, 14) for j in range(0, 14)])


# In[59]:


similarity_dataframe = pd.DataFrame(data = similarity_matrix.reshape(14, 14))
similarity_dataframe


# In[ ]:




