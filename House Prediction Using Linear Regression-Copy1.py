#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression


# In[2]:


data = pd.read_csv(r'C:\Users\91944\OneDrive\Desktop\TASK\train_data.csv')

test_data = pd.read_csv(r'C:\Users\91944\OneDrive\Desktop\TASK\test_data.csv')

data.head()


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


data.isnull().sum()[data.isnull().sum() > 0]


# In[9]:


data.drop(['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)


# In[10]:


null_columns = data.isnull().sum()[data.isnull().sum() > 0]


# In[11]:


data[null_columns.index].describe()


# In[12]:


data['LotFrontage'].fillna(70, inplace=True)
data['MasVnrArea'].fillna(103, inplace=True)
data['GarageYrBlt'].fillna(1978, inplace=True)


# In[13]:


data.isnull().sum()[data.isnull().sum() > 0]


# In[14]:


data.dropna(inplace=True)


# In[15]:


data.shape


# In[16]:


data.isnull().sum()[data.isnull().sum() > 0]


# In[17]:


data.columns


# In[18]:


data.head()


# In[19]:


df_cat = data.select_dtypes(include = ['object'])
df_cat.columns


# In[20]:


def map_categories_to_nums(df_cat):
    for col in df_cat.columns:
        mapping_dict = {val: idx for idx, val in enumerate(df_cat[col].unique())}
        df_cat[col] = df_cat[col].map(mapping_dict)
    return df_cat


# In[21]:


df_cat_num = map_categories_to_nums(df_cat)
df_cat_num.head()


# In[22]:


data.head()


# In[23]:


data[df_cat.columns] = df_cat_num
data.head()


# In[24]:


data.info()


# In[25]:


test_data.head()


# In[26]:


test_data.shape


# In[27]:


test_data.drop(['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)


# In[28]:


test_data.isnull().sum()[test_data.isnull().sum() > 0]


# In[29]:


test_data[null_columns.index].describe()


# In[30]:


data['LotFrontage'].fillna(68, inplace=True)
data['MasVnrArea'].fillna(100, inplace=True)
data['GarageYrBlt'].fillna(1977, inplace=True)


# In[31]:


test_data.isnull().sum()[test_data.isnull().sum() > 0]


# In[32]:


test_data.dropna(inplace=True)


# In[33]:


test_data.shape


# In[34]:


test_df_cat = test_data.select_dtypes(include = ['object'])


# In[35]:


test_df_cat_num = map_categories_to_nums(test_df_cat)


# In[36]:


test_df_cat_num.head()


# In[37]:


test_data[test_df_cat.columns] = test_df_cat_num
test_data.head()


# In[38]:


X_train = data.drop(['SalePrice'], axis=1)
y_train = data['SalePrice']


# In[39]:


LinearRegressionModel = LinearRegression()
LinearRegressionModel.fit(X_train, y_train)


# In[40]:


y_pred = LinearRegressionModel.predict(test_data)


# In[42]:


sample_submission_df = pd.read_csv(r'C:\Users\91944\OneDrive\Desktop\TASK\sample_submission.csv')
sample_submission_df = sample_submission_df[sample_submission_df['Id'].isin(test_data['Id'])]
sample_submission_df['SalePrice'] = y_pred
sample_submission_df = pd.read_csv(r'C:\Users\91944\OneDrive\Desktop\TASK\sample_submission.csv')

sample_submission_df.head()


# In[ ]:




