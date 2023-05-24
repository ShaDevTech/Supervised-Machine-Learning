#!/usr/bin/env python
# coding: utf-8

# ## Predicting House Sale Prices for Ames, Iowa
# ### We will work with housing data for the city of Ames, Iowa, United States from 2006 to 2010. 

# Let is begin with pipeline of functions we will apply.
# We will apply different techniques K Fold, One Hot encoded 
# ![Screenshot%202023-05-22%20190640.png](attachment:Screenshot%202023-05-22%20190640.png)

# ### Predictor
# 
# * SalePrice: The property's sale price in dollars. 
# 
# ### Features
# 
# * MoSold: Month Sold
# * YrSold: Year Sold   
# * SaleType: Type of sale
# * SaleCondition: Condition of sale
# * MSSubClass: The building class
# * MSZoning: The general zoning classification
# * ...
# 

# ## Import required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# ## Read data

# In[2]:


df = pd.read_csv("AmesHousing.tsv", delimiter="\t")
df.head()


# In[3]:


df.shape


# ### We are dividing data into train and test in half

# ### We are using only 1 input 'Gr Liv Area' and 1 output 'Sale Price'

# ### Create functions

# In[4]:


def transform_features(df):
    return df

def select_features(df):
    return df[['Gr Liv Area','SalePrice']]

def train_and_test(df):
    train = df.loc[:1460] # Selects the first 1460 rows from from data and assign to train.
    test = df.loc[1460:]  # Selects the remaining rows from data and assign to test.
    
    # select numerical columns only from train and test data
    num_train = train.select_dtypes(['int','float'])
    num_test = train.select_dtypes(['int','float'])
    
    # features (X), drop target variables SalePrice
    features = num_train.columns.drop('SalePrice')
    
    # instantiate Linear Regression
    lr = LinearRegression()
    
    # fit the model
    lr.fit(train[features],train['SalePrice'])
    
    # predict the model
    y_predict = lr.predict(test[features])
    
    # assess model
    mse = mean_squared_error(test['SalePrice'],y_predict)
    rmse = np.sqrt(mse)
    
    return rmse

transform_df = transform_features(df)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df)
rmse


# train_features
# ![Screenshot%202023-05-22%20193518.png](attachment:Screenshot%202023-05-22%20193518.png)

# ##  Feature Engineering

# Let us start removing features with many missing values, diving deeper into potential categorical features, and transforming text and numerical columns. 
# 
# * remove features that we don't want to use in the model, just based on the number of missing values or data leakage
# * transform features into the proper format (numerical to categorical, scaling numerical, filling in missing values, etc)
# * create new features by combining other features
# 
# - Handle missing values:
#     - All columns:
#         - Drop any with 5% or more missing values **for now**.
#     - Text columns:
#         - Drop any with 1 or more missing values **for now**.
#     - Numerical columns:
#         - For columns with missing values, fill in with the most common value in that column

# ### 1. All columns: drop any with 5% (2930 * 0.05 = 146.5 or 2930/146.5 = 20) more missing values for now.

# In[5]:


num_missing = df.isnull().sum()
num_missing


# In[6]:


missing_cols = num_missing[num_missing>len(df)/20].sort_values()
missing_cols


# In[7]:


df = df.drop(missing_cols.index, axis = 1)


# 71 columns are left

# ### 2: Text columns: drop any with any missing values for now.

# In[8]:


# find text columns 
text_col = df.select_dtypes(include = 'object')
text_col


# In[9]:


# count missing values in the columns
text_col_missing = text_col.isnull().sum().sort_values(ascending=False)
text_col_missing 


# In[10]:


# drop text columns with any missing value
df = df.drop(text_col_missing[text_col_missing> 0].index, axis=1)
df


# In[11]:


df.shape


# ### 3: Numerical columns: for columns with missing values, fill in with the most common value in that column

# In[12]:


# Count missing values in the columns
num_col = df.select_dtypes(include=['int', 'float']).isnull().sum()
missing_num_col = num_col[num_col > 0]
missing_num_col


# In[13]:


df[df.isnull().any(axis=1)]


# In[14]:


# take average of columbs with missing values to fill in with most common value i.e mode
replacement_values = df[missing_num_col.index].mode().to_dict(orient='records')[0]
# In this case, 'records' means we want each row in the DataFrame to be converted into a dictionary
replacement_values


# In[15]:


df = df.fillna(replacement_values)
df


# In[16]:


df.isnull().sum().value_counts()


# ### What new features can we create, that better capture the information in some of the features?

# In[17]:


# years sold
years_sold = df['Yr Sold'] - df['Year Built']
years_sold[years_sold < 0]


# In[18]:


# years_until_remod
years_until_remod = df['Yr Sold'] - df['Year Remod/Add']
years_until_remod[years_until_remod<0]


# In[19]:


years_until_remod [years_until_remod  < 3]


# In[20]:


# add new columns
df['years_before_sale'] = years_sold
df['years_since_remod'] = years_until_remod

# drop rows with negative values
df = df.drop([2180,1702,2181], axis = 0)


# In[21]:


df.head()


# ## Drop columns that:
# 
# - that aren't useful for ML
# - leak data about the final sale, read more about columns here

# In[22]:


# that aren't useful for ML
not_useful_col = ['PID', 'Order']
df= df.drop(not_useful_col, axis = 1)


# In[23]:


# leak data about the final sale, read more about columns here
col_leak_data = ["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"]
df = df.drop(col_leak_data, axis=1)


# In[24]:


df.head()


# ## Let's update transform_features()

# In[25]:


def transform_features(df):
    # drop any columns contain less than 5% missing values
    num_missing = df.isnull().sum()
    missing_cols = num_missing[num_missing > len(df)/20].sort_values()
    df = df.drop(missing_cols.index, axis = 1)
    
    # drop text columns with any missing value
    missing_text_col = df.select_dtypes(include='object').isnull().sum().sort_values(ascending=False)
    df = df.drop(missing_text_col[missing_text_col > 0].index, axis = 1)
    
    # for numerical columns with missing values, fill in with the most common value in that column
    missing_num_col = df.select_dtypes(include=['int','float']).isnull().sum()
    missing_num_col = missing_num_col[(missing_num_col > 0) & (missing_num_col < len(df)/20)].sort_values()
    replacement_num_col = df[missing_num_col.index].mode().to_dict(orient='records')[0]
    df = df.fillna(replacement_num_col)  
    
    # add new features
    df['years_before_sale'] = df['Yr Sold'] - df['Year Built']
    df['years_since_remod'] = df['Yr Sold'] - df['Year Remod/Add']
    
    #drop rows with negative values
    df = df.drop([1702, 2180, 2181], axis=0)
    
    # drop useless columns
    df = df.drop(["PID", "Order", "Mo Sold", "Sale Condition", "Sale Type", "Year Built", "Year Remod/Add"], axis=1)
    
    return df  

def select_features(df):
    return df[["Gr Liv Area", "SalePrice"]]

def train_and_test(df):  
    train = df[:1460]
    test = df[1460:]
    
    ## You can use `pd.DataFrame.select_dtypes()` to specify column types
    ## and return only those columns as a DataFrame.
    numeric_train = train.select_dtypes(include=['integer', 'float'])
    numeric_test = test.select_dtypes(include=['integer', 'float'])
    
    ## You can use `pd.Series.drop()` to drop a value.
    features = numeric_train.columns.drop("SalePrice")
    lr = LinearRegression()
    lr.fit(train[features], train["SalePrice"])
    predictions = lr.predict(test[features])
    mse = mean_squared_error(test["SalePrice"], predictions)
    rmse = np.sqrt(mse)
    
    return rmse

df = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_df = transform_features(df)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df)

rmse


# ## Feature Selection

# ### Now that we have cleaned and transformed a lot of the features in the data set, it's time to move on to feature selection for numerical features.

# In[26]:


corr_with_target = df.corr(numeric_only=True)['SalePrice'].abs().sort_values(ascending=False)
corr_with_target 


# In[27]:


transform_df = df.drop(corr_with_target[corr_with_target < 0.4].index, axis = 1)
transform_df.head(3)


# - **Which columns are currently numerical but need to be encoded as categorical instead (because the numbers don't have any semantic meaning)?**
# - **If a categorical column has hundreds of unique values (or categories), should we keep it? When we dummy-code this column, hundreds of columns will need to be added back to the DataFrame.**

# In[28]:


# Create a list of column names from documentation that are *meant* to be categorical.
nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]


# **Which column we already have from nominal_features list**

# In[29]:


transform_cat_col = []
for col in nominal_features:
    if col in df.columns:
        transform_cat_col.append(col)


# **How many unique values in each categorical column?**

# In[30]:


cat_unique_vals= df[transform_cat_col].apply(lambda x : len(x.value_counts())).sort_values(ascending=False)
cat_unique_vals


# If a categorical column has hundreds of unique values (or categories), should you keep it? When you dummy code this column, hundreds of columns will need to be added back to the data frame.

# **Aribtrary cutoff of 10 unique values (worth experimenting)**

# In[31]:


df = df.drop(cat_unique_vals[cat_unique_vals>10].index, axis=1)


# **Select only the remaining text columns, and convert to categorical**

# In[32]:


text_col = df.select_dtypes(include='object')
for col in text_col:
    df[col] = df[col].astype('category')

# Create dummy columns, and add back to the DataFrame!
df = pd.concat([df, pd.get_dummies(df.select_dtypes(include='category'))], axis = 1).drop(text_col, axis = 1)


# In[33]:


df


# In[34]:


def transform_features(df):
    # drop any columns contain less than 5% missing values
    num_missing = df.isnull().sum()
    missing_cols = num_missing[num_missing > len(df)/20].sort_values()
    df = df.drop(missing_cols.index, axis = 1)
    
    # drop text columns with any missing value
    missing_text_col = df.select_dtypes(include='object').isnull().sum().sort_values(ascending=False)
    df = df.drop(missing_text_col[missing_text_col > 0].index, axis = 1)
    
    # for numerical columns with missing values, fill in with the most common value in that column
    missing_num_col = df.select_dtypes(include=['int','float']).isnull().sum()
    missing_num_col = missing_num_col[(missing_num_col > 0) & (missing_num_col < len(df)/20)].sort_values()
    replacement_num_col = df[missing_num_col.index].mode().to_dict(orient='records')[0]
    df = df.fillna(replacement_num_col)  
    
    # add new features
    df['years_before_sale'] = df['Yr Sold'] - df['Year Built']
    df['years_since_remod'] = df['Yr Sold'] - df['Year Remod/Add']
    
    #drop rows with negative values
    df = df.drop([1702, 2180, 2181], axis=0)
    
    # drop useless columns
    df = df.drop(["PID", "Order", "Mo Sold", "Sale Condition", "Sale Type", "Year Built", "Year Remod/Add"], axis=1)
    
    return df  

def select_features(df, coeff_threshold=0.4, uniq_threshold=10):
    df_corr = df.corr(numeric_only=True)['SalePrice'].abs().sort_values()
    df.drop(df_corr[df_corr < coeff_threshold].index, axis = 1)
    
    #nominal features
    nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]
    
   # how many features we already have in df
    transform_cat_cols = []
    for col in nominal_features:
        if col in df.columns:
            transform_cat_cols.append(col)
    
    # How many unique values in each categorical column?
    unique_count = df[transform_cat_cols].apply(lambda x : len(x.value_counts())).sort_values()
    df = df.drop(unique_count[unique_count>10].index, axis = 1)
    
    # convert remaining columns to categorical
    text_cols = df.select_dtypes('object')
    for col in text_cols:
        df[col] = df[col].astype('category')
        
    # add dummy columns to data
    df = pd.concat([df, pd.get_dummies(df.select_dtypes('category'))],axis=1).drop(text_cols,axis=1)
    
    return df

def train_and_test(df):  
    train = df[:1460]
    test = df[1460:]
    
    ## You can use `pd.DataFrame.select_dtypes()` to specify column types
    ## and return only those columns as a DataFrame.
    numeric_train = train.select_dtypes(include=['integer', 'float'])
    numeric_test = test.select_dtypes(include=['integer', 'float'])
    
    ## You can use `pd.Series.drop()` to drop a value.
    features = numeric_train.columns.drop("SalePrice")
    lr = LinearRegression()
    lr.fit(train[features], train["SalePrice"])
    predictions = lr.predict(test[features])
    mse = mean_squared_error(test["SalePrice"], predictions)
    rmse = np.sqrt(mse)
    
    return rmse

df = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_df = transform_features(df)
filtered_df = select_features(transform_df)
rmse = train_and_test(filtered_df)

rmse


# ## Use K fold

# In[35]:


def transform_features(df):
    # drop any columns contain less than 5% missing values
    num_missing = df.isnull().sum()
    missing_cols = num_missing[num_missing > len(df)/20].sort_values()
    df = df.drop(missing_cols.index, axis = 1)
    
    # drop text columns with any missing value
    missing_text_col = df.select_dtypes(include='object').isnull().sum().sort_values(ascending=False)
    df = df.drop(missing_text_col[missing_text_col > 0].index, axis = 1)
    
    # for numerical columns with missing values, fill in with the most common value in that column
    missing_num_col = df.select_dtypes(include=['int','float']).isnull().sum()
    missing_num_col = missing_num_col[(missing_num_col > 0) & (missing_num_col < len(df)/20)].sort_values()
    replacement_num_col = df[missing_num_col.index].mode().to_dict(orient='records')[0]
    df = df.fillna(replacement_num_col)  
    
    # add new features
    df['years_before_sale'] = df['Yr Sold'] - df['Year Built']
    df['years_since_remod'] = df['Yr Sold'] - df['Year Remod/Add']
    
    #drop rows with negative values
    df = df.drop([1702, 2180, 2181], axis=0)
    
    # drop useless columns
    df = df.drop(["PID", "Order", "Mo Sold", "Sale Condition", "Sale Type", "Year Built", "Year Remod/Add"], axis=1)
    
    return df  

def select_features(df, coeff_threshold=0.4, uniq_threshold=10):
    df_corr = df.corr(numeric_only=True)['SalePrice'].abs().sort_values()
    df.drop(df_corr[df_corr < coeff_threshold].index, axis = 1)
    
    #nominal features
    nominal_features = ["PID", "MS SubClass", "MS Zoning", "Street", "Alley", "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air", "Garage Type", 
                    "Misc Feature", "Sale Type", "Sale Condition"]
    
   # how many features we already have in df
    transform_cat_cols = []
    for col in nominal_features:
        if col in df.columns:
            transform_cat_cols.append(col)
    
    # How many unique values in each categorical column?
    unique_count = df[transform_cat_cols].apply(lambda x : len(x.value_counts())).sort_values()
    df = df.drop(unique_count[unique_count>10].index, axis = 1)
    
    # convert remaining columns to categorical
    text_cols = df.select_dtypes('object')
    for col in text_cols:
        df[col] = df[col].astype('category')
        
    # add dummy columns to data
    df = pd.concat([df, pd.get_dummies(df.select_dtypes('category'))],axis=1).drop(text_cols,axis=1)
    
    return df

def train_and_test(df, k=0):  
    #features X and y
    numeric_df = df.select_dtypes(include=['integer', 'float'])
    features = numeric_df.columns.drop('SalePrice')
    
    # Initialize a model (e.g., Linear Regression)
    lr = LinearRegression()
    
    # Define the number of folds (K)
    if k == 0:
        train= df[:1460]
        test = df[1460:]
        
        # fit the model
        lr.fit(train[features],train['SalePrice'])
        
        # predict
        y_predict = lr.predict(test[features])
        
        #assess the model
        mse = mean_squared_error(test['SalePrice'], y_predict)
        rmse = np.sqrt(mse)
        
        return rmse
    
    # K==1 approach provides an estimate of the model's performance using a single train-test split of the data. 
    
    if k == 1:
        
        # Randomize *all* rows by sampling from dataframe using sample(), specofy fraction of rows to sample
        # frac=1 means we want to sample all the rows, which is equivalent to shuffling the entire DataFrame.
        df = df.sample(frac=1)
        train= df[:1460]
        test = df[1460:]
        
        # Train the model on the training data and make predictions on the test data
        lr.fit(train[features],train['SalePrice'])
        y_predict = lr.predict(test[features])
        
        # Calculate the mean squared error (MSE) and root mean squared error (RMSE) for the first split
        mse1 = mean_squared_error(test['SalePrice'], y_predict)
        rmse1 = np.sqrt(mse1)
        
        # Train the model on the test data and make predictions on the train data
        lr.fit(test[features],test['SalePrice'])
        y_predict2 = lr.predict(train[features])
        
        # Calculate the mean squared error (MSE) and root mean squared error (RMSE) for the second split
        mse2 = mean_squared_error(train['SalePrice'], y_predict2)
        rmse2 = np.sqrt(mse2)
        
        avg_rmse = np.mean([rmse1,rmse2])
        return avg_rmse
    
    else:
        # Create a KFold object
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        
        # iterate over K folds
        for train_index,test_index in kf.split(df):
            # Split the data into training and test sets based on the fold indices
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            
            # Train the model using the training data and predict
            lr.fit(train[features], train["SalePrice"])
            y_predict = lr.predict(test[features])
    
            # assess the model
            mse = mean_squared_error(test["SalePrice"], y_predict)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
            avg_rmse = np.mean(rmse_values)
            return avg_rmse

df = pd.read_csv("AmesHousing.tsv", delimiter="\t")
transform_df = transform_features(df)
filtered_df = select_features(transform_df)

rmse = train_and_test(filtered_df , k=4)

k_val = np.arange(0,50).tolist()
rmse_val = []
for k in k_val:
    rmse = train_and_test(filtered_df , k=k)
    rmse_val.append(rmse)

# find the lowest RMSE val and its corresponding k value
lowest_rmse = min(rmse_val)
lowest_k = k_val[rmse_val.index(lowest_rmse)]

plt.plot(k_val, rmse_val)
plt.scatter(lowest_k, lowest_rmse, color='red')
plt.annotate(f'({lowest_k},{lowest_rmse:.2f})', xy = (lowest_k, lowest_rmse), xytext = (lowest_k, lowest_rmse+20), 
            arrowprops = dict(arrowstyle='-', color='red'))

plt.title('RMSE for different K values')
plt.xlabel('K fold values')
plt.ylabel('RMSE values')
plt.show()


# <details><summary><strong>K Fold explained using examples </strong></summary>
# Child: Have you ever played a game where you had to divide your friends into teams and compete against each other?
# 
# You: Yes, it's like playing a game of cricket where we have two teams competing against each other.
# 
# Child: Great! Now, imagine we want to see which team is better at playing cricket. One way to find out is by having them play against each other. But what if we want to make sure our judgment is fair and accurate?
# 
# You: That's where K-fold cross-validation comes in. Instead of just playing one match between the two teams, we divide the teams into different groups and have them play multiple matches.
# 
# Child: So, it's like having many matches to make sure we get a good idea of which team is better?
# 
# You: Exactly! In K-fold cross-validation, we divide the teams into K groups or folds. Let's say we have 10 teams, and we decide to use K=5. We would divide the teams into 5 equal groups, with 2 teams in each group.
# 
# Child: Oh, I see! So, each group will play against the other groups?
# 
# You: Yes, that's correct! We will take one group as the test group and the remaining groups as the training groups. The teams in the training groups will practice and improve their skills, and then they will compete against the test group.
# 
# Child: That sounds fair! But what about the other groups?
# 
# You: After the first match, we will rotate the groups. The test group will become a training group, and one of the training groups will become the new test group. We repeat this process until every group has had a chance to be the test group.
# 
# Child: So, each team gets to play against different teams, and they all have a chance to be in the test group?
# 
# You: Yes, that's correct! By playing against different teams and taking turns as the test group, we can make sure that every team has a fair opportunity to show their skills and compete against a variety of opponents.
# 
# Child: I get it now! It's like having a tournament where every team gets to play against each other and show their abilities.
# 
# You: Exactly! K-fold cross-validation helps us evaluate and compare the performance of different teams or models in a more reliable and fair way by giving them multiple chances to showcase their skills.
# 
# Child: That's really cool! It ensures that we have a better understanding of how good each team is by testing them against different opponents.
# 
# You: Absolutely! K-fold cross-validation helps us make more informed decisions by testing our models or teams on different subsets and providing a more accurate assessment of their performance.
#     
# Also,
#     for train_index,test_index in Kf.split(X):
# Typically, in machine learning tasks, you have a set of features X (input variables) and a corresponding target variable y (output variable). When using kf.split(X) to generate the train and test indices, it assumes that the rows in X and y are aligned, meaning that the target variable y has the same number of rows and follows the same order as the feature data X.
# 
# So, when you iterate over kf.split(X), the generated train and test indices correspond to both the feature data and the target variable. This allows you to split both X and y into training and testing sets in the same manner.
# </details>

# ## Training and Test Splits

# ## Question 1
# 
# * Import the data using Pandas and examine the shape. There are 79 feature columns plus the predictor, the sale price (`SalePrice`). 
# * There are three different types: integers (`int64`), floats (`float64`), and strings (`object`, categoricals). Examine how many there are of each data type. 
# 

# ### Categorical Columns

# In[36]:


data = pd.read_csv("AmesHousing.tsv", delimiter="\t")
data.shape


# In[37]:


data.dtypes.value_counts()


# ### null columns and fill them

# In[38]:


# find cols with null values
data_null = data.isnull().sum()
data_missing_val = data_null [data_null  > 0]
data_missing_val


# #### fill numeric values with mean

# In[39]:


replacement_values = data[data_missing_val.index].mean(numeric_only=True)
replacement_values


# In[40]:


data[data_missing_val.index]= data[data_missing_val.index].fillna(replacement_values)
data[data_missing_val.index].isnull().sum()


# #### fill categorical values

# In[41]:


categroical_cols = data[data_missing_val.index].columns[data[data_missing_val.index].dtypes == object].tolist()
categroical_cols_filled = data[categroical_cols].fillna(data[categroical_cols].mode()).iloc[0]
data[categroical_cols] = categroical_cols_filled

# check for missing values
data.isnull().sum()


# ### Question 2 

# A significant challenge, particularly when dealing with data that have many columns, is ensuring each column gets encoded correctly. 
# 
# This is particularly true with data columns that are ordered categoricals (ordinals) vs unordered categoricals. Unordered categoricals should be one-hot encoded, however this can significantly increase the number of features and creates features that are highly correlated with each other.
# 
# Determine how many total features would be present, relative to what currently exists, if all string (object) features are one-hot encoded. Recall that the total number of one-hot encoded columns is `n-1`, where `n` is the number of categories.
# 

# In[42]:


#select the object (string) columns
categorical_columns = data.columns[data.dtypes == object]

# Determine how many extra columns would be created- see number of unique columns as each unique value is converted to column
num_ohc_cols = data[categorical_columns].apply(lambda x:  x.nunique()).sort_values(ascending=False)

# No need to encode if there is only one value so we do not keep them
small_num_ohc_cols = num_ohc_cols[num_ohc_cols > 1]

# Number of one-hot columns is 1 less than number of categories n-1
small_num_ohc_cols -=1

# total number of columns created
small_num_ohc_cols.sum()


# ### Code breakdown

# ### Select the object (string) columns

# In[43]:


categorical_cols = data.columns[data.dtypes==object]
categorical_cols


# ### Determine how many extra columns would be created

# In[44]:


num_ohc_cols = data[categorical_cols].apply(lambda x : x.nunique()).sort_values(ascending=False)
num_ohc_cols 


# ### total number of columns including original

# In[45]:


num_ohc_cols.sum()


# ### No need to encode if there is only one value so we do not keep them

# In[46]:


small_num_ohc_cols = num_ohc_cols[num_ohc_cols > 1]
small_num_ohc_cols


# ### Number of one-hot columns is one less than the number of categories

# In[47]:


small_num_ohc_cols -= 1
small_num_ohc_cols


# ### number of columns willl be created by one hot encoder

# In[48]:


small_num_ohc_cols.sum()


# ## Question 3
# 
# Let's create a new data set where all of the above categorical features will be <b>one-hot encoded</b>. We can fit this data and see how it affects the results.
# 
# * Used the dataframe `.copy()` method to create a completely separate copy of the dataframe for one-hot encoding
# * On this new dataframe, one-hot encode each of the appropriate columns and add it back to the dataframe. Be sure to drop the original column.
# * For the data that are not one-hot encoded, drop the columns that are string categoricals.
# 
# For the first step, numerically encoding the string categoricals, either Scikit-learn;s `LabelEncoder` or `DictVectorizer` can be used. However, the former is probably easier since it doesn't require specifying a numerical value for each category, and we are going to one-hot encode all of the numerical values anyway. (Can you think of a time when `DictVectorizer` might be preferred?)
# 

# In[49]:


from sklearn.preprocessing import OneHotEncoder

# Create a copy of the original data as we are making changes by doing One Hot Encoding
data_ohc = data.copy()

# Initialize the OneHotEncoder
ohc = OneHotEncoder() # drop=first option to drop 1st value to avoid multicollinearity similar to pd.get_dummies()

# iterate through the unique categorical columns for one hot encoding

for col in num_ohc_cols.index:
    # Encode the column data-make it a datafrmae 2d array to avoid error.this returns a sparse matrix
    new_dat = ohc.fit_transform(data_ohc[[col]])
    # sparse matrix is created, Compressed Sparse Row format>
    
    # drop the original column from the data
    data_ohc = data_ohc.drop(col, axis = 1)
    
    # get names of all unique columns so we can use them as an identifier
    categ = ohc.categories_
    
    # create column name for each One hot encoded column by values (joining column name with value name)
    # use categ[0] because it is an array. Our list is in an array
    #new_col = ["{}_{}".format(col, cat) for cat in categ[0]]
    #new_col = ['_'.join([col,cat]) for cat in categ[0]]
    new_col = [f"{col}_{cat}" for cat in categ[0]]
    
    # create a new dataframe
    new_df = pd.DataFrame(new_dat.toarray(), columns = new_col)
    
    # append the new data to original data
    data_ohc = pd.concat([data_ohc, new_df], axis = 1)
        


# In[50]:


# calcualte the difference in original and one hot encoded columns
data_ohc.shape[1] - data.shape[1]


# In[51]:


data_ohc.shape[1]- data.shape[1]


# 241 columns are created

# In[52]:


print(data.shape[1])
# Remove the string columns from the dataframe
data = data.drop(num_ohc_cols.index, axis=1)

print(data.shape[1])


# In[53]:


data


# In[54]:


data_ohc


# ### Code breakdown

# In[55]:


num_ohc_cols.index


# In[56]:


pd.DataFrame(new_dat.toarray())


# In[57]:


new_df


# In[58]:


categ[0]


# In[59]:


new_col


# ## Question 4
# 
# * Create <b>train and test splits </b> of both data sets. To ensure the data gets split the same way, use the same `random_state` in each of the two splits.
# * For each data set, fit a basic linear regression model on the training data. 
# * Calculate the mean squared error on both the train and test sets for the respective models. Which model produces smaller error on the test data and why?
# 

# In[60]:


from sklearn.model_selection import train_test_split

y_col = 'SalePrice'

# split the data that is not one hot encoded
# X = data.drop(y_col, axis=1)
feature_cols = [x for x in data.columns if x != y_col]
X = data[feature_cols]
y = data[y_col]

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 42)

# split the data that is one hot encoded
X = data_ohc.drop(y_col, axis = 1)
y = data_ohc[y_col]

X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X,y, test_size=0.3, random_state=42)


# #### Compare the indices to ensure they are identical

# In[61]:


(X_train_ohc.index == X_train.index).all()


# ### Code breakdown

# In[62]:


feature_cols = [x for x in data.columns if x != y_col]
feature_cols


# ### Build the Linear Regression Model

# In[63]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

# Create an instance of the imputer
# imputer = SimpleImputer(strategy='mean')
# Impute missing values in X_train and X_test
# X_train_imputed = imputer.fit_transform(X_train)
# X_test_imputed = imputer.transform(X_test)

# Initialize the linear regression model
lr = LinearRegression()

# Storage for error values
error_df = []

# Fit and predict the model with encoded data
lr.fit(X_train_ohc, y_train_ohc)
y_train_ohc_pred = lr.predict(X_train_ohc)
y_test_ohc_pred = lr.predict(X_test_ohc)

# Calculate and append the mean squared errors to the error_df DataFrame
error_df.append(pd.Series({"train":mean_squared_error(y_train_ohc,y_train_ohc_pred),
                          "test":mean_squared_error(y_test_ohc,y_test_ohc_pred)}, name ='one hot encoded'))

# Fit and predict the model with non encoded data
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Calculate and append the mean squared errors to the error_df DataFrame
error_df.append(pd.Series({"train":mean_squared_error(y_train,y_train_pred),
                          "test":mean_squared_error(y_test,y_test_pred)}, name ='no encoding'))

# assemble the results
error_df = pd.concat(error_df, axis = 1)

error_df


# error values for encoded and not encoded data is very different. This mean, one hot encoding is over fitting the data. 

# ### Code breakdown

# #### rows and columns in X, y training and testing data

# In[64]:


X_train.shape, y_train.shape, X_test.shape , y_test.shape


# In[65]:


X_train_ohc.shape, y_train_ohc.shape, X_test_ohc.shape , y_test_ohc.shape


# more columns in one hot encoded data as each unique value is converted to a column

# #### Compare the indices to ensure they are identical

# In[66]:


X_train_ohc.index == X_train.index


# In[67]:


(X_train_ohc.index == X_train.index).all()


# In[68]:


X_train_ohc.index


# In[69]:


X_train.index


# ## Question 5
# 
# For each of the data sets (one-hot encoded and not encoded):
# 
# * Scale the all the non-hot encoded values using one of the following: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`.
# * Compare the error calculated on the test sets
# 
# Be sure to calculate the skew (to decide if a transformation should be done) and fit the scaler on *ONLY* the training data, but then apply it to both the train and test data identically.
# 

# In[79]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


scalers = {'standard': StandardScaler(),
           'minmax': MinMaxScaler(),
           'maxabs': MaxAbsScaler()}

training_test_sets = {
    'not_encoded': (X_train, y_train, X_test, y_test),
    'one_hot_encoded': (X_train_ohc, y_train_ohc, X_test_ohc, y_test_ohc)}


# Get the list of float columns, and the float data
# so that we don't scale something we already scaled. 
# We're supposed to scale the original data each time
mask = X_train.dtypes == float
float_columns = X_train.columns[mask]

# initialize model
LR = LinearRegression()

# iterate over all possible combinations and get the errors
errors = {}
for encoding_label, (X_train, y_train, X_test, y_test) in training_test_sets.items():
    for scaler_label, scaler in scalers.items():
        X_train_scaled = X_train.copy()  # copy because we dont want to scale this more than once.
        X_test_scaled = X_test.copy()
        X_train_scaled[float_columns] = scaler.fit_transform(X_train_scaled[float_columns])
        X_test_scaled[float_columns] = scaler.transform(X_test_scaled[float_columns])
        LR.fit(X_train_scaled, y_train)
        y_pred = LR.predict(X_test_scaled)
        key = encoding_label + ' - ' + scaler_label + 'scaling'
        errors[key] = mean_squared_error(y_test, y_pred)

errors = pd.Series(errors)
print(errors.to_string())
print('-' * 80)
for key, error_val in errors.items():
    print(key, error_val)


# ### Code breakdown

# In[71]:


scalers = {'standard': StandardScaler(),
           'minmax': MinMaxScaler(),
           'maxabs': MaxAbsScaler()
          }

for scaler_labels, scaler in scalers.items():
    print(scaler_labels)


# In[72]:


for encoding_label, data_tuple in training_test_sets.items():
    _X_train, _y_train, _X_test, _y_test = data_tuple
    print("Encoding Label:", encoding_label)


# ## Question 6
# 
# Plot predictions vs actual for one of the models.
# 

# In[96]:


# modify appearance
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

ax = plt.axes()
ax.scatter(y_test,y_pred)

ax.set(xlabel='Ground truth', 
       ylabel='Predictions',
       title='Ames, Iowa House Price Predictions vs Truth, using Linear Regression');

