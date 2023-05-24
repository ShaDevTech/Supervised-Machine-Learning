#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Foundation: Linear Regression
# 
# ## Part a: Regression Intro: Transforming Target 
# 

# ## Learning objectives
# 
# By the end of this lesson, you will be able to:
# 
# * Run Simple Linear Regression using Sklearn.
# * Apply transformations to make target variable more normally distributed for regression
# * Apply inverse transformations to be able to use these in a regression context
# 

# ## Import Required Libraries

# In[133]:


import pandas as pd #import pandas
import numpy as np # import numpy
import matplotlib.pyplot as plt # import matplotlib
import sklearn # import sklearn
from scipy.stats.mstats import normaltest # D'Agostino K^2 Test # import scipy for statistics
from scipy.stats import boxcox
from scipy.special import inv_boxcox


# ## Read boston data

# In[35]:


# data type is pickle
boston_data = pd.read_csv("boston_housing.csv")
boston_data.head(15)


# ## We want to predict MEDV (Median Value of household sold in the area)

# ## Number of rows and cols

# In[39]:


boston_data.info()


# In[41]:


boston_data.shape


# There are 506 rows and 14 columns in the dataset. 2 are interger and 12 are float datatypes

# ## Determining Normality
# 
# ### For Linear Regression, it is not essential to have target normally distributed although it might help. It is essential that errors are normally distributed.
# 
# Making our target variable normally distributed often will lead to better results
# 
# If our target is not normally distributed, we can apply a transformation to it and then fit our regression to predict the transformed values.
# 
# How can we tell if our target is normally distributed? There are two ways:
# 
# * Visually
# * Using a statistical test
# 
# **Visually**
# * Plotting a histogram:

# **We want to see column MEDV (target) is normally distributed or not**

# In[44]:


boston_data['MEDV'].hist()

# or boston_data.MEDV.hist()


# It does not look normally distributed due to right tail. We can use statistical test from scipy to veriy it

# ### from scipy.stats.mstats import normaltest - # D'Agostino K^2 Test
# 
# Without getting into Bayesian vs. frequentist debates, for the purposes of this lesson, the following will suffice:
# 
# This is a statistical test that tests whether a distribution is normally distributed or not. It isn't perfect, but suffice it to say:
# * This test outputs a "p-value". The higher this p-value is the closer the distribution is to normal.
# * Frequentist statisticians would say that you accept that the distribution is normal (more specifically: fail to reject the null hypothesis that it is normal) if p > 0.05.

# In[50]:


normaltest(boston_data.MEDV.values)


# p value is extremely low and less than 0.05. Therfore, we reject the null hypothesis and conclude MEDV (target variable) is not normally disctributed.

# Linear Regression assumes a normally distributed residuals which can be aided by transforming y variable. 
# Let's try some common transformations to try and get y to be normally distributed:
# 
# * Log
# * Square root
# * Box cox

# In[52]:


def plot_exponential_data():
    data = np.exp(np.random.normal(size=1000))
    plt.hist(data)
    plt.show()
    return data
    
def plot_square_normal_data():
    data = np.square(np.random.normal(loc =5,size=1000))
    plt.hist(data)
    plt.show()
    return data


# ## Testing Log

# The log transform can transform data that is significantly skewed right to be more normally distributed:

# In[54]:


data = plot_exponential_data()


# **original data**

# In[58]:


data[0:10]


# **log transformation is applied on the data**

# In[59]:


np.log(data)[0:10]


# In[62]:


plt.hist(np.log(data))


# **Apply transformation to the boston data**

# In[64]:


log_medv = np.log(boston_data.MEDV)
log_medv.hist()


# **test statistaically**

# In[65]:


normaltest(log_medv)


# **Closer but still not normally distributed**

# ## Testing square root transformation
# The square root transformation is another transformation that can transform non-normally distributed data into normally distributed data:

# In[67]:


data= plot_square_normal_data()


# slightly skewed to right

# In[69]:


plt.hist(np.sqrt(data))


# **Apply the square root transformation to the Boston data target and test whether the result is normally distributed.**

# In[70]:


sqrt_medv = np.sqrt(boston_data.MEDV)
plt.hist(sqrt_medv)


# In[71]:


normaltest(sqrt_medv)


# **closer but still not normal**

# ## Testing box cox transformation 
# 
# The box cox transformation is a parametrized transformation that tries to get distributions "as close to a normal distribution as possible".
# 
# It is defined as:
# 
# $$ \text{boxcox}(y_i) = \frac{y_i^{\lambda} - 1}{\lambda} $$
# 
# You can think of as a generalization of the square root function: the square root function uses the exponent of 0.5, but box cox lets its exponent vary so it can find the best one.
# 

# In[78]:


bc_result = boxcox(boston_data.MEDV) # it gives two results, 1 is transformed array and seconds is lambda value
boxcox_medv = bc_result[0]
lam = bc_result[1]
lam


# in square root, it is 0.5 but best value is 0.21 to get normally distributed target variable

# In[79]:


plt.hist(boxcox_medv)


# In[80]:


normaltest(boxcox_medv)


# p value is greater than 0.05 and we fail to reject the $ \text H_{0}$ null hypothesis.
# 
# Now that we have a normally distributed y-variable, let's try a regression!

# ## Testing Regression

# In[86]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score #it tell how good is our model
from sklearn.model_selection import train_test_split # split the data for training and testing
from sklearn.preprocessing import (StandardScaler,PolynomialFeatures) #bring values to same scale / add anotter degree to deal with complexity


# ### initiate the model

# In[90]:


lr = LinearRegression()


# after model is fit, it will have predict method so it is called predictor

# ### Create X and y

# In[88]:


y_col = 'MEDV'
X = boston_data.drop(y_col,axis =1) # keep all columns except MEDV
y = boston_data[y_col]


# In[91]:


X


# In[92]:


y


# In[99]:


X.shape


# There are 13 columns and 506 rows in data

# ## Create polynomial features

# In[101]:


pf = PolynomialFeatures(degree=2, include_bias=False) # drop B0 term as Linear regression comes up on its own with it
X_pf = pf.fit_transform(X)


# In[102]:


X_pf


# X_pf is an array. 

# In[103]:


X_pf.shape


# There are lot more columns as it invovles interation and squared terms

# ## Train test Split

# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X_pf,y,test_size=0.3, random_state=72018)


# In[114]:


X_train, X_test, y_train, y_test


# In[115]:


X_train.shape


# 70% of data is training data from X_pf

# ## Fit StandardScaler on X_train

# In[119]:


s = StandardScaler()
X_train_s = s.fit_transform(X_train)


# In[120]:


X_train_s.shape


# ## Apply box-cox transformation to transform non normal to normal target variable

# In[126]:


# we will apply boxcox

bc_result = boxcox(y_train)
y_train_bc = bc_result[0] # transofmed values
y_train_lam = bc_result[1] #lambda value


# In[124]:


y_train_lam 


# As before, we'll now:
# 
# 1. Fit regression
# 1. Transform testing data
# 1. Predict on testing data
# 

# In[125]:


y_train_bc.shape


# **StandardScaler must be done after train test split and Polynomial Feature can be done before it**

# ## Fit regression

# In[128]:


lr.fit(X_train_s, y_train_bc) # fit model on the data
X_test_s = s.transform(X_test) # transform test set using fit we defined on training set
y_pred_bc = lr.predict(X_test_s)


# In[129]:


X_test_s.shape


# In[130]:


y_test


# In[131]:


y_pred_bc


# We have transfomed Y using boxcox so scaling is off. Therefore, we have to inverse transformation

# ## Inverse Transformation

# Every transformation has an inverse transformation. The inverse transformation of $f(x) = \sqrt{x}$ is $f^{-1}(x) = x^2$, for example. Box cox has an inverse transformation as well: notice that we have to pass in the lambda value that we found from before:
# 

# In[143]:


bc_result = boxcox(boston_data.MEDV)
boxcox_medv = bc_result[0]
boxcox_lam= bc_result[1]
print(boston_data.MEDV[:10])
print(boxcox_medv[0:10])


# In[140]:


inv_boxcox(boxcox_medv,boxcox_lam)[:10]


# same as above

# ## Apply inverse transformation on predicted y

# In[152]:


y_pred_tran = inv_boxcox(y_pred_bc,y_train_lam)
y_pred_tran[0:10]


# ## find $ r^{2}$

# In[153]:


r2_score(y_pred_tran,y_test)


# ## Find $ r^{2}$ without boxcox applied on y. Is it higher or lower?

# In[154]:


lr = LinearRegression()
lr.fit(X_train_s,y_train)
y_pred = lr.predict(X_test_s)
r2_score(y_pred,y_test)


# $ r^{2}$ score is lower when no box-cox transformation is applied. Therefore, coming up with normal distribution for outcome variable improved the score

# # Machine Learning Foundation: Linear Regression
# 
# ## Part b: Regression analysis on a car price dataset
# 

# # Linear Regression
# 
# If you are consulting an automobile company, you are trying to understand the factors that influence the sale price of the cars. Specifically, **which factors drive the car prices up? And how accurately can you predict the sale price based on the car's features?**
# 
# In this notebook, we will perform a simple linear regression analysis on a car price dataset, show how this prediction analysis is done and what are the important assumptions that must be satisfied for linear regression. We will also look at different ways to transform our data.
# 
# ## Objectives
# 
# After completing this lab you will be able to:
# 
# *   Select the significant features based on the visual analysis
# *   Check the assumptions for Linear Regression model
# *   Apply the Linear Regression model and make the predictions
# *   Apply the pipelines to transform the data
# 

# ## Import the required libraries

# In[157]:


import pandas as pd # pandas for managing data
import numpy as np # numpy for mathemtaical operations
import seaborn as sns # for visualizing the data.
import matplotlib.pyplot as plt # for visualizing the data.

from scipy.stats import boxcox # for statistcal computations
from scipy.stats.mstats import normaltest

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (r2_score, mean_squared_error)
from sklearn.preprocessing import (StandardScaler,PolynomialFeatures)
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits, load_wine


# ## Reading and understanding our data
# 
# For this lab, we will be using the car sales dataset, hosted on IBM Cloud object storage. The dataset contains all the information about cars, the name of the manufacturer, the year it was launched, all car technical parameters, and the sale price.

# In[159]:


URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv'
carprice_data = pd.read_csv(URL)
carprice_data.head()


# find info about the data

# In[161]:


carprice_data.info()


# * According to the output above, we have 205 entries or rows, as well as 26 features. 
# * The "Non-Null Count" column shows the number of complete entries. If the count is 205 then there is no missing values for that particular feature. 
# * The 'price' is our target, or response variable, and the rest of the features are our predictor variables.
# * We also have a mix of numerical (8 int64 and 8 float64) and object data types (10 object).
# 
# **The describe() function will provide the statistical information about all numeric values.**

# In[162]:


carprice_data.describe()


# ## Data Cleaning and Wrangling
# 
# * Check for missing values
# * Check for duplicated values
# * Check for typos

# In[165]:


carprice_data.isnull().sum()


# No missing value in the dataset

# Also, check for any duplicates by running duplicated() function through 'car_ID' records, since each row has a unique car ID value.

# In[169]:


sum(carprice_data.duplicated('car_ID')) == 0


# No duplicated car_ID

# Next, let's look into some of our object variables first. Using unique() function, we will describe all categories of the 'CarName' attribute.

# In[172]:


carprice_data['CarName'].unique()


# We can see that the 'CarName' includes both the company name (brand) and the car model. Next, we want to split a company name from the model of a car, as for our model building purpose, we will focus on a company name only.

# In[183]:


carprice_data['CarName'].str.split(' ')


# In[187]:


carprice_data['CarName'].str.split(' ').str.get(0).str.lower()


# In[188]:


carprice_data['brand'] = carprice_data['CarName'].str.split(' ').str.get(0).str.lower()


# Let's view all the unique() brands now.

# In[189]:


carprice_data['brand'].unique()


# There are some typos in the names of the cars, so they should be corrected.

# In[190]:


carprice_data['brand'] = carprice_data['brand'].replace(['vokswagen','vw'],'volkswagen')
carprice_data['brand'] = carprice_data['brand'].replace('toyouta','toyota')
carprice_data['brand'] = carprice_data['brand'].replace('porcshce','porsche')
carprice_data['brand'] = carprice_data['brand'].replace('maxda','mazda')


# In[191]:


carprice_data['brand'].unique()


# Let's plot and sort the total number of Brands.

# In[193]:


carprice_data['brand'].value_counts()


# In[219]:


fig, ax = plt.subplots(figsize=(10,5))
plt1 = sns.countplot(x=carprice_data['brand'], order = pd.value_counts(carprice_data['brand']).index)
plt1.set(xlabel='Brand', ylabel='Count of cars')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()


# Count plot: Displays the count or frequency of observations in each category.
# 
# Bar plot: Displays aggregated numerical data (mean, sum, etc.) for different categories.

# We can drop 'car_ID', 'symboling', and 'CarName' from our data frame, since they will no longer be needed.

# In[220]:


carprice_data.drop(['car_ID', 'symboling','CarName'], axis=1, inplace= True)


# In[222]:


carprice_data.info()


# In[223]:


carprice_data.shape


# In[215]:


#If you need to save this partially processed data, uncomment the line below.
#carprice_data.to_csv('cleaned_car_data.csv',index=False)


# ## explore any (or all) object variables of your interest.

# In[237]:


carprice_data.head(2)


# In[234]:


carprice_data.fueltype.unique()


# In[236]:


carprice_data['enginelocation'].value_counts()


# Next, we need to engineer some features, for better visualizations and analysis. We will group our data by 'brand', calculate the average price for each brand. We rename price column to brand_avg_price and split these prices into 3 bins: 'Budget', 'Mid-Range', and 'Luxury' cars, naming the newly created column - the 'brand_category'.

# By default in pandas, when we perform groupby() operation, grouped column becomes new index. The as_index paramater allows to specify where to want to keep grouped columns as index or not. If not, we we have a 0,1,2,....index created

# In[259]:


data_comp_avg_price= carprice_data[['brand', 'price']].groupby('brand', as_index=False).mean().rename(columns={'price':'brand_avg_price'})

#data_comp_avg_price.sort_values(by='brand_avg_price', ascending=False)
data_comp_avg_price


# **Combine original data with the average price data for each brand. We look for brand name in original data, write average price for it from our above list. Since in original data, brands are repeated, so we'll have duplicated values in multiple row for average price**

# In[262]:


carprice_data = carprice_data.merge(data_comp_avg_price, on='brand')
carprice_data


# **We will now check the statistics of our average car price per car brand.**

# In[265]:


carprice_data['brand_avg_price'].describe()


# **split these prices into 3 bins: 'Budget', 'Mid-Range', and 'Luxury' cars, naming the newly created column - the 'brand_category'.**

# In[270]:


carprice_data['brand_category'] = carprice_data['brand_avg_price'].apply(lambda x: "Budget" if x < 10000
                                       else ("Mid_Range" if 10000<= x <20000
                                       else "Luxury")
                                      )


# In[271]:


carprice_data.head()


# ## Exploratory Data Analysis

# **List catgeorical variables as a list and use sorted(...) function to sort the elements of a list in alphabetical order.**

# In[276]:


sorted(carprice_data.select_dtypes('object').columns.to_list())


# We will use the `boxplot()` function on the above mentioned categorical variables, to display the mean, variance, and possible outliers, with respect to the price.

# In[314]:


# List of variables for boxplots
cat_var = ['aspiration', 'brand','brand_category','carbody', 'cylindernumber', 'doornumber', 'drivewheel', 'enginelocation',
 'enginetype', 'fuelsystem', 'fueltype']

# Set up the figure and subplots
fig,axes = plt.subplots(nrows= 6, ncols = 2, figsize=(10,20))

# Iterate through variables and create boxplots
for i, var in enumerate(cat_var):
    row = i // 2 # determine row index for the subplot- integer division- return quotient- whole integer
    col = i % 2  # determine column index for the subplot - moduluo operation- return remainder
    ax = axes[row,col] # select the appropriate subplot

    # Create the boxplot
    sns.boxplot(x=var, y='price', data= carprice_data, ax=ax)
    
    # rotate x-axis lables
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # get_xticklabels() retrieves the current x-axis tick labels, 
    # set_xticklabels() allows you to set new x-axis tick labels for the subplot.

# remove unused subplot
fig.delaxes(axes[5,1])
# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# **Next, let's view the list of top features that have high correlation coefficient. The `corr()` function calculates the Pearson's correlation coefficients with respect to the 'price'.**

# In[317]:


#  to specify that only numeric columns should be used for correlation calculations.
corr_matrix = carprice_data.corr(numeric_only=True)
corr_matrix


# **see correlation for price columns only and sort it in highest to lowest**

# In[321]:


corr_matrix['price'].sort_values(ascending=False)


# **After creating correlation matrix, We can also use the heatmap() or pairplot() to further explore the relationship between all features and the target variables.**

# ### Plot heatmap, add labels and color map as 'coolwarm' to indicate the colors to be used for positive and negative correlations.

# In[331]:


plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# add title
plt.title("Price vs features")


# ### Plot pairplot
# 
# It shows scatter plot for each numerical variable against other numerical variables and histograms on the diagonal

# In[340]:


plt.figure(figsize=(10,20))

# plot the pairplot
pairplot = sns.pairplot(carprice_data)

# set the y axis label font size
for ax in pairplot.axes.flat:
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel,fontsize=12, rotation = 90)
        


# **axes represent 2 d array of axes where each corresponds to specifc subplot in pairplot grid. We convert it to 1 d , individual axes using flat**

# ## **Testing Assumptions for Linear Regression**
# 
# Since we fit a linear model, we assume that the relationship between the target (price) and other features is linear.
# 
# We also expect that the errors, or residuals, are pure random fluctuations around the true line, in other words, the variability in the response (dependent) variable doesn't increase as the value of the predictor (independent) variable increases. This is the assumption of equal variance, also known as **Homoscedasticity**.
# 
# We also assume that the observations are independent of one another (no **multicollinearity**), and there is no correlation between the sequential observations.
# 
# If we see one of these assumptions in the dataset are not met, it's more likely that the other ones, mentioned above, will also be violated. Luckily, we can check and fix these assumptions with a few unique techniques.
# 
# Now, let's briefly touch upon each of these assumptions in our example.
# 

# <details>
# <summary> explaination in simple words of concept Homoscedasticity and multicollinearity </summary>
#     
#     Imagine you are playing a game where you have to throw a ball into a basket. The distance from which you throw the ball represents the predictor variable, and the number of points you score represents the response variable. Now, let's consider two important concepts related to this game.
# 
# <b>Equal Variance or Homoscedasticity</b>
# Imagine you are throwing the ball from different distances and recording the number of points you score each time. If the variability in your scores remains the same, regardless of the distance from which you throw the ball, then we can say that the scores have equal variance or homoscedasticity. It means that the amount of variation in your scores doesn't increase or decrease as you change the distance. In other words, the game is fair and the results don't depend on how far you throw the ball.
# 
# <b>Independence and No Correlation</b>:
# Now, let's consider that you are playing this game with a group of friends. Each person takes turns throwing the ball, and you record the scores for everyone. It's important to note that each person's score is independent of the other person's score. This means that one person's score doesn't affect another person's score. Additionally, there is no correlation between the scores of consecutive players. It means that the scores of the players don't follow any particular pattern or trend. Each person's score is unique and unrelated to what others have scored.
# 
# In statistical analysis, we make similar assumptions when studying relationships between variables. We assume that the variability in the response variable (e.g., scores) doesn't change as the predictor variable (e.g., distance) changes. We also assume that the observations are independent of each other, meaning they don't influence each other's values, and there is no correlation or pattern between consecutive observations. These assumptions help us make valid conclusions and draw meaningful insights from our analysis, just like understanding the fairness of a game by considering equal variance and independence.
#     
# </details>
# 

# ### 1. Linearity Assumption
# 
# Linear regression needs the relationship between independent variable and the dependent variable to be linear. We can test this assumption with some scatter plots and regression lines.
# 
# We will start with the 'enginesize', 'curb weight' and 'horsepower' features.
# 
# **Build scatter and regression plot**
# 

# In[369]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,3), sharey=False)
sns.scatterplot(x='enginesize', y='price', data=carprice_data, ax= axes[0])
sns.regplot(x='enginesize', y='price', data=carprice_data, ax= axes[0])

sns.scatterplot(x='horsepower', y='price', data=carprice_data, ax= axes[1])
sns.regplot(x='horsepower', y='price', data=carprice_data, ax= axes[1])

sns.scatterplot(x='curbweight', y='price', data=carprice_data, ax= axes[2])
sns.regplot(x='curbweight', y='price', data=carprice_data, ax= axes[2])

# remove label from 2 and 3 plot
axes[1].set_ylabel(' ')
axes[2].set_ylabel(' ')

# adjust horizontal space between plots
plt.subplots_adjust(wspace=0.8)

# show plots
plt.show()


# Based on above plots, we can say 'enginesize' and 'horsepower' features have linear relationshwip with price. Points are close to line

# In[392]:


test_var = ['enginesize', 'horsepower', 'curbweight']

# Define plot and axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 4))
axes = np.ravel(axes)  # Convert axes to 1-dimensional array

for i, var in enumerate(test_var):
    # Determine the row and column index for the subplot
    row = i // 3  # Integer division to get row index
    col = i % 3   # Modulo division to get column index

    # Select the appropriate subplot
    ax = axes[i] # array is 1d

    # Build scatter plot
    sns.scatterplot(x=var, y='price', data=carprice_data, ax=ax)
    sns.regplot(x=var, y='price', data=carprice_data, ax=ax)
    
    # remove labels from 2 and 3 y axis of
    if i>0:
        ax.set_ylabel(" ")  # Set an empty string as the y-label

# remove unused plots
for j in range(len(test_var), len(axes)):
    fig.delaxes(axes[j])

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# <details>
#     <b> ravel() </b> In this specific case, it is not mandatory to use np.ravel(axes) to convert the axes to a 1-dimensional array. You can directly access the individual axes using axes[i] without the need for flattening.
# 
# However, if you were to use a different layout for subplots, such as having more than one row or column, or a combination of both, then using np.ravel(axes) would ensure that you can access the axes correctly in a 1-dimensional manner regardless of the layout. It provides flexibility and makes the code more generalizable.
# 
# So, in short, if you have a fixed layout of 1 row and 3 columns as in your example, you can directly use ax = axes[i]. But if you want to make your code more flexible and adaptable to different subplot arrangements, it is recommended to use np.ravel(axes).
#     
#     <b> delete unused plot</b>
#     In the provided code, the loop for j in range(len(test_var), len(axes)): is used to iterate over the indices of the unused plots in the axes array.
# 
# Here's an explanation of each part:
# 
# len(test_var) returns the length of the test_var list, which is the number of variables used for plotting.
# len(axes) returns the total number of subplots in the axes array.
# range(len(test_var), len(axes)) generates a range of indices starting from the length of test_var up to the length of axes. This range represents the indices of the unused plots in the axes array.
# for j in range(len(test_var), len(axes)): iterates over these indices.
# Inside the loop, fig.delaxes(axes[j]) is used to delete the unused plots from the figure. It accesses the subplot at index j in the axes array and removes it from the figure.
# 
# By using this loop, any extra plots that were not assigned a variable from test_var will be removed from the figure, ensuring that only the desired plots are displayed.
# </details>

# ### 2. *Homoscedasticity*
# 
# The assumption of *homoscedasticity* (constant variance), is crucial to linear regression models. *Homoscedasticity* describes a situation in which the error term or variance or the "noise" or random disturbance in the relationship between the independent variables and the dependent variable is the same across all values of the independent variable. In other words, there is a constant variance present in the response variable as the predictor variable increases. If the "noise" is not the same across the values of an independent variable, we call it *heteroscedasticity*, opposite of *homoscedasticity*.
# 

# <details> 
# 
#     Imagine you have a bunch of toy cars, and you want to understand how their speed (independent variable) affects their distance traveled (dependent variable). You decide to do an experiment where you measure the distance each car travels at different speeds.
# 
# Now, let's say all the cars are the same and should behave similarly. If everything goes perfectly, you would expect that for every increase in speed, the distance traveled would increase by the same amount. This means that if one car goes twice as fast as another car, it should also travel twice as far.
# 
# But sometimes, things don't go perfectly. Imagine if some cars started acting differently as their speed increased. For example, one car might start to slow down or veer off course when it goes really fast, while another car might stay on track and keep going smoothly. This would create a difference in the distance traveled for the same increase in speed.
# 
# When we talk about homoscedasticity, we mean that all the cars behave the same way as their speed increases. In other words, the variability or "noise" in the relationship between speed and distance is the same for all the cars. This would be like all the cars having the same level of randomness or unpredictability in how far they can travel.
# 
# On the other hand, if the cars start behaving differently as their speed increases, we call it heteroscedasticity. It means that some cars have more randomness or variability in their distance traveled compared to others.
# 
# In summary, when we assume homoscedasticity, we expect that all the cars, or data points, have the same level of variability in how far they can travel for each increase in speed. This assumption is important because it helps us make accurate predictions and understand the relationship between variables in linear regression models.
# </details>

# ### Plotting redisual/error plots

# In[394]:


sns.residplot(x='enginesize', y='price', data= carprice_data)


# From the above plot, we can tell the error variance across the true line is dispersed somewhat not uniformly, but in a funnel like shape. So, the assumption of the homoscedasticity is more likely not met.

# In[399]:


carprice_data.select_dtypes(['float','int']).columns.to_list()


# In[408]:


num_var = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'boreratio', 'stroke',
           'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']

# Set up figure and labels
fig, axes = plt.subplots(figsize=(20, 40), nrows=7, ncols=2)

# Iterate through each variable
for i, var in enumerate(num_var):
    row = i // 2  # Integer division for row axes
    col = i % 2   # Modulo division for column axes
    ax = axes[row, col]
    
    # Plot residual plot
    sns.residplot(x=var, y='price', data=carprice_data, ax=ax)

# Remove unused plots
for j in range(len(num_var), len(axes.flatten())):
    fig.delaxes(axes.flatten()[j])

# Adjust the layout and display the plot
plt.tight_layout()
plt.show()


# No variable shows random distibrution. All have some pattern.Therefore, the assumption of the homoscedasticity is more likely not met.

# ### 3. Normality
# 
# The linear regression analysis requires the dependent variable, 'price', to be normally distributed. A histogram, box plot, or a Q-Q-Plot (probability plot) can check if the target variable is normally distributed. The goodness of fit test, e.g., the Kolmogorov-Smirnov test can check for normality in the dependent variable.

# In[453]:


def test_normality(data,features):
    # import required libraires
    from scipy import stats 
    import matplotlib.gridspec as gridspec
    
    # make canvas look stylish
    import matplotlib.style as style
    style.use('fivethirtyeight')
    
    # create a customized chart
    fig = plt.figure(figsize=(12,8)) # create a figure/blank board with grid of rows and cols 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig) ## creating a grid of 3 cols and 3 rows. 
    
    ## build historgram
    ax1 = fig.add_subplot(grid[0,:2]) # place chart on top row and first 2 columns in the grid
    #set the title
    ax1.set_title("Histogram")
    #plot the histogram
    sns.histplot(data.loc[:, features], kde=True, ax=ax1)
    
    # add some vertical space
    plt.subplots_adjust(hspace=0.6)
    
    ## build a probability plot Q-Q plot
    ax2 = fig.add_subplot(grid[1,:2])
    #set the title
    ax2.set_title('Q-Q Plot')
    #plot the probability plot
    stats.probplot(data.loc[:,features], plot=ax2)
    
    # add some horizontal space
    plt.subplots_adjust(wspace=0.4)
    
    ## build a box plot
    ax3= fig.add_subplot(grid[:,2])
    #set the title
    ax3.set_title("Box Plot")
    #plot the box plot
    sns.boxplot(carprice_data.loc[:,features], ax=ax3, orient='v')
    

test_normality(carprice_data,'price')


# These three charts above can tell us a lot about our target variable:
# 
# - our target variable 'price' is not normally distributed
# - our target variable 'price' is right - skewed.
# - As per box plot, there are outliers present in the data and 75% of the cars has prices around 16000.Max value is vary far from 75% quanitle statistic.
# 
# Next, we will perform the log transformation to correct our target variable and to make it more normally distributed.

# save our data before we apply any transformation

# In[454]:


archived_carprice_data = carprice_data.copy()


# ## Try applying log transformation to correct our target variable and to make it more normally distributed.

# We can also check statistically if the target is normally distributed, using normaltest() function. If the p-value is large (>0.05), the target variable is normally distributed.

# In[456]:


normaltest(carprice_data['price'].values)


# p value is super small. Thus, it is not normally distributed.

# In[457]:


carprice_data['log_price'] = np.log(carprice_data['price'])


# In[458]:


test_normality(carprice_data,'log_price')


# In[459]:


normaltest(carprice_data.log_price.values)


# data looks much more normally distributed. It looks more symmetical but it is still not normally distributed.

# ## Try applying square root transformation to correct our target variable and to make it more normally distributed.

# In[461]:


carprice_data['sqrt_price'] = np.sqrt(carprice_data.price)


# In[462]:


test_normality(carprice_data,'sqrt_price')


# In[463]:


normaltest(carprice_data.sqrt_price.values)


# It has become worse than log transformation.

# ## Try Box Cox Transformation to correct our target variable and to make it more normally distributed.

# In[471]:


bc_result = boxcox(carprice_data.price)
bc_price = bc_result[0]
lam1_price = bc_result[1]


# In[475]:


normaltest(bc_price)


# Higher the p value, closer to normal distribution. It is still less than 0.05 so it is still not normally distributed.

# ### 4. *Multicollinearity*
# 
# *Multicollinearity* is when there is a strong correlation between the independent variables. Linear regression or multilinear regression requires independent variables to have little or no similar features. *Multicollinearity* can lead to a variety of problems, including:
# 
# *   The effect of predictor variables estimated by our regression will depend on what other variables are included in our model.
# *   Predictors can have widely different results depending on the observations in our sample, and small changes in samples can   result in very different estimated effects.
# *   With very high multicollinearity, the inverse matrix, the computer calculations may not be accurate.
# *   We can no longer interpret a coefficient on a variable because there is no scenario in which one variable can change without a conditional change in another variable.
# 
# Using `heatmap()` function is an excellent way to identify whether there is *multicollinearity* present or not. The best way to solve for *multicollinearity* is to use the regularization methods like *Ridge* or *Lasso*, which we will introduce in the **Regularization** lab.
# 

# <details>
#     Imagine you have a toy car collection, and you want to know which factors make a toy car more expensive. You decide to collect data on different features of the toy cars, such as the size, color, brand, and weight. You think that these features might affect the price of the toy cars.
# 
# Now, let's say you start analyzing the data and find that the size and weight of the toy cars are strongly correlated. This means that when the size of the car increases, the weight also tends to increase. It's like saying that bigger cars tend to be heavier.
# 
# In this case, multicollinearity is present between the size and weight variables. It can cause some issues when trying to understand the individual effects of these variables on the price. For example, if you try to figure out how much the size of the car affects the price while considering the weight, it might be challenging because these two factors are strongly related.
# 
# To identify multicollinearity, you can use a heatmap, which shows how strongly each pair of variables is correlated. If you notice that some variables have high correlations, it indicates multicollinearity.
# 
# To handle multicollinearity, you can use techniques like Ridge or Lasso, which help in reducing the impact of correlated variables. These techniques help in finding the best combination of variables that contribute to predicting the price without being overly influenced by multicollinearity.
# 
# So, to sum it up, multicollinearity occurs when some variables are strongly related to each other. It can make it difficult to understand the individual effects of those variables. Using techniques like Ridge or Lasso can help manage multicollinearity and find the most important factors affecting the price of toy cars.
# 
# </details>

# There are various colormap (cmap) options that you can use for sns.heatmap() function. Some popular ones are:
# 
# - "viridis": A perceptually uniform colormap that ranges from deep blue to vibrant yellow.
# - "plasma": A colormap that ranges from dark purple to bright yellow.
# - "coolwarm": A diverging colormap that ranges from cool blue to warm red.
# - "RdYlGn": A diverging colormap that ranges from red to yellow to green.
# - "BuPu": A sequential colormap that ranges from blue to purple.
# - "Greens": A sequential colormap that ranges from light green to dark green.
# - "Oranges": A sequential colormap that ranges from light orange to dark orange.

# In[492]:


# find correlation
plt.figure(figsize=(20,10))
corr_matrix = carprice_data.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True,cmap ='viridis')


# List of significant variables after Exploratory Data Analysis :
# 
# Numerical:
# 
# *   Curbweight
# *   Car Length
# *   Car width
# *   Engine Size
# *   Boreratio
# *   Horse Power
# *   Wheel base
# *   City mpg (miles per gallon)
# *   Highway mpg (miles per gallon)
# 
# Categorical:
# 
# *   Engine Type
# *   Fuel type
# *   Car Body
# *   Aspiration
# *   Cylinder Number
# *   Drivewheel
# *   Brand Category
# 

# save all significant features into a dataframe named `selected`

# In[510]:


columns=['log_price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase', 'brand_category',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 'carlength','carwidth','citympg','highwaympg']



selected = carprice_data[columns]
selected.info()


# **Categorical Columns**

# In[511]:


categorical_columns = selected.select_dtypes('object').columns.to_list()
categorical_columns 


# **Numerical Columns**

# In[513]:


numerical_columns = sorted(selected.select_dtypes(['int64', 'float64']).columns.to_list())
##numeric_columns=list(set(columns)-set(categorical_columns))
numeric_columns


# ## database is now called as 'selected'. Split the data into X and y

# In[531]:


X= selected.drop('log_price', axis=1)
y = selected['log_price']


# In[532]:


X.head()


# In[533]:


y.head()


# ## Apply One-Hot encoding to the categorical data

# <details>
# Child: Hey! Do you know what "one hot encoding" is?
# 
# Child: No, I've never heard of it. What is it?
# 
# You: Well, imagine you have a box of different fruits, like apples, bananas, and oranges. Each fruit has its own special characteristics, right?
# 
# Child: Yeah, apples are red and crunchy, bananas are yellow and smooth, and oranges are orange and juicy!
# 
# You: Exactly! Now, let's say we want to put these fruits into separate boxes based on their colors. We want one box for red fruits, another box for yellow fruits, and another box for orange fruits. How can we do that?
# 
# Child: Hmm, we can look at each fruit and decide which box it belongs to based on its color!
# 
# You: That's right! We do the same thing with data in computers. Sometimes, we have information that can be in different categories, just like the fruits. But computers like numbers more than colors or names, so we need a way to represent the categories as numbers.
# 
# Child: How do we do that?
# 
# You: We use something called "one hot encoding." It's like making special boxes for each category, just like we did for the fruits. Instead of putting the fruits directly into the boxes, we use numbers to represent them. We make a new box for each category, and if a fruit belongs to that category, we put a "1" in the box. If it doesn't belong to the category, we put a "0" in the box.
# 
# Child: Oh, I get it! So, for example, we can have a box for "red" and put a "1" in it if a fruit is red, and a "0" if it's not red?
# 
# You: That's exactly it! With one hot encoding, we can convert different categories into numbers that computers can understand. It helps the computers analyze and make decisions based on the categories.
# 
# Child: Wow, that's cool! It's like giving the fruits their own special boxes, but in a computer way!
# 
# You: Yes, exactly! It's like giving the data its own special boxes so that the computer can work with it easily.
# </details>

# ### Load required libraries

# In[517]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# To perform one-hot encoding, we use the `ColumnTransformer` class, this allows different columns or column subsets to be transformed separately.
# 
# The input is as follows:
# 
# The `transformerslist` is the number of tuples.
# The list of `(name, transformer, columns)` tuples specify the transformer objects to be applied to the subsets of the data.
# 
# *   name: name of the operation that can be used later
# *   `transformer`: estimator must support fit and transform, in this case we will use `OneHotEncoder()`
# *   `‘drop’`: to  drop the columns
# *   `‘passthrough’`: to pass them through untransformed data
# *   `remainder`: specifies the columns that are not transformed are being set to `passthrough`. They are  combined in the output, and the non-specified columns are dropped.
# 
# We apply `fit_transform()` to transform the data.
# 

# <details>
# Child: Now that we understand one hot encoding, let's learn about something called the ColumnTransformer class. It helps us perform the one hot encoding process on different parts of our data separately.
# 
# Child: What do you mean by "different parts"?
# 
# You: Good question! Imagine you have a big box full of different things like toys, books, and clothes. Each item in the box is different and needs to be treated differently. The ColumnTransformer class helps us do that with our data.
# 
# Child: So, it's like having special instructions for each part of the data?
# 
# You: Exactly! With the ColumnTransformer, we can give special instructions for each part, or subset, of our data. We use a list of tuples to specify these instructions.
# 
# Child: Tuples? What are those?
# 
# You: Tuples are like small packages that hold different pieces of information together. In our case, each tuple in the list has three things: a name, a transformer, and the columns it applies to.
# 
# Child: Can you explain each part of the tuple?
# 
# You: Of course! The name is like a special label we give to the instructions. It helps us remember what we did later on. The transformer is like a special tool that we use to transform or change the data. In this case, we use the OneHotEncoder as our transformer. And the columns part tells us which specific columns in our data we want to apply the transformation to.
# 
# Child: Oh, I see! So, we can apply different transformations to different parts of the data!
# 
# You: Yes, exactly! We can specify different transformations for different parts of our data. And if we want to keep some parts of the data unchanged, we can use the special words "drop" or "passthrough".
# 
# Child: What do "drop" and "passthrough" mean?
# 
# You: "Drop" means we want to remove those columns from our data after applying the transformation. "Passthrough" means we want to keep those columns as they are, without any changes.
# 
# Child: That's cool! So, it's like having a magic box that can do different things to different items inside!
# 
# You: That's a great way to think about it! The ColumnTransformer helps us perform different transformations on different parts of our data, just like a magic box with special instructions for each item. And then we use the fit_transform() function to apply all these transformations to our data.
# 
# </details>

# In[534]:


one_hot = ColumnTransformer(transformers=[("one_hot", OneHotEncoder(), categorical_columns) ],remainder="passthrough")
X=one_hot.fit_transform(X)
type(X)


# <details>
# <summary>fit_transform() vs transform()</summary>
# Child: You might be wondering about the difference between fit_transform() and transform(). They are both methods used in machine learning to modify our data, but they are used in slightly different ways.
# 
# You: Imagine you have a coloring book with different pictures. When you want to color one of the pictures, you first need to learn the colors and where to use them. In machine learning, this process of learning is called "fitting" the data.
# 
# Child: So, "fitting" is like learning the colors?
# 
# You: Exactly! When we use fit_transform(), we are teaching the machine learning model how to transform the data based on the patterns it finds in the training data. It learns what transformations to apply to the data.
# 
# Child: And what about "transform"?
# 
# You: Great question! After we have taught the model how to transform the data using fit_transform(), we can then use the transform() method to apply those transformations to new, unseen data. It's like taking what we learned about coloring and using that knowledge to color new pictures without having to learn the colors again.
# 
# Child: So, "fit_transform()" is for teaching the model, and "transform()" is for using what it learned?
# 
# You: That's exactly right! "fit_transform()" is used during the training phase to teach the model, and "transform()" is used during the prediction phase to apply the learned transformations to new data.
# 
# Child: I get it now! It's like learning how to color and then using that knowledge to color different pictures without starting from scratch each time.
# 
# You: Exactly! You're getting the hang of it! fit_transform() is for teaching the model, and transform() is for using what it learned to transform new data.
# 
# </details>

# - We see the output is a NumPy array, so let's get the feature names from the one_hot object using `get_feature_names_out()` method. 
# - The output will be the feature name with the prefix of the name of the transformer. 
# - For one-hot encoding, the prefix will also include the name of the column that generated that feature.

# <details>
# Child: Have you ever seen a collection of different fruits?
# 
# You: Yes, there are many different kinds of fruits like apples, oranges, and bananas.
# 
# Child: Exactly! Now, let's imagine we want to create a list of all the fruits we have seen. But instead of just writing down the names of the fruits, we want to include some extra information about each fruit.
# 
# You: So, for each fruit, we want to know its color, size, and taste, right?
# 
# Child: Yes, that's right! One way we can do this is by using something called "one-hot encoding." It's like creating a special code for each fruit that tells us its color, size, and taste.
# 
# You: Oh, that sounds interesting! How does it work?
# 
# Child: Well, let's say we have a list of fruits, and we want to encode their colors. We start by creating a separate column for each possible color, like red, yellow, and green. If a fruit is red, we put a 1 in the red column, and if it's not red, we put a 0. We do the same for the other colors.
# 
# You: So, if we have an apple that is red, we would have a 1 in the red column and 0 in the yellow and green columns?
# 
# Child: That's right! And if we have an orange that is yellow, we would have 0 in the red column, 1 in the yellow column, and 0 in the green column.
# 
# You: I see! So, each fruit will have a unique combination of 0s and 1s in the color columns.
# 
# Child: Exactly! And by doing this, we can represent the color of each fruit using these special codes. We can do the same for the size and taste as well.
# 
# You: That's really cool! So, one-hot encoding helps us represent different characteristics of fruits using special codes.
# 
# Child: Yes, you got it! And when we use the get_feature_names_out() method, we can get the names of these special codes. The names will include the prefix of the transformer (like "color_") and the name of the column that generated that feature.
# 
# You: Ah, so if we had a column called "color" and one-hot encoded it, we would get feature names like "color_red," "color_yellow," and "color_green."
# 
# Child: That's correct! We can use these feature names to understand which characteristics are represented by each code.
# 
# You: I understand now! One-hot encoding helps us create special codes for different characteristics, and the get_feature_names_out() method gives us the names of these codes.
# 
# Child: Yes, exactly! It's like having a secret language for fruits that tells us their colors, sizes, and tastes.
# </details>

# In[559]:


names = one_hot.get_feature_names_out()
names


# Let's strip out the prefix of the string.

# In[564]:


[[name[name.find("__")+2:]]for name in names]


# In[566]:


column_names=[name[name.find("_")+1:] for name in  [name[name.find("__")+2:] for name in names]]
column_names


# **We can save the result as a dataframe to be used later.**

# In[572]:


df_carprice = pd.DataFrame(data=X,columns=column_names)
# to download file, uncomment below
# df_carprice.to_csv("cleaned_car_data.csv", index=False)


# ### Write the lines of code that performs same task as  `ColumnTransformer` using `pd.get_dummies`.
# 

# In[ ]:





# ## Train Test Split data

# ### Import libraries

# In[578]:


from sklearn.model_selection import train_test_split


# ### split the data

# In[593]:


X_train,X_test,y_train,y_test = train_test_split(df_carprice,y, test_size=0.3, random_state=0)


# In[594]:


X_train.shape


# In[595]:


X_test.shape


# In[596]:


y_train.shape


# In[597]:


y_test.shape


# ## Standardize the data

# We standardize features by removing the mean and scaling to unit variance using `StandardScaler`, we create a
# `StandardScaler` object:
# 

# In[599]:


from sklearn.preprocessing import StandardScaler


# In[600]:


s = StandardScaler()


# ## fit the data

# In[602]:


X_train = s.fit_transform(X_train)


# ## Linear Regression

# Finally, we apply the LinearRegression() model and fit() our X and y data.

# In[604]:


lr = LinearRegression()


# In[605]:


lr.fit(X_train,y_train)


# ## Make Predictions

# #### apply Standard Scaler on test data too

# In[607]:


X_test = s.transform(X_test)


# ## Inverse the log transformation using the exponential function

# In[616]:


np.exp(y_predict)


# In[622]:


y_predict


# ## Model Evaluation

# ### R2 Score

# In[617]:


r2_score(y_test, y_predict)


# Score is 0.84, model looks good.

# ### MSE (mean Squared Error)

# In[621]:


mse = mean_squared_error(y_test, y_predict)
mse


# ### Linear Regression score

# In[623]:


lr.score(X_test, y_test)


# **The r2_score method returns the same statistic, also known as the goodness of fit of the model.**

# ## Pipeline Object

# We can also create a Pipeline object and apply a set of transforms sequentially. Then, we can apply linear regression. 
# - Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaleras a step in our pipeline.
# 

# <details>
# You: Have you ever played with building blocks? Imagine you have different types of building blocks like cubes, triangles, and circles. Each of these blocks has a special function or purpose.
# 
# Child: Yes, I have played with building blocks before!
# 
# You: Great! Now, think of a Pipeline as a way to combine different types of building blocks in a specific order to create something cool. In machine learning, a Pipeline is like a sequence of building blocks that perform different operations on the data.
# 
# Child: So, it's like putting the blocks together to make something?
# 
# You: Exactly! In a Pipeline, each building block is called a "transformer". Each transformer has a specific task, like cleaning the data or transforming it in some way. These transformers are combined in a sequence, and the data flows through them, just like when you build something with your blocks.
# 
# Child: Can you give me an example?
# 
# You: Sure! Let's say we want to build a model to predict whether it will rain tomorrow. We need to do some steps before making predictions, like cleaning the data, scaling the features, and training a model. Each of these steps can be represented by a transformer in the Pipeline.
# 
# Child: How does the data flow through the Pipeline?
# 
# You: Good question! The data starts at the beginning of the Pipeline and goes through each transformer one by one. Each transformer performs its specific task on the data and passes it to the next transformer. This process continues until the data reaches the end of the Pipeline.
# 
# Child: So, it's like an assembly line for data?
# 
# You: Exactly! It's like an assembly line where each transformer does its part to process the data. And just like when you build something with your blocks, the final result is the output of the last transformer in the Pipeline.
# 
# Child: That sounds really cool! So, a Pipeline helps us organize and automate different steps in machine learning?
# 
# You: That's right! A Pipeline makes it easier for us to organize and automate the different tasks involved in machine learning. It helps us streamline the process and ensures that each step is applied in the correct order.
# 
# Child: I want to try building my own Pipeline with my blocks now!
# 
# You: That's a great idea! Building your own Pipeline with blocks can be a fun way to understand how it works. And who knows, maybe one day you'll be building machine learning Pipelines with real data!
# 
# Child: That would be awesome! Thanks for explaining it to me!
# 
# You: You're welcome! Have fun with your blocks and keep exploring new things!
#     </details>

# We create the pipeline, by creating a list of tuples including the name of the model or estimator and its corresponding constructor.

# In[626]:


steps = [('s', StandardScaler()),("lr",LinearRegression())]


# We input the list as an argument to the constructor

# In[630]:


pipe = Pipeline(steps = steps)
pipe


# We fit the constructor

# In[631]:


pipe.fit(X_train,y_train)


# We make predictions and evaluate the model

# In[648]:


y_predict = pipe.predict(X_test)
y_predict


# In[649]:


np.exp(y_predict)


# In[636]:


mse = mean_squared_error(y_test,y_predict)
rmse = np.sqrt(mse)
mse, rmse


# In[639]:


r2_score(y_test, y_predict)

