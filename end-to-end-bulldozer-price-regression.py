#!/usr/bin/env python
# coding: utf-8

# # Predicting the Sale Price of bulldozers using machine learning
# 
# In this notebook, we are going to go through a machine learning project with the goal of predicting the sale price of bulldozers.
# 
# ## 1. Problem definition
# 
# > how well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?
# 
# ## 2. Data
# 
# > the data is downloased from the kaggle competition https://www.kaggle.com/c/bluebook-for-bulldozers/data 
# 
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# * Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
# 
# ## 3. Evaluation
# 
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices. 
# 
# For more on the evaluation https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation 
# 
# **Note:** The goal for most regression evaluation metrics is to minimise the error. for example, our goal for this project will be to build a machine learning model which minimizes RMSLE.
# 
# ## 4. Features
# 
# Kaggle provides a data dictionary detailing all of the features of the dataset. You can view the data on google sheets. xxx
# https://www.kaggle.com/c/bluebook-for-bulldozers/data

# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[43]:


# import training and validation sets
df = pd.read_csv("TrainAndValid.csv",
                low_memory=False)


# In[44]:


df.info()


# In[45]:


df.isna().sum()


# In[46]:


df.columns


# In[47]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])


# In[48]:


df.saledate[:1000]


# In[49]:


df.SalePrice.plot.hist()


# ### Parsing date
# 
# When we work with time series dat, we ant to enrich the time and date component as much as possible
# 
# We can do that by telling pandas which of our columns has dates in it using the 'pase_dates' parameter.

# In[50]:


# import data again but this time parse dates
df = pd.read_csv("TrainAndValid.csv",
                low_memory=False,
                parse_dates=["saledate"])


# In[51]:


# With parse_dates... check dtype of "saledate"
df.info()


# In[52]:


df.saledate.dtype


# In[53]:


df.saledate[:1000]


# In[54]:


fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])


# In[55]:


df.head()


# In[56]:


df.head().T


# In[57]:


df.saledate.head(20)


# ### sort dataframme by saledate
# 
# when working with time series data, it's a good idea to sort it out by date

# In[58]:


# sort dataframe in date order
df.sort_values(by=["saledate"], inplace=True, ascending=True)
df.saledate.head(20)


# In[59]:


df.head()


# ### make a copy of the og dataframe
# 
# we make a copy of the og so that when we manipulate the copy, we still have the og

# In[60]:


# make a copy
df_tmp = df.copy()


# In[61]:


df_tmp


# ### add datetime parameters for 'saledate' column
# 
# So we can enrich our dataset with as much information as possible.
# 
# Because we imported the data using read_csv() and we asked pandas to parse the dates using parase_dates=["saledate"], we can now access the different datetime attributes of the saledate column.

# In[62]:


df_tmp[:1].saledate.dt.year


# In[63]:


df_tmp[:1].saledate.dt.day


# In[64]:


df_tmp[:1].saledate


# In[65]:


# Add datetime parameters for saledate
df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayofweek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayofyear"] = df_tmp.saledate.dt.dayofyear

# Drop original saledate
df_tmp.drop("saledate", axis=1, inplace=True)


# In[66]:


df_tmp.head().T


# In[67]:


# check the values of different columns
df_tmp.state.value_counts()


# ## 5. Modelling
# 
# We've done enough EDA(we could always do more) but let's start to do some model_driven EDA
# 
# Why model so early?
# 
# We know the evaluation metric we're heading towards. We could spend more time doing exploratory data analysis (EDA), finding more out about the data ourselves but what we'll do instead is use a machine learning model to help us do EDA.
# 
# Remember, one of the biggest goals of starting any new machine learning project is reducing the time between experiments.
# 
# Following the Scikit-Learn machine learning map, we find a RandomForestRegressor() might be a good candidate.
# 

# In[68]:


# Check for missing categories and different datatypes
df_tmp.info()


# In[69]:


# Check for missing values
df_tmp.isna().sum()


# ### Convert strings to categories
# 
# One way to help turn all of our data into numbers is to convert the columns with the string datatype into a category datatype.
# 
# To do this we can use the pandas types API which allows us to interact and manipulate the types of data.

# In[70]:


df_tmp.head().T


# In[71]:


pd.api.types.is_string_dtype(df_tmp["UsageBand"])


# In[72]:


# These columns contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[73]:


# If you're wondering what df.items() does, let's use a dictionary as an example
random_dict = {"key1": "hello",
               "key2": "world!"}

for key, value in random_dict.items():
    print(f"This is a key: {key}")
    print(f"This is a value: {value}")


# In[74]:


# This will turn all of the string values into category values
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()


# In[75]:


df_tmp.info()


# In[76]:


df_tmp.state.cat.categories


# In[77]:


df_tmp.state.cat.codes


# 
# All of our data is categorical and thus we can now turn the categories into numbers, however it's still missing values...

# In[78]:


df_tmp.isnull().sum()/len(df_tmp)


# 
# In the format it's in, it's still good to be worked with, let's save it to file and reimport it so we can continue on.

# ### Save Processed Data

# In[85]:


# Save preprocessed data
df_tmp.to_csv("sample_project/bulldozer-price-prediction-project/data/bluebook-for-bulldozers/file.csv.csv",
              index=False)


# In[45]:


# Import preprocessed data
df_tmp = pd.read_csv("http://localhost:8889/tree/Desktop/sample_project/bulldozer-price-prediction-project",
                     low_memory=False)
df_tmp.head().T


# # Let's build a machine learning model

# In[42]:


len(df_tmp)


# In[101]:


# Let's build a machine learning model
from sklearn.ensemble import RandomForestRegeressor

model = RandomForestRegeressor(n_jobs=-1,
                              random_state=42)

model.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])


# In[46]:


df.info()


# In[48]:


df_tmp["UsageBand"].dtype


# In[49]:


df.isna().sum()


# ### convert strings into categories
# 
# one way to turn all our data into numbers is to covert strings into categories.

# In[50]:


df_tmp.head().T


# In[51]:


pd.api.types.is_string_dtype(df_tmp["UsageBand"])


# In[52]:


# find colums which contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[53]:


# if you're wondering what df.items() does, here's an example

random_dict = {"key1": "hello",
              "key2": "world!"}

for key, value in random_dict.items():
    print(f"this is a key: {key}",
         f"this is a value: {value}")


# In[56]:


# this will turn all of the string value into category values
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label]=content.astype("category").cat.as_ordered()


# In[57]:


df_tmp.info()


# In[58]:


df_tmp.state.cat.categories


# In[60]:


df_tmp.state.value_counts()


# In[62]:


df_tmp.state.cat.codes


# Thanks to pandas categories, we can now access all our data in numbers.
# 
# But we still have a bunch of missing data...

# In[65]:


# check missing data
df_tmp.isnull().sum()/len(df_tmp)


# ### save preprocessed data

# In[66]:


# export current tmp dataframe
df_tmp.to_csv("train_tmp.csv",
             index=False)


# In[67]:


# import preprocessed data
df_tmp = pd.read_csv("train_tmp.csv",
             low_memory=False)

df_tmp.head().T


# In[68]:


df_tmp.isna().sum()


# ### fill missing values
# 
# ### fill numerical missing values first

# In[71]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)


# In[70]:


df_tmp.ModelID


# In[72]:


# check for which numeric columns have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[74]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
           # add a binary column which tells us if the data was missing or not
           df_tmp[label+"_is_missing"] = pd.isnull(content)
            # fill missing numeric values with median
        df_tmp[label] = content.fillna(content.median())


# In[75]:


# demonstrate how median is more robust that mean
hundreds = np.full((1000), 100)
hundreds_billion = np.append(hundreds, 1000000000)
np.mean(hundreds), np.mean(hundreds_billion), np.median(hundreds), np.median(hundreds_billion)


# In[77]:


hundreds_billion


# In[78]:


# check if there is any null numeric values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[79]:


# check to see how many examples were missing
df_tmp.auctioneerID_is_missing.value_counts()


# In[80]:


df_tmp.isna().sum()


# ### filling and turning categorical variables into numbers

# In[81]:


for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)


# In[82]:


df_tmp.isna().sum()


# In[89]:


# turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # add binary column to indicate whether sample had missing values
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # turn categories into numbers and add +1
        df_tmp[label] = pd.Categorical(content).codes + 1


# In[90]:


pd.Categorical(df_tmp["state"]).codes + 1


# In[92]:


pd.Categorical(df_tmp["UsageBand"]).codes + 1


# In[94]:


df_tmp.info()


# In[95]:


df_tmp.head().T


# In[96]:


df_tmp.isna().sum()


# Now that all our data is numeric and our dataframe has no missing values, we should be able to build a machine learning model.

# In[97]:


df_tmp.head()


# In[98]:


len(df_tmp)


# In[102]:


get_ipython().run_cell_magic('time', '', '# instantiate model\nmodel = RandomForestRegressor(n_jobs=-1,\n                             random_state=42)\n\n# fit the model\nmodel.fit(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])')


# In[103]:


# score the model
model.score(df_tmp.drop("SalePrice", axis=1), df_tmp["SalePrice"])


# **question** why doesnt the metric above hold water?
# 

# ### Splitting data into train/validation sets

# In[104]:


df_tmp.head()


# In[105]:


df_tmp.saleYear


# In[106]:


df_tmp.saleYear.value_counts()


# In[107]:


# split data into training  and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

len(df_val), len(df_train)


# In[109]:


# split data into x and y
x_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
x_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

x_train.shape, y_train.shape, x_valid.shape, y_valid.shape


# In[110]:


y_train


# ## building an evaluation function

# In[125]:


# create evaluation function (the competition uses  RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

def rmsle(y_test, y_preds):
    """
    calculates root mean squared log error between predictions and true labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# create function to evaluate model on a few differeent levels
def show_scores(model):
    train_preds = model.predict(x_train)
    val_preds = model.predict(x_valid)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds)
             "Valid MAE": mean_absolute_error(y_valid, val_preds),
             "Training RMSLE": rmsle(y_train, train_preds),
             "Valid RMSLE": rmsle(y_valid, val_preds),
             "Training R^2": r2_score(y_train, train_preds),
             "Valid R^2": r2_score(y_valid, val_preds)}

    return scores


# ## Testing our model on a subset(to tune the hyperparameters)

# In[114]:


# this takes long

# %%time
# model = RandomForestRegressor(n_jobs=-1,
                             #random_state=42)

# model.fit(x_train, y_train)


# In[115]:


len(x_train)


# In[118]:


# change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                             random_state=42)

model


# In[123]:


get_ipython().run_cell_magic('time', '', '# cutting down on the max number of samples each estimator can see\nmodel.fit(x_train, y_train)')


# In[122]:


(x_train.shape[0] * 100) / 100000


# In[121]:


10000* 100


# In[126]:


show_scores(model)


# ### Hyperparameter tuning with RandomizedSearchCV

# In[128]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import RandomizedSearchCV\n\n# Different RandomForestRegressor hyperparameters\nrf_grid = {"n_estimators": np.arrange(10, 100, 10),\n          "max_depth": [None, 3, 5, 10],\n          "min_samples_split": np.arrange(2, 20, 2),\n          "min_samples_leaf": np.arrange(1, 20, 2),\n          "max_features": [0.5, 1 "sqrt", "auto"],\n          "max_samples": [10000]}\n\n# instantiate RandomizedSearchCV model\nrs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,\n                                                   random_state=42),\n                             param_distributions=rf_grid,\n                             n_iter=2,\n                             cv=5,\n                             verbose=True)\n\n# fit the RandomisedSearchCV\nrs_model.fit(x_train, y_train)')


# In[129]:


# FIND THE BEST MODEL hyperparameters
rs_model.best_params_


# In[130]:


# evaluate the RandomizedSEarch model
show_scores(rs_model)


# ### train the model with the best hyperparameters
# 
# **note**: these were found after 100 iterations of RandomizedSearchCV 

# In[131]:


get_ipython().run_cell_magic('time', '', '\n# most ideal hyperparameters\nideal_model = RandomForestRegressor(n_estimators=40,\n                                   min_samples_leaf=1,\n                                   min_samples_split=14,\n                                   max_features=0.5,\n                                   n_jobs=-1,\n                                   max_samples=None,\n                                   random_state=42)\n\n# fit the ideal model\nideal_model.fit(x_train, y_train)')


# In[132]:


# scores for ideal model (trained on all data)

show_scores(ideal_model)


# In[134]:


# scores on rs_model (only trained on -10, 000 examples)
show_scores(rs_model)


# ## make predictions on test data

# In[135]:


# import the test data
df_test = pd.read_csv("Test.csv",
                     low_memory=False,
                     parse_dates=["saledate"])

df_test.head()


# In[136]:


# make predictions on the test data
test_preds = ideal_model.predict(df_test)


# In[138]:


df_test.isna().sum()


# In[139]:


df_test.info()


# In[141]:


df_test.columns


# In[143]:


x_train.columns


# ### preprocessing the data

# In[146]:


def process_data(df):
    """
    Performs transformations on df and returns transformed df.
    """
    df["saleYear"] = df_tmp.saledate.dt.year
    df["saleMonth"] = df_tmp.saledate.dt.month
    df["saleDay"] = df_tmp.saledate.dt.day
    df["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
    df["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear
    
    df.drop("saledate", axis=1, inplace=True)
    
    # fill numeric rows with medium
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # add a binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # fill missing numeric values with median
                df[label] = content.fillna(content.median())
        
    # filled categorical missing data and turn categories into numbers
    if not pd.api.types.is_missing_dtypes(content):
        df[label+"_is_missing"] = pd.isnull(content)
        
        # we add +1 to the category code because pandas enncode missing categories
        df[label] = pd.Categorical(content).codes+1
    
    return df


# In[147]:


# process the test data
df_test = preprocess_data(df_test)
df_test.head()


# In[148]:


x_train.head()


# In[150]:


# make predictions on updated test data
test_preds = ideal_model.predict(df_test)


# In[151]:


x_train.head()


# In[152]:


# we cam find how the columns differ using sets
set(x_train.columns) - set(df_test.columns)


# In[ ]:


# manually  adjust df_test
df_test[""] = False
df_test.head()


# finally our test dataframe has the same features as our training data, we can make predictions!

# In[153]:


# make predictions on the test data
test_preds = ideal_model.predict(df_test)


# we've made predictions but they are not in the same format kaggle is asking for:

# In[155]:


# format predictions into the same format Kaggle is after
df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds


# In[ ]:


# export data
df_preds.to_csv


# ### feature importance
# 
# Feature importance seeks to figure out which different attributions of the data were most important when it comes to predicting the **target variable** (SalePrice)

# In[156]:


# find feature importance of our best model

ideal_model.feature_importances_


# In[157]:


x_train


# In[161]:


def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                       "feature_importances": importances})
         .sort_values("feature_importances", ascending=False)
         .reset_index(drop=True))
    
    # plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:20])
    ax.set_ylabel("feature")
    ax.set_xlabel("feature_importances")
    ax.inver_yaxis()


# In[160]:


plot_features(x_train.columns, ideal_model.faeture_importance)


# In[163]:


x_train["ProductSize"].value_counts()


# **question to finish:** why might knowing the feature importances of a trained machine learning be helpful?
# 
# **final challenge:** what other machine learning models could you try on our dataset?

# In[ ]:




