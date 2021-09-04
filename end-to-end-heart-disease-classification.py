#!/usr/bin/env python
# coding: utf-8

# # Predicting heart disease using machine learning
# 
# This notebook looks into using various Python-based and data science libraries in an attempt to build a machine learning model or not capable of predicting whether or not someone has heart disease based on the medical attributes.
# 
# We're going to take the following approach
# 1. Problem definition
# 2. Data
# 3. Evaluation
# 4. Features
# 5. Modelling 
# 6. Experimentation
# 
# ## 1. Problem Definition
# 
# In a statement,
# > Given clinical parameters about a patient, can we predict whether or not they have heart disease?
# 
# 
# ## 2. Data
# 
# The original data came from the Cleavland data from the UCI Machine Learning Repository.
# 
# There is alos a version of it available on Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci
# 
# 
# ## 3. Evaluation
# 
# > if we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.
# 
# ## 4. Features
# 
# This is where you get different information about each of the features in the data.
# 
# **Create data dictionary**
# 
# * age - age in years
# * sex - (1=male 0=female)
# * chest pain type (4 values) 
# * resting blood pressure 
# * serum cholestoral in mg/dl 
# * fasting blood sugar > 120 mg/dl
# * resting electrocardiographic results (values 0,1,2)
# * maximum heart rate achieved 
# * exercise induced angina 
# * oldpeak = ST depression induced by exercise relative to rest 
# * the slope of the peak exercise ST segment 
# * number of major vessels (0-3) colored by flourosopy 
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# 
# 

# ## Prepearing the tools
# 
# we're going to use pandas, matplotlib and numpy for data analysis ad manipulation.

# In[9]:


# Import all the tools we need

# Regular EDA and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# we want our plots to appear inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ## load the data

# In[10]:


df = pd.read_csv("heart-disease.csv")
df.shape # (rows, colums)


# ## data exploration (exploratory data analysis or EDA)
# 
# The goal is to find out more about the data and become a subject matter expert on the dataset you are working with. 
# 
# 1. What questions are you trying to solve?
# 2. What kind of data do you have and how to treat differenty types?
# 3. What's missing from the data and how to deal with it?
# 4. where are the outliers and why you should care about them?
# 5. How can you add, change or remove features to get more out of your data?

# In[11]:


df.head()


# In[12]:


df.tail()


# In[13]:


# let us find out how many of each class there are
df["target"].value_counts()


# In[16]:


# Normalized value counts
df.target.value_counts(normalize=True)


# In[18]:


# Plot the value counts with a bar graph
df["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"]);


# In[19]:


df.info()


# In[20]:


# are there any missing values
df.isna().sum()


# In[21]:


df.describe()


# ## Heart Disease Frequency according to sex

# In[22]:


df.sex.value_counts()


# In[23]:


# compare target column with sex column
pd.crosstab(df.target, df.sex)


# In[24]:


# craete a plot of crosstab
pd.crosstab(df.target, df.sex).plot(kind="bar",
                                   figsize=(10, 6),
                                   color=["salmon", "lightblue"])

plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"]);
plt.xticks(rotation=0)


# In[25]:


df.head()


# In[26]:


df["thalach"].value_counts()


# ### Age vs. Max Heart Rate for Heart Disease

# In[27]:


# create another figure
plt.figure(figsize=(10, 6))

# scatter with positive example
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon")


# scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue");

# add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# In[28]:


df.age[df.target==1]


# In[29]:


# check the distribution of the age column with the histogram
df.age.plot.hist()


# ### Heart Disease Frequency per Chest Pain Type
# 
# add data description from above

# In[30]:


pd.crosstab(df.cp, df.target)


# In[31]:


# make the crosstab more visual
pd.crosstab(df.cp, df.target).plot(kind="bar",
                                  figsize=(10, 6),
                                  color=["salmon", "lightblue"])

# add some communication
plt.title("Heart Disease Frequency Per Chesr Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease", "Disease"]);
plt.xticks(rotation=0);


# In[32]:


df.head()


# In[33]:


#  make a correlation matrix
df.corr()


# In[34]:


# let's make our corr matrix pretty
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidths=0.5,
                fmt=".2f",
                cmap="YlGnBu");

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# ## 5. Modelling 

# In[35]:


df.head()


# In[36]:


# Everything except target variable
X = df.drop("target", axis=1)

# Target variable
y = df.target.values


# In[41]:


# Independent variables (no target column)
X


# In[39]:


# targets
y


# In[43]:


# Random seed for reproducibility
np.random.seed(42)

# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(X, # independent variables 
                                                    y, # dependent variable
                                                    test_size = 0.2) # percentage of data to use for test set


# In[47]:


X_train.head()


# In[48]:


y_train, len(y_train)


# Now we've got our data split into training and test sets, it's time to build a machine learning model.
# 
# We'll train it (find patterns), test it (use patterns).
# 
# we are going to try three different machine learning models:
# 
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier
# 

# In[49]:


X_test.head()


# In[50]:


y_test, len(y_test)


# Model choices
# 
# Now we've got our data prepared, we can start to fit models. We'll be using the following and comparing their results.
# 
#     1. Logistic Regression - LogisticRegression()
#     2. K-Nearest Neighbors - KNeighboursClassifier()
#     3. RandomForest - RandomForestClassifier()

# In[54]:


# Put models in a dictionary
models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier()}

# Create function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[59]:


model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores


# ## Model Comparison

# In[60]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();


# Now we've got a baseline model... and we know a model's first predictions aren't always what we should base our next steps off. What should we do?
# 
#  Let's look at the following:
#  
#  * Hyperparameter tuning
#  * Feature importance
#  * Confusion matrix
#  * Cross-validation
#  * Precision
#  * Recall
#  * F1 score
#  * Classification report
#  * ROC curve
#  * Area under the curve (AUC)
#  
#  ### Hyperparameter tuning

# In[62]:


# Create a list of train scores
train_scores = []

# Create a list of test scores
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1, 21) # 1 to 20

# Setup algorithm
knn = KNeighborsClassifier()

# Loop through different neighbors values
for i in neighbors:
    knn.set_params(n_neighbors = i) # set neighbors value
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # Update the training scores
    train_scores.append(knn.score(X_train, y_train))
    
    # Update the test scores
    test_scores.append(knn.score(X_test, y_test))
    


# In[63]:


train_scores


# In[64]:


test_scores


# In[65]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ## Hyperparameter tuning with RandomizedSearchCV
# 
# We're going to tune:
# * LogisticRegression()
# * RandomForestClassifier()
# 
# ... using RandomizedSearchCV 

# In[69]:


# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Different RandomForestClassifier hyperparameters
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# Now let's use RandomizedSearchCV to try and tune our LogisticRegression model.
# 
# We'll pass it the different hyperparameters from log_reg_grid as well as set n_iter = 20. This means, RandomizedSearchCV will try 20 different combinations of hyperparameters from log_reg_grid and save the best ones.

# In[70]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model
rs_log_reg.fit(X_train, y_train);


# In[71]:


rs_log_reg.best_params_


# In[72]:


rs_log_reg.score(X_test, y_test)


# In[67]:


np.arange(10, 1000, 50)


# Now we've got hyperparameter grids setup for each of our models, let's tune them using RandomizedSearchCV...

# In[73]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model
rs_rf.fit(X_train, y_train);


# In[75]:


# Find the best parameters
rs_rf.best_params_


# In[76]:


# Evaluate the randomized search random forest model
rs_rf.score(X_test, y_test)


# Tuning a model with GridSearchCV
# 
# The difference between RandomizedSearchCV and GridSearchCV is where RandomizedSearchCV searches over a grid of hyperparameters performing n_iter combinations, GridSearchCV will test every single possible combination.
# 
# In short:
#     
#     * RandomizedSearchCV - tries n_iter combinations of hyperparameters and saves the best.
#     * GridSearchCV - tries every single combination of hyperparameters and saves the best.
# 

# In[78]:


# Different LogisticRegression hyperparameters
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train);


# In[79]:


# Check the best parameters
gs_log_reg.best_params_


# In[80]:


# Evaluate the model
gs_log_reg.score(X_test, y_test)


# Evaluating a classification model, beyond accuracy
# 
# Now we've got a tuned model, let's get some of the metrics we discussed before.
# 
# We want:
# 
#     * ROC curve and AUC score - plot_roc_curve()
#     * Confusion matrix - confusion_matrix()
#     * Classification report - classification_report()
#     * Precision - precision_score()
#     * Recall - recall_score()
#     * F1-score - f1_score()
#     
# Luckily, Scikit-Learn has these all built-in.
# 
# To access them, we'll have to use our model to make predictions on the test set. You can make predictions by calling predict() on a trained model and passing it the data you'd like to predict on.
# 
# We'll make predictions on the test data.

# In[81]:


# Make preidctions on test data
y_preds = gs_log_reg.predict(X_test)


# In[82]:


y_preds


# In[83]:


y_test


# ROC Curve and AUC Scores
# 
# What's a ROC curve?
# 
# It's a way of understanding how your model is performing by comparing the true positive rate to the false positive rate.
# 
# In our case...
# 
#         To get an appropriate example in a real-world problem, consider a diagnostic test that seeks to determine whether a person has a certain disease. A false positive in this case occurs when the person tests positive, but does not actually have the disease. A false negative, on the other hand, occurs when the person tests negative, suggesting they are healthy, when they actually do have the disease.
#         
# Scikit-Learn implements a function plot_roc_curve which can help us create a ROC curve as well as calculate the area under the curve (AUC) metric.
# 
# Reading the documentation on the plot_roc_curve function we can see it takes (estimator, X, y) as inputs. Where estiamator is a fitted machine learning model and X and y are the data you'd like to test it on.
# 
# In our case, we'll use the GridSearchCV version of our LogisticRegression estimator, gs_log_reg as well as the test data, X_test and y_test.

# In[84]:


# Import ROC curve function from metrics module
from sklearn.metrics import plot_roc_curve

# Plot ROC curve and calculate AUC metric
plot_roc_curve(gs_log_reg, X_test, y_test);


# **Confusion matrix**
# 
# A confusion matrix is a visual way to show where your model made the right predictions and where it made the wrong predictions (or in other words, got confused).
# 
# Scikit-Learn allows us to create a confusion matrix using confusion_matrix() and passing it the true labels and predicted labels.

# In[85]:


# Display confusion matrix
print(confusion_matrix(y_test, y_preds))


# In[86]:


# Import Seaborn
import seaborn as sns
sns.set(font_scale=1.5) # Increase font size

def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    plt.xlabel("true label")
    plt.ylabel("predicted label")
    
plot_conf_mat(y_test, y_preds)


# **Classification report**
# 
# We can make a classification report using classification_report() and passing it the true labels as well as our models predicted labels.
# 
# A classification report will also give us information of the precision and recall of our model for each class.

# In[87]:


# Show classification report
print(classification_report(y_test, y_preds))


# Let's get a refresh.
# 
#    * **Precision** - Indicates the proportion of positive identifications (model predicted class 1) which were actually correct. A model which produces no false positives has a precision of 1.0.
#     * **Recall** - Indicates the proportion of actual positives which were correctly classified. A model which produces no false negatives has a recall of 1.0.
#     * **F1 score** - A combination of precision and recall. A perfect model achieves an F1 score of 1.0.
#     * **Support** - The number of samples each metric was calculated on.
#     * **Accuracy** - The accuracy of the model in decimal form. Perfect accuracy is equal to 1.0.
#     * **Macro avg** - Short for macro average, the average precision, recall and F1 score between classes. Macro avg doesn’t class imbalance into effort, so if you do have class imbalances, pay attention to this metric.
#     * **Weighted avg** - Short for weighted average, the weighted average precision, recall and F1 score between classes. Weighted means each metric is calculated with respect to how many samples there are in each class. This metric will favour the majority class (e.g. will give a high value when one class out performs another due to having more samples).

# Ok, now we've got a few deeper insights on our model. But these were all calculated using a single training and test set.
# 
# What we'll do to make them more solid is calculate them using cross-validation.
# 
# How?
# 
# We'll take the best model along with the best hyperparameters and use cross_val_score() along with various scoring parameter values.
# 
# cross_val_score() works by taking an estimator (machine learning model) along with data and labels. It then evaluates the machine learning model on the data and labels using cross-validation and a defined scoring parameter.
# 
# Let's remind ourselves of the best hyperparameters and then see them in action

# In[88]:


# Check best hyperparameters
gs_log_reg.best_params_


# In[89]:


# Import cross_val_score
from sklearn.model_selection import cross_val_score

# Instantiate best model with best hyperparameters (found with GridSearchCV)
clf = LogisticRegression(C=0.23357214690901212,
                         solver="liblinear")


# 
# Now we've got an instantiated classifier, let's find some cross-validated metrics.

# In[90]:


# Cross-validated accuracy score
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5, # 5-fold cross-validation
                         scoring="accuracy") # accuracy as scoring
cv_acc


# Since there are 5 metrics here, we'll take the average.

# In[91]:


cv_acc = np.mean(cv_acc)
cv_acc


# Now we'll do the same for other classification metrics.

# In[92]:


# Cross-validated precision score
cv_precision = np.mean(cross_val_score(clf,
                                       X,
                                       y,
                                       cv=5, # 5-fold cross-validation
                                       scoring="precision")) # precision as scoring
cv_precision


# In[93]:


# Cross-validated recall score
cv_recall = np.mean(cross_val_score(clf,
                                    X,
                                    y,
                                    cv=5, # 5-fold cross-validation
                                    scoring="recall")) # recall as scoring
cv_recall


# In[94]:


# Cross-validated F1 score
cv_f1 = np.mean(cross_val_score(clf,
                                X,
                                y,
                                cv=5, # 5-fold cross-validation
                                scoring="f1")) # f1 as scoring
cv_f1


# In[95]:


# Visualizing cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                            "Precision": cv_precision,
                            "Recall": cv_recall,
                            "F1": cv_f1},
                          index=[0])
cv_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);


# **Feature importance**
# 
# Feature importance is another way of asking, "which features contributing most to the outcomes of the model?"
# 
# Or for our problem, trying to predict heart disease using a patient's medical characterisitcs, which charateristics contribute most to a model predicting whether someone has heart disease or not?
# 
# Unlike some of the other functions we've seen, because how each model finds patterns in data is slightly different, how a model judges how important those patterns are is different as well. This means for each model, there's a slightly different way of finding which features were most important.
# 
# You can usually find an example via the Scikit-Learn documentation or via searching for something like "[MODEL TYPE] feature importance", such as, "random forest feature importance".
# 
# Since we're using LogisticRegression, we'll look at one way we can calculate feature importance for it.
# 
# To do so, we'll use the coef_ attribute. Looking at the Scikit-Learn documentation for LogisticRegression, the coef_ attribute is the coefficient of the features in the decision function.
# 
# We can access the coef_ attribute after we've fit an instance of LogisticRegression.

# In[96]:


# Fit an instance of LogisticRegression (taken from above)
clf.fit(X_train, y_train);


# In[97]:


# Check coef_
clf.coef_


# In[98]:


# Match features to columns
features_dict = dict(zip(df.columns, list(clf.coef_[0])))
features_dict


# In[99]:


# Visualize feature importance
features_df = pd.DataFrame(features_dict, index=[0])
features_df.T.plot.bar(title="Feature Importance", legend=False);


# You'll notice some are negative and some are positive.
# 
# The larger the value (bigger bar), the more the feature contributes to the models decision.
# 
# If the value is negative, it means there's a negative correlation. And vice versa for positive values.
# 
# For example, the sex attribute has a negative value of -0.904, which means as the value for sex increases, the target value decreases.
# 
# We can see this by comparing the sex column to the target column.

# In[100]:


pd.crosstab(df["sex"], df["target"])


# You can see, when sex is 0 (female), there are almost 3 times as many (72 vs. 24) people with heart disease (target = 1) than without.
# 
# And then as sex increases to 1 (male), the ratio goes down to almost 1 to 1 (114 vs. 93) of people who have heart disease and who don't.
# 
# What does this mean?
# 
# It means the model has found a pattern which reflects the data. Looking at these figures and this specific dataset, it seems if the patient is female, they're more likely to have heart disease.
# 
# How about a positive correlation?

# In[101]:


# Contrast slope (positive coefficient) with target
pd.crosstab(df["slope"], df["target"])


# Looking back the data dictionary, we see slope is the "slope of the peak exercise ST segment" where:
# 
#     * 0: Upsloping: better heart rate with excercise (uncommon)
#     * 1: Flatsloping: minimal change (typical healthy heart)
#     * 2: Downslopins: signs of unhealthy heart
#     
# According to the model, there's a positive correlation of 0.470, not as strong as sex and target but still more than 0.
# 
# This positive correlation means our model is picking up the pattern that as slope increases, so does the target value.

# ## 6. Experimentation
# 
# If you haven't hit your evaluation metric yet... ask yourself...
# 
# * could you collect more data?
# * could you try a better model? like CatBoost XGBoost?
# * could you improve the current models?
# * if good enough, how do you export it and share it with others?
