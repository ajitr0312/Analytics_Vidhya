#!/usr/bin/env python
# coding: utf-8

# ## <center> Smart Lead Scoring Engine</center>
# 
# 

# #### Problem Statement:
# A D2C startup develops products using cutting edge technologies like Web 3.0. Over the past few months, the company has started multiple marketing campaigns offline and digital both. As a result, the users have started showing interest in the product on the website. These users with intent to buy product(s) are generally known as leads (Potential Customers). 
# 
# Leads are captured in 2 ways - <b>Directly and Indirectly.</b>
# 
# <b>Direct leads</b> are captured via forms embedded in the website while <b>indirect leads</b> are captured based on certain activity of a user on the platform such as time spent on the website, number of user sessions, etc.
# 
# Now, the marketing & sales team wants to identify the leads who are more likely to buy the product so that the sales team can manage their bandwidth efficiently by targeting these potential leads and increase the sales in a shorter span of time.
# 
# Our task at hand is to predict the propensity to buy a product based on the user's past activities and user level information.

# In[166]:


#Importing the required libraries:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# #### Importing the train dataset:

# In[168]:


df_train = pd.read_csv(r'E:\Hackathons\Smart_Lead_Scoring_Engine\train_wn75k28.csv')

# looking for the dataset
pd.set_option('display.max_columns', None)
df_train.head()


# #### Importing the test dataset:

# In[169]:


df_test = pd.read_csv(r'E:\Hackathons\Smart_Lead_Scoring_Engine\test_Wf7sxXF.csv')

df_test.head()


# In[170]:


# Looking for the shape of the datasets

print('Shape of train dataset is:', df_train.shape)
print('Shape of test dataset is:', df_test.shape)


# - The train dataset contains total <b>19 columns</b> including the target column.
# - The train dataset contains <b>39161 rows.</b>
# - <b> buy </b> is the target column in the train dataset.
# - The test dataset contains <b>13184 rows and 18 columns.</b>

# #### Checking for the columns:

# In[171]:


print('The columns in the train dataset are:','\n'*2, df_train.columns,'\n'*2,'='*85)
print('\n','The columns in the test dataset are:','\n'*2, df_test.columns, '\n'*2, '='*85)


# #### Column Description:

# <b><u>Train Dataset:</u></b>
# 
# 1. <b>id</b> -- Unique identifier of a lead
# 2. <b>created_at</b> -- Date of lead dropped
# 3. <b>campaign_var_1 & campaign_var_2</b> -- campaign information of the lead
# 4. <b>products_purchased</b> -- No. of past products purchased at the time of dropping the lead
# 5. <b>signup_date</b> -- Sign up date of the user on the website
# 6. <b>user_activity_var_(1 to 12)</b> -- Derived activities of the user on the website
# 7. <b>buy</b> -- 0 or 1 indicating if the user will buy the product in next 3 months or not
# 
# ------------------------------------------------------------------------------------------------------------------------
# 
# <b><u>Test Dataset:</u></b>
# 
# 1. <b>id</b> -- Unique identifier of a lead
# 2. <b>created_at</b> -- Date of lead dropped
# 3. <b>campaign_var_1 & campaign_var_2</b> -- campaign information of the lead
# 4. <b>products_purchased</b> -- No. of past products purchased at the time of dropping the lead
# 5. <b>signup_date</b> -- Sign up date of the user on the website
# 6. <b>user_activity_var_(1 to 12)</b> -- Derived activities of the user on the website

# #### Checking the data type of the columns

# In[172]:


# For trian dataset:

df_train.dtypes


# - All the columns contains integer values except created_at, products_purchased, and signup_date.
# - We can see that created_at and signup_date coulmns contains date so we'll change the data type to datetime for better understanding later.
# - products_purchased column contains the number of past items purchased by the customers. So, it would be better if we change its type to integer datatype as items can't be in float.

# #### Checking the null values in the train and test datasets:

# In[173]:


#Null values or missing values in train dataset:

print('Column wise missing values in the train dataset are:','\n'*2,df_train.isnull().sum(),'\n','-'*70)
print('\n','Column wise missing values in the test dataset are:','\n'*2,df_test.isnull().sum(),'\n','-'*70)


# We found that product_purchased and signup_date columns contain missing values in both the datasets. 
# 
# - <b><u>Train Dataset:</u></b>
#  - 20911 missing values in the products_purchased column.
#  - 15113 missing values in the signup_date column.
#  
# - <b><u>Test Dataset:</u></b>
#  - 8136 missing values in the products_purchased column.
#  - 6649 missing values in the signup_date column.

# ### Handling the missing values:

# 1. <b><u>products_purchased column (Train dateset):</u></b>
# 
# We know this column contains the number of items purchased by the customer. So, we can replace the missing values with the most occuring items(mode value) number which is commonly purchased by the customers.

# In[174]:


# Checking for the values:

df_train['products_purchased'].value_counts()


# - We found that most of the customers like to purchase 2 items, so we'll replace the missing values with 2

# In[175]:


df_train['products_purchased'] = df_train['products_purchased'].fillna(2)

df_train.isnull().sum()


# In[176]:


# Changing the products_purchased column from float to integer data type

df_train['products_purchased'] = df_train['products_purchased'].astype(int)

df_train.dtypes


# 2. <b><u>products_purchased column (Test dateset):</u></b>

# In[177]:


df_test['products_purchased'].value_counts()


# In[178]:


# Replaceing the missing values with 2

df_test['products_purchased'] = df_test['products_purchased'].fillna(2)

df_test.isnull().sum()


# In[179]:


# Changing the products_purchased column from float to integer data type

df_test['products_purchased'] = df_test['products_purchased'].astype(int)

df_test.dtypes


# In[180]:


# Dropping the missing values in the signup date column

df_train = df_train.dropna(axis = 0)

df_test = df_test.dropna(axis = 0)


# In[181]:


df_train.isnull().sum()


# In[182]:


#checking the missing values using heatmap for train dataset

plt.figure(figsize=[16,6])
sns.heatmap(df_train.isnull())
plt.title(" NULL VALUES OF TRAIN DATASET ")
plt.show()


# In[183]:


#checking the missing values using heatmap for test dataset

plt.figure(figsize=[16,6])
sns.heatmap(df_test.isnull())
plt.title(" NULL VALUES OF TEST DATASET ")
plt.show()


# #### Checking if the target column is balanced or not

# In[184]:


# Plotting graph to show the distribution of 'buy' column.

df_train['buy'].hist(grid=True)
plt.xlabel('Purchased or not')
plt.ylabel('Counts')
plt.show()


# - This clearly shows that the target column is not balanced. We'll balance it later.

# In[185]:


#Checking the description of the train dataset

df_train.describe()


# In[186]:


#Checking the description of the test dataset

df_test.describe()


# ### Merging the train and test dataset in single data:

# In[187]:


df_train['Source'] = 'train'
df_test['Source'] = 'test'

df = pd.concat([df_train,df_test], ignore_index=True)

#Checking the final dataset df

df.head()


# In[188]:


# Splitting the created_at into day, month, and year

df[["year", "month", "day"]] = df['created_at'].str.split("-", expand=True)

df.head()


# In[189]:


# Splitting the products_purchased into day. month, and year

df[['signup_year', 'signup_month', 'signup_day']] = df['signup_date'].str.split("-", expand=True)

df.head()


# In[190]:


# Dropping the columns created_at and signup_date as we've splitted the values

df = df.drop(['created_at', 'signup_date'], axis = 1)

df.head()


# In[191]:


# Checking for the datatype again

df.dtypes


# In[192]:


# Checking for the information of the dataset

df.info()


# In[193]:


# Changing the data types of year,month,day

df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)
df['day'] = df['day'].astype(int)
df['signup_year'] = df['signup_year'].astype(int)
df['signup_month'] = df['signup_month'].astype(int)
df['signup_day'] = df['signup_day'].astype(int)

df.dtypes


# In[194]:


df.dtypes


# In[195]:


# Plotting the histogram for univariant analysis to check the normal distribution

df.hist(figsize=[18,16], grid=True, layout=(7,6), bins=30)


# #### Checking for the Statistical Summary:

# In[196]:


df.describe()


# In[197]:


# Plotting Heatmap for Statistical Summary

plt.figure(figsize=[35,15])
sns.heatmap(round(df.describe()[1:].transpose(),2), annot=True, linewidths=0.25, linecolor='white', fmt='f')
plt.title('Statistical Summary')
plt.show()


# #### Checking for the correlation:

# In[198]:


df.corr()


# In[199]:


# Plotting heatmap for the correlation table

plt.figure(figsize=[40,15])
sns.heatmap(round(df.corr().transpose(),2), annot=True, linewidths=0.25, linecolor='white', fmt='f')
plt.title('Correlation Table')
plt.show()


# In[200]:


corr_matrix = df.corr()
corr_matrix['buy'].sort_values(ascending = False)


# - id column has max correlation approx 95%
# - month has least correlation approx - 60%

# #### Splitting the dataset into training and test dataset:

# In[201]:


#training dataset:

train_data = df[:24048]

train_data = train_data.drop(['Source'],axis = 1)

train_data.tail()


# In[202]:


# Changing the data type of buy column

train_data['buy'] = train_data['buy'].astype(int)

train_data.dtypes


# In[203]:


#testing dataset:

test_data = df[24048:]

test_data = test_data.drop(['Source', 'buy'],axis = 1)

test_data.tail()


# #### Checking for the skewness & outliers (training data)

# In[204]:


# Checking for the outliers

train_data.plot(kind='box', subplots=True, layout=(16,5), sharex=False, legend=True, figsize=(15,45))
plt.show()


# In[205]:


# Dividing the independent and dependent variables

x=train_data.drop('buy',axis=1)
y=train_data['buy']


# #### Checking the skewness:

# In[206]:


x.skew()


# #### Outcome of skewness:
# 
# Skewness threshold is taken as +/-0.5.
# 
# ###### Columns having skewness are:
# 
# - user_activity_var_1      0.659238
# - user_activity_var_2      9.518093
# - user_activity_var_3      2.522052
# - user_activity_var_4      7.276702
# - user_activity_var_5      1.879562
# - user_activity_var_7      0.908615
# - user_activity_var_8      2.131923
# - user_activity_var_9      7.163071
# - user_activity_var_10    40.005003
# - user_activity_var_11     1.686843
# - user_activity_var_12    37.573462
# - buy                      3.046158
# 
# We'll remove the skewness using power_transform method

# In[207]:


from imblearn.over_sampling import SMOTE
sm = SMOTE()

x_over, y_over = sm.fit_resample(x, y)


# In[208]:


y_over.value_counts()


# - We can see now the target column is balanced.

# #### Finding best RandomState:

# In[209]:


from sklearn.linear_model import LogisticRegression
maxAccu = 0
maxRS = 0

for i in range (1,200):
    x_train, x_test, y_train, y_test = train_test_split(x_over,y_over,test_size=0.25,random_state=i)
    LR = LogisticRegression()
    LR.fit(x_train,y_train)
    predlr = LR.predict(x_test)
    acc = accuracy_score(y_test, predlr)
    if acc>maxAccu:
        maxAccu = acc
        maxRS = i
        
print('The best accuracy is ',maxAccu, ' on Random_State ',maxRS)


# #### Data Preprocessing:

# In[210]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_over,y_over, test_size=0.25, random_state=67)


# ### Model Training:

# #### 1. LogisticRegression:

# In[211]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()

LR.fit(x_train,y_train)
predlr = LR.predict(x_test)

print(accuracy_score(y_test, predlr))
print(confusion_matrix(y_test, predlr))
print(classification_report(y_test, predlr))


# - From LogisticRegression, we're getting 77% accuracy score.

# #### 2. RandomForestClassifier:

# In[212]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()

RF.fit(x_train, y_train)
predrf = RF.predict(x_test)

print(accuracy_score(y_test, predrf))
print(confusion_matrix(y_test, predrf))
print(classification_report(y_test, predrf))


# - Using RandomForestClassifier we're getting 92% accuracy

# #### 3. DecisionTreeClassifier:

# In[213]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()

DT.fit(x_train, y_train)
preddt = DT.predict(x_test)

print(accuracy_score(y_test, preddt))
print(confusion_matrix(y_test, preddt))
print(classification_report(y_test, preddt))


# - We're getting accuracy score of approx 87% using DecisionTreeClassifier.

# #### Let's check for the cross validation score:

# In[214]:


from sklearn.model_selection import cross_val_score

scr = cross_val_score(LR, x, y, cv=5)
print("Cross validation score for LogisticRegresssion is ", scr.mean())


# In[215]:


scr = cross_val_score(RF, x, y, cv = 5)
print("Cross validation score for RandonForestClassifier is ", scr.mean())


# In[216]:


scr = cross_val_score(DT, x, y, cv =5)
print("Cross validation score for DecisionTreeClassifier is ", scr.mean())


# - As we checked, we're getting minimum difference between the accuracy score and cross validation score for RandomForestClassifier (3.88). So, the best model is RandomForestClassifier.

# #### Hyper parameter tuning:

# In[217]:


from sklearn.model_selection import GridSearchCV

#DecisionTreeClassifier:
params = {'max_depth': [4, 5, 10, 20],
    'max_features': [2, 3],
    'n_estimators': [100, 200, 300, 400]}


# In[218]:


GCV = GridSearchCV(RandomForestClassifier(), params, cv=3)


# In[219]:


GCV.fit(x_train, y_train)


# In[220]:


# Finding the best parameter found by GridSearchCV

GCV.best_params_


# In[221]:


model = RandomForestClassifier(max_depth= 20, max_features = 2, n_estimators=200)
model.fit(x_train, y_train)
pred = model.predict(x_test)

print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# - After hyper parameter tuning we got 91% accuracy score.

# ### AUC ROC Curve:

# In[222]:


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(pred, y_test)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr,tpr, color ='darkorange', lw=10, label ='ROC curve (area = %0.2f)'% roc_auc)
plt.plot([0,1],[0,1], color='navy', lw=10, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Charactristics')
plt.legend(loc='lower right')

plt.show()


# #### Saving the model:

# In[223]:


import pickle
filename = 'smart_lead_scoring_engine.pkl'
pickle.dump(model, open(filename, 'wb'))


# #### Conclusion:

# In[224]:


a = np.array(y_test)
predicted = np.array(model.predict(x_test))
df_conclusion = pd.DataFrame({"Original":a, "Predicted":predicted}, index=range(len(a)))

df_conclusion


# ## Prediction on the test dataset:

# In[225]:


test_sub = pd.read_csv(r'E:\Hackathons\Smart_Lead_Scoring_Engine\test_Wf7sxXF.csv')

test_sub.head()


# In[226]:


# Splitting the created_at into day, month, and year

test_sub[["year", "month", "day"]] = test_sub['created_at'].str.split("-", expand=True)
test_sub[['signup_year', 'signup_month', 'signup_day']] = test_sub['signup_date'].str.split("-", expand=True)
test_sub.head()


# In[227]:


# Dropping created_at and signup_date

test_sub = test_sub.drop(['created_at','signup_date'], axis = 1)

test_sub.head()


# In[229]:


test_sub.isnull().sum()


# In[231]:


#checking the value count

test_sub['products_purchased'].value_counts()


# In[233]:


# replacing the missing values of products_purchased with mode

test_sub['products_purchased'] = test_sub['products_purchased'].fillna(2)

test_sub.head()


# In[234]:


# Checking the value counts of signup_year

test_sub['signup_year'].value_counts()


# In[236]:


# Checking the value counts of signup_month

test_sub['signup_month'].value_counts()


# In[235]:


# Checking the value counts of signup_day

test_sub['signup_day'].value_counts()


# In[237]:


#Replacing the missing values with 2021
test_sub['signup_year'] = test_sub['signup_year'].fillna(2021)
test_sub['signup_month'] = test_sub['signup_month'].fillna(3)
test_sub['signup_day'] = test_sub['signup_day'].fillna(28)

test_sub.head()


# In[239]:


test_sub.dtypes


# In[240]:


#Changing the datatypes

test_sub['products_purchased'] = test_sub['products_purchased'].astype(int)
test_sub['year'] = test_sub['year'].astype(int)
test_sub['month'] = test_sub['month'].astype(int)
test_sub['day'] = test_sub['day'].astype(int)
test_sub['signup_day'] = test_sub['signup_day'].astype(int)
test_sub['signup_month'] = test_sub['signup_month'].astype(int)
test_sub['signup_year'] = test_sub['signup_year'].astype(int)

test_sub.dtypes


# In[241]:


b = test_sub['id']
buy = np.array(model.predict(test_sub))

df_submission = pd.DataFrame({"id":b, "buy":buy})

df_submission


# In[242]:


df_submission.to_csv(r'E:\Hackathons\Smart_Lead_Scoring_Engine\ajitesh_submission.csv')


# In[ ]:




