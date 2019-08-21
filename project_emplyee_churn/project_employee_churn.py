#!/usr/bin/env python
# coding: utf-8

# # Predicting Employee Churn

# # Analyze employee churn. Find out why employees are leaving the company, and learn to predict who will leave the company.

# # Employee Churn Analysis
# 

# Employee churn can be defined as a leak or departure of an intellectual asset from a company or organization. Alternatively, in simple words, you can say, when employees leave the organization is known as churn. Another definition can be when a member of a population leaves a population, is known as churn.

# In Research, it was found that employee churn will be affected by age, tenure, pay, job satisfaction, salary, working conditions, growth potential and employee’s perceptions of fairness. Some other variables such as age, gender, ethnicity, education, and marital status, were essential factors in the prediction of employee churn. In some cases such as the employee with niche skills are harder to replace. It affects the ongoing work and productivity of existing employees. Acquiring new employees as a replacement has its costs such as hiring costs and training costs. Also, the new employee will take time to learn skills at the similar level of technical or business expertise knowledge of an older employee. Organizations tackle this problem by applying machine learning techniques to predict employee churn, which helps them in taking necessary actions.

# In[3]:


#import modules
import numpy as np
import pandas as pd  # for dataframes
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data=pd.read_csv('HR_comma_sep.csv')
data.head()


# In[5]:


data.tail()


# In[6]:


data.info()


# # You can describe 10 attributes in detail as:
# 
# satisfaction_level: It is employee satisfaction point, which ranges from 0-1.
# 
# last_evaluation: It is evaluated performance by the employer, which also ranges from 0-1.
# 
# number_projects: How many numbers of projects assigned to an employee?
# 
# average_monthly_hours: How many average numbers of hours worked by an employee in a month?
# 
# time_spent_company: time_spent_company means employee experience. The number of years spent by an employee in the company.
# 
# work_accident: Whether an employee has had a work accident or not.
# 
# promotion_last_5years: Whether an employee has had a promotion in the last 5 years or not.
# 
# Departments: Employee's working department/division.
# 
# Salary: Salary level of the employee such as low, medium and high.
# 
# left: Whether the employee has left the company or not.

# In[7]:


left = data.groupby('left')
left.mean()


# Here you can interpret, Employees who left the company had low satisfaction level, low promotion rate, low salary, and worked more compare to who stayed in the company.

# In[8]:


data.describe().transpose()


# # Data Visualization

# Employees Left
# 
# Let's check how many employees were left?

# In[11]:


left_count=data.groupby('left').count()
plt.bar(left_count.index.values, left_count['satisfaction_level'])
plt.xlabel('Employees Left Company')
plt.ylabel('Number of Employees')


# In[12]:


data.left.value_counts()


# Here, you can see out of 15,000 approx 3,571 were left, and 11,428 stayed. The no of employee left is 23 % of the total employment.

# In[13]:


3571/15000


# Number of Projects
# 
# Similarly, you can also plot a bar graph to count the number of employees deployed on How many projects?

# In[14]:


num_projects=data.groupby('number_project').count()
plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
plt.xlabel('Number of Projects')
plt.ylabel('Number of Employees')


# Most of the employee is doing the project from 3-5

# Time Spent in Company
# 
# Similarly, you can also plot a bar graph to count the number of employees have based on how much experience?

# In[15]:


time_spent=data.groupby('time_spend_company').count()
plt.bar(time_spent.index.values, time_spent['satisfaction_level'])
plt.xlabel('Number of Years Spend in Company')
plt.ylabel('Number of Employees')


# Most of the employee experience between 2-4 years. Also, there is a massive gap between 3 years and 4 years experienced employee.

# In[16]:


features=['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','Departments ','salary']
fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data)
    plt.xticks(rotation=90)
    plt.title("No. of employee")


# You can observe the following points in the above visualization:
# 
# Most of the employee is doing the project from 3-5.
# 
# There is a huge drop between 3 years and 4 years experienced employee.
# 
# The no of employee left is 23 % of the total employment.
# 
# A decidedly less number of employee get the promotion in the last 5 year.
# 
# The sales department is having maximum no.of employee followed by technical and support
# 
# Most of the employees are getting salary either medium or low.
# 

# In[17]:


fig=plt.subplots(figsize=(10,15))
for i, j in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data, hue='left')
    plt.xticks(rotation=90)
    plt.title("No. of employee")


# You can observe the following points in the above visualization:
# 
# Those employees who have the number of projects more than 5 were left the company.
# 
# The employee who had done 6 and 7 projects, left the company it seems to like that they were overloaded with work.
# 
# The employee with five-year experience is leaving more because of no promotions in last 5 years and more than 6 years experience are not leaving because of affection with the company.
# 
# Those who promotion in last 5 years they didn't leave, i.e., all those left they didn't get the promotion in the previous 5 years.

# # Cluster Analysis:
# 
# Let's find out the groups of employees who left. 
# 
# You can observe that the most important factor for any employee to stay or leave is satisfaction and performance in the company. 
# 
# So let's bunch them in the group of people using cluster analysis.

# In[18]:


#import module
from sklearn.cluster import KMeans
# Filter data
left_emp =  data[['satisfaction_level', 'last_evaluation']][data.left == 1]
# Create groups using K-means clustering.
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(left_emp)


# In[19]:


# Add new column "label" annd assign cluster labels.
left_emp['label'] = kmeans.labels_
# Draw scatter plot
plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'],cmap='Accent')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('3 Clusters of employees who left')


# Here, Employee who left the company can be grouped into 3 type of employees:
# 
# High Satisfaction and High Evaluation(Shaded by green color in the graph), you can also call them Winners.
# 
# Low Satisfaction and High Evaluation(Shaded by blue color(Shaded by green color in the graph), you can also call them Frustrated.
# 
# Moderate Satisfaction and moderate Evaluation (Shaded by grey color in the graph), you can also call them 'Bad match'.

# # Building a Prediction Model

# pre-processing

# In[20]:


# Import LabelEncoder
from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data['salary']=le.fit_transform(data['salary'])
data['Departments ']=le.fit_transform(data['Departments '])


# In[21]:


data.head()


# In[22]:


#Spliting data into Feature and
X=data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Departments ', 'salary']]
y=data['left']


# In[23]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test


# # Model Building
# Let's build employee an churn prediction model.
# 
# Here, you are going to predict churn using Gradient Boosting Classifier.
# 
# First, import the GradientBoostingClassifier module and create Gradient Boosting classifier object using GradientBoostingClassifier() function.
# 
# Then, fit your model on train set using fit() and perform prediction on the test set using predict().

# In[24]:


#Import Gradient Boosting Classifier model
from sklearn.ensemble import GradientBoostingClassifier

#Create Gradient Boosting Classifier
gb = GradientBoostingClassifier()

#Train the model using the training sets
gb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gb.predict(X_test)


# # Evaluating Model Performance

# In[25]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall
print("Recall:",metrics.recall_score(y_test, y_pred))


# Well, you got a classification rate of 97%, considered as good accuracy.
# 
# Precision: Precision is about being precise, i.e., how precise your model is. In other words, you can say, when a model makes a prediction, how often it is correct. In your prediction case, when your Gradient Boosting model predicted an employee is going to leave, that employee actually left 95% of the time.
# 
# Recall: If there is an employee who left present in the test set and your Gradient Boosting model can identify it 92% of the time.

# In[ ]:




