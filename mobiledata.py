#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train=pd.read_csv("train.csv")
train


# In[2]:


print(train.columns)


# In[3]:


train.isnull().sum()


# In[4]:


train.price_range.value_counts()


# In[5]:


import matplotlib.pyplot as plt
plt.boxplot(train.fc                                        )


# In[6]:


plt.boxplot(train.wifi                    )


# In[10]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# In[ ]:


train.isnull().sum()


# In[7]:


x=train.iloc[:,:20]
y=train.iloc[:,20]


# In[8]:


y


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.25, random_state = 0)


# In[ ]:


len(x_train)


# In[ ]:


len(x_test)


# In[11]:


from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression()
logistic.fit(x_train,y_train)


# In[12]:


x_test


# In[13]:


train_score =logistic.score(x_train,y_train)
train_score


# In[14]:


test_score =logistic.score(x_test,y_test)
test_score


# In[15]:


from sklearn.naive_bayes import GaussianNB
naive= GaussianNB()
naive.fit(x_train, y_train)


# In[16]:


train_score =naive.score(x_train,y_train)
train_score


# In[17]:


test_score =naive.score(x_test,y_test)
test_score


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(x_train, y_train)


# In[19]:


train_score =knn.score(x_train,y_train)
train_score


# In[20]:


test_score =knn.score(x_test,y_test)
test_score


# In[21]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(x_train, y_train)


# In[22]:


train_score =tree.score(x_train,y_train)
train_score


# In[23]:


test_score =tree.score(x_test,y_test)
test_score


# In[24]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
tree.fit(x_train, y_train)


# In[25]:


train_score =tree.score(x_train,y_train)
train_score


# In[26]:


test_score =tree.score(x_test,y_test)
test_score


# In[27]:


from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
random.fit(x_train, y_train)


# In[28]:


train_score =random.score(x_train,y_train)
train_score


# In[29]:


test_score =random.score(x_test,y_test)
test_score


# In[31]:


from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', random_state = 0)
svc.fit(x_train, y_train)


# In[32]:


train_score =svc.score(x_train,y_train)
train_score


# In[33]:


test_score =svc.score(x_test,y_test)
test_score


# In[34]:


from sklearn.svm import SVC
svc2 = SVC(kernel = 'poly', random_state = 0)
svc2.fit(x_train, y_train)


# In[35]:


train_score =svc2.score(x_train,y_train)
train_score


# In[36]:


test_score =svc.score(x_test,y_test)
test_score


# In[37]:


new=pd.read_csv("test.csv")
new


# In[38]:


print(new.columns)


# In[39]:


x=train.iloc[:,:20]


# In[40]:


len(x)


# In[41]:


new=pd.read_csv("test.csv")
new


# In[42]:


print(train.columns)


# In[43]:


print(new.columns)


# In[46]:


x_new=new.iloc[:,1:21]
x_new


# In[47]:


x_new["predict"]=logistic.predict(x_new)
x_new

