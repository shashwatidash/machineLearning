
# coding: utf-8

# In[1]:

import pandas as pda


# In[2]:

mydataset = pda.read_csv(r"C:\Users\student\Desktop\DataSets-master\500_Person_Gender_Height_Weight_Index.csv")


# In[3]:

mydataset


# In[4]:

mydataset.shape


# In[6]:

index_name=pda.Series(["Extremely weak","Weak","Normal","Overweight","Obesity","Extreme Obesity"])


# In[7]:

index_name


# In[8]:

female_num=mydataset[mydataset["Gender"]=='Female']
male_num=mydataset[mydataset["Gender"]=='Male']


# In[9]:

lf=len(female_num)
lm=len(male_num)


# In[10]:

lf


# In[11]:

lm


# In[14]:

#female in each...class
female0=len(female_num[female_num["Index"]==0])
female1=len(female_num[female_num["Index"]==1])
female2=len(female_num[female_num["Index"]==2])
female3=len(female_num[female_num["Index"]==3])
female4=len(female_num[female_num["Index"]==4])
female5=len(female_num[female_num["Index"]==5])


# In[15]:

female0


# In[16]:

#male in each...class
male0=len(male_num[male_num["Index"]==0])
male1=len(male_num[male_num["Index"]==1])
male2=len(male_num[male_num["Index"]==2])
male3=len(male_num[male_num["Index"]==3])
male4=len(male_num[male_num["Index"]==4])
male5=len(male_num[male_num["Index"]==5])


# In[17]:

male5


# In[18]:

male_class=pda.Series([male0,male1,male2,male3,male4,male5])
female_class=pda.Series([female0,female1,female2,female3,female4,female5])


# In[20]:

m_f_class=pda.DataFrame({"Male":male_class,"Female":female_class})


# In[24]:

pm_0=male0/245
pm_1=male1/245
pm_2=male2/245
pm_3=male3/245
pm_4=male4/245
pm_5=male5/245
pm_list=pda.Series([pm_0,pm_1,pm_2,pm_3,pm_4,pm_5])


# 

# In[26]:

m_f_class["%Male"]=pm_list


# In[27]:

pf_0=female0/255
pf_1=female1/255
pf_2=female2/255
pf_3=female3/255
pf_4=female4/255
pf_5=female5/255
pf_list=pda.Series([pf_0,pf_1,pf_2,pf_3,pf_4,pf_5])


# In[28]:

m_f_class["%Female"]=pf_list


# In[29]:

m_f_class


# In[30]:

#female_classification
female_num.head()


# In[31]:

female_HWI=female_num.drop("Gender",axis=1)
male_HWI=male_num.drop("Gender",axis=1)


# In[32]:

male_HWI.head()


# In[33]:

#female/male input and target
F_HW=female_HWI.drop("Index",axis=1)
F_I=female_HWI["Index"]
M_HW=male_HWI.drop("Index",axis=1)
M_I=male_HWI["Index"]


# In[34]:

#F_HW(Feature) and F_I(target)...split into train and test
from sklearn.model_selection import train_test_split


# In[42]:

F_X_train,F_X_test,F_Y_train,F_Y_test=train_test_split(F_HW,F_I,test_size=.20,random_state=10)
M_X_train,M_X_test,M_Y_train,M_Y_test=train_test_split(M_HW,M_I,test_size=.20,random_state=10)


# In[43]:

F_X_train.head()


# In[44]:

from sklearn.neighbors import KNeighborsClassifier


# In[48]:

female_teacher=KNeighborsClassifier()
female_learner=female_teacher.fit(F_X_train,F_Y_train)


# In[49]:

male_teacher=KNeighborsClassifier()
male_learner=male_teacher.fit(M_X_train,M_Y_train)


# In[52]:

#Prediction
p_for_male=male_learner.predict([[145,160]])
index_name[p_for_male]
p_for_female=female_learner.predict([[185,110]])
index_name[p_for_female]


# In[53]:

from sklearn.metrics import accuracy_score


# In[56]:

Yfa=F_Y_test
Yma=M_Y_test
#predicted value for male and female
Ypm=male_learner.predict(M_X_test)
Ypf=female_learner.predict(F_X_test)


# In[57]:

macc=float(accuracy_score(Yma,Ypm))*100
facc=float(accuracy_score(Yfa,Ypf))*100


# In[58]:

print("Male accuracy is {} female accuracy {}".format(macc,facc))


# In[ ]:



