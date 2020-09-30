#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install -U Cython


# In[1]:


pip install -U git+https://github.com/scikit-multiflow/scikit-multiflow


# In[1]:


from skmultiflow.data.random_rbf_generator import RandomRBFGenerator


# In[2]:


RBF = RandomRBFGenerator(n_classes=2, n_features=10)


# In[3]:


RBF.prepare_for_use()


# In[4]:


RBF_data = RBF.next_sample(10000)


# In[6]:


RBF_X = pd.DataFrame(RBF_data[0])
RBF_Y = pd.DataFrame(RBF_data[1])


# In[7]:


pd.DataFrame(RBF_data[1])


# In[10]:


RBF0 = pd.concat([RBF_X, RBF_Y], axis = 1, ignore_index = True)


# In[32]:


RBF0.to_csv("RBF Dataset.csv", index = False)


# In[4]:


from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift
import matplotlib.pyplot as plt


# In[14]:


RBF10 = RandomRBFGeneratorDrift(n_classes = 2, n_features = 10, change_speed = 10)
RBF10.prepare_for_use()
RBF10_data = RBF10.next_sample(10000)


# In[16]:


RBF10_X = pd.DataFrame(RBF10_data[0])
RBF10_Y = pd.DataFrame(RBF10_data[1])
RBF10 = pd.concat([RBF10_X, RBF10_Y], axis = 1, ignore_index = True)


# In[33]:


RBF10.to_csv("RBF Dataset 10.csv", index = False)


# In[19]:


RBF70 = RandomRBFGeneratorDrift(n_classes = 2, n_features = 10, change_speed = 70)
RBF70.prepare_for_use()
RBF70_data = RBF70.next_sample(10000)


# In[20]:


RBF70_X = pd.DataFrame(RBF70_data[0])
RBF70_Y = pd.DataFrame(RBF70_data[1])
RBF70 = pd.concat([RBF70_X, RBF70_Y], axis = 1, ignore_index = True)


# In[34]:


RBF70.to_csv("RBF Dataset 70.csv", index = False)


# In[ ]:


###Importing and manipulating the Data from the .csv files ###


# In[15]:


import numpy as np
import pandas as pd
from skmultiflow.trees.hoeffding_tree import HoeffdingTree


# In[16]:


RBF_df = pd.read_csv("RBF Dataset.csv", delimiter = ',')
RBF10_df = pd.read_csv("RBF Dataset 10.csv", delimiter = ',')
RBF70_df = pd.read_csv("RBF Dataset 70.csv", delimiter = ',')


# In[17]:


###Separating Target values and Feature Values
RBF_X = np.array(RBF_df.iloc[:,:10])

RBF_Y = np.array(RBF_df.iloc[:,10])

RBF10_X = np.array(RBF10_df.iloc[:,:10])

RBF10_Y = np.array(RBF10_df.iloc[:,10])

RBF70_X = np.array(RBF70_df.iloc[:,:10])

RBF70_Y = np.array(RBF70_df.iloc[:,10])


# In[18]:


###Hoeffding Tree Online Classifier###


# In[19]:


### HT Online for RBF###


# In[233]:


HT = HoeffdingTree()
positive = 0
cnt=1
temp_accuracy = []
itr = []
HT_RBF_prediction = []
for i in range(len(RBF_X)):
    tempx = np.array([RBF_X[i]])
    tempy = np.array([RBF_Y[i]])
    prediction = HT.predict(tempx)
    if tempy == prediction:
        positive += 1
    temp_accuracy.append(positive/cnt)
    
    HT_RBF_prediction.append(np.int(HT.predict(tempx)))
    
    HT.partial_fit(tempx, tempy) #Fitting(training) the model
    cnt += 1
    itr.append(cnt) 

ACC_HT_RBF = positive/len(RBF_X)
print(ACC_HT_RBF)
plt.plot(itr, temp_accuracy)
plt.title("Temporal Accuracy of RBF HT Online Classifier")


# In[234]:


### Hoeffding Tree Online Classifier for RBF 10 ###


# In[235]:


HT = HoeffdingTree()
positive = 0
cnt=1
temp_accuracy = []
itr = []
HT_RBF10_prediction = []
for i in range(len(RBF10_X)):
    tempx = np.array([RBF10_X[i]])
    tempy = np.array([RBF10_Y[i]])
    prediction = HT_RBF10.predict(tempx) #Predicting and testing the model
    if tempy == prediction:
        positive += 1
    
    HT_RBF10_prediction.append(np.int(HT.predict(tempx))) #Storing the prediction
    
    temp_accuracy.append(positive/cnt)  
    HT.partial_fit(tempx, tempy) #Fitting(training) the model
    cnt += 1
    itr.append(cnt) 

ACC_HT_RBF10 = positive/len(RBF10_X)
print(ACC_HT_RBF10)
plt.plot(itr, temp_accuracy)
plt.title("Temporal Accuracy of RBF 10 HT Online Classifier")


# In[236]:


### HT Online for RBF 70 ###


# In[321]:


HT = HoeffdingTree()
positive = 0
cnt=1
temp_accuracy = []
itr = []
HT_RBF70_prediction = []
for i in range(len(RBF70_X)):
    tempx = np.array([RBF70_X[i]])
    tempy = np.array([RBF70_Y[i]])
    prediction = HT.predict(tempx) #Predicting and testing the model
    if tempy == prediction:
        positive += 1
    temp_accuracy.append(positive/cnt)
    
    HT_RBF70_prediction.append(np.int(HT.predict(tempx))) #Storing the prediction
    
    HT.partial_fit(tempx, tempy) #Fitting(training) the model
    cnt += 1
    itr.append(cnt) 

ACC_HT_RBF70 = positive/len(RBF70_X)
print(ACC_HT_RBF70)    
plt.plot(itr, temp_accuracy)
plt.title("Temporal Accuracy of RBF 70 HT Online Classifier")


# In[212]:


###Naive Bayes Online Classifier###
from skmultiflow.bayes.naive_bayes import NaiveBayes


# In[246]:


### NB online Classifier (RBF)###
NB = NaiveBayes()
positive = 0
cnt=1
temp_accuracy = []
itr = []
NB_RBF_prediction = []

for i in range(len(RBF_X)):
    tempx = np.array([RBF_X[i]])
    tempy = np.array([RBF_Y[i]])
    prediction = NB.predict(tempx) 
    if tempy == prediction:
        positive += 1
    temp_accuracy.append(positive/cnt) #Testing The model
    

    NB_RBF_prediction.append(np.int(NB.predict(tempx))) #Storing predictions
    
    
    NB.partial_fit(tempx, tempy) #Fitting(training) the model
    cnt += 1
    itr.append(cnt)
    
ACC_NB_RBF = positive/len(RBF_X) #Accuracy of the model
print(positive/len(RBF_X))
plt.plot(itr, temp_accuracy)
plt.title("RBF Temporal Accuracy of NB Classifier")


# In[247]:


### NB online classifier (RBF10) ###

NB = NaiveBayes()
positive = 0
cnt=1
temp_accuracy = []
itr = []
NB_RBF10_prediction = []

for i in range(len(RBF_X)):
    tempx = np.array([RBF10_X[i]])
    tempy = np.array([RBF10_Y[i]])
    prediction = NB.predict(tempx)
    if tempy == prediction:
        positive += 1
    temp_accuracy.append(positive/cnt) #Testing The model
    

    NB_RBF10_prediction.append(np.int(NB.predict(tempx))) #Storing predictions
    
    
    NB.partial_fit(tempx, tempy) #Fitting(training) the model
    cnt += 1
    itr.append(cnt) 

ACC_NB_RBF10 = positive/len(RBF10_X) #Accuracy of the model
print(positive/len(RBF10_X))
plt.plot(itr, temp_accuracy)
plt.title("RBF 10 Temporal Accuracy of NB Classifier")


# In[248]:


### NB Online Classifier (RBF 70) ###

NB = NaiveBayes()
positive = 0
cnt=1
temp_accuracy = []
itr = []
NB_RBF70_prediction = []

for i in range(len(RBF70_X)):
    tempx = np.array([RBF70_X[i]])
    tempy = np.array([RBF70_Y[i]])
    prediction = NB.predict(tempx)
    if tempy == prediction:
        positive += 1
    temp_accuracy.append(positive/cnt) #Testing The model
    

    NB_RBF70_prediction.append(np.int(NB.predict(tempx))) #Storing predictions
    
    
    NB.partial_fit(tempx, tempy) #Fitting(training) the model
    cnt += 1
    itr.append(cnt) 

ACC_NB_RBF70 = positive/len(RBF70_X) #Accuracy of the model
print(positive/len(RBF70_X))
plt.plot(itr, temp_accuracy)
plt.title("RBF 70 Temporal Accuracy of NB Classifier")


# In[249]:


###Multi-Layer Perceptron Online Classifier###
from sklearn.neural_network import MLPClassifier


# In[250]:


###MLP Online Classifier (RBF)###

MLP = MLPClassifier(activation = 'logistic', max_iter=2, hidden_layer_sizes =(4,200) )
positive = 0
cnt=1
temp_accuracy = []
itr= []

MLP_RBF_prediction = []

for i in range(len(RBF_X)):
    tempx = np.array([RBF_X[i]])
    tempy = np.array([RBF_Y[i]])
    if i==0:
        MLP.partial_fit(tempx, tempy, classes=(0,1)) #Initialization of model just for the 1st instance before testing
    

    MLP_RBF_prediction.append(np.int(MLP.predict(tempx))) 
    
    prediction = MLP.predict(tempx) #Testing the model
    if tempy == prediction:
        positive += 1
    MLP.partial_fit(tempx, tempy, classes=(0,1)) #Training(fitting) the Model
    temp_accuracy.append(positive/cnt)
    cnt += 1
    itr.append(cnt) 


# In[251]:


print("Mean Accuracy of Online MLP RBF ")
ACC_MLP_RBF = positive/len(RBF_X)
print(positive/len(RBF_X))
plt.plot(itr, temp_accuracy)
plt.title("RBF Dataset Temporal Accuracy of MLP Classifier")
print(len(temp_accuracy))


# In[324]:


###MLP Online Classifier (RBF 10)###

MLP = MLPClassifier(max_iter=2, hidden_layer_sizes =(4,200) )
positive = 0
cnt=1
temp_accuracy = []
itr = []

MLP_RBF10_prediction = []

for i in range(len(RBF_X)):
    tempx = np.array([RBF10_X[i]])
    tempy = np.array([RBF10_Y[i]])
    if i==0:
        MLP.partial_fit(tempx, tempy, classes=(0,1))

        

    MLP_RBF10_prediction.append(np.int(MLP.predict(tempx)))  #Initialization of model just for 
                                                            #the 1st instance before testing
    
    prediction = MLP.predict(tempx) #Testing the model
    if tempy == prediction:
        positive += 1
    
    MLP.partial_fit(tempx, tempy, classes=(0,1)) #Training(fitting) the Model

    temp_accuracy.append(positive/cnt)
    cnt += 1
    itr.append(cnt) 


# In[325]:


print("Mean Accuracy of Online MLP RBF 10")
ACC_MLP_RBF10 = positive/len(RBF10_X)
print(positive/len(RBF10_X))
plt.plot(itr, temp_accuracy)
plt.title("RBF 10 Dataset Temporal Accuracy of MLP Classifier")


# In[254]:


###MLP Online Classifier (RBF 70 )###

MLP = MLPClassifier(max_iter=2, hidden_layer_sizes =(4,200) )
positive = 0
cnt=1
temp_accuracy = []
itr = []

MLP_RBF70_prediction = []

for i in range(len(RBF70_X)):
    tempx = np.array([RBF70_X[i]])
    tempy = np.array([RBF70_Y[i]])
    if i==0:
        MLP.partial_fit(tempx, tempy, classes=(0,1))

    
    MLP_RBF70_prediction.append(np.int(MLP.predict(tempx))) #Initialization of model just for 
                                                            #the 1st instance before testing
    
    
    prediction = MLP.predict(tempx) #Testing the model
    if tempy == prediction:
        positive += 1
    MLP.partial_fit(tempx, tempy, classes=(0,1)) #Training(fitting) the Model
    temp_accuracy.append(positive/cnt)
    cnt += 1
    itr.append(cnt) 


# In[255]:


print("Mean Accuracy of Online MLP RBF 70")
ACC_MLP_RBF70 = positive/len(RBF70_X)
print(positive/len(RBF70_X))
plt.plot(itr, temp_accuracy)
plt.title("RBF 70 Dataset Temporal Accuracy of MLP Classifier")


# In[266]:


###Online Ensemble Classifier###
HT_RBF_prediction = np.asarray(HT_RBF_prediction)
HT_RBF10_prediction = np.asarray(HT_RBF10_prediction)
HT_RBF70_prediction = np.asarray(HT_RBF70_prediction)

NB_RBF_prediction = np.asarray(NB_RBF_prediction)
NB_RBF10_prediction = np.asarray(NB_RBF10_prediction)
NB_RBF70_prediction = np.asarray(NB_RBF10_prediction)

MLP_RBF_prediction = np.asarray(MLP_RBF_prediction)
MLP_RBF10_prediction = np.asarray(MLP_RBF10_prediction)
MLP_RBF70_prediction = np.asarray(MLP_RBF70_prediction)


# In[ ]:


### Majority Voting ###


# In[242]:


###RBF Online MV###

RBF_MV = []
positive = 0

for i in range(len(RBF_X)):

    temp_sum = np.int(HT_RBF_prediction[i]) + NB_RBF_prediction[i] + MLP_RBF_prediction[i]
    if temp_sum < 2:
        RBF_MV.append(0)
    else:
        RBF_MV.append(1)
    if RBF_MV[i] == RBF_Y[i]:
        positive += 1 

print("MV online Ensemble Accuracy of RBF Dataset")
print(positive/len(RBF_X))
        


# In[243]:


###RBF10 Online MV###

RBF10_MV = []
positive = 0

for i in range(len(RBF10_X)):

    temp_sum = np.int(HT_RBF10_prediction[i]) + NB_RBF10_prediction[i] + MLP_RBF10_prediction[i]
    if temp_sum < 2:
        RBF10_MV.append(0)
    else:
        RBF10_MV.append(1)
    if RBF10_MV[i] == RBF10_Y[i]:
        positive += 1 

print("MV online Ensemble Accuracy of RBF 10 Dataset")
print(positive/len(RBF10_X))
        


# In[244]:


###RBF70 Online MV###

RBF70_MV = []
positive = 0

for i in range(len(RBF70_X)):

    temp_sum = np.int(HT_RBF70_prediction[i]) + NB_RBF70_prediction[i] + MLP_RBF70_prediction[i]
    if temp_sum < 2:
        RBF70_MV.append(0)
    else:
        RBF70_MV.append(1)
    if RBF70_MV[i] == RBF70_Y[i]:
        positive += 1 

print("MV online Ensemble Accuracy of RBF 70 Dataset")
print(positive/len(RBF70_X))
        


# In[ ]:


### Online WMV ###


# In[272]:


### WMV RBF Online ###
RBF_WMV = []
positive = 0

Tot_ACC = ACC_HT_RBF + ACC_NB_RBF + ACC_MLP_RBF 
#weights are given proportionate to accuracy fractions sums up to 1
w_HT = ACC_HT_RBF/Tot_ACC
w_NB = ACC_NB_RBF/Tot_ACC
w_MLP = ACC_MLP_RBF/Tot_ACC

for i in range(len(RBF_X)):

    temp_weighted_sum = HT_RBF_prediction[i]*w_HT + NB_RBF_prediction[i]*w_NB + MLP_RBF_prediction[i]*w_MLP
    if temp_sum < 0.5:
        RBF_WMV.append(0)
    else:
        RBF_WMV.append(1)
    if RBF_WMV[i] == RBF_Y[i]:
        positive += 1 

print("WMV online Ensemble Accuracy of RBF Dataset")
print(positive/len(RBF_X))
        


# In[273]:


### WMV RBF 10 Online ###
RBF10_WMV = []
positive = 0

Tot_ACC = ACC_HT_RBF10 + ACC_NB_RBF10 + ACC_MLP_RBF10
#weights are given proportionate to accuracy fractions sums up to 1
w_HT = ACC_HT_RBF10/Tot_ACC
w_NB = ACC_NB_RBF10/Tot_ACC
w_MLP = ACC_MLP_RBF10/Tot_ACC

for i in range(len(RBF10_X)):

    temp_weighted_sum = HT_RBF10_prediction[i]*w_HT + NB_RBF10_prediction[i]*w_NB + MLP_RBF10_prediction[i]*w_MLP
    if temp_weighted_sum < 0.5:
        RBF10_WMV.append(0)
    else:
        RBF10_WMV.append(1)
    if RBF10_WMV[i] == RBF10_Y[i]:
        positive += 1 

print("WMV online Ensemble Accuracy of RBF 10 Dataset")
print(positive/len(RBF10_X))
        


# In[274]:


### WMV RBF 70 Online ###
RBF70_WMV = []
positive = 0

Tot_ACC = ACC_HT_RBF70 + ACC_NB_RBF70 + ACC_MLP_RBF70 
#weights are given proportionate to accuracy fractions sums up to 1
w_HT = ACC_HT_RBF70/Tot_ACC 
w_NB = ACC_NB_RBF70/Tot_ACC
w_MLP = ACC_MLP_RBF70/Tot_ACC

for i in range(len(RBF70_X)):

    temp_weighted_sum = HT_RBF70_prediction[i]*w_HT + NB_RBF70_prediction[i]*w_NB + MLP_RBF70_prediction[i]*w_MLP
    if temp_weighted_sum < 0.5:
        RBF70_WMV.append(0)
    else:
        RBF70_WMV.append(1)
    if RBF70_WMV[i] == RBF70_Y[i]:
        positive += 1 

print("WMV online Ensemble Accuracy of RBF 70 Dataset")
print(positive/len(RBF10_X))


# In[ ]:


### Batch classification ###


# In[291]:


from sklearn.model_selection import train_test_split
X_train_RBF, X_test_RBF, y_train_RBF, y_test_RBF = train_test_split(RBF_X, RBF_Y, test_size=0.3)

X_train_RBF10, X_test_RBF10, y_train_RBF10, y_test_RBF10 = train_test_split(RBF10_X, RBF10_Y, test_size=0.3)

X_train_RBF70, X_test_RBF70, y_train_RBF70, y_test_RBF70 = train_test_split(RBF70_X, RBF70_Y, test_size=0.3)


# In[292]:


### Hoeffding Tree Batch Classification ###


# In[293]:


### RBF HT ###
HT = HoeffdingTree()

HT.fit(X_train_RBF, y_train_RBF)

HT.score(X_test_RBF, y_test_RBF)


# In[294]:


### RBF 10 HT ###
HT = HoeffdingTree()

HT.fit(X_train_RBF10, y_train_RBF10)

HT.score(X_test_RBF10, y_test_RBF10)


# In[295]:


### RBF HT 70###
HT = HoeffdingTree()

HT.fit(X_train_RBF70, y_train_RBF70)

HT.score(X_test_RBF70, y_test_RBF70)


# In[ ]:


### Naive Bayes Batch Classification ###


# In[296]:


### RBF NB ###
NB = NaiveBayes()

NB.fit(X_train_RBF, y_train_RBF)

NB.score(X_test_RBF, y_test_RBF)


# In[297]:


### RBF 10 NB  ###
NB = NaiveBayes()

NB.fit(X_train_RBF10, y_train_RBF10)

NB.score(X_test_RBF10, y_test_RBF10)


# In[298]:


### RBF 70 NB ###
NB = NaiveBayes()

NB.fit(X_train_RBF70, y_train_RBF70)

NB.score(X_test_RBF70, y_test_RBF70)


# In[ ]:


### Multi Layer Perceptron Batch Classifier ###


# In[299]:


### RBF MLP ###
MLP = MLPClassifier(hidden_layer_sizes = (4,200))

MLP.fit(X_train_RBF, y_train_RBF)

MLP.score(X_test_RBF, y_test_RBF)


# In[300]:


### RBF 10 MLP ###
MLP = MLPClassifier(hidden_layer_sizes = (4,200))

MLP.fit(X_train_RBF10, y_train_RBF10)

MLP.score(X_test_RBF10, y_test_RBF10)


# In[302]:


### RBF 70 MLP ###
MLP = MLPClassifier(hidden_layer_sizes = (4,200))

MLP.fit(X_train_RBF70, y_train_RBF70)

MLP.score(X_test_RBF70, y_test_RBF70)


# In[303]:


### Ensemble batch classification ###
from sklearn.ensemble import VotingClassifier


# In[ ]:


HT = HoeffdingTree()
NB = NaiveBayes()
MLP = MLPClassifier(hidden_layer_sizes = (4,200))


# In[313]:


### RBF MV Ensemble ###

###SOFT
MV_RBF_soft = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'soft') 
MV_RBF_soft = MV_RBF_soft.fit(X_train_RBF, y_train_RBF)
print("Accuracy of MV Ensemble batch classification of RBF dataset-soft")
print(MV_RBF_soft.score(X_test_RBF, y_test_RBF))

###Hard
MV_RBF_hard = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'hard') 
MV_RBF_hard = MV_RBF_hard.fit(X_train_RBF, y_train_RBF)
print("Accuracy of MV Ensemble batch classification of RBF dataset-hard")
print(MV_RBF_hard.score(X_test_RBF, y_test_RBF))


# In[312]:


### RBF 10 MV Ensemble ###

###Soft
MV_RBF10_soft = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'soft') 
MV_RBF10_soft = MV_RBF10_soft.fit(X_train_RBF10, y_train_RBF10)
print("Accuracy of MV Ensemble batch classification of RBF 10 dataset-soft")
print(MV_RBF10_soft.score(X_test_RBF10, y_test_RBF10))

###Hard
MV_RBF10_hard = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'hard') 
MV_RBF10_hard = MV_RBF10_hard.fit(X_train_RBF10, y_train_RBF10)
print("Accuracy of MV Ensemble batch classification of RBF 10 dataset-hard")
print(MV_RBF10_hard.score(X_test_RBF10, y_test_RBF10))


# In[317]:


### RBF 70 MV Ensemble ###
###Soft
MV_RBF70_soft = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'soft') 
MV_RBF70_soft = MV_RBF70_soft.fit(X_train_RBF70, y_train_RBF70)
print("Accuracy of MV Ensemble batch classification of RBF 70 dataset-soft")
print(MV_RBF70_soft.score(X_test_RBF70, y_test_RBF70))
###Hard
MV_RBF70_hard = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'hard') 
MV_RBF70_hard = MV_RBF70_hard.fit(X_train_RBF70, y_train_RBF70)
print("Accuracy of MV Ensemble batch classification of RBF 70 dataset-hard")
print(MV_RBF70_hard.score(X_test_RBF70, y_test_RBF70))


# In[ ]:


### WMV ensemble Batch Classification ##


# In[318]:


### WMV RBF ###
WMV_RBF_soft = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'soft', weights = [2,2,3]) 
WMV_RBF_soft = WMV_RBF_soft.fit(X_train_RBF, y_train_RBF)
print("Accuracy of WMV Ensemble batch classification of RBF dataset-soft")
print(WMV_RBF_soft.score(X_test_RBF, y_test_RBF))


# In[319]:


### WMV RBF 10 ###
WMV_RBF10_soft = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'soft', weights=[2,2,3]) 
WMV_RBF10_soft = WMV_RBF10_soft.fit(X_train_RBF10, y_train_RBF10)
print("Accuracy of WMV Ensemble batch classification of RBF 10 dataset-soft")
print(WMV_RBF10_soft.score(X_test_RBF10, y_test_RBF10))


# In[320]:


### WMV RBF 70 ###
WMV_RBF70_soft = VotingClassifier(estimators=[('HT', HT), ('NB', NB), ('MLP', MLP )], voting = 'soft', weights=[2,2,3]) 
WMV_RBF70_soft = WMV_RBF70_soft.fit(X_train_RBF70, y_train_RBF70)
print("Accuracy of WMV Ensemble batch classification of RBF 70 dataset-soft")
print(WMV_RBF70_soft.score(X_test_RBF70, y_test_RBF70))


# In[ ]:




