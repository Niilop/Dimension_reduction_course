#!/usr/bin/env python
# coding: utf-8

# A1 a) knn, leave one out and leave one out error (ssd)

# In[204]:


def k_nn (x_test, x_train, lrtd_train):
    labels = []
    distances = np.sqrt(np.sum((x_test[:, np.newaxis, :] - x_train)**2, axis=-1))
    
    for dist in distances:
        min_index = np.argmin(dist)
        # picks the index of the closest object and appends the train data angles from this index
        labels.append(lrtd_train[min_index])
    return labels


# In[205]:


def ssd (pred, test):
    sum_of_squared_diff = np.sum((pred-test)**2)
    # returns the sums between all predicted faces over the true angles
    return sum_of_squared_diff


# In[220]:


def result_compare(old_sum, new_sum):
    if old_sum > new_sum:
        return True
    else:
        return False


# In[206]:


import numpy as np
from sklearn.neighbors import KNeighborsRegressor


# Read the contents of the file into a np array
data = np.loadtxt("noisy_sculpt_faces.txt")

data.shape


# In[287]:


from sklearn.model_selection import train_test_split

X = data[:,0:256]

angles = data[:,256:259]


# Split the data for the model
# The results may vary since the function splits the data randomly and due to very small sample size (100) it affects the results drastically
x_train, x_test = train_test_split(X, test_size=0.2)
angles_train, angles_test =train_test_split(angles, test_size=0.2)


print(angles.shape,X.shape)


# In[288]:


# prediction as a list of angles
pred = k_nn(x_test, x_train, angles_train)

# sum the squared error over all the faces
ssd(pred, angles_test)


# A1 b) forward selection

# In[336]:


def feature_selection(x_train,x_test,angles_train,angles_test):
    all_SSD = []
    filter_indices = []
    current_model = float('inf')


    is_better = True
    while is_better:
        best_index = 0
        current_ssd = float('inf')
        is_better = False
        for i in range(X.shape[1]):
            if i not in filter_indices:

                if not filter_indices:
                    new_indices = [i]
                else:
                    new_indices = np.concatenate((filter_indices, [i]))

                new_x_train = x_train[:,new_indices]
                new_x_test = x_test[:,new_indices]
                # Now the data has been filtered with the chosen indices (selected features) and 
                # a new index across all indexes in the data
                new_ssd = ssd(k_nn(new_x_test, new_x_train, angles_train), angles_test)

                if new_ssd < current_ssd:
                    current_ssd = new_ssd
                    best_index = i

        # Adds the best index to the filter index list            
        filter_indices.append(best_index)           
        # If the new selected features propose a better model than current, the loop will stay on
        new_model = ssd(k_nn(new_x_test, new_x_train, angles_train), angles_test)

        
        if new_model < current_model:
            
            all_SSD.append(new_model)
            current_model = new_model
            is_better = True

    return all_SSD


# A1 c) add best feature in each iteration until the end

# In[466]:


def feature_selection_variant(x_train,x_test,angles_train,angles_test):
   all_SSD = []
   filter_indices = []
   current_model = float('inf')

   for j in range(X.shape[1]):    
       best_index = 0
       current_ssd = float('inf')
       for i in range(X.shape[1]):
           if i not in filter_indices:

               if not filter_indices:
                   new_indices = [i]
               else:
                   new_indices = np.concatenate((filter_indices, [i]))

               new_x_train = x_train[:,new_indices]
               new_x_test = x_test[:,new_indices]
               # Now the data has been filtered with the chosen indices (selected features) 
               # and a new index across all indexes in the data
               new_ssd = ssd(k_nn(new_x_test, new_x_train, angles_train), angles_test)

               if new_ssd < current_ssd:
                   current_ssd = new_ssd
                   best_index = i

       # Adds the best index to the filter index list            
       filter_indices.append(best_index)           
       # If the new selected features propose a better model than current, the loop will stay on
       new_model = ssd(k_nn(new_x_test, new_x_train, angles_train), angles_test)

       all_SSD.append(new_model)
       if new_model < current_model:
           current_model = new_model

   return all_SSD, filter_indices


# In[467]:


import matplotlib.pyplot as plt
SSD = feature_selection(x_train,x_test,angles_train,angles_test)
SSD_var, indices = feature_selection_variant(x_train,x_test,angles_train,angles_test)

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,10))
x = np.arange(1, len(SSD) + 1, 1)
ax1.plot(x,SSD)
ax1.set_xticks(range(1,len(SSD)+1))
ax1.set_ylabel("sum of squared differences")
ax1.set_xlabel("Total features")

x2 = np.arange(1, len(SSD_var) + 1, 1)
ax2.plot(x2,SSD_var)
ax2.set_xticks(range(1,len(SSD_var)+1),10)
ax2.set_ylabel("sum of squared differences")
ax2.set_xlabel("Total features")


# 1A d) 
# 
# With this split in training data and test data, the later algorithm gets better results quite early in the iterations due to there being a small change in the beginning. At 4th feature summed ssd score rises which stops the first algorithm. We can see from graph 2 that 2nd algorithm gets quite stable good results at 100-150 features and then starts getting worse again. Keep in mind that the order of the samples in test and train data will vary if the program is run from the start again, hence the results will be different as well.
# 
# 2A Variable ranking
# 
# 1.
# 
# Pearsons correlation could be used as a simple ranking method in feature selection by showing what features may have correlation to the target variables. The features with high values could be chosen as the variables to be used in a model. However in this case, the correlation would assume that the variables are independent of each other, which might cause problems. Correlation can also only detect linear dependencies between vaariable and the target.
# 
# 2.

# In[475]:


def variable_ranking(x_train,x_test,angles_train,angles_test):
    import operator
    
    all_ssd = {}
    
    for i in range(X.shape[1]):
        
        indences = [i]
        new_x_train = x_train[:,indences]
        new_x_test = x_test[:,indences]
        
        new_ssd = ssd(k_nn(new_x_test, new_x_train, angles_train), angles_test)
        
        # adds a pair, index and the squared error sum to a list
        all_ssd[i]=new_ssd
    sorted_ssd = sorted(all_ssd.items(), key=operator.itemgetter(1))
    return sorted_ssd


# In[476]:


ranking = variable_ranking(x_train,x_test,angles_train,angles_test)
for i in range(len(indices)):
    print(indices[i], ranking[i][0])


# Basically the first number is the same, but after that, the list is completely different
