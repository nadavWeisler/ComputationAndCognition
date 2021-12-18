#!/usr/bin/env python
# coding: utf-8

# In[99]:


## read xlsx file and create a matrix
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## reading the file and processing it into a numpy array
## write here the path of your file and the name of your file below
## if the file is in the same folder as this code, leave cd an empty string, i.e., cd = ""

### FILL HERE:
cd = "C:\\Users\\user\\Documents\\cognition_computationex\\6_files\\" 
filename = "quiz_12345.xlsx" 
###
quiz_ans = pd.read_excel(cd+filename)
if 'Timestamp' in quiz_ans.columns:
    print("yes")
    quiz_ans.drop(columns=['Timestamp'], inplace = True)

if len(quiz_ans.columns)>6:
    print('You have more than six columns in your excel, not including timestamp')
    
    
## visualizing the results
plt.figure()
quiz_ans.plot.bar()
plt.show()

## create a matrix of the results number_of_samples X number of features (questions)
quiz_mat = quiz_ans.to_numpy()

# CONTINUE FROM HERE

