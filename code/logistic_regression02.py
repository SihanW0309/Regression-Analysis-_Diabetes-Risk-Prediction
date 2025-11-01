# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 19:24:49 2025

@author: Sihan
"""


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

column_threshold = 0.05 # percentage of column missing values allowed
row_threshold = 0.2 # percentage of row missing values allowed
lasso_cv=10

base_dir = os.path.dirname(__file__)
folder_path_Y= os.path.join(base_dir, 'Dataset/Y')
df_y = None


for filename in os.listdir(folder_path_Y):
    if filename.endswith('.xpt'):
        file_path = os.path.join(folder_path_Y, filename)
        df = pd.read_sas(file_path, format='xport')
        
    
        if 'SEQN' not in df.columns:
            continue
        if df_y is None:
            df_y = df
        else:
            df_y = pd.merge(df_y, df, on='SEQN', how='outer', suffixes=('_left', '_right'))

df_y['Value']=0
df_y.Value[df_y.DIQ010==3]=1
df_y.Value[df_y.DIQ010==1]=2

df_y['feat_selection']=0
df_y.feat_selection[df_y.DIQ010==1]=1


folder_path_Questionnaire_Data= os.path.join(base_dir, 'Dataset/X/Questionnaire Data')
folder_path_Laboratory_Data= os.path.join(base_dir, 'Dataset/X/Laboratory Data')
folder_path_Demo_Data= os.path.join(base_dir, 'Dataset/X/Demographic Variables and Sample Weights')
# folder_path_Dietary_Data= os.path.join(base_dir, 'Dataset/X/Dietary Data')
folder_path_Examination_Data= os.path.join(base_dir, 'Dataset/X/Examination Data')

# folder_path_Data=[folder_path_Questionnaire_Data, folder_path_Laboratory_Data, folder_path_Demo_Data, 
#                   folder_path_Dietary_Data, folder_path_Examination_Data]
folder_path_Data=[folder_path_Questionnaire_Data, folder_path_Laboratory_Data,  
                   folder_path_Examination_Data, folder_path_Demo_Data]

# folder_path_Data=[folder_path_Examination_Data]

df_x = None
columns_to_rename = ['DRDINT', 'DRABF', 'WTDR2D', 'WTDRD1','WTPH2YR']

dropped_columns = []
files_to_exclude=[]

for folder_path in folder_path_Data:
    for filename in os.listdir(folder_path):
        if filename.endswith('.xpt'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_sas(file_path, format='xport')
            if df.columns.isin(columns_to_rename).any():
                df = df.rename(columns={col: col+filename for col in columns_to_rename if col in df.columns})        
            
            if 'SEQN' not in df.columns:
                continue
            df_sub=df[df['SEQN'].isin(df_y['SEQN'])]
            if (df_sub.shape[0]>df_y.shape[0]):
                files_to_exclude.extend([filename])
                continue
            

            df_sub=df_sub.select_dtypes(exclude=['object', 'category'])
            
            #delete columns with more than column_threshold % missing values
            missing = df_sub.isna().mean() 
            cols_to_drop = missing[missing > column_threshold].index
            df_sub=df_sub.drop(columns=cols_to_drop)
            dropped_columns.extend([filename, cols_to_drop])
            
            #delete columns with unique value
            cols_to_drop = df_sub.columns[df_sub.nunique() == 1]
            dropped_columns.extend([filename, cols_to_drop])
            df_sub = df_sub.drop(columns=cols_to_drop)
            
              
            if df_x is None:
                df_x = pd.merge(df_y[['SEQN','Value','feat_selection']], df_sub,  how='left', on='SEQN',suffixes=('_x', '_y'))
            else:
                df_x = pd.merge(df_x, df_sub,  how='left', on='SEQN',suffixes=('_x', '_y'))
            



df_x = df_x.drop_duplicates()

#Loosly delete columns with more than 50% missing values
#This drops largely sparse columns
missing = df_x.isna().mean() 
cols_to_drop = missing[missing > 0.5].index
df_x=df_x.drop(columns=cols_to_drop)
dropped_columns.extend([filename, cols_to_drop])

# Drop rows with more than row_threshold% missing values
threshold = (1 - row_threshold) * df_x.shape[1]
df_x = df_x.dropna(thresh=threshold)
df_x.reset_index(drop=True, inplace=True)

X = df_x.drop(columns=['SEQN','feat_selection','Value'])
y = df_x['feat_selection']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
 

pipeline = Pipeline([('imp', SimpleImputer(strategy='median')),       
                     ('scaler', StandardScaler()),                     
                     ('lassoCV', LassoCV(cv=lasso_cv, random_state=1))])

pipeline.fit(X_train, y_train)
lassoCV = pipeline.named_steps['lassoCV']
sns.set_theme()
plt.plot(lassoCV.alphas_, lassoCV.mse_path_.mean(axis=1), color='green', marker='o')
plt.axvline(lassoCV.alpha_,label=f"Best alpha: {round(lassoCV.alpha_,3)}")
plt.xlabel("alpha")
plt.ylabel("MSE")
plt.title("LassoCV Feature Selection - MSE")
plt.legend()
plt.grid(True)
plt.show()

alpha=lassoCV.alpha_
coeffs=lassoCV.coef_
features = X.columns

# plotting the Column Names and Importance of Columns. 
plt.figure(figsize=(14, 12))
plt.bar(features[np.abs(coeffs>0.0001)], np.abs(coeffs[np.abs(coeffs>0.0001)]))
plt.xticks(rotation=75)
plt.grid()
plt.title("LassoCV Feature Selection - Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.ylim(0, 0.05)
plt.grid(True)
plt.show()


X_test_transformed = pipeline.named_steps['scaler'].transform(
    pipeline.named_steps['imp'].transform(X_test)
)
y_pred = pipeline.named_steps['lassoCV'].predict(X_test_transformed)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("LassoCV CV n:", lasso_cv)
print("LassoCV alpha:", round(alpha,4))
print("LassoCV MSE:", round(mse,4))
print("LassoCV R²:", round(r2,4))


model = SelectFromModel(pipeline.named_steps['lassoCV'], prefit=True)
features = X.columns[model.get_support()]

df_y=df_x['Value']
df_x=df_x[features]
 
 
#Delete columns with more than column_threshold% missing values
missing = df_x.isna().mean() 
cols_to_drop = missing[missing > column_threshold].index
df_x=df_x.drop(columns=cols_to_drop)
dropped_columns.extend([filename, cols_to_drop])         
