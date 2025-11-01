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
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from Column_map import column_map
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix,roc_curve,ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

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
df_y.loc[df_y.DIQ010==1, 'Value'] = 1
df_y.loc[df_y.DIQ010==3, 'Value'] = 2

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

X_age40 = X.loc[X['RIDAGEYR']>=40]
Y_age40 = y.loc[X['RIDAGEYR']>=40]

# drop duplicated columns
X_age40 = X_age40.loc[:, ~X_age40.T.duplicated()] 


# VIF prio to lasso
# Step 1: Impute missing values (median)
imputer = SimpleImputer(strategy='median')  
X_imputed = pd.DataFrame(imputer.fit_transform(X_age40), columns=X_age40.columns)

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_age40.columns)

# Step 3: Calculate VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

vif_df = calculate_vif(X_scaled)
#vif_df['readable_feature'] = vif_df['feature'].map(column_map)
high_vif_df = vif_df[vif_df['VIF'] > 5]
X_reduced = X_age40.drop(columns=high_vif_df['feature'])

#Visualize VIF
plt.figure(figsize=(10, 6))
sns.barplot(x="VIF", y="feature", data=high_vif_df.sort_values("VIF", ascending=False))
plt.title("Variance Inflation Factor (VIF >5) for Features")
plt.xlabel("VIF")
plt.ylabel("Feature")
plt.show()

# Logistic Regression CV
X_train, X_test, y_train, y_test = train_test_split(X_reduced, Y_age40, test_size=0.2, random_state=1, stratify=Y_age40)

##imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Step 2: Apply SMOTE on the imputed training data
#smote = SMOTE(random_state=1)
#X_resampled, y_resampled = smote.fit_resample(X_train_imputed, y_train)


pipeline = Pipeline([('imp', SimpleImputer(strategy='median')),       
                     ('scaler', StandardScaler()),                     
                     ('logistic', LogisticRegressionCV(
                      penalty='l1',                   
                      solver='liblinear',             
                      Cs=10,                         
                      cv=5,                                  
                      scoring='neg_log_loss',
                      class_weight='balanced',
                      max_iter = 1000,         
                      random_state=1
                      ))
                     ])



pipeline.fit(X_train, y_train)

#Extract CV model
logreg_cv = pipeline.named_steps['logistic']

#Convert C to alpha (alpha = 1/C)
alphas = 1 / logreg_cv.Cs_  
optimal_alpha = 1/logreg_cv.C_[0]

# Get the mean negative log loss score
scores = logreg_cv.scores_[1].mean(axis=0)  # 1 is the positive class label


# Plotting alpha vs. scoring
sns.set_theme()
plt.figure(figsize=(8, 5))
plt.plot(alphas, scores, marker='o', linestyle='-', color='green')
plt.axvline(optimal_alpha, linestyle='--', label=f'Best alpha = {optimal_alpha:.4f}')
plt.xscale('log')
plt.xlabel("Alpha (1/C)")
plt.ylabel("Mean CV Score (Negative Log Loss)")
plt.title("Alpha vs. CV Score (Lasso Logistic Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# plotting the Column Names and Importance of Columns. 
scaled_coefs =logreg_cv.coef_.flatten()
features = X_train.columns

plt.figure(figsize=(14, 12))
plt.bar(features[np.abs(scaled_coefs>0.0001)], np.abs(scaled_coefs[np.abs(scaled_coefs>0.0001)]))
plt.xticks(rotation=75)
plt.grid()
plt.title("LassoCV Feature Selection - Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.ylim(0, 0.05)
plt.grid(True)
plt.show()


# Prediction on test data for confusion matrix
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # for binary classification
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()


# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()


# Output the selected feature and coefficients
scaler = pipeline.named_steps['scaler']
original_scale_coefficients = scaled_coefs/ scaler.scale_

selected_coefficients = {
    name: coef for name, coef in zip(X_test.columns, original_scale_coefficients) if np.abs(coef) > 0.0001}

print("Selected Features and Original-Scale Coefficients:")
for name, coef in selected_coefficients.items():
    readable_name = column_map.get(name, name)
    print(f"{name:<20} {readable_name:<45} {coef:>12.4f}")