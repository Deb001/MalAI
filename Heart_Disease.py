##################################################################################
#
# Predicts heart disease from the heart disease data set published
# in Kaggle https://www.kaggle.com/johnsmith88/heart-disease-dataset
#
# Following are data elements:
# age - age in years
# sex - (1 = male; 0 = female)
# cp - chest pain type
# trestbps - resting blood pressure (in mm Hg on admission to the hospital)
# chol - serum cholestoral in mg/dl
# fbs - (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
# restecg - resting electrocardiographic results
# thalach - maximum heart rate achieved
# exang - exercise induced angina (1 = yes; 0 = no)
# oldpeak - ST depression induced by exercise relative to rest
# slope - the slope of the peak exercise ST segment
# ca - number of major vessels (0-3) colored by flourosopy
# thal - 1 = normal; 2 = fixed defect; 3 = reversable defect
# target - 1 or 0
#
###################################################################################

# Import the tools needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

# Import Models from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

# Import Data
df = pd.read_csv('heart_disease.csv')

# Get some quick stats

print(df.describe())
#print(df.shape)
#print(df.info())
#print (df.head())

# Analyze  data for missing values
print(df.isna().sum())
print(df['sex'].value_counts())
ct_sex = pd.crosstab(df.target, df.sex)
# Plot the data
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(12,12))
ax1.set(title="Heart Disease Condition",
        xlabel="Disease",
        ylabel ="Number of People")

ax1 = df['target'].value_counts().plot(kind="bar")

ax2.set(title="Sex Based Disease Condition",
        xlabel="Sex",
        ylabel="Number of People")

ax2 = ct_sex.plot(kind="bar")

plt.show()
