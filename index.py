#USING LOGISTIC REGRESSION

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv('winequalityN.csv')
pd.set_option('display.max_columns', 100)

#dataset info
print(df.head())
print(df.shape)
print(df.info())

print(df['type'].unique())


#fill null values in num columns with mean()
for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())

df.hist(figsize=(20, 10))
plt.show()

#get correlation between features and labels by heatmap
# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

correlation_matrix = numeric_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()

print(df.quality.unique())
"""there are 7 unique values which is not good. So, 
what we can do is, we can consider 1 if quality is above 5 and 0 if it is below 5.
It’s a classification problem."""

df['best quality'] = [1 if x > 5 else 0 for x in df['quality']]
df = df.drop(columns='quality') #dropping quality


#To handle the categorical column 
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# make the splitting.

y = df['best quality']
X = df.drop(columns='best quality')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)

print(X_train.shape, X_test.shape) # (5197, 12), (1300, 12))

#Normalize the data using minmaxscaler
norm = MinMaxScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.fit_transform(X_test)


#model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_prediction = lr_model.predict(X_test)
print(lr_prediction)


#Accuracy
lr_accuracy = accuracy_score(y_test, lr_prediction)
print(lr_accuracy) # 0.7453846153846154

confmat = confusion_matrix(y_test,lr_prediction)
print(confmat)

# Plotting the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()