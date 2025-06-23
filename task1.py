import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


try:
    df=pd.read_csv('train.csv')
except FileNotFoundError:
    print("Error: 'train.csv' not found. please make sure the dataset is in the same directory or provide the full path.")
    exit()


print("Initial DataFrame Info:")
df.info()
print("\nMissing Values before handling:")
print(df.isnull().sum())
print("\nFirst 5 rows of the dataset:")
print(df.head())

#handle missing values

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('cabin', axis=1, inplace=True)

print("\nMissing Values after handling (Age, Embarked, Cabin dropped):")
print(df.isnull().sum())


#Convert categorical.....
df = pd.get_dummies(df, columns=['sex', 'Embarked'], drop_first=True)

df.drop(['Name', 'Ticket','PassengerI'], axis=1, inplace='True')

print("\nDataFrame after categorical encoding and dropping unneeded columns:")
print(df.head())
print("\nDataFrame Info after encoding:")
df.info()


#Normalize/standardize.....

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
numerical_cols = ['Age','Fare','SibSp','Parch']
df[numerical_cols]=scaler.fit_transform(df[numerical_cols])

print("\nDataFrame after numerical feature standardization:")
print(df.head())


#visualize.....

df_original = pd.read_csv('train.csv')
df_original['Age'].fillna(df_original['Age'].median(), inplace=True)
df_original['Fare'].fillna(df_original['Fare'].median(), inplace=True)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df_original['Age'])
plt.title('Boxplot of Age (Original Scale)')

plt.subplot(1, 2, 2)
sns.boxplot(y=df_original['Fare'])
plt.title('Boxplot of Fare (Original Scale)')
plt.tight_layout()
plt.show()


Q1 = df_original['Fare'].quantile(0.25)
Q3 = df_original['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned_outliers = df_original[(df_original['Fare'] >= lower_bound) & (df_original['Fare'] <= upper_bound)]

print(f"\nOriginal number of rows: {len(df_original)}")
print(f"Number of rows after removing Fare outliers: {len(df_cleaned_outliers)}")


print("\n--- Data Cleaning & Preprocessing Complete ---")
print("Final processed DataFrame (df) head:")
print(df.head())
print("\nFinal processed DataFrame (df) info:")
df.info()
