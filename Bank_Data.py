
import  pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/Data Analytics - BankData/bank_transactions.csv")

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)

# Impute missing values for numerical columns with the mean
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Impute missing values for categorical columns with the mode
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Verify if there are any missing values left
missing_values_after_imputation = df.isnull().sum()
print("Missing values in the dataset after imputation:")
print(missing_values_after_imputation)

# Filter rows where bank balance is more than 40000 and transaction amount is more than 2000
filtered_df = df[(df['CustAccountBalance'] > 50121) & (df['TransactionAmount (INR)'] > 80101)]

# Calculate the total number of customers who meet the criteria
total_customers = len(filtered_df)

# Print the filtered data and total number of customers
print("People with bank balance more than 40000 and transaction amount more than 20000:")
print(filtered_df)
print("\nTotal number of customers: ", total_customers)


# Convert 'CustomerDOB' to datetime
df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'])

# Calculate age from date of birth without datetime
current_year = pd.Timestamp.now().year
df['Age'] = current_year - pd.DatetimeIndex(df['CustomerDOB']).year

# Bifurcate age between 18-40
df['Age_Category'] = pd.cut(df['Age'], bins=[18, 40, np.inf], labels=['18-40', 'Above 40'], right=False)

# Count the number of customers in each age category
age_counts = df['Age_Category'].value_counts()

# Print the counts of customers in each age category
print("Counts of customers in each age category:")
print(age_counts)
import pandas as pd

# Load banking dataset from Kaggle (replace 'bank_dataset.csv' with the actual filename)
bank_df = pd.read_csv("C:/Users/DELL/Downloads/Data Analytics - BankData/bank_transactions.csv")

# Impute missing values
print("Missing values in the dataset:")
print(bank_df.isnull().sum())

# Impute missing values
bank_df['CustGender'] = bank_df['CustGender'].ffill()
bank_df['CustLocation'] = bank_df['CustLocation'].ffill()
bank_df['CustAccountBalance'] = bank_df['CustAccountBalance'].ffill()


print("Missing values in the dataset after imputation:")
print(bank_df.isnull().sum())

# Convert CustomerDOB to age
from datetime import datetime

bank_df['CustomerDOB'] = pd.to_datetime(bank_df['CustomerDOB'])
bank_df['Age'] = (datetime.now() - bank_df['CustomerDOB']).astype('<m8[Y]').astype(int)

# Filter customers with bank balance more than 40000 and transaction amount more than 20000
high_value_customers = bank_df[(bank_df['CustAccountBalance'] > 40000) & (bank_df['TransactionAmount (INR)'] > 20000)]

print("People with bank balance more than 40000 and transaction amount more than 20000:")
print(high_value_customers)
print("\nTotal number of customers: ", len(high_value_customers))







