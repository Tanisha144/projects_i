'''import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Read the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/Black Friday Sales/test.csv")

# Handling missing values
print(df.isnull().sum())

# Plot boxplot before handling missing values
def plot_boxplot(data, column):
    sns.boxplot(y=column, data=data)
    plt.title(f"Boxplot of {column}")
    plt.show()

graphs = ['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',
          'Stay_In_Current_City_Years', 'Marital_Status',
          'Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Purchase']

for column in graphs:
    plot_boxplot(df, column)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Age', 'City_Category'], drop_first=True)

# Convert 'Stay_In_Current_City_Years' to numerical
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace('4+', 4)
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)

# Prepare data for linear regression
X = df.drop(columns=["Purchase", "Product_ID"])  # Drop the 'Product_ID' column
y = df["Purchase"]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=1, test_size=0.2)

# Train the linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predictions
y_pred = reg.predict(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the linear regression model on scaled features
reg.fit(X_train_scaled, y_train)

# Predictions on scaled features
y_pred_scaled = reg.predict(X_test_scaled)

# Calculate Mean Squared Error on scaled predictions
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
print("Mean Squared Error (scaled features):", mse_scaled)'''

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Read the dataset
df = pd.read_csv("C:/Users/DELL/Downloads/Black Friday Sales/test.csv")

# Handling missing values
df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Age', 'City_Category'], drop_first=True)

# Convert 'Stay_In_Current_City_Years' to numerical
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace('4+', 4)
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].astype(int)

# Prepare data for linear regression
X = df.drop(columns=["Purchase", "Product_ID"])  # Drop the 'Product_ID' column
y = df["Purchase"]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, random_state=1, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Ridge regression model with hyperparameter tuning
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters
best_alpha = grid_search.best_params_['alpha']

# Train the Ridge regression model with the best hyperparameters
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train)

# Predictions on test set
y_pred = ridge.predict(X_test_scaled)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)