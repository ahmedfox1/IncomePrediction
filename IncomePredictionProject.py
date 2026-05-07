import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer


# Load Data from excel
train_df = pd.read_csv(r".\train_data.csv")
test_df = pd.read_csv(r".\test_data.csv")

X_train = train_df.drop("Income", axis=1)
y_train = train_df["Income"]

X_test = test_df.drop("Income", axis=1)
y_test = test_df["Income"]

#remove dots from income values
train_df["Income"] = train_df["Income"].str.strip().str.replace(".", "", regex=False)
test_df["Income"] = test_df["Income"].str.strip().str.replace(".", "", regex=False)

#remove duplicates
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()

#check missing values(no missing)
missing = train_df.isna().sum()
#print(missing)

#encoding
#split the categorical columns
train_cat = train_df.select_dtypes('str')
test_cat = test_df.select_dtypes('str')

#OneHotEncoder(categorical data)
onh = OneHotEncoder(sparse_output=False,drop='first')
trian_cat_onh = onh.fit_transform(train_cat)
test_cat_onh = onh.transform(test_cat)

#scaling
train_num = train_df.select_dtypes('int64')
test_num = test_df.select_dtypes('int64')

scaler = MinMaxScaler() #0 to 1 range
train_num_scaled = scaler.fit_transform(train_num)
test_num_scaled = scaler.transform(test_num)




