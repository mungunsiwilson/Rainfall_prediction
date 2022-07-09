from turtle import color
from click import style
import pandas as pd 
import numpy as np
from requests import head 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt   
import seaborn as sns  
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import pickle

#upload csv dataset
df = pd.read_csv("weatherAUS.csv")
#print(df.shape)
#print(df.isna().sum())
'''
#print(df.info)

#Exploratory data analysis and visualization
#plt.scatter(df.MinTemp, df.MaxTemp, color='blue', marker='.')
#plt.show()
#plt.scatter(df.MaxTemp, df.Temp3pm, color="red", marker='.')
#plt.show()'''
#data cleaning
#creating a list of numeric columns
df_num_cols = df.select_dtypes(include=np.number).columns.tolist()
# filling null values in MaxTemp column with Temp3pm+1
df.MaxTemp = np.where(df.MaxTemp.isnull(), 1.0027043*df.Temp3pm+1.46210074, df.MaxTemp)

#filling null values in Temp3pm using MaxTemp-1

df.Temp3pm = np.where(df.Temp3pm.isnull(), (df.MaxTemp-1.46210074)/1.0027043, df.Temp3pm)
#rplacing missing values in other columns with the mean 
list_col = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
for col in list_col:
    df[col] = np.where(df[col].isnull(), df[col].mean(), df[col])
    

#df = df.dropna(subset=['Rainfall'])
df = df.dropna(subset=['RainTomorrow'])
df = df.dropna(subset=['WindDir9am'])
df = df.dropna(subset=['WindGustDir'])
df = df.dropna(subset=['WindDir3pm'])
df = df.dropna(subset=['RainToday'])

#print(df.isna().sum())
df.to_csv('cleaned.csv', index=False)
#lets scale our numerical data
dfe1 = df[df_num_cols]
dfe1.to_csv('file_name2.csv', index=False)
scaler = MinMaxScaler()
scaler.fit(df[df_num_cols])
df[df_num_cols] = scaler.transform(df[df_num_cols])

# lets encode our categorical data
'''
le = LabelEncoder()
#df.RainToday = le.fit_transform(df['RainToday'])
#df.RainTomorrow = le.fit_transform(df['RainTomorrow'])
cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
for a in cat_cols:
    df[a] = le.fit_transform(df[a])

'''
cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(df[cat_cols])
dfe = df[cat_cols]
dfe.to_csv('file_name.csv', index=False)
#generate colmn names for our new encoded cols
encoded_cols = list(encoder.get_feature_names_out(cat_cols))

df[encoded_cols] = encoder.transform(df[cat_cols])


#split dataset into variables and target
X = df.drop(['RainTomorrow','Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'],axis=1)
Y = df['RainTomorrow','Date']
#splitting my dataset into train val and test sets
#train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
#train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)
#since we are dealing with a time series data we want to split our dataset with respect to time
year = pd.to_datetime(df.Date).dt.year
#plt.hist(year, bins=5)
#plt.show()
x_train = X[year<=2015]
#x_val = X[year==2015]
x_test = X[year>2015]
y_train = Y[year<=2015]
#y_val = Y[year==2015]
y_test = Y[year>2015]
x_train = x_train.drop(['Date'], axis=1)
#x_val = x_val.drop(['Date'], axis=1)
x_test = x_test.drop(['Date'], axis=1)
train = y_train.drop(['Date'], axis=1)
y_test = y_test.drop(['Date'], axis=1)
#y_val = y_val.drop(['Date'], axis=1)
#Training our model
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)
#making predictions 
y_pred = model.predict(x_test)
#find out how good our model performs
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
#save model
with open("model_1", 'wb') as f:
    pickle.dump(model, f)