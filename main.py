from calendar import month
from re import X
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sympy import true
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
# import plotly.express as px
from sklearn.metrics import r2_score

data = pd.read_csv("airline-price-prediction.csv")

# print(data.isnull().sum())
x = data.iloc[:, :-1]  # Futures
y = data.iloc[:, -1]  # Goal

# price preprocessing
y1 = y.str.split(",", expand=True)[0]
y2 = y.str.split(",", expand=True)[1]
y = y1 + y2
y = pd.to_numeric(y, downcast="float")
y = y.to_frame()
y.columns = ['price']
data['price'] = y
# before removing outliers
sns.boxplot(data['price'])


''' Removing the Outliers '''
Q1 = np.percentile(data['price'], 25, interpolation = 'midpoint')
 
Q3 = np.percentile(data['price'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
 
# print("Old Shape: ", data.shape)
 
# Upper bound
upper = np.where(data['price'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(data['price'] <= (Q1-1.5*IQR))

# print(upper[0])

data.drop(upper[0], axis=0,inplace = True)
data.drop(lower[0], axis=0,inplace = True)
data.reset_index(drop=True, inplace=True)
# print("New Shape: ", data.shape)
# after removing outliers
sns.boxplot(data['price'])

x = data.iloc[:, :-1]  # Futures
y = data.iloc[:, -1]  # Goal

# y1 = y.str.split(",", expand=True)[0]
# y2 = y.str.split(",", expand=True)[1]
# y = y1 + y2
y = pd.to_numeric(y, downcast="integer")
y = y.to_frame()
y.columns = ['price']
data['price'] = y

# drop airline // ch_code == airline
x.drop(["airline"], axis=1, inplace=True)


# drop arr_time // arr_time == dept_time + time_taken
x.drop(["arr_time"], axis=1, inplace=True)
# print(x)


#  dates preprocessing
dates = data["date"].str.replace("/", "-")
dates = dates.str.replace("-0", "-")

days = dates.str.split("-", expand=True)[0]
days = pd.to_numeric(days, downcast="integer")
days = days.to_frame()
days.columns = ['days']

months = dates.str.split("-", expand=True)[1]
months = pd.to_numeric(months, downcast="integer")
months = months.to_frame()
months.columns = ['months']
# dates = days + "/" + months
# x["date"] = dates
x.drop(["date"], axis=1, inplace=True)
x = pd.concat([x, days, months], axis=1)
# print(x)


# stop preprocessing
stop = data['stop'].str.strip()
stop.replace("\s+",'',regex=True, inplace = True)
stop.replace('^non.*','0',regex=True, inplace = True)
stop.replace('^1-st.*','1',regex=True, inplace = True)
stop.replace('^2.*','2',regex=True, inplace = True)
stop = pd.to_numeric(stop, downcast="integer")
x['stop'] = stop


# dep_time preprocessing
hours = data["dep_time"].str.split(":", expand=True)[0]
minutes = data["dep_time"].str.split(":", expand=True)[1]
for i in range(len(minutes)):
    minutes[i] = float(minutes[i]) / 60
    hours[i] = float(hours[i]) + float(minutes[i])

# print(hours)
hours = pd.to_numeric(hours, downcast="float")
x["dep_time"] = hours
# print(x)


# time_taken preprocessing
hours = data["time_taken"].str.split("h ", expand=True)[0]
minutes = data["time_taken"].str.split("h ", expand=True)[1]
minutes = minutes.str.split("m", expand=True)[0]
minutes = minutes.str.replace("", "0")
# print(hours)
# print(minutes)
for i in range(len(minutes)):
    hours[i] = float(hours[i]) * 60
    minutes[i] = float(minutes[i]) + hours[i]

minutes = pd.to_numeric(minutes, downcast="float")
x["time_taken"] = minutes

# x before encoding
# print(x)

# Encoding Data
x_obj = x.select_dtypes(include=["object"])
x_non_obj = x.select_dtypes(exclude=["object"])
la = LabelEncoder()

for i in range(x_obj.shape[1]):
    x_obj.iloc[:, i] = la.fit_transform(x_obj.iloc[:, i])

x_encoded = pd.concat([x_non_obj, x_obj], axis=1)
print('X encoded\n', x_encoded)
# x = x_encoded

# correlation
corr = x_encoded.corrwith(y['price'])
print('correlation with price\n', corr)


def select_k_features(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectKBest(score_func=mutual_info_regression, k=9)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs


# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x_encoded, y, test_size=0.20, shuffle=True)


# K best
X_train_fs, X_test_fs, fs = select_k_features(X_train, y_train, X_test)
# print('feature selection with K best: ', fs.feature_names_in_())
# fit the model
model = LinearRegression()
model.fit(X_train_fs, y_train)
# evaluate the model
yhat = model.predict(X_test_fs)
# plt.scatter(X_test_fs[:,0], y_test)
# plt.plot(X_test_fs[:,0], yhat, color='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
trueValue = np.asarray(y_test)[0]
predictedValue = yhat[0]
print('Linear regression')
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), yhat))
print("True value  in the test set: " + str(trueValue))
print("Predicted value  in the test set: " + str(predictedValue))
print('score:', r2_score(y_test, yhat))
print('---------------------------------------------------------------')


# linear regression
# cls = linear_model.LinearRegression(normalize=True).fit(X_train,y_train)
# # cls.fit(X_train,y_train)
# prediction= cls.predict(X_test)
# print('Co-efficient of linear regression',cls.coef_)
# print('Intercept of linear regression model',cls.intercept_)

# trueValue = np.asarray(y_test)[0]
# predictedValue = prediction[0]
# print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
# print("True value  in the test set: " + str(trueValue))
# print("Predicted value  in the test set: " + str(predictedValue))

# %%

# feature selection by correlation
final_data = pd.concat([x_encoded, y], axis=1)
corr = final_data.corr()
topFeature = corr.index[abs(corr['price']) > 0.05]
top_corr = final_data[topFeature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
topFeature = topFeature.delete(-1)
x = x_encoded[topFeature]
print('feature selection with correlation\n', x.head())

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, shuffle=True)

# polynomial regression
poly_features = PolynomialFeatures(degree=4)


# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred = poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

# print("Co-efficient of linear regression", poly_model.coef_)
# print("Intercept of linear regression model", poly_model.intercept_)

trueValue = np.asarray(y_test)[0]
predictedValue = prediction[0]
print('Polynomial regression')
print("Mean Square Error", metrics.mean_squared_error(y_test, prediction))
print("True value  in the test set: " + str(trueValue))
print("Predicted value  in the test set: " + str(predictedValue))
print('score:', r2_score(y_test, prediction))
print('---------------------------------------------------------------')

