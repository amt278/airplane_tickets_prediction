import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
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


def preprocessingfunTestRegCorr(data):
    print(data.isnull().sum())
    for column in data.columns:
        data[column].fillna(0, inplace=True)
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
    x_encoded.drop(['dep_time','time_taken','days','route'],inplace=True,axis=1)
    return x_encoded,y


def preprocessingfunTestRegK(data):
    print(data.isnull().sum())
    for column in data.columns:
        data[column].fillna(0, inplace=True)
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
    
    return x_encoded,y



def preprocessingfunTest(dtatSet):
    proc = preprocessing.LabelEncoder()
    
    dtatSet['TicketCategory'] = proc.fit_transform(dtatSet['TicketCategory'])
    print(dtatSet)
    for column in dtatSet.columns:
        dtatSet[column].fillna(0, inplace=True)
    x = dtatSet.iloc[:, :-1]  # Futures
    y = dtatSet.iloc[:, -1]  # Goal
    
    # print(y)
    
    
    # drop airline // ch_code == airline
    x.drop(["airline"], axis=1, inplace=True)
    
    
    # drop arr_time // arr_time == dept_time + time_taken
    x.drop(["arr_time"], axis=1, inplace=True)
    
    
    #  dates preprocessing
    dates = dtatSet["date"].str.replace("/", "-")
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
    stop = dtatSet['stop'].str.strip()
    stop.replace("\s+",'',regex=True, inplace = True)
    stop.replace('^non.*','0',regex=True, inplace = True)
    stop.replace('^1-st.*','1',regex=True, inplace = True)
    stop.replace('^2.*','2',regex=True, inplace = True)
    stop = pd.to_numeric(stop, downcast="integer")
    x['stop'] = stop
    
    
    # dep_time preprocessing
    hours = dtatSet["dep_time"].str.split(":", expand=True)[0]
    minutes = dtatSet["dep_time"].str.split(":", expand=True)[1]
    for i in range(len(minutes)):
        minutes[i] = float(minutes[i]) / 60
        hours[i] = float(hours[i]) + float(minutes[i])
    
    # print(hours)
    hours = pd.to_numeric(hours, downcast="float")
    x["dep_time"] = hours
    # print(x)
    
    
    # time_taken preprocessing
    hours = dtatSet["time_taken"].str.split("h ", expand=True)[0]
    minutes = dtatSet["time_taken"].str.split("h ", expand=True)[1]
    minutes = minutes.str.split("m", expand=True)[0]
    minutes = minutes.str.replace("", "0")
    
    for i in range(len(minutes)):
        hours[i] = float(hours[i]) * 60
        minutes[i] = float(minutes[i]) + hours[i]
    
    minutes = pd.to_numeric(minutes, downcast="float")
    x["time_taken"] = minutes
    
    # Encoding dtatSet
    x_obj = x.select_dtypes(include=["object"])
    x_non_obj = x.select_dtypes(exclude=["object"])
    la = LabelEncoder()
    
    for i in range(x_obj.shape[1]):
        x_obj.iloc[:, i] = la.fit_transform(x_obj.iloc[:, i])
    
    x_encoded = pd.concat([x_non_obj, x_obj], axis=1)
    print('X encoded\n', x_encoded)
    # x = x_encoded
    
    # correlation
    corr = x_encoded.corrwith(y)
    # print('correlation with TicketCategory\n', corr)
    
    
    
    final_data = pd.concat([x_encoded, y], axis=1)
    corr = final_data.corr()
    x_encoded.drop(['dep_time','days'],axis=1,inplace=True)
    x = x_encoded
    print('feature selection with correlation\n', x.head())
    return x,y