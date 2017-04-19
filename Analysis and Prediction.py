# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 22:23:36 2017

@author: Aniket
"""



global fig    #To keep track of figure counts
fig = 0

#import collections
import pandas as pd
#Import module for plotting
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
import numpy as np
#from sklearn.preprocessing import normalize
import seaborn as sb
sb.set_style('darkgrid')

#########################################################################################################

'''
Importing the given data
'''

path = r'E:\Users\Dell\Desktop\Pason\interview_dataset.csv'    #Change this to local path where data is stored
df = pd.read_csv(path)   


#########################################################################################################
'''
Plot Average Usage/15 min Intervals throughout the year
'''

times = pd.DatetimeIndex(df['datetime'])
usage_averages_per15min = []

for i in df.groupby(by = 'HOUR_OF_DAY'):
    times = pd.DatetimeIndex(i[1]['datetime'])
    for j in i[1].groupby([times.hour, times.minute]):
        usage_averages_per15min.append(np.mean(j[1]['USAGE_KWH']))

plt.figure()
fig_handle = sb.tsplot(usage_averages_per15min, color="g")
fig_handle.set_title('Average Usage/15 min Intervals')
fig_handle.set_xlabel('Number of Time Interval')
fig_handle.set_ylabel('Average Usage in That 15 min Interval')
fig_handle.figure.savefig(r'1. Average Usage per 15 min Intervals.png', dpi=600)

#########################################################################################################
'''
Plot Average Usage per Week Day
'''

usage_averages_per_weekday = []

for k in df.groupby(by = 'DAY_OF_WEEK'):
    usage_averages_per_weekday.append(np.mean(k[1]['USAGE_KWH']))

plt.figure()
fig_handle = sb.barplot(y = usage_averages_per_weekday, x = range(7))
fig_handle.set_title('Average Usage per Week Day')
fig_handle.set_xlabel('Week Days')
fig_handle.set_ylabel('Average Usage on the Week Day')
fig_handle.figure.savefig(r'2. Average Usage per Week Day.png', dpi=600)   

#########################################################################################################

'''
Plot for each week day, the distribution of Usage
'''

for i in df.groupby(by = 'DAY_OF_WEEK'):
    usage_averages_for_week = []
    for j in i[1].groupby(by = 'HOUR_OF_DAY'):
        times = pd.DatetimeIndex(j[1]['datetime'])
        for k in j[1].groupby([times.hour, times.minute]):
            usage_averages_for_week.append(np.mean(k[1]['USAGE_KWH']))
            
    plt.figure()
    fig_handle = sb.tsplot(usage_averages_for_week, color="g")
    fig_handle.set_title('Average Usage/15 min Intervals for ' + str(int(i[0])) + ' day')
    fig_handle.set_xlabel('Number of Time Interval')
    fig_handle.set_ylabel('Average Usage in That 15 min Interval')
    fig_handle.figure.savefig(str(int(i[0] + 3)) +'. Average Usage for ' + str(int((i[0]))) + ' Day.png', dpi=600)   
    
#########################################################################################################

'''
Visualizing pairwise relationships in a dataset
'''


plt.figure()
fig_handle = sb.pairplot(df.dropna())
plt.savefig(r'10. Visualizing pairwise relationships in a dataset.png', dpi=600)


#########################################################################################################

'''
Visualize Visibility
'''

plt.figure()
fig_handle = sb.distplot(df['VISIBILITY'], color = 'm', hist = False)
fig_handle.set_title('Distribution of Visibility')
fig_handle.figure.savefig(r'11. Distribution of Visibility.png', dpi=600) 


plt.figure()
fig_handle = sb.distplot(df['VISIBILITY'], kde = False, color = 'r')
fig_handle.set_title('Histogram of Visibility')
fig_handle.figure.savefig(r'12. Histogram of Visibility.png', dpi=600) 


#########################################################################################################

'''
Visualize Relative Humidity
'''

plt.figure()
fig_handle = sb.distplot(df['RELATIVE_HUMIDITY'], color = 'm', hist = False)
fig_handle.set_title('Distribution of Relative Humidity')
fig_handle.figure.savefig(r'15. Distribution of Relative Humidity.png', dpi=600) 


plt.figure()
fig_handle = sb.distplot(df['RELATIVE_HUMIDITY'], color = 'g', kde = False)
fig_handle.set_title('Histogram of Relative Humidity')
fig_handle.figure.savefig(r'16. Histogram of Relative Humidity.png', dpi=600) 

#########################################################################################################

'''
Monthwise Analysis: need for adding the month feature
'''

dfmons = pd.DataFrame(columns = range(96))
times = pd.DatetimeIndex(df['datetime'])
for i in df.groupby([times.month]):
    times = pd.DatetimeIndex(i[1]['datetime'])
    mon = []
    for j in i[1].groupby([times.hour, times.minute]):
        mon.append(np.mean(j[1]['USAGE_KWH']))
    dfmons.loc[i[0]] = mon

plt.figure()
fig_handle = dfmons.iloc[list(range(12)[:6])].T.plot()
fig_handle.set_title('Average Usage/15 min Intervals as per month(Jan through Jun)')
fig_handle.set_xlabel('Number of Time Interval')
fig_handle.set_ylabel('Average Usage in That 15 min Interval in particular month')
fig_handle.figure.savefig(r'13. Average Usage per 15 min Intervals as per month(Jan through Jun).png', dpi=600)

plt.figure()
fig_handle = dfmons.iloc[list(range(12)[6:])].T.plot()
fig_handle.set_title('Average Usage/15 min Intervals as per month(Jul through Dec)')
fig_handle.set_xlabel('Number of Time Interval')
fig_handle.set_ylabel('Average Usage in That 15 min Interval in particular month')
fig_handle.figure.savefig(r'14. Average Usage per 15 min Intervals as per month(Jul through Dec).png', dpi=600)


#########################################################################################################

'''
Monthwise Analysis: need for keeping the visibility feature???
'''

dfmons = pd.DataFrame(columns = range(96))
times = pd.DatetimeIndex(df['datetime'])
for i in df.groupby([times.month]):
    times = pd.DatetimeIndex(i[1]['datetime'])
    mon = []
    for j in i[1].groupby([times.hour, times.minute]):
        mon.append(np.mean(j[1]['VISIBILITY']))
    dfmons.loc[i[0]] = mon

plt.figure()
fig_handle = dfmons.iloc[list(range(12)[:6])].T.plot()
fig_handle.set_title('Visibility in Intervals as per month(Jan through Jun)')
fig_handle.set_xlabel('Number of Time Interval')
fig_handle.set_ylabel('Visibility in That 15 min Interval in particular month')
fig_handle.figure.savefig(r'17. Visibility in Intervals as per month(Jan through Jun).png', dpi=600)

plt.figure()
fig_handle = dfmons.iloc[list(range(12)[6:])].T.plot()
fig_handle.set_title('Visibility in Intervals as per month(Jul through Dec)')
fig_handle.set_xlabel('Number of Time Interval')
fig_handle.set_ylabel('Visibility in That 15 min Interval in particular month')
fig_handle.figure.savefig(r'18. Visibility in Intervals as per month(Jul through Dec).png', dpi=600)

#########################################################################################################

'''
Monthwise Analysis: need for keeping the Relative Humidity feature
'''


dfmons = pd.DataFrame(columns = range(96))
times = pd.DatetimeIndex(df['datetime'])
for i in df.groupby([times.month]):
    times = pd.DatetimeIndex(i[1]['datetime'])
    mon = []
    for j in i[1].groupby([times.hour, times.minute]):
        mon.append(np.mean(j[1]['RELATIVE_HUMIDITY']))
    dfmons.loc[i[0]] = mon

plt.figure()
fig_handle = dfmons.iloc[list(range(12)[:6])].T.plot()
fig_handle.set_title('Relative Humidity in Intervals as per month(Jan through Jun)')
fig_handle.set_xlabel('Number of Time Interval')
fig_handle.set_ylabel('Relative Humidity in That 15 min Interval in particular month')
fig_handle.figure.savefig(r'19. Relative Humidity in Intervals as per month(Jan through Jun).png', dpi=600)

plt.figure()
fig_handle = dfmons.iloc[list(range(12)[6:])].T.plot()
fig_handle.set_title('Relative Humidity in Intervals as per month(Jul through Dec)')
fig_handle.set_xlabel('Number of Time Interval')
fig_handle.set_ylabel('Relative Humidity in That 15 min Interval in particular month')
fig_handle.figure.savefig(r'20. Relative Humidity in Intervals as per month(Jul through Dec).png', dpi=600)


#########################################################################################################

'''
Build a predictive model no given data
'''

from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.linear_model import LinearRegression


#makiing a temporary duplicate
temp = df.copy()

#Extract Predictions
train_targets = np.nan_to_num(temp['USAGE_KWH'])

del temp['USAGE_KWH']
del temp['datetime']

to_be_tested = normalize(np.nan_to_num(temp.iloc[:96]), norm = 'l2', axis = 1)

#To make every sample, a unit vector. This is due to different ranges of feature values
temp = normalize(np.nan_to_num(temp), norm = 'l2', axis = 1)

train = np.nan_to_num(np.array(temp))

model_LR = LinearRegression()
model_LR.fit(train, train_targets)


pred = model_LR.predict(to_be_tested)
print np.sum(abs(pred - df['USAGE_KWH'][:96]))


#########################################################################################################

'''
Build predictive model on modified data 
'''


temp = df.copy()
del temp['TEMP_F']
temp['Number of current interval'] = range(96) * 366
temp['Month'] = pd.DatetimeIndex(temp['datetime']).month
temp['Day'] = pd.DatetimeIndex(temp['datetime']).day

from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import decomposition

#makiing a temporary duplicate
#temp = df.copy()

#Extract Predictions
train_targets = np.nan_to_num(temp['USAGE_KWH'])

del temp['USAGE_KWH']
del temp['datetime']

to_be_tested = (np.nan_to_num(temp.iloc[:96]))

temp = (np.nan_to_num(temp))

train = np.nan_to_num(np.array(temp))

model_LR = svm.SVR()
model_LR.fit(train, train_targets)


pred = model_LR.predict(to_be_tested)
print np.sum(abs(pred - df['USAGE_KWH'][:96]))
