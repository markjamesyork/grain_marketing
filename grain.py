# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:42:19 2016

@author: markj
"""
import datetime as dt
import math
import numpy as np
import os
import pandas as pd
import urllib.request as urllib
from matplotlib import pyplot as plt
wd = os.getcwd()
crops = {'C':'Corn','S':'Soybeans','W':'Wheat'}
colors = {'C':'b','S':'g','W':'r'}

def download_pdfs():
    #Example URL: 'http://usda.mannlib.cornell.edu/usda/waob/wasde//2000s/2009/wasde-12-10-2009.pdf'
    root_url = 'http://usda.mannlib.cornell.edu/usda/waob/wasde//2010s/'
    dates = pd.date_range(dt.date(2016,9,12),dt.date(2016,9,12))
    for date in dates:
        url = root_url+str(date.year)+'/wasde-'+date.strftime('%m-%d-%Y')+'.pdf'
        try:
            response = urllib.urlopen(url)
            file = open(wd+'/html/'+url[-20:],'wb')
            file.write(response.read())
            file.close()
            print(url)
        except: continue
        
    return

#download_pdfs()

def px_charts():
    #0 Set Parameters
    year_0 = 1993
    year_n = 1999
    commodity = 'S'
    coef = {'C':[0.7868374 ,  4.42726076,  0.55595749,  0.24694035],\
            'S':[7.79982942e-01,   9.72787393e+00,   2.97931979e-01, 3.17969598e+00],\
            'W':[1.49397056,  4.65126329,  8.08304669e-01, 4.90732173e-02]} #[k,v1,eta,phi]
            #MRS Model 'C':[ 0.74518114,  6.17917109,  0.34528775,  0.25962526]
            #MRS Model 'S':[ 0.63498273,  7.05160847,  0.38416453,  0.111981  ]
            #MRS Model 'W':[ 1.22322022,  6.41970721, -0.12312364,  0.15037818]
    #1 Read data
    data = pd.read_csv('grain_marketing/'+commodity+'_cont.csv',index_col=2)
    data.index = pd.to_datetime(data.index,format='%Y/%m/%d')
    data.index = pd.DatetimeIndex(data.index).date

    #2 Add month and DOY columns
    month = []
    WOY = []
    for date in data.index:
        month.append(date.month)
        WOY.append(int((date - dt.date(date.year,1,1)).days/7))
    data['month'] = pd.Series(month,index=data.index)
    data['WOY'] = pd.Series(WOY,index=data.index)
    
    #3 Create averages by month and DOY
    trim = data[(data.index >= dt.date(year_0,1,1))&(data.index <= dt.date(year_n,12,31))]
    monthly = trim.groupby('month').mean()
    weekly = trim.groupby('WOY').mean()

    #4 Make seasonal trend line based on coefficients from the mean reversion with seasonality model
    t = np.asarray([(7*i+3.5)/365 for i in range(52)])
    coef = coef[commodity]
    coef[1] = np.mean(np.log(trim.Settle))
    print('coef[1]',coef[1])
    seasonal = np.exp(coef[1] + coef[2]*np.sin(2*np.dot(np.pi,(t+coef[3]))))
    print('seasonal',seasonal)
    
    #5 Print and chart results
    print(weekly)
    print(monthly)

    #plt.bar(range(1,13),monthly.Settle,align='center',color='b',width=1)

    fig = plt.figure()
    plt.plot(range(52),weekly.Settle[:-1],colors[commodity],linewidth=2) #Weekly Average
    plt.plot(range(52),seasonal,color='y',linewidth=2) #Seasonality equation
    plt.title('Fitted '+crops[commodity]+' Seasonality Equation vs Average by Week of Year')
    plt.ylabel('Nearby Futures Price in cents/bu')
    plt.xlabel('Week of Year')
    plt.legend(['Avg by Week of Year','Seasonality Equation'],loc='lower right')
    return

px_charts()

def psd_chart():
    #data = pd.read_csv('ag_data//psd_grains_pulses_8.12.16.csv')
    #print('keys',data.keys())
    #trim = data[(data['Country_Name'] == 'Argentina')&(data['Commodity_Description']=='Corn')]
    psd = pd.read_csv('ag_data\psd_grains_pulses_9.12.16.csv')
    crop = 'Wheat' #'Oilseed, Soybean'
    print(psd.Commodity_Description.value_counts().keys())
    trim = psd[(psd.Commodity_Description==crop)&(psd.Attribute_Description=='Ending Stocks')]
    trim2 = psd[(psd.Commodity_Description==crop)&(psd.Attribute_Description=='Domestic Consumption')]
    stocks = trim.groupby(['Market_Year']).sum().Value
    print('Stocks:',stocks)
    consumption = trim2.groupby(['Market_Year']).sum().Value
    print('Consumption:',consumption)
    stu = stocks/consumption
    plt.figure()
    plt.plot(stocks.index,stocks,linewidth=2,color='r')
    plt.title('Global Wheat Stocks as of Market Year End')
    plt.ylabel('Thousands of Metric Tons')
    plt.xlabel('Marketing Year')
    #plt.plot(stocks.index,stocks,linewidth=2,color='r')
    #plt.plot(consumption.index,consumption,linewidth=2,color='g')
    return
    
def fac(n):
    if n == 0:
        return 1
    else:
        return n * fac(n-1)
        
#psd_chart()

