
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 22:24:44 2016

@author: markj
"""
import datetime as dt
import numpy as np
import pandas as pd
import pdb
#import pykalman #https://pykalman.github.io/
import quandl
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize #http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
quandl.ApiConfig.api_key = "jbXKzkbT2fyxtoW6AcSm"
#Data pre-processing with Quandl https://www.quandl.com/tools/python
contract_months = ['F','G','H','J','K','M','N','Q','U','V','X','Z']
months = {'CME/C':[3,5,7,9,12],'CME/S':[1,3,5,7,9,11],'CME/W':[3,5,7,9,12]}
commodity = 'W'

def daily_price_fit(commodity, model, st_date, end_date):
    '''This function takes daily data for the given commodity (defined globally without
    market prefix) and fits parameters for a given price model to it
    commodity: commodity price to model. Options: [C,S,W]
    model: type of price model to fit. Options: [mr, mrs, mrsv, mrsvsm]
        mr = mean reversion
        mrs = mean reversion with seasonality
        mrsv = mean reversion with seasonality and volatility
        mrsvsm = mean reversion with seasonality, volatility and short-term mean evolution
    st_date: first day from which to use prices
    end_date: last day from which to use prices
    '''
    global data
    global returns
    data = pd.read_csv(commodity + '_cont.csv')
    data = data.loc[lambda df: df.Date >= st_date,:]
    data = np.asarray([np.log(x) for x in list(data['Settle'])])
    returns = [a-b for a,b in zip(data[1:],data[:-1])]
    constraints_min = [0,0,0,-.5,-inf,0,0,-.5,0,0,-inf,0]
    constraints_max = [inf,inf,inf,.5,inf,inf,inf,.5,inf,inf,inf,inf]
    Phi_0 = {}
    Phi_0[mr] = [.5,np.mean(data[1:])] # [k, v1]
    Phi_0[mrs] = Phi_0[mr] + [1, .1] # [k,v1,eta,phi]
    Phi_0[mrsv] = Phi_0[mrs] + [] # [k,v1,eta,phi,lambda_x,sigma,theta,squiggle]
    Phi_0[mrsvsm] = Phi_0[mrsv] + [] # [k,v1,eta,phi,lambda_x,sigma,theta,squiggle,a_1,b_1,lambda_1,epsilon_1]
    Phi = minimize(model, Phi_0[model], method='Nelder-Mead', options={'disp':True})
    print Phi
    return Phi

def mean_reversion_fit(commodity, st_date, end_date):
    '''This function takes daily data for the given commodity (defined globally without
    market prefix) and fits parameters for a single-state mean-reverting model to the
    daily changes.
    k = speed of mean reversion
    v1 = long-term mean
    '''
    global data
    global returns
    data = pd.read_csv(commodity + '_cont.csv')
    data = data.loc[lambda df: df.Date >= st_date,:]
    data = np.asarray([np.log(x) for x in list(data['Settle'])])
    returns = [a-b for a,b in zip(data[1:],data[:-1])]
    print returns
    Phi_0 = [.5,np.mean(data[1:])] #Initial states for k, and theta
    print 'Phi_0',Phi_0
    Phi = minimize(mr, Phi_0, method='Nelder-Mead', options={'disp':True})
    print Phi
    return Phi
    
def mr(Phi_0):
    print Phi_0
    returns_est = (Phi_0[0]/252)*(Phi_0[1]-data[:-1]) #dpt = k(theta - p(t-1))dt
    sse = returns_est - returns
    sse = np.sum([x**2 for x in sse]) #np.sum([abs(x) for x in sse]) #
    print sse
    return sse
    
def mrs_fit(commodity, st_date, end_date):
    '''This function takes daily data for the given commodity (defined globally without
    market prefix) and fits parameters for a two-state mean-reverting model with
    seasonality to the daily changes.
    k = speed of mean reversion
    v1 = long-term mean
    '''
    global data
    global returns
    data = pd.read_csv(commodity + '_cont.csv')
    data = data.loc[lambda df: df.Date >= st_date,:]
    data = np.asarray([np.log(x) for x in list(data['Settle'])])
    returns = [a-b for a,b in zip(data[1:],data[:-1])]
    print returns
    Phi_0 = [.5,np.mean(data[1:]),1,.5] #Initial state: [k,v1,eta,phi]
    print 'Phi_0',Phi_0
    Phi = minimize(mrs, Phi_0, method='Nelder-Mead', options={'disp':True})
    print Phi
    return Phi
    
def mrs(Phi):
    print 'Phi',Phi
    t = np.array(range(len(data)-1))/252.
    h_t = Phi[2]*np.sin(2*np.pi*(t+Phi[3])) #h_t = eta*sin(2*pi*(t+phi))
    returns_est = (Phi[0]/252)*(Phi[1]-(data[:-1]-h_t)) #dpt = k(v1 - (pt-h_t))dt
    sse = returns_est - returns
    sse = np.sum([x**2 for x in sse])
    print 'sse', sse
    return sse

def mrsv(Phi):
    print 'Phi',Phi
    t = np.array(range(len(data)-1))/252.
    h_t = Phi[2]*np.sin(2*np.pi*(t+Phi[3])) #h_t = eta*sin(2*pi*(t+phi))
    returns_est = (Phi[0]/252)*(Phi[1]-(data[:-1]-h_t)) #dpt = k(v1 - (pt-h_t))dt
    sse = returns_est - returns
    sse = np.sum([x**2 for x in sse])
    print 'sse', sse
    return sse

def mrsvsm(Phi):
    print 'Phi',Phi
    t = np.array(range(len(data)-1))/252.
    h_t = Phi[2]*np.sin(2*np.pi*(t+Phi[3])) #h_t = eta*sin(2*pi*(t+phi))
    returns_est = (Phi[0]/252)*(Phi[1]-(data[:-1]-h_t)) #dpt = k(v1 - (pt-h_t))dt
    sse = returns_est - returns
    sse = np.sum([x**2 for x in sse])
    print 'sse', sse
    return sse

def quandl_futures(commodity, month, year):
    call = commodity + contract_months[month-1] + str(year)
    data = quandl.get(call) #"FRED/GDP" "CME/WN2016"
    return data

def quandl_monthly(commodity, c_yr_st, c_mo_st, c_yr_end, c_mo_end):
    '''This function pulls price data from Quandl, calculates monthly averages,
    and writes them to a .csv file.
    '''
    #1 Variable Declarations
    fields = ['Contract_mo','Date','High','Low','Settle','Volume','Open Interest'] #fields to pull from price data
    prices = {}
    for field in fields: prices[field] = []
    #2 Pull data for each contract
    for year in range(c_yr_st, c_yr_end+1):
        for month in months[commodity]:
            if year==c_yr_st and month < c_mo_st: continue #skip months before first month in year 0
            if year== c_yr_end and month > c_mo_end: continue #skip months after last month in year -1
            print 'Gathering:', commodity, year, month
            data = quandl_futures(commodity, month, year)
            #3 Calculate monthly values for each price variable
            px_months = []
            for date in data.index: px_months.append((date.year,date.month))
            px_months = np.sort(pd.value_counts(px_months).keys()) #unique price months, sorted from earliest to latest
            for px_mo in px_months:
                if px_mo == px_months[0] or px_mo == (year,month): continue #skips first and last months of contract
                date_range = pd.date_range(dt.date(px_mo[0],px_mo[1],1),dt.date(px_mo[0],
                                           px_mo[1],1)+relativedelta(months=1,days=-1))
                print 'price_mo', px_mo
                print 'date_range', date_range
                print 'keys', data.keys()
                prices['Date'].append(date_range[0])
                prices['Contract_mo'].append(dt.date(year,month,1))
                prices['Settle'].append(np.mean(data['Settle'][date_range]))
                prices['High'].append(np.max(data['Settle'][date_range]))                
                prices['Low'].append(np.min(data['Settle'][date_range]))                
                prices['Volume'].append(np.sum(data['Volume'][date_range]))
                try: prices['Open Interest'].append(np.mean(data['Open Interest'][date_range]))
                except: prices['Open Interest'].append(np.mean(data['Prev. Day Open Interest'][date_range]))
    #4 Properly format data and print to a .csv
    prices = pd.DataFrame(prices)
    prices.to_csv(commodity[commodity.find('/')+1:]+'_price.csv')
    return prices

def quandl_continuous(commodity, st_date, end_date):
    '''This function pulls nearby daily price data from Quandl and writes them to a .csv file.
    '''
    #1 Variable Declarations
    fields = ['Contract_mo','Date','High','Low','Settle','Volume','Open Interest'] #fields to pull from price data
    prices = {}
    for field in fields: prices[field] = []
    #2 Pull data for each contract
    new_st_date = st_date
    while st_date <= end_date:
        st_date += relativedelta(months=1)
        while st_date.month not in months[commodity]: st_date += relativedelta(months=1)
        print 'Gathering:', commodity, st_date.year, st_date.month
        data = quandl_futures(commodity, st_date.month, st_date.year)
        new_end_date = new_st_date + relativedelta(months=1)
        while new_end_date.month not in months[commodity]: new_end_date += relativedelta(months=1)
        new_end_date = dt.date(new_end_date.year, new_end_date.month,1) - relativedelta(days=1)
        date_range = data.ix[new_st_date:new_end_date].index
        print date_range
        new_st_date = new_end_date + relativedelta(days=1)        
        #3 Select only nearby data from contract data
        prices['Date'].extend(date_range)
        prices['Contract_mo'].extend([st_date.month]*len(date_range))
        print data['Settle'][date_range]
        prices['Settle'].extend(data['Settle'][date_range])
        prices['High'].extend(data['High'][date_range])
        prices['Low'].extend(data['Low'][date_range])
        prices['Volume'].extend(data['Volume'][date_range])
        try: prices['Open Interest'].extend(data['Open Interest'][date_range])
        except: prices['Open Interest'].extend(data['Prev. Day Open Interest'][date_range])

    #4 Properly format data and print to a .csv
    prices = pd.DataFrame(prices)
    prices.to_csv(commodity[commodity.find('/')+1:]+'_cont.csv')
    return prices

#Phi = mrs_fit(commodity, '1985-03-01', dt.date(2016,6,13))
Phi = mean_reversion_fit(commodity, '1985-03-01', dt.date(2016,6,13))
#data = quandl_monthly('CME/W',1959,12,2018,5)
#data = quandl_continuous('CME/W',dt.date(1959,9,1),dt.date.today())
'''Function Results:
MR: [k,v1] - assumes constant short-term mean price v1
    C: 2.202766, [ 0.23177989,  5.74287083]
    S: 1.839071, [ 0.21900827,  6.68320196]
    W: 2.577176, [ 0.34068019,  6.02876028]
MRS: [k,v1,eta,phi] - assumes constant short-term mean price v1
    C: 2.19841, [ 0.2357774 ,  5.74165212,  1.12680647,  0.36587832]
    S: 1.835637, [ 0.22456354,  6.67781703,  1.04945373,  0.3041474 ]
    W: 2.575747, [ 0.34057448,  6.03190998,  0.44480096,  0.68616499]
MRSV: [k,v1,eta,phi,lambda_x,sigma,theta,squiggle] - assumes constant short-term mean price v1
    C: 
    S:
    W: 
MRSVSM: [k,v1,eta,phi,lambda_x,sigma,theta,squiggle,a_1,b_1,lambda_1,epsilon_1]
    C: 
    S:
    W: 
'''
