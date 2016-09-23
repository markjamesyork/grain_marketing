# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 22:24:44 2016

@author: markj
"""
import datetime as dt
import numpy as np
import os
import pandas as pd
import pdb
#import pykalman #https://pykalman.github.io/
import quandl
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from scipy.optimize import minimize #http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
#Minimization methods that do not require a gradient: 'nelder-mead','powell','BFGS'
quandl.ApiConfig.api_key = "jbXKzkbT2fyxtoW6AcSm"
#Data pre-processing with Quandl https://www.quandl.com/tools/python
contract_months = ['F','G','H','J','K','M','N','Q','U','V','X','Z']
months = {'CME/C':[3,5,7,9,12],'CME/S':[1,3,5,7,9,11],'CME/W':[3,5,7,9,12]}
mkt_yr_start = {'C':dt.date(1960,9,1),'S':dt.date(1960,9,1),'W':dt.date(1960,6,1)}
#months = {'CME/C':[12],'CME/S':[11],'CME/W':[7]} #Use for harvest contracts only
com = 'S'

def daily_price_fit(com, model, st_date, end_date):
    '''This function takes daily data for the given commodity (defined globally without
    market prefix) and fits parameters for a given price model to it
    com: commodity price to model. Options: [C,S,W]
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
    global v1
    global error
    error = []
    data = pd.read_csv('grain_marketing/'+com+'_cont.csv') #('grain_marketing/'+com+'_cont.csv')
    data = data.loc[lambda df: df.Date >= st_date,:]
    dates = pd.to_datetime(data.Date,format='%Y-%m-%d')
    dates = pd.DatetimeIndex(dates).date
    data = np.asarray([np.log(x) for x in list(data['Settle'])])
    returns = [a-b for a,b in zip(data[1:],data[:-1])]
    if model == mrsvsms: #Incorporation of PSD annual stocks
        crops = {'C':'Corn','S':'Oilseed, Soybean','W':'Wheat'}
        crop = crops[com]
        psd = pd.read_csv('ag_data\psd_oilseeds_9.12.16.csv')
        trim = psd[(psd.Commodity_Description==crop)&(psd.Attribute_Description=='Ending Stocks')]
        stocks = trim.groupby(['Market_Year']).sum().Value
        global scarcity
        scarcity = []
        for date in dates[:-1]:
            yr = date.year
            if date < dt.date(date.year,mkt_yr_start[com].month,mkt_yr_start[com].day):
                yr -= 1
            scarcity.append(stocks[yr])
        scarcity = np.asarray([1/i for i in scarcity]) #s_t = 1/(world stocks at time t)

    #constraints_min = [0,0,0,-.5,-inf,0,0,-.5,0,0,-inf]
    #constraints_max = [inf,inf,inf,.5,inf,inf,inf,.5,inf,inf,inf]
    cons = [(0,None),(0,None),(0,None),(-.5,.5),(None,None),(0,None),(0,None),
            (-.5,.5),(0,None),(0,None),(None,None),(0,None),(0,None),(0,None),
            (0,None),(0,None),(None,None),(0,None)]
    Phi_0 = {}
    Phi_0[mr] = [.5,np.mean(data[1:])] # [k, v1]
    Phi_0[mrs] = Phi_0[mr] + [1, np.random.random()-.5] # [k,v1,eta,phi]
    Phi_0[mrsv] = Phi_0[mrs] + [np.random.random()-.5,.5,.5,np.random.random()-.5] # [k,v1,eta,phi,lambda_x,sigma,theta,zeta]
    Phi_0[mrsvsm] = Phi_0[mrsv] + [.2,np.mean(data[1:]),.1] # [k,v1,eta,phi,lambda_x,sigma,theta,zeta,a_1,b_1,lambda_1]  
    Phi_0[mrsvsms] = [.0015,7927,.019,-.0025,.6831,-.0005,.6575,-.0815] #Mod Parameters: [alpha,beta,eta,psi,kappa,lambda_X,theta,zeta] #Old Parameters: [k,v1,eta,phi,lambda_x,sigma_1,theta,zeta,a_1,b_1,lambda_1,a_2,b_2,lambda_2,sigma_2,alpha,beta]
    if model == mrsvsms:
        cons = [(None,None),(0,None),(0,None),(-.5,.5),(0,None),(None,None),(0,None),(-.5,.5)]
    #Phi = minimize(model, Phi_0[model], method='powell',
    #               options={'disp':True}, bounds=cons)
    #print(Phi)
    
    '''
    #Error Plotting Loop
    cutoff = np.percentile(error,90)
    error = np.clip(error,0,cutoff)
    plt.figure()
    plt.plot(range(len(error)),error,linewidth=2,color='r')
    plt.title('Sum of Squared Errors vs Iterations for Returns')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Squared Errors')
    '''
    Phi = [  -1.28918884e-01,   7.77389204e+03,   2.97931979e-01, 3.17969598e+00,   7.81798294e-01,   5.15704638e-02, 3.80323195e+00,   2.94038844e+00]
    sse = mrsvsms(Phi)
    return Phi
    
def mr(Phi_0):
    print(Phi_0)
    returns_est = (Phi_0[0]/252)*(Phi_0[1]-data[:-1]) #dpt = k(theta - p(t-1))dt
    sse = returns_est - returns
    sse = np.sum([x**2 for x in sse]) #np.sum([abs(x) for x in sse]) #
    print(sse)
    return sse
    
def mrs(Phi):
    #Phi: [k,v1,eta,phi]
    print('Phi',Phi)
    t = np.array(range(len(data)-1))/252.
    h_t = Phi[2]*np.sin(2*np.pi*(t+Phi[3])) #h_t = eta*sin(2*pi*(t+phi))
    returns_est = (Phi[0]/252)*(Phi[1]-(data[:-1]-h_t)) #dpt = k(v1 - (pt-h_t))dt
    sse = returns_est - returns
    sse = np.sum([x**2 for x in sse])
    print('sse', sse)
    return sse

def mrsv(Phi):
    #Phi: [k,v1,eta,phi,lambda_x,sigma,theta,zeta]
    print('Phi',Phi)
    t = np.array(range(len(data)-1))/252.
    h_t = Phi[2]*np.sin(2*np.pi*(t+Phi[3])) #h_t = eta*sin(2*pi*(t+phi))
    psi_t = Phi[6]*np.sin(2*np.pi*(t+Phi[7]))
    returns_est = (Phi[0]/252)*(Phi[1]-(data[:-1]-h_t))+Phi[4]*Phi[5]*np.exp(psi_t) #dpt = (k(v1 - (pt-h_t))+lambda_x*sigma*exp(psi_t))dt
    sse = returns_est - returns
    sse = np.sum([x**2 for x in sse])
    print('sse', sse)
    return sse

def mrsvsm(Phi):
    #Phi: [k,v1,eta,phi,lambda_x,sigma,theta,zeta,a_1,b_1,lambda_1]
    print('Phi',Phi)
    t = np.array(range(len(data)-1))/252.
    h_t = Phi[2]*np.sin(2*np.pi*(t+Phi[3])) #h_t = eta*sin(2*pi*(t+phi))
    psi_t = Phi[6]*np.sin(2*np.pi*(t+Phi[7]))
    v1 = [Phi[1]]
    for i in range(1,len(returns)):
        v1.append(Phi[8]*(Phi[9]-v1[-1])+Phi[10]*np.exp(psi_t[i])) #dv1t = (a_1(b_1-v1t)+lambda_1*exp(psi_t))dt
    returns_est = (Phi[0]/252)*(v1-(data[:-1]-h_t))+Phi[4]*Phi[5]*np.exp(psi_t) #dpt = (k(v1 - (pt-h_t))+lambda_x*sigma*exp(psi_t))dt
    sse = returns_est - returns
    sse = np.sum([x**2 for x in sse])
    print('sse', sse)
    return sse

def mrsvsms(Phi):
    #Phi: [alpha,beta,eta,psi,kappa,lambda_X,theta,zeta]
    print('Phi',Phi)
    plot = False
    t = np.array(range(len(data)-1))/252.
    h_t = Phi[2]*np.sin(2*np.pi*(t+Phi[3])) #h_t = eta*sin(2*pi*(t+phi))
    psi_t = Phi[6]*np.sin(2*np.pi*(t+Phi[7]))
    #v1 = [Phi[1]]
    #for i in range(1,len(returns)):
    #    v1.append(Phi[8]*(Phi[9]-v1[-1])+Phi[10]*np.exp(psi_t[i])) #dv1t = (a_1(b_1-v1t)+lambda_1*exp(psi_t))dt
    v1 = np.convolve(data,np.ones((63,))/63,mode='valid')
    v1 = np.reshape(v1,(1,v1.shape[0]))
    v1 = np.append([v1[0,0]]*61,v1)
    if plot == True:
        v12 = v1
        plt.figure()
        plt.plot(range(len(v1)),v1,linewidth=2,color='r')    
        plt.plot(range(len(v12)),v12,linewidth=2,color='b')
        plt.plot(range(len(h_t)),data[:-1]-h_t,linewidth=2,color='y')
        plt.title('Price deviations, averages and models')
        plt.ylabel('Deviation from seasonal mean (log-log)')
        plt.xlabel('Days after start of period')
        plt.legend(['v1_blind','1-month rolling avg','actual'],loc='lower left')
        plt.show()
    v2 = np.asarray([(Phi[0] + Phi[1]*i)**2 for i in scarcity]) #v_2t = (alpha + beta*s_t)^2
    returns_est = (Phi[4]/252)*(v1-(data[:-1]-h_t))+Phi[5]*np.exp(psi_t)*v2 #dX_t = (k(v1 - X_t)+lambda_x*exp(psi_t)*v_2t)dt
    sse = returns_est - returns
    '''Error printing module'''
    plt.figure()
    returns_trim = np.clip(returns,-.08,.08)
    plt.plot(range(len(returns_est)),returns_trim,linewidth=2,color='b')
    plt.plot(range(len(returns_est)),returns_est,linewidth=2,color='r')
    plt.title('Soybean Returns and Model Expecations 2006-2015')
    plt.xlabel('Market Days After Jan 1, 2006')
    plt.ylabel('Change in Log Price')
    plt.legend(['Returns','Model-Estimated Returns'])

    sse = np.sum([x**2 for x in sse])
    print('sse', sse)
    error.append(sse)
    return sse
    
def quandl_futures(com, month, year):
    call = com + contract_months[month-1] + str(year)
    data = quandl.get(call) #"FRED/GDP" "CME/WN2016"
    return data

def quandl_monthly(com, c_yr_st, c_mo_st, c_yr_end, c_mo_end):
    '''This function pulls price data from Quandl, calculates monthly averages,
    and writes them to a .csv file.
    '''
    #1 Variable Declarations
    fields = ['Contract_mo','Date','High','Low','Settle','Volume','Open Interest'] #fields to pull from price data
    prices = {}
    for field in fields: prices[field] = []
    #2 Pull data for each contract
    for year in range(c_yr_st, c_yr_end+1):
        for month in months[com]:
            if year==c_yr_st and month < c_mo_st: continue #skip months before first month in year 0
            if year== c_yr_end and month > c_mo_end: continue #skip months after last month in year -1
            print('Gathering:', com, year, month)
            data = quandl_futures(com, month, year)
            #3 Calculate monthly values for each price variable
            px_months = []
            for date in data.index: px_months.append((date.year,date.month))
            px_months = np.sort(pd.value_counts(px_months).keys()) #unique price months, sorted from earliest to latest
            for px_mo in px_months:
                if px_mo == px_months[0] or px_mo == (year,month): continue #skips first and last months of contract
                date_range = pd.date_range(dt.date(px_mo[0],px_mo[1],1),dt.date(px_mo[0],
                                           px_mo[1],1)+relativedelta(months=1,days=-1))
                print('price_mo', px_mo)
                print('date_range', date_range)
                print('keys', data.keys())
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
    prices.to_csv(com[com.find('/')+1:]+'_price.csv')
    return prices

def quandl_continuous(com, st_date, end_date):
    '''This function pulls nearby daily price data from Quandl and writes them to a .csv file.
    '''
    offset = 1 #Number of months before to cut off contract. E.g. 2 means we will switch from the July to Sep contract on June 1st.
    #1 Variable Declarations
    fields = ['Contract_mo','Date','High','Low','Settle','Volume','Open Interest'] #fields to pull from price data
    prices = {}
    for field in fields: prices[field] = []
    #2 Pull data for each contract
    new_st_date = st_date
    while st_date <= end_date:
        st_date += relativedelta(months=1)
        while st_date.month not in months[com]: st_date += relativedelta(months=1)
        print('Gathering:', com, st_date.year, st_date.month)
        data = quandl_futures(com, st_date.month, st_date.year)
        new_end_date = new_st_date + relativedelta(months=1)
        while new_end_date.month not in (np.asarray(months[com])-offset)%12+1:
            new_end_date += relativedelta(months=1)
        new_end_date = dt.date(new_end_date.year, new_end_date.month,1) - relativedelta(days=1)
        date_range = data.ix[new_st_date:new_end_date].index
        print(date_range)
        new_st_date = new_end_date + relativedelta(days=1)        
        #3 Select only nearby data from contract data
        prices['Date'].extend(date_range)
        prices['Contract_mo'].extend([st_date.month]*len(date_range))
        print(data['Settle'][date_range])
        prices['Settle'].extend(data['Settle'][date_range])
        prices['High'].extend(data['High'][date_range])
        prices['Low'].extend(data['Low'][date_range])
        prices['Volume'].extend(data['Volume'][date_range])
        try: prices['Open Interest'].extend(data['Open Interest'][date_range])
        except: prices['Open Interest'].extend(data['Prev. Day Open Interest'][date_range])

    #4 Properly format data and print to a .csv
    prices = pd.DataFrame(prices)
    prices.to_csv('grain_marketing/'+com[com.find('/')+1:]+'_cont.csv')
    return prices

Phi = daily_price_fit(com, mrsvsms, '2006-01-01', dt.date(2015,12,31))
#data = quandl_monthly('CME/W',1959,12,2018,5)
#data = quandl_continuous('CME/'+com,dt.date(1959,9,1),dt.date.today())
'''Function Results:
Dates: 1985-03-01, 2016-06-13
MR: [k,v1] - assumes constant short-term mean price v1
    C: 2.202766, [ 0.23177989,  5.74287083]
    S: 1.839071, [ 0.21900827,  6.68320196]
    W: 2.577176, [ 0.34068019,  6.02876028]
MRS: [k,v1,eta,phi] - assumes constant short-term mean price v1
    C: 2.19841, [ 0.2357774 ,  5.74165212,  1.12680647,  0.36587832]
    S: 1.835637, [ 0.22456354,  6.67781703,  1.04945373,  0.3041474 ]
    W: 2.575747, [ 0.34061308,  6.03187541,  0.44473761, -0.3138355 ]
MRSV: [k,v1,eta,phi,lambda_x,sigma,theta,zeta] - assumes constant short-term mean price v1
    C: 2.197940, [ 0.26892418,  7.17248678,  1.84960223,  0.37217049, -0.10886533,
        0.0128228 ,  0.64416224, -0.61705236]
    S: 1.834981, [ 0.18451708,  5.35722162,  1.2386509 ,  0.4208389 ,  0.13988522,
        0.00596087,  0.82726892,  0.09791779]
    W: 2.575703, [  3.25816875e-01,   7.48143866e+00,   7.02221950e-02,
         6.64634543e-02,   1.26550094e-01,  -6.57403339e-03, 1.05445785e+00,  -2.00404163e-01]
MRSVSM: [k,v1,eta,phi,lambda_x,sigma,theta,zeta,a_1,b_1,lambda_1]
    C:* 2.198332, [ 0.2471702 ,  2.55719511,  2.337884  ,  0.46633101, -0.46275572,
       -0.01824248,  0.18613135,  0.0303311 , -0.28632821,  7.39277839, 0.01527546]
    S:
    W: 
'''
'''Function Results:
Dates: 2006-03-01, 2016-06-13
MR: [k,v1] - assumes constant short-term mean price v1
    C: 1.066929, [ 0.76227154,  6.1833194 ]
    S: 0.744529, [ 0.74595825,  7.06606417]
    W: 1.250436, [ 1.22006393,  6.41940251]
MRS: [k,v1,eta,phi] - assumes constant short-term mean price v1
    C: 1.065449, [ 0.79015205,  6.18246685,  0.3441067 , -0.58553226]
    S: 0.743053, [ 0.75885829,  7.06017242,  0.35581246, -0.72378875]
    W: 1.249975, [ 1.23218307,  6.42036613,  0.12262585, -0.20291042]
MRSV: [k,v1,eta,phi,lambda_x,sigma,theta,zeta] - assumes constant short-term mean price v1
    C: 1.065299, [ 0.6787206 ,  6.66275248,  0.79128527,  0.36038556, -0.10590679,
        0.00919458,  1.06632978, -0.68503681]
    S: 0.74210019, [ 0.75771812,  6.17246406, -0.97051661, -0.01153361, -0.14935378,
       -0.0123308 ,  1.26854251,  0.04824805]
    W: 
MRSVSM: [k,v1,eta,phi,lambda_x,sigma,theta,zeta,a_1,b_1,lambda_1]
    C:* 1.065012, [ 0.75909553,  2.75026617,  1.51799327,  0.14866136,  0.02909125,
        0.60973422,  0.26964599, -0.38665453,  0.05714488,  3.16050006, 0.01586621]
    S:
    W: 
***Function Results:
Dates: 2006-01-01, 2015,12,31    
MRSVSMS: [alpha,beta,eta,psi,kappa,lambda_X,theta,zeta] #v1 and v2 are calculated from data. v1 is 3 month rolling avg
    C: 1.102 [ -2.70654629e-01,   2.58628698e+04,   4.24673492e-01, 6.78227515e+00,
              -4.42632123e-01,  -5.44462246e-03, 3.18390984e+00,   4.72379216e+00]
    S: .7516 [  -1.28918884e-01,   7.77389204e+03,   2.97931979e-01,
              3.17969598e+00,   7.81798294e-01,   5.15704638e-02, 3.80323195e+00,   2.94038844e+00]
    W: 
    [(None,None),(0,None),(0,None),(-.5,.5),(0,None),(None,None),(0,None),(-.5,.5)]
'''