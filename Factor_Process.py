import pandas as pd
import numpy as np
import xlrd
from scipy import stats
import pandas_datareader as web
import datetime
import math
import time
from pulp import *
from pandas import ExcelWriter
import random


def bloomberg_reader(filename):

    #global factor_data
    sector_file = pd.ExcelFile('Sector Assignment.xlsx')
    sector_list = sector_file.parse('Sheet1')
    print 'Reading workbook 1....'
    workbook = pd.ExcelFile(filename)
    print 'Reading workbook 2....'
    book = xlrd.open_workbook(filename)
    num_sheets = len(workbook.sheet_names)
    sheet_names = workbook.sheet_names
    num_periods = 220
    num_factors = 5
    num_dummy = 11

    sector_dict = pd.Series(['Consumer Discretionary', 'Consumer Staples', 'Energy', 'Financials', 'Health Care', 'Industrials', 'Information Technology', 'Materials', 'Real Estate', 'Telecommunications Services', 'Utilities'], index=np.arange(1,12))

    count_tickers = 0
    for i in sheet_names:
        sheet = workbook.parse(i)
        if (len(sheet) > num_periods):
            count_tickers += 1

    num_tickers = count_tickers
    tickers = np.array([]).astype(str)
    factor_data = np.zeros([num_tickers, num_periods, num_factors+num_dummy])

    ticker_count = 0
    for i in sheet_names:
        print 'Processing sheet:', i
        sheet_ind = book.sheet_by_index(ticker_count)
        sheet = workbook.parse(i)
        #if ticker_count == 0:
        if (len(sheet) > num_periods):
            tickers = np.append(tickers, sheet_ind.cell(0, 0).value)
            sector_assign = sector_list['GICS Sector'].loc[sector_list['Ticker'] == str.split(str(tickers[ticker_count])," ")[0]]
            s1 = sector_assign.values
            s2 = sector_dict.values
            sector_index = np.where(s2 == s1)
            start_limit = len(sheet)-num_periods
            for j in range(num_factors):
                factor_data[ticker_count, :, j] = sheet.ix[:, j+1][start_limit:]
            factor_data[ticker_count, :, num_factors+sector_index[0]] = 1
            ticker_count += 1
    factor_data.dump('FactorData.dat')
    tickers.dump('Tickers.dat')

    return factor_data, tickers

def data_transform(factor_data, transform_type, factor):
    num_tickers, num_periods, num_factors = factor_data.shape
    num_sectors = 11            #Potentially hide this declaration as its defined as a global variable
    num_factors = num_factors - num_sectors
    if transform_type == 0:     #Log Transform
        factor_data[:,:,factor] = np.log(factor_data[:,:,factor])
        factor_data.dump('FactorData_C_T_0.dat')

    if transform_type == 1:     # Cross-sectional z-score
        for i in range(num_periods):
            factor_data[:,i,factor] = (factor_data[:,i,factor] - np.mean(factor_data[:, i, factor][~np.isnan(factor_data[:, i, factor])]))/np.std(factor_data[:, i, factor][~np.isnan(factor_data[:, i, factor])], ddof=0)
        factor_data.dump('FactorData_C_T_1.dat')

    if transform_type == 2:     # Time series z-score
        for i in range(num_tickers):
            factor_data[i,:,factor] = (factor_data[i,:,factor] - np.mean(factor_data[i, :, factor][~np.isnan(factor_data[i, :, factor])]))/np.std(factor_data[i, :, factor][~np.isnan(factor_data[i, :, factor])], ddof=0)
        factor_data.dump('FactorData_C_T_2.dat')

    # here is another approach we can take. Read each cross-sectional period and then load that into a Dataframe.
    if transform_type == 3:     # Sector Neutral Cross-sectional z-score
        sector_count = 0
        for i in range(num_periods):
            print 'Processing period:', i
            period_slice = pd.DataFrame(factor_data[:,i,:], index = np.arange(len(factor_data[:,i,:])))
            for j in range(num_sectors):
                sector_slice = period_slice.loc[period_slice[num_factors+j] == 1.0]
                sector_slice[[0,1,2,3,4]] = (sector_slice[[0,1,2,3,4]] - np.mean(sector_slice[[0,1,2,3,4]][~np.isnan(sector_slice[[0,1,2,3,4]])]))/np.std(sector_slice[[0,1,2,3,4]][~np.isnan(sector_slice[[0,1,2,3,4]])], ddof = 0)
                period_slice.loc[period_slice[num_factors+j] == 1.0] = sector_slice
            factor_data[:,i,:] = period_slice
        factor_data.dump('FactorData_C_T_3.dat')
    return factor_data

def forward_returns(factor_data, num_periods):
    # We want to get the next 12 month forward return data calculated and ready.
    price = pd.ExcelFile('stock_data.xlsx')
    price_data = price.parse('Close Price Data')

    price_data = price_data[len(price_data)-num_periods:]
    #price_data = np.arange(489*220).reshape(220,489)
    #price_data[0,0] = 12

    num_periods, num_tickers = price_data.shape
    return_data = np.zeros([num_periods, num_tickers-1])
    for i in range(num_tickers-1):
        return_data[:,i] = (price_data.ix[:,i+1].shift(-1)/price_data.ix[:,i+1]) - 1

    # Construct forward return matrix
    fret_periods = 12               # Number of periods of forward returns
    fret = np.zeros([num_tickers-1, num_periods, fret_periods])
    col = 0
    for i in range(num_tickers-1):
        for j in range(1,fret_periods+1):
            fret[i,:,j-1] = pd.Series(return_data[:,col]).shift(-j)
        col += 1

    fret.dump('Forward_Returns.dat')

def price_download(tickers, start_date, end_date):
    writer = ExcelWriter('stock_data.xlsx', engine='xlsxwriter')
    for i in range(len(tickers)):
        tickers[i],b,c = str.split(tickers[i].astype('str'), ' ')
    g = web.DataReader(tickers, 'yahoo', start_date, end_date)
    close = g['Adj Close']
    close = close.resample('W-Fri', how = 'last')
    close.to_excel(writer, sheet_name='Close Price Data')

    writer.close()

def data_clean(factor_data):
    num_tickers, num_periods, num_factors = factor_data.shape
    for i in range(num_periods):
        for j in range(num_factors - 1):
            q75, q25 = np.percentile(factor_data[:, i, j][~np.isnan(factor_data[:, i, j])], [75, 25])
            iqr = q75 - q25
            data_median = np.median(factor_data[:, i, j][~np.isnan(factor_data[:, i, j])])
            factor_data[:, i, j][factor_data[:, i, j] < data_median - N * iqr] = data_median - N * iqr
            factor_data[:, i, j][factor_data[:, i, j] >= data_median + N * iqr] = data_median + N * iqr
    return factor_data

#------------------- Code Starting ------------------------#
start = datetime.datetime(2010, 1, 8)
end = datetime.datetime(2016, 12, 30)
num_periods = 220
#------------ Working with a new dataset ---------------#
#filename = 'Factor Data.xlsx'
#factor_data, tickers = bloomberg_reader(filename)
#price_download(tickers, start, end)
#fret = forward_returns(factor_data, num_periods)
#-------------------------------------------------------#

#---------- Pre-loading existing dataset ---------------#
factor_data = np.load('FactorData.dat')
tickers = np.load('Tickers.dat')
fret = np.load('Forward_Returns.dat')
#-------------------------------------------------------#

N = 3
num_sectors = 11
num_tickers, num_periods, num_factors = factor_data.shape
num_factors = num_factors-num_sectors

# --------------------------= Cleaning the data ------------------------------------------------#
factor_data = data_clean(factor_data)
# for i in range(num_periods):
#     for j in range(num_factors-1):
#         q75,q25 = np.percentile(factor_data[:,i,j][~np.isnan(factor_data[:,i,j])], [75,25])
#         iqr = q75-q25
#         data_median = np.median(factor_data[:, i, j][~np.isnan(factor_data[:, i, j])])
#         factor_data[:,i,j][factor_data[:,i,j] < data_median - N*iqr] = data_median-N*iqr
#         factor_data[:, i, j][factor_data[:,i,j] >= data_median + N * iqr] = data_median + N*iqr
#-----------------------------------------------------------------------------------------------#


# ------------------- Apply Data Transforms ------------------------------#
# 0 = Log Transform
# 1 = Cross-sectional z-score
# 2 = Time-series z-score
# 3 = Sector-neutral Cross-sectional z-score
# ------------------------------------------------------------------------#
factor_data[:,:,0] = 1/factor_data[:,:,0]
factor_data[:,:,3] = 1/factor_data[:,:,3]
factor_data = data_transform(factor_data, 1,0)
factor_data = data_transform(factor_data, 1,1)
factor_data = data_transform(factor_data, 1,2)
factor_data = data_transform(factor_data, 1,3)
factor_data = data_transform(factor_data, 1,4)
#-------------------------------------------------------------------------#

# ------------------- Quick Load File for Analysis -----------------------#
#factor_data = np.load('FactorData_C_T_3.dat')
# ------------------------------------------------------------------------#

# ------------------- Loading Forward Return Data ------------------------#

a,b,c = fret.shape
#-------------------------------------------------------------------------#

#------------- Calculating the IC decay for each factor--------------------#
# The code below performs 12 forward period IC decay for each factor.
IC_matrix = np.zeros([num_factors, num_periods,c])
for f in range(num_factors):
    for p in range(num_periods):
        for per in range(c):
            IC_frame = pd.DataFrame()
            IC_frame[0] = factor_data[:,p,f]
            IC_frame[1] = fret[:,p,per]
            IC_matrix[f,p,per] = IC_frame[0].corr(IC_frame[1], method='spearman')
# -------------------------------------------------------------------------#



# --------- Section for the Analysis of IC matrix------------------------#
num_simulations = 10000
hold_period = 4       #In Weeks
int_IC = np.zeros([num_factors, num_simulations, c])

for s in range(num_simulations):
    rand_ind = random.randint(0, num_periods)
    for f in range(num_factors):
        avg_IC = np.average(IC_matrix[f,rand_ind:rand_ind+hold_period ,: ], axis = 0)
        std_IC = np.std(IC_matrix[f,rand_ind:rand_ind+hold_period,:], axis = 0)
        int_IC[f,s,: ] = (avg_IC/std_IC)*(math.sqrt(52))
int_IC.dump('IntIR.dat')

#------------ Load Information Co-efficient Data into Excel ---------------#
writer = pd.ExcelWriter('IC_data.xlsx', engine='xlsxwriter')
for i in range(num_factors):
    temp = pd.DataFrame(IC_matrix[i])
    temp1 = pd.DataFrame(int_IC[i])
    temp.to_excel(writer, sheet_name='Factor_IC'+str(i))
    temp1.to_excel(writer, sheet_name='Factor_IR'+str(i)+'_hp_'+str(hold_period))
writer.close()
#--------------------------------------------------------------------------#
















