import pandas as pd
import numpy as np
import datetime
import math
from pandas import ExcelWriter
import pandas_datareader as web

## reading the portfolio data file from CSV
filename                                = 'Q4 2016 - December Monthly Report.csv'
data_reader                             = pd.read_csv(filename)
data_reader['Port contribution']        = 0.00
data_reader['Bench contribution']       = 0.00
num_sectors = 11
num_periods = 59
num_tickers = 63


### Module to get the historical data for tickers in the
start = datetime.datetime(2016,12,30)
end = datetime.datetime(2017,3,28)


#
# stocks = pd.unique(data_reader['Ticker'])
#
# print len(stocks)
#
# g = web.DataReader(stocks,'yahoo', start, end)
#
# h = g['Adj Close']
# writer = ExcelWriter('HistoricalData.xlsx', engine='xlsxwriter')
#
# h.to_excel(writer, sheet_name='Results')
# writer.save()

####################################

#### Read Return Data
return_data = pd.read_csv("Return Data.csv")
data_reader = data_reader.sort('Ticker', ascending=True)
total_effect = np.zeros(num_periods)
active_return = np.zeros(num_periods)
sectors = pd.unique(data_reader['GICS Sector'])
mkt_val = np.array(data_reader['Market Value']).astype('float32')
attr_result = np.zeros([num_periods, num_sectors, 3])
attr_summ = np.zeros([num_periods, 2])

for i in range(num_periods):
    ### Assignments
    port_wts = np.zeros(num_tickers).astype('float32')
    bench_wts = np.zeros(num_tickers).astype('float32')
    returns = np.zeros(num_tickers).astype('float32')
    portfolio_contr = np.zeros(num_tickers).astype('float32')
    bench_contr = np.zeros(num_tickers).astype('float32')
    data_reader['Port contribution'] = 0.00
    data_reader['Bench contribution'] = 0.00
    data_reader['Portfolio Wt'] = 0.00
    data_reader['Bench Wt'] = 0.00
    active_wts = np.zeros(num_sectors)
    port_sector_ret = np.zeros(num_sectors)
    bench_sector_ret = np.zeros(num_sectors)

    tot_port_return = 0.00
    tot_bench_return = 0.00
    returns = np.array(return_data.iloc[[i]])

    if i == 0:
        port_wts = mkt_val / mkt_val.sum()
    else:
        mkt_val = mkt_val*(1+returns)
        port_wts = mkt_val / mkt_val.sum()

    bench_wts[:] = 1.0/num_tickers


    ### Refinements
    portfolio_contr = (port_wts*returns).astype('float32')
    bench_contr = (bench_wts*returns).astype('float32')
    tot_port_return = portfolio_contr.sum()
    tot_bench_return = bench_contr.sum()
    data_reader['Port contribution'] = portfolio_contr[0,:]
    data_reader['Bench contribution'] = bench_contr[0,:]
    if i == 0:
        data_reader['Portfolio Wt'] = port_wts[:]

    else:
        data_reader['Portfolio Wt'] = port_wts[0, :]
    data_reader['Bench Wt'] = bench_wts[:]

    # Sector Level Initializations


    # Calculating the attribution
    count = 0
    for j in sectors:
         data_slice                 = data_reader.loc[data_reader['GICS Sector'] == j]
         active_wts[count]          = sum(data_slice['Portfolio Wt']-data_slice['Bench Wt'])
         port_sector_ret[count]     = sum(data_slice['Port contribution'])/sum(data_slice['Portfolio Wt'])
         bench_sector_ret[count]    = sum(data_slice['Bench contribution']) / sum(data_slice['Bench Wt'])
         attr_result[i,count,0]     = active_wts[count]*(bench_sector_ret[count] - tot_bench_return)
         attr_result[i,count,1]     = sum(data_slice['Portfolio Wt']*(port_sector_ret[count]-bench_sector_ret[count]))
         attr_result[i,count,2]     = attr_result[i,count, 0] + attr_result[i,count,1]
         count += 1

    attr_summ[i,0]   = attr_result[i,:,2].sum()
    attr_summ[i,1]   = tot_port_return - tot_bench_return

print attr_result[1]

###### Risk Attribution ########
std_sec_stock = np.zeros([num_sectors,2])
corr_coef = np.zeros([num_sectors, 2])
ir_input_return = np.zeros([num_sectors,2])
ir_input_risk = np.zeros([num_sectors,2])
std_total = 0.00
total_per_sect_effect = 0.00
total_per_stock_effect = 0.00
total_risk = 0.00
total_active_return = 0.00
Direct_IR = 0.00
IR_Attr = np.zeros([num_sectors,2])
total_sector_IR_contrib = 0.00
total_stock_IR_contrib = 0.00

std_total = np.std(attr_summ[:,0])

for i in range(num_sectors):
    for j in range(2):
        std_sec_stock[i,j] = np.std(attr_result[:,i, j])
        corr_coef[i,j] = np.corrcoef(attr_result[:, i,j], attr_summ[:, j])[0,1]
        ir_input_return[i,j] = np.sum(attr_result[:, i, j])
        ir_input_risk[i,j] = std_sec_stock[i,j]*corr_coef[i,j]


total_risk = np.sum(ir_input_risk)
total_active_return = np.sum(ir_input_return)

for i in range(num_sectors):
    for j in range(2):
        IR_Attr[i,j] = (ir_input_return[i,j]/num_periods)/ir_input_risk[i,j]

total_sector_IR_contrib = np.sum((ir_input_risk[:,0]/total_risk)*IR_Attr[:,0])
total_stock_IR_contrib = np.sum((ir_input_risk[:,1]/total_risk)*IR_Attr[:,1])
IR_Attr_total = total_sector_IR_contrib + total_stock_IR_contrib

Direct_IR = (total_active_return/num_periods)/total_risk

print IR_Attr_total, Direct_IR

























