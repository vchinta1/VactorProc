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
num_sectors                             = 11                # Number of sectors
num_periods                             = 59                # Number of periods of data
num_tickers                             = 63                # Number of stocks being analyzed


# ------ Module to get the historical data for portfolio tickers ---------------
# start         = datetime.datetime(2016,12,30)
# end           = datetime.datetime(2017,3,28)
# stocks        = pd.unique(data_reader['Ticker'])
# g             = web.DataReader(stocks,'yahoo', start, end)
# h             = g['Adj Close']


# --------- Writing the data to excel ------------------------------------------
# writer        = ExcelWriter('HistoricalData.xlsx', engine='xlsxwriter')
# h.to_excel(writer, sheet_name='Results')
# writer.save()
#------------------------------------------------------------------------------

#--------------- Initializing data variables ----------------------------------
return_data         = pd.read_csv("Return Data.csv")
data_reader         = data_reader.sort('Ticker', ascending=True)
attr_result         = np.zeros([num_periods, num_sectors, 3])
attr_summ           = np.zeros([num_periods, 2])
total_effect        = np.zeros(num_periods)
active_return       = np.zeros(num_periods)
sectors             = pd.unique(data_reader['GICS Sector'])
mkt_val             = np.array(data_reader['Market Value']).astype('float32')

#-------------- Iterating through the periods --------------------------#
for i in range(num_periods):
    # --------- Initializing variables for periodic calculations -------#
    port_wts            = np.zeros(num_tickers).astype('float32')
    bench_wts           = np.zeros(num_tickers).astype('float32')
    returns             = np.zeros(num_tickers).astype('float32')
    returns             = np.array(return_data.iloc[[i]])
    portfolio_contr     = np.zeros(num_tickers).astype('float32')
    bench_contr         = np.zeros(num_tickers).astype('float32')
    active_wts          = np.zeros(num_sectors)
    port_sector_ret     = np.zeros(num_sectors)
    bench_sector_ret    = np.zeros(num_sectors)
    tot_port_return     = 0.00
    tot_bench_return    = 0.00

    #--------- Initializing new columns on the dataframe ------------#
    data_reader['Bench Wt']             = 0.00
    data_reader['Portfolio Wt']         = 0.00
    data_reader['Port contribution']    = 0.00
    data_reader['Bench contribution']   = 0.00

    #-------- Calculating Portfolio Weights -------------------------#
    # For the initial period, portfolio weights are determined by
    # their current market value in the portfolio. For subsequent
    # periods, the portfolio value is adjusted for returns in the
    # last period.
    # Bench weights are set to be equally weighted (Issue - needs fixing)
    # --------------------------------------------------------------#
    if i == 0:
        port_wts        = mkt_val / mkt_val.sum()
    else:
        mkt_val         = mkt_val*(1+returns)
        port_wts        = mkt_val / mkt_val.sum()

    bench_wts[:]        = 1.0/num_tickers

    #-------- Calculating Portfolio and Bench Contributions --------#
    portfolio_contr     = (port_wts*returns).astype('float32')
    bench_contr         = (bench_wts*returns).astype('float32')
    tot_port_return     = portfolio_contr.sum()
    tot_bench_return    = bench_contr.sum()

    #-------- Assigning the calculated values to the dataframe------#
    data_reader['Port contribution']    = portfolio_contr[0,:]
    data_reader['Bench contribution']   = bench_contr[0,:]
    if i == 0:
        data_reader['Portfolio Wt']     = port_wts[:]

    else:
        data_reader['Portfolio Wt']     = port_wts[0, :]

    data_reader['Bench Wt']             = bench_wts[:]
    #--------------------------------------------------------------#

    #------- Calculating the period and sector attributions -------#
    # The attr_results array is 59 x 11 x 3 matrix that holds the
    # following data: Sector Effect, Stock Effect and Total Effect
    # for each sector across all periods.
    #--------------------------------------------------------------#
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

    #--------------------------------------------------------------#
    # The attr_summ array is a 59 x 2 array that holds the
    # attribution active return and the true active return
    # values
    #--------------------------------------------------------------#
    attr_summ[i,0]   = attr_result[i,:,2].sum()
    attr_summ[i,1]   = tot_port_return - tot_bench_return


#--------------- IR Attribution -----------------------------------#
# The IR Attribution section, aggregates the data from the return and
# risk information from the periodic sector attributions and decomposes
# the IR across the 11 sectors.
# The IR_Attr array holds the Sector and Stock level IR decomposition
# data.
#-------------------------------------------------------------------#
std_sec_stock           = np.zeros([num_sectors,2])
corr_coef               = np.zeros([num_sectors,2])
ir_input_return         = np.zeros([num_sectors,2])
ir_input_risk           = np.zeros([num_sectors,2])
IR_Attr                 = np.zeros([num_sectors,2])
total_per_sect_effect   = 0.00
total_per_stock_effect  = 0.00
std_total               = np.std(attr_summ[:,0])

#--------Iterating through sectors to calculate the IR Attribution ---#
for i in range(num_sectors):
    for j in range(2):
        std_sec_stock[i,j]      = np.std(attr_result[:,i, j])
        corr_coef[i,j]          = np.corrcoef(attr_result[:, i,j], attr_summ[:, j])[0,1]
        ir_input_return[i,j]    = np.sum(attr_result[:, i, j])
        ir_input_risk[i,j]      = std_sec_stock[i,j]*corr_coef[i,j]
        IR_Attr[i, j]           = (ir_input_return[i, j] / num_periods) / ir_input_risk[i, j]

#-------------------------------------------------------------------#

total_risk                  = np.sum(ir_input_risk)
total_active_return         = np.sum(ir_input_return)

total_sector_IR_contrib     = np.sum((ir_input_risk[:,0]/total_risk)*IR_Attr[:,0])
total_stock_IR_contrib      = np.sum((ir_input_risk[:,1]/total_risk)*IR_Attr[:,1])
IR_Attr_total               = total_sector_IR_contrib + total_stock_IR_contrib

Direct_IR                   = (total_active_return/num_periods)/total_risk

print IR_Attr_total, Direct_IR

























