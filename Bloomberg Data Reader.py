import pandas as pd
import numpy as np
import xlrd


global factor_data
workbook = pd.ExcelFile('Factor Data.xlsx')
num_sheets = len(workbook.sheet_names)
sheet_names = workbook.sheet_names
num_periods = 220
num_factors = 5

count_tickers = 0
for i in sheet_names:
    sheet = workbook.parse(i)
    if (len(sheet) > num_periods):
        count_tickers += 1

num_tickers = count_tickers
factor_data = np.zeros([num_tickers, num_periods, num_factors])

ticker_count = 0
for i in sheet_names:
    print 'Processing sheet:',i
    sheet = workbook.parse(i)
    if (len(sheet) > num_periods):
        start_limit = len(sheet)-220
        for j in range(num_factors):
            factor_data[ticker_count, :, j] = sheet.ix[:, j+1][start_limit:]
        ticker_count += 1

print factor_data

#----- Factor backtesting -------#
