import pandas_datareader as web
import pandas as pd
import numpy as np
import datetime
import math
import time
from pulp import *


from xlsxwriter import *

from pandas import ExcelWriter


writer = ExcelWriter('ETF_data.xlsx', engine='xlsxwriter')


start = datetime.datetime(2017,3,1)
end = datetime.datetime(2017,4,9)
# Analysis for the latest quarter
ETFs = np.array(['XLK', 'XLF', 'XLY', 'XLP', 'XLE', 'XLV', \
                 'XLI', 'XLRE', 'XLB', 'XTL', 'XLU', 'SHW',\
                 'HPQ', 'FISV', 'EMN', 'HCA', 'HST'])

g = web.DataReader(ETFs,'yahoo', start, end, )
close = g['Adj Close']

close.to_excel(writer, sheet_name='Close Price Data')


writer.close()

