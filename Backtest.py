import pandas as pd
import numpy as np

num_tickers = 10
num_sectors = 2
num_periods = 4

file_name = pd.ExcelFile("Portfolio_Month1.xlsx")
data = file_name.parse('Sheet1')

port_struct         = np.zeros([num_periods, num_tickers*2, 10])
bench_risk          = np.zeros(3)
port_wts            = np.zeros(num_tickers*2)
alpha_pickup        = 5.0
sector_offset       = 0.02
security_offset     = 0.05
lncap_offset        = 0.1
stocks              = pd.unique(data['Asset'])


for i in range(0,4):
    port_struct[0,:num_tickers, i-1] = data.ix[:,i+1]

port_struct[0,:num_tickers,5] = np.log(port_struct[0,:num_tickers,3])                                    # Ln of the Market Cap
port_struct[0,:num_tickers,6] = port_struct[0,:num_tickers,3]/np.sum(port_struct[0,:num_tickers,3])      # Bench Weights
port_struct[0,:num_tickers,7] = port_struct[0,:num_tickers,6] + security_offset                         # Up Limit
port_struct[0,:num_tickers,8] = port_struct[0,:num_tickers,6] - security_offset                         # Down Limit

port_struct[0,num_tickers:,0] = port_struct[0,:num_tickers,0]
port_struct[0,num_tickers:,9] = 1/num_tickers

# Bench risk factor stats
bench_risk[0] = np.sum(port_struct[0,:num_tickers,1]*port_struct[0,:num_tickers,6])
bench_risk[1] = np.sum(port_struct[0,:num_tickers,2]*port_struct[0,:num_tickers,6])
bench_risk[2] = np.sum(port_struct[0,:num_tickers,5]*port_struct[0,:num_tickers,6])

# Portfolio Optimization

prob1 = LpProblem('Portfolio Optimization', LpMaximize)
port_wts = LpVariable.dicts('PWTS', stocks)
# ----------------------------------------------------------------------
# Defining the Objective function.
# Maximize(Sum(portfolio weights*alpha))
# ----------------------------------------------------------------------

prob1 += lpSum([port_struct[0,i,0]*port_wts[stocks[i]] for i in range(0, num_tickers*2)])), "Total Alpha" ##******************
# prob1 += lpSum(date1['ALPHA']*port_wts[date1['TICKERz']])

# ----------------------------------------------------------------------
# Constraint #7 from Prof. Cooper Worksheet
# The sum of all the portfolio weights should be equal to 100%
# ----------------------------------------------------------------------

prob1 += lpSum([port_wts[stocks[i]] + port_wts[stocks[i + num_tickers]] for i in range(0, num_tickers)]) == 1.0, "PercentagesSum"
prob1 += lpSum([port_wts[stocks[i]] for i in range(0, num_tickers*2)]) == 1.0

# ----------------------------------------------------------------------
# Constraint #1: Each portfolio weight has to be greater than or equal to 0.00
# Constraint #2: For z stocks, the portfolio wt cannot be greater than begin weight
# Constraint #3 & #4: The consolidated portfolio wts has to be within the upper and lower limit
# ----------------------------------------------------------------------
for i in range(0, num_tickers*2):
    prob1 += port_wts[stocks[i]] >= 0.00
    if (i < num_tickers):
        prob1 += (port_wts[stocks[i]] + port_wts[stocks[i + num_tickers]]) <= port_struct[0,i,7]
        prob1 += (port_wts[stocks[i]] + port_wts[stocks[i + fs_size]]) >= port_struct[0,i,8]

    if (i >= num_tickers):
        prob1 += port_wts[stocks[i]] <= port_struct[0,i,9]


# ----------------------------------------------------------------------
# Constraint #5 & 6: Sum product of Sector & lncap with portfolio wt
# has to be within the upper and lower limits of the sector and lncap
# constraints
# Note: The sector and lncap contraints allow up to control systematic risk
# posed by these factors. Controlling for these factors minimizes a key source
# of systematic risk.
# ----------------------------------------------------------------------

for g in range(1, num_sectors + 1):
    sector_list = np.array(date1['Sector' + str(g)])
    prob1 += lpSum([(port_wts[stocks[i]] + port_wts[stocks[i + fs_size]]) * sector_list[i] for i in range(0, fs_size)]) <= Sector_UL[g - 1]
    prob1 += lpSum([(port_wts[stocks[i]] + port_wts[stocks[i + fs_size]]) * sector_list[i] for i in range(0, fs_size)]) >= Sector_LL[g - 1]
# prob += lpSum([(port_wts[row['TICKERz']]+port_wts[date1['TICKERz'][index+fs_size]])*row['lncap'] for index,row in date1.iterrows() if row['TICKER'] == row['TICKERz']]) <= lncap_UL
# prob += lpSum([(port_wts[row['TICKERz']]+port_wts[date1['TICKERz'][index+fs_size]])*row['lncap'] for index,row in date1.iterrows() if row['TICKER'] == row['TICKERz']]) >= lncap_LL

# ----------------------------------------------------------------------
# Solving for the objective statement
# IF PROB1.SOLVE() FAILS: Check for blanks cells in the data.
# Status of the optimizer can be checked by LpStatus[prob1.status]
# Final Value of objective function can be checked by value(prob1.objective)
# ----------------------------------------------------------------------
# pdb.set_trace()
prob1.writeLP("OptimizeStocks1.lp")
prob1.solve()




















