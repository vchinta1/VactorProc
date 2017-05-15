import xlrd
#import xlwt
#from xlutils.copy import copy
#import xlsxwriter
import pandas as pd
from pandas import Series, DataFrame
from pandas import ExcelWriter
import numpy as np
#import sklearn
from pulp import *
#from sklearn.cross_validation import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.feature_selection import f_regression
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
#import scipy.stats
#from scipy import *
#import warnings
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#import sys
#import subprocess as sp
#import pdb


#############################################################################
#                   Perform spearman correlations by Date                   #
#############################################################################
def bloomberg_reader(filename):

    global factor_data
    workbook = pd.ExcelFile(filename)
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

    return factor_data


def spearmancorr(first_sheet, dates, test_factor):
    rolling_ic = pd.DataFrame()
    ic_sig = pd.DataFrame()
    datelen = len(dates)
    print 'Inside function:',test_factor
    for i in dates:
        for j in range(1,13):
            #pdb.set_trace()
            date1 = first_sheet.loc[first_sheet['Date'] == str(i)]
            ic = date1['fret'+str(j)].corr(date1[test_factor], method='spearman')
            tstat = ic*(math.sqrt((998)/(1-ic**2)))
            pval = scipy.stats.t.sf(abs(tstat), 11)*2
            s1 = pd.Series([i,'fret'+str(j), ic,tstat,pval])
            ic_list = pd.DataFrame([list(s1)], columns = ["Date","fret","IC","tstat","pval"])
            rolling_ic = rolling_ic.append(ic_list)

    #pdb.set_trace()
    print 'Averaging ICs and checking for Significance...'
    for j in range(1,13):
        date1 = rolling_ic.loc[rolling_ic.fret == 'fret'+str(j)]
        averageIC = date1['IC'].mean()
        stdevIC = date1['IC'].std()
        pval = scipy.stats.t.sf((math.fabs(averageIC)*math.sqrt(datelen)/stdevIC), datelen-1)
        s1 = pd.Series([averageIC, stdevIC, pval])
        ic_list = pd.DataFrame([list(s1)], columns = ["Average IC","Standard Deviation","p-Value"])
        ic_sig = ic_sig.append(ic_list)
    print ic_sig
    sig_count = 0
    for index, row in ic_sig.iterrows():
        if(row['p-Value'] < 0.10):
            sig_count = sig_count+1
    return sig_count, ic_sig, rolling_ic

def plothistog(first_sheet, test_factor):
    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1)
    n, bins, patches = ax2.hist(first_sheet[test_factor].dropna(), bins = 50)
    ax2.set_title(test_factor)
    ax2.set_xlabel('Bin')
    ax2.set_ylabel('Frequency')
    plt.show()

def begin_wt_vector(a, b, Port_Return):
     x = np.array(a)
     y = np.array(b)
     return x*(1+y)/(1+Port_Return)

def mktcap_vector(a,b):
    x = np.array(a)
    y = np.array(b)
    return x*(1+y)



#############################################################################


    
warnings.filterwarnings("ignore")
file_list = list()
final_sheet1 = pd.DataFrame()
final_data1 = pd.DataFrame()
final_stats = pd.DataFrame()
#file_list.append('MSF546HistoricalData.xlsx')
#file_list.append('PortOptimizeDF_cooper.xlsx')
#file_list.append('MSF546HistoricalData1.xlsx')
file_list.append('Factor Data.xlsx')
file_select = input('Which file do you wish to use?')
print 'Loading Excel file into DataFrame......'
print 'Using file '+file_list[file_select-1]+' for the analysis'
xls_file = pd.ExcelFile(file_list[file_select-1])
first_sheet = xls_file.parse('Sheet1')
#first_sheet = pd.read_csv(".csv")
list_columns = list(first_sheet)
ic_list = pd.DataFrame()
rolling_ic1 = pd.DataFrame()
rolling_ic2 = pd.DataFrame()
spread_summary = pd.DataFrame()
q75 = 0
q25 = 0
log_incomp = 0
sqrt_incomp = 0
Q5_fret = 0.00
Q4_fret = 0.00
Q3_fret = 0.00
Q2_fret = 0.00
Q1_fret = 0.00
num_periods = 1
num_facts = 6
alpha_f = list()
alpha_f_w = list()
first_sheet.set_index(keys=['Date','Ticker'],drop=False)#,inplace=True)
dates = first_sheet['Date'].unique().tolist()
ticker = pd.unique(first_sheet['Ticker'])
writer = ExcelWriter('FactorResults.xlsx', engine='xlsxwriter')
## first_sheet has all the data in it. By Ticker, by Date factor data. We need to
# load it into a numpy multidimentional array. There are three dimensions tickers X period X factors
first_sheet_nump = bloomberg_reader('Factor Data.xlsx')
#first_sheet_nump = np.zeros([num_periods, len(ticker), num_facts])

backtest_results = pd.DataFrame()
count = 0
avgActiveReturn = 0.00
avgTurnover = 0.00
avgWin = 0.00
avgLoss = 0.00
avgTE = 0.00

if (file_select == 1):
    num_sectors = 10
if(file_select == 2):
    num_sectors = 2



while True:
    print 'Main Menu'
    print 'What would you like to do?'
    print '1. Clean and Transform Data'
    print '2. Factor Testing'
    print '3. Create Alpha'
    print '4. Backtest Alpha'
    print '5. Quit'
    main_select = input('Enter Selection: ')

    ################  Test Factor Assignment #######################################
    # <<n>> represents the number of ranges for for the IQR (Interquartile Range)
    # <<skip_clean>> is a conditional field that allows the user to skip data cleaning
    #
    #------------------------------------------------------------------------------
    # Loading the csv file into a dataframe
    # <<dates>> represents a unique list of dates in the data set
    # <<ticker>> represents a unique list of tickers in the data set
    #------------------------------------------------------------------------------
    
    if main_select == 3:
        print 'Constructing An Alpha'
        num_factors = input('Enter number of factors:')
        print 'List of Transformed Factors:'
        tr_list_columns = list(first_sheet)
        for i in range(0,len(tr_list_columns)):
            if(tr_list_columns[i][-2:] == '_T' or tr_list_columns[i][:5] == 'ALPHA'):
                print str(i)+'.'+str(tr_list_columns[i])
        alpha_select_id = raw_input('Enter Factor(s):')
        alpha_factor_wts = raw_input('Enter Factor Weight(s) (0.00 - 1.00):')

        alpha_select = alpha_select_id.split(',')
        for j in range(0,num_factors):
            alpha_f.append(tr_list_columns[int(alpha_select[j])])

        alpha_factor_wt = alpha_factor_wts.split(',')
        alpha_name = raw_input('Enter name of Alpha factor(ALPHA will be appended):')
        first_sheet['ALPHA_c_'+alpha_name] = 0.000

        for index, row in first_sheet.iterrows():
            first_sheet.set_value(index, 'ALPHA_c_'+alpha_name, sum(row[str(alpha_f[j])]*float(alpha_factor_wt[j]) for j in range(0, num_factors)))        
        

    if main_select == 5:
        writer.save()
        print 'Thank you for using the AlphaCreation Tool'
        sys.exit(0)
    
    while main_select == 1:
        print '**************************************************'
        print 'Clean and Transform Menu Options:'
        print '1. Clean Factor'
        print '2. Transform Factor'
        print '3. Return to Main Menu'
        print '4. View Cleaned Factors'
        print '5. View Transformed Factors'
        print '6. Plot Factor'
        cl_tr_select = input('Enter Selection: ')

        if cl_tr_select == 3:
            main_select = 0
            sp.call('clear',shell=True)
            
        if cl_tr_select == 4:
            print 'List of Cleaned Factors:'
            tr_list_columns = list(first_sheet)
            for i in range(0,len(tr_list_columns)):
                if(tr_list_columns[i][-2:] == '_c'):
                    print str(i)+'.'+str(tr_list_columns[i])

        if cl_tr_select == 5:
            print 'List of Transformed Factors:'
            tr_list_columns = list(first_sheet)
            for i in range(0,len(tr_list_columns)):
                if(tr_list_columns[i][-2:] == '_T'):
                    print str(i)+'.'+str(tr_list_columns[i])

        if cl_tr_select == 6:
            test_factor = list()
            print 'Which factor(s) do you wish to plot?'
            for i in range(0, len(list_columns)):
                if(list_columns[i] != 'Date' and list_columns[i] != 'Ticker' and list_columns[i][0:4]!='fret'):
                    print str(i)+'.'+str(list_columns[i])
            test_selection = raw_input('Enter selection(s): ')
            test_factor_id = test_selection.split(',')



        if cl_tr_select == 1:
            test_factor = list()
            print 'Which factor(s) do you wish to clean?'
            for i in range(0, len(list_columns)):
                if(list_columns[i] != 'Date' and list_columns[i] != 'Ticker' and list_columns[i][0:4]!='fret'):
                    print str(i)+'.'+str(list_columns[i])
            test_selection = raw_input('Enter selection(s): ')
            test_factor_id = test_selection.split(',')
            n = input('Enter the value for N: ')
            charts = raw_input('Show charts(Y/N)?')
            for j in range(0, len(test_factor_id)):
                test_factor.append(list_columns[int(test_factor_id[j])])

            for j in range(0, len(test_factor)):
                first_sheet[test_factor[j]+'_c'] = first_sheet[test_factor[j]]
                q75,q25 = np.percentile(first_sheet[test_factor[j]].dropna(), [75,25])
                iqr = q75-q25
                median = first_sheet[test_factor[j]].median()
                data = np.array(first_sheet[test_factor[j]+'_c'])
                data[data < median-(n*iqr)] = median - (n*iqr)
                data[data >= median+(n*iqr)] = median + (n*iqr)
                first_sheet[test_factor[j]+'_c'] = data
                print 'Data Cleaning complete. New column created:'+test_factor[j]+'_c'
                if(charts == 'Y'):
                    plothistog(first_sheet, test_factor[j]+'_c')

        #############################################################################
        #                           Data Transformations                            #
        #############################################################################
        # Transform only clean data example list_columns[-5:] is 'clean' 

        if cl_tr_select == 2:
            print 'Which factor do you wish to transform?'
            tr_list_columns = list(first_sheet)
            for i in range(0,len(tr_list_columns)):
                if(tr_list_columns[i][-2:] == '_c' or tr_list_columns[i][-2:] == '_T'):
                    print str(i)+'.'+str(tr_list_columns[i])
            test_selection = input('Enter a selection: ')
            test_factor = tr_list_columns[test_selection]
            
            transform_selection = 0

        
            print 'Select from the following types of transformations:'
            print '1. Log'
            print '2. Sqrt'
            print '3. z-Score by Cross Section'
            print '4. z-Score by Time'
            print '5. z-Score by Sector'
            print '6. Return to Test List'
            
            transform_selection = raw_input('Enter a selection: ')
            transform_select = transform_selection.split(',')
            charts = raw_input('Show Charts(Y/N)?')
            
            print 'Transforming the data....'
            ## Log Transform
            for j in range(0, len(transform_select)):
                if transform_select[j] == '1':
                    print 'Performing a log transform....'
                    column = test_factor
                    first_sheet[column+'log_T'] = first_sheet[column]
                    log_list = np.array(first_sheet[column+'log_T'])
                    if log_list.min()<0:
                        print row[column]
                        log_incomp = -1
                    
                    if log_incomp != -1:
                        first_sheet[column+'log_T'] = np.log(log_list)
                        print 'Log Transform complete.'
                    else:
                        print 'Log tranformation incompatible.'

                if transform_select[j] == '2':
                    ## Square Root Transformation
                    column = test_factor
                    first_sheet[column+'sqrt_T'] = first_sheet[column]
                    for index, rows in first_sheet.iterrows():
                        if rows[column] <0:
                            sqrt_incomp = -1
                    if sqrt_incomp != -1:
                        for index, rows in first_sheet.iterrows():
                            first_sheet.set_value(index, column+'sqrt_T', math.sqrt(rows[column]))
                        print 'Sqrt Transform complete.'
                    else:
                        print 'Sqrt tranformation incompatible.'

                if transform_select[j] == '3':
                    ##z-score by crosssection
                    print 'zscoring '+test_factor+' by Cross Section'
                    column = test_factor
                    first_sheet[column+'_zCS_T'] = first_sheet[column]
                    #pdb.set_trace()
                    for i in dates:
                        #pdb.set_trace()
                        date1 = first_sheet[column+'_zCS_T'].loc[first_sheet['Date'] == i]
                        date1 = (date1 - date1.mean())/date1.std(ddof = 0)
                        first_sheet.loc[first_sheet['Date'] == i, column+'_zCS_T'] = date1
                    #pdb.set_trace()
                    print 'Cross-sectional z-score complete.'
                    if(charts == 'Y'):
                        plothistog(first_sheet, column+'_zCS_T')
                    #print first_sheet[column+'_zCS_T'].head()
                    test_factor = column+'_zCS_T'

                if transform_select[j] == '4':
                    print 'zscoring '+test_factor+' by Time'
                    ##z-score by time
                    column = test_factor
                    first_sheet[column+'_zT_T'] = first_sheet[column]
                    zscoreT = pd.DataFrame()
                    for i in ticker:
                        date1 = first_sheet[column+'_zT_T'].loc[first_sheet.Ticker == i]
                        date1 = (date1 - date1.mean())/date1.std(ddof = 0)
                        first_sheet.loc[first_sheet['Ticker'] == i, column+'_zT_T'] = date1
                    print 'Time series z-score complete.'
                    if(charts == 'Y'):
                        plothistog(first_sheet, column+'_zT_T')
                    #print first_sheet[column+'_zT_T'].head()
                    test_factor = column+'_zT_T'

                if transform_select[j] == '5':
                    print 'zscoring '+test_factor+' by Sector'
                    column = test_factor
                    first_sheet[column+'_zS_T'] = first_sheet[column]
                    zscoreSCS = pd.DataFrame()
                    for i in dates:
                        for j in range(1, num_sectors+1):
                           date1 = first_sheet[column+'_zS_T'].loc[(first_sheet['Date'] == i) & (first_sheet['Sector'+str(j)] == 1)]
                           date1 = (date1 - date1.mean())/date1.std(ddof = 0)
                           first_sheet.loc[(first_sheet['Date'] == i) & (first_sheet['Sector'+str(j)] == 1), column+'_zS_T'] = date1
        
                    print 'Sector z-score complete.'
                    
                    if(charts == 'Y'):
                        plothistog(first_sheet, column+'_zS_T')
                    #print first_sheet[['Date', 'Ticker', column+'_zS_T']].head()
                    test_factor = column+'_zS_T'
                        

                

    while main_select == 2:
        print '************** Factor Test Menu **************************'
        print '1. IC Decay Test'
        print '2. Spread Test'
        print '3. Confusion Matrix'
        print '4. Return to Main Menu'
        fact_test_select = input('Enter test selection:')

        if fact_test_select == 1:
            print 'Select the Factor to test'
            tr_list_columns = list(first_sheet)
            for i in range(0,len(tr_list_columns)):
                if(tr_list_columns[i][-2:] == '_T' or tr_list_columns[i][:5] == 'ALPHA'):
                    print str(i)+'.'+str(tr_list_columns[i])
            test_select = input('Select Factor:')
            test_factor = tr_list_columns[test_select]
            sig_count, rolling_ic1, rolling_ic2 = spearmancorr(first_sheet, dates, test_factor)
            print 'Number of significant months:',sig_count
            plothistog(first_sheet, test_factor)
            rolling_ic1.to_excel(writer, sheet_name='ICD'+test_factor)
            rolling_ic2.to_excel(writer, sheet_name='ICD1'+test_factor)

        if fact_test_select == 4:
            main_select = 0

        if fact_test_select == 2:
            test_factor1 = list()
            spread_list = pd.DataFrame()
            fsbuffer = pd.DataFrame()
            spread = pd.DataFrame( columns = ['Date','Fret','Mean'])

            print 'Select the Factor to test'
            tr_list_columns = list(first_sheet)
            for i in range(0,len(tr_list_columns)):
                if(tr_list_columns[i][-2:] == '_T'):
                    print str(i)+'.'+str(tr_list_columns[i])
            test_select = raw_input('Select Factor:')
            #test_factor = tr_list_columns[test_select]
            test_select_len = test_select.split(',') #n
            for k in range(0, len(test_select_len)): #n
                test_factor1.append(tr_list_columns[int(test_select_len[k])]) #n
                print test_factor1[k]


            ###Calculating Quantiles
            print 'Calculating Quintiles....'
            ''' Original single entry
            for i in dates:
                date1 = first_sheet.loc[first_sheet.Date == i]
                #date1['alpha1_q']= pd.qcut(date1['alpha1'], 5, labels = ["5", "4","3","2","1"] )
                date1[test_factor+'_q'] = pd.qcut(date1[test_factor], 5, labels = ["5", "4","3","2","1"] )
                fsbuffer = fsbuffer.append(date1)
            '''

            for i in dates:
                date1 = first_sheet.loc[first_sheet.Date == i]
                #date1['alpha1_q']= pd.qcut(date1['alpha1'], 5, labels = ["5", "4","3","2","1"] )
                for j in range (0, len(test_select_len)):
                    date1[test_factor1[j]+'_q'] = pd.qcut(date1[test_factor1[j]], 5, labels = ["5", "4","3","2","1"] )
                fsbuffer = fsbuffer.append(date1)

            first_sheet = fsbuffer
            print first_sheet.head()
            print 'Quantile calc. complete.','First Sheet shape:', first_sheet.shape

            print 'Calculating Spreads....'
            for k in range(0, len(test_factor1)):
                print 'Value of K:'+str(k)
                for i in dates:
                    date1 = first_sheet.loc[first_sheet.Date == i]
                    for j in range(1,13):
                        date2 = date1.loc[date1[test_factor1[k]+'_q'] == "5"]
                        RetMean2 = date2['fret'+str(j)].mean()
                        s2 = pd.Series([i, test_factor1[k], 'fret'+str(j), 'Q5', RetMean2])
                        spread_list2 = pd.DataFrame([list(s2)], columns = ["Date","Factor", "Fret","Quintile","Mean"])
                        spread = spread.append(spread_list2)
                        
                        date3 = date1.loc[date1[test_factor1[k]+'_q'] == "4"]
                        RetMean3 = date3['fret'+str(j)].mean()
                        s3 = pd.Series([i, test_factor1[k],'fret'+str(j), 'Q4', RetMean3])
                        spread_list3 = pd.DataFrame([list(s3)], columns = ["Date","Factor","Fret","Quintile","Mean"])
                        spread = spread.append(spread_list3)
                        
                        date4 = date1.loc[date1[test_factor1[k]+'_q'] == "3"]
                        RetMean4 = date4['fret'+str(j)].mean()
                        s4 = pd.Series([i, test_factor1[k],'fret'+str(j), 'Q3', RetMean4])
                        spread_list4 = pd.DataFrame([list(s4)], columns = ["Date","Factor","Fret","Quintile","Mean"])
                        spread = spread.append(spread_list4)

                        date5 = date1.loc[date1[test_factor1[k]+'_q'] == "2"]
                        RetMean5 = date5['fret'+str(j)].mean()
                        s5 = pd.Series([i, test_factor1[k],'fret'+str(j), 'Q2', RetMean5])
                        spread_list5 = pd.DataFrame([list(s5)], columns = ["Date","Factor","Fret","Quintile","Mean"])
                        spread = spread.append(spread_list5)

                        date6 = date1.loc[date1[test_factor1[k]+'_q'] == "1"]
                        RetMean6 = date6['fret'+str(j)].mean()
                        s6 = pd.Series([i, test_factor1[k],'fret'+str(j), 'Q1', RetMean6])
                        spread_list6 = pd.DataFrame([list(s6)], columns = ["Date","Factor","Fret","Quintile","Mean"])
                        spread = spread.append(spread_list6)
                        
                spread_factor_filter = spread.loc[spread['Factor'] == test_factor1[k]]
               
                spread_date_filter = spread_factor_filter.loc[spread['Date'] == i]
                for j in range(1,13):
                    spread_fret_filter = spread_date_filter.loc[spread_date_filter['Fret'] == 'fret'+str(j)]
                    for index, row in spread_fret_filter.iterrows():
                        if(row['Quintile'] == 'Q5'):
                            Q5_fret = row['Mean']
                        if(row['Quintile'] == 'Q4'):
                            Q4_fret = row['Mean']
                        if(row['Quintile'] == 'Q3'):
                            Q3_fret = row['Mean']
                        if(row['Quintile'] == 'Q2'):
                            Q2_fret = row['Mean']
                        if(row['Quintile'] == 'Q1'):
                            Q1_fret = row['Mean']
                    Total_Spread = Q5_fret - Q1_fret
                    Middle_Spread = Q5_fret - Q3_fret
                    Lower_Spread = Q3_fret - Q1_fret
                    s1 = pd.Series([i, test_factor1[k], 'fret'+str(j), Total_Spread, Middle_Spread, Lower_Spread])
                    spread_list = pd.DataFrame([list(s1)], columns = ["Date","Factor", "Fret","Total Spread","Middle Spread", "Lower Spread"])
                    spread_summary = spread_summary.append(spread_list)
                        
                    
                    
            print 'Spread calculation complete.', 'Spread shape:', spread.shape
            print spread.head()
            spread.to_excel(writer, sheet_name='Spd'+test_factor)
            spread_summary.to_excel(writer, sheet_name='SpdS'+test_factor)

        if fact_test_select == 3:
            fsbuffer = pd.DataFrame()
            print 'Select the Factor to test'
            tr_list_columns = list(first_sheet)
            print tr_list_columns
            for i in range(0,len(tr_list_columns)):
                if(tr_list_columns[i][-2:] == '_T'):
                    print str(i)+'.'+str(tr_list_columns[i])
            test_select = input('Select Factor:')
            test_factor = tr_list_columns[test_select]
            print 'Generating confusion matrix....'

            for i in dates:
                date1 = first_sheet.loc[first_sheet.Date == i]
                date1['alpha1_q']= pd.qcut(date1['alpha1'], 5, labels = ["5", "4","3","2","1"] )
                date1[test_factor+'_q'] = pd.qcut(date1[test_factor], 5, labels = ["5", "4","3","2","1"] )
                fsbuffer = fsbuffer.append(date1)
            first_sheet = fsbuffer


            total = 0
            conf_matrix = pd.DataFrame(index = ['1','2','3','4','5'], columns=['1','2','3','4','5'])
            conf_matrix['1'] = 0.00
            conf_matrix['2'] = 0.00
            conf_matrix['3'] = 0.00
            conf_matrix['4'] = 0.00
            conf_matrix['5'] = 0.00

            for i in range(1,6):
                for j in range(1,6):
                    datei = first_sheet.loc[first_sheet[test_factor+'_q'] == str(i)]
                    datej = datei.loc[datei.alpha1_q == str(j)]
                    count = len(datej)
                    total = total+count
                    conf_matrix.set_value(str(i),str(j), count)

            print 'Total:', total

            for i in range(1,6):
                for j in range(1,6):
                    conf_matrix.set_value(str(i),str(j), conf_matrix[str(j)][str(i)]/total)

            print conf_matrix
            conf_matrix.to_excel(writer, sheet_name='Confusion'+test_factor)

    ##################### Menu Option - Backtesting Alpha #################################
    # The purpose of this section is to select an Alpha column and perform backtesting against it
    
    if main_select == 4:
        #----------------------------------------------------------------------------------
        # Input backtest variables.
        # <<months_select>> = The section takes inputs for the number of months to backtest.
        # <<security_off>> = Security offset to be applied to the portfolio
        # <<sector_off>> = Sector offset to be applied to the portfolio
        # <<alphaPick>> = Alpha pickup to be applied to the Portfolio
        num_dates = len(dates)
        months_select = input('Enter Number of months of Backtest:')
        security_off = input('Enter Security Offset:')
        sector_off = input('Enter Sector Offset:')
        alphaPick = input('Enter Alpha Pickup:')

        iter_num = input('Enter number of iterations:')
        secu_iter_off = input('Enter security realted offset per iteration:')
        sect_iter_off = input('Enter sector related offset per iteration:')
        alp_iter_off = input('Enter alpha pickup related offset per iteration:')
        minAlpha = alphaPick-(alp_iter_off*iter_num)
        security_off_original = security_off
        #---------------------------------------------------------------------------------
        # Starting loop to run the number of iterations entered by the user
        while alphaPick != minAlpha:
            
            alphaPick = alphaPick - alp_iter_off
            security_off = security_off_original
            #-----------------------------------------------------------------------------
            # Starting loop to 
            for k in range(0, iter_num):
                print 'Iteration Number:'+str(k+1)
                if count>0:
                    security_off = security_off - secu_iter_off
                    sector_off = sector_off - sect_iter_off

                count = 0           # Count is used as a way to skip the first month in every iteration
                # DONT CHANGE COUNT!!! ITS GOING TO BREAK A LOT OF THINGS!!
                # COUNT is used to skip the first month for portfolio weighting, average turnover calculations

                #------------------------------------------------------------------------
                # Starting loop to iterate through the months and perform backtesting
                # The loop is limited to the number of months requested by the user
                for m in dates[num_dates - months_select:]:
                    ctr = 0
                    winner_count = 0
                    loser_count = 0
                    date1 = first_sheet[['Date', 'Ticker', 'MKTCAP', 'FWDR', 'ALPHA', 'lncap', 'Sector1','Sector2', 'Sector3', 'Sector4', 'Sector5', 'Sector6', 'Sector7', 'Sector8', 'Sector9', 'Sector10']].loc[first_sheet['Date'] == m]
                    count = count+1
                    print 'Backtesting Alpha...'+str(m)
                    date1['bench_wt'] = 0.00
                    tot_cap = date1['MKTCAP'].sum()
                    if count >1:
                        fwdr = np.array(final_port['FWDR'])
                    #------------------------------------------------------------------------
                    # Cap weighting the stocks, and using that as the bench weight.
                    # For the first month the bench weight is based on the starting MKTCAP
                    # For subsequent months, the MKTCAP is modified to account for changes over the
                    # previous period.
                    if count == 1:
                        date1['bench_wt'] = date1['MKTCAP']/tot_cap
                    else:
                        mktcap_func = np.vectorize(mktcap_vector)
                        mktcap = np.array(final_port['MKTCAP'])
                        mktcap_upd = mktcap_func(mktcap, fwdr)
                        date1['MKTCAP'] = mktcap_upd
                    #bench wt assignment complete.
                    #------------------------------------------------------------------------

                    #------------------------------------------------------------------------
                    # ALPHA PICKUP Assignment. The purpose of the Alpha Pickup is to introduce
                    # variability into the Alpha to reduce the turnover experienced by the Portfolio.
                    # We want to avoid the portfolio chasing every small change in Alpha. This could lead
                    # to major transaction costs.
                    # <<first_append>> is the DataFrame that holds a duplicate copy of the original list
                    # of stocks.
                    fs_size = len(date1)
                    first_append = pd.DataFrame()
                    first_append = date1.copy()
                    first_append['Begin_PW'] = 0.000
                    first_append['Tickerz'] = 'dummy'
                    first_append['Up Limit'] = -1.00
                    first_append['Low Limit'] = -1.00
                    
                    #-------------------------------------------------------------
                    # Initializing the value for <<first_append>>
                    #-------------------------------------------------------------
                    first_append['Tickerz'] = first_append['Ticker']+'z'
                    first_append['ALPHA'] += alphaPick
                    first_append['bench_wt'] = -1.00
                    first_append['Port_Wt'] = 0.00
                    begin_wt_func = np.vectorize(begin_wt_vector)
                    if count == 1:
                        first_append['Begin_PW'] = date1['bench_wt'][:fs_size]
                    else:
                        cons_wt = np.array(final_port['Cons_Port_Wt'])
                        begin_wt = begin_wt_func(cons_wt, fwdr, Port_Return)
                        first_append['Begin_Wt'] = begin_wt

                    date1['Up Limit'] = date1['bench_wt'] + float(security_off)
                    date1['Low Limit'] = date1['bench_wt'] - float(security_off)
                    date1.loc[date1['Low Limit']<0.00, 'Low Limit'] = 0.00

                    #-------------------------------------------------------------
                    # Calculating the Upper and Lower limits of the
                    # Sector and lncap contraints
                    #-------------------------------------------------------------
                    Sector_UL = list()
                    Sector_LL = list()

                    for i in range(1, num_sectors+1):        
                        Sector_UL.append(sum(date1['bench_wt']*date1['Sector'+str(i)]) + security_off)#row['bench_wt']*row['Sector'+str(i)] for index,row in date1.iterrows())+security_off)
                        Sector_LL.append(sum(date1['bench_wt']*date1['Sector'+str(i)]) - security_off)
                    #lncap_UL = sum(row['bench_wt']*row['lncap'] for index,row in date1.iterrows()) + float(lncap_off)
                    #lncap_LL = sum(row['bench_wt']*row['lncap'] for index,row in date1.iterrows()) - float(lncap_off)


                    date1['Begin_PW'] = 0.000
                    date1['Tickerz'] = date1['Ticker']
                    date1 = date1.append(first_append, ignore_index = True)
                    stocks = date1['Tickerz'].unique().tolist()
                    #print date1

                    #print 'Stock List:', stocks
                    fs_size_app = len(date1)
                    alpha_list = np.array(date1['ALPHA'])
                    begin_pw = np.array(date1['Begin_PW'])
                    row_up = np.array(date1['Up Limit'])
                    row_down = np.array(date1['Low Limit'])
                    
                    ################### Initializing Portfolio Optimization ###############
                    # Using the python Pulp module to perform a linear program optimization.
                    # The objective of the optimization is to maximize the Alpha associated
                    # to the portfolio.
                    #----------------------------------------------------------------------
                    #print 'Initializing Portfolio Optimization'
                    prob1 = LpProblem('Portfolio Optimization', LpMaximize)
                    port_wts = LpVariable.dicts('PWTS',stocks)
                    #----------------------------------------------------------------------
                    # Defining the Objective function.
                    # Maximize(Sum(portfolio weights*alpha))
                    #----------------------------------------------------------------------

                    prob1 += lpSum([alpha_list[i]*port_wts[stocks[i]] for i in range(0, fs_size_app)]), "Total Alpha"
                    #prob1 += lpSum(date1['ALPHA']*port_wts[date1['Tickerz']])

                    #----------------------------------------------------------------------
                    # Constraint #7 from Prof. Cooper Worksheet
                    # The sum of all the portfolio weights should be equal to 100%
                    #----------------------------------------------------------------------

                    prob1 += lpSum([port_wts[stocks[i]]+port_wts[stocks[i+fs_size]] for i in range(0, fs_size)]) == 1.0, "PercentagesSum"
                    prob1 += lpSum([port_wts[stocks[i]] for i in range(0, fs_size_app)]) == 1.0

                    #----------------------------------------------------------------------
                    # Constraint #1: Each portfolio weight has to be greater than or equal to 0.00
                    # Constraint #2: For z stocks, the portfolio wt cannot be greater than begin weight
                    # Constraint #3 & #4: The consolidated portfolio wts has to be within the upper and lower limit
                    #----------------------------------------------------------------------
                    for i in range(0, fs_size_app):
                        prob1 += port_wts[stocks[i]] >= 0.00
                        if(i < fs_size):
                            prob1 += (port_wts[stocks[i]] + port_wts[stocks[i+fs_size]]) <= row_up[i]
                            prob1 += (port_wts[stocks[i]] + port_wts[stocks[i+fs_size]]) >= row_down[i]
                        if(i >= fs_size):
                            prob1 += port_wts[stocks[i]] <= begin_pw[i]
                            
                    #----------------------------------------------------------------------
                    # Constraint #5 & 6: Sum product of Sector & lncap with portfolio wt
                    # has to be within the upper and lower limits of the sector and lncap
                    # constraints
                    # Note: The sector and lncap contraints allow up to control systematic risk
                    # posed by these factors. Controlling for these factors minimizes a key source
                    # of systematic risk.
                    #----------------------------------------------------------------------

                    for g in range(1, num_sectors+1):
                        sector_list = np.array(date1['Sector'+str(g)])
                        prob1 += lpSum([(port_wts[stocks[i]]+port_wts[stocks[i+fs_size]])*sector_list[i] for i in range(0, fs_size)]) <= Sector_UL[g-1]
                        prob1 += lpSum([(port_wts[stocks[i]]+port_wts[stocks[i+fs_size]])*sector_list[i] for i in range(0, fs_size)]) >= Sector_LL[g-1]
                    #prob += lpSum([(port_wts[row['Tickerz']]+port_wts[date1['Tickerz'][index+fs_size]])*row['lncap'] for index,row in date1.iterrows() if row['Ticker'] == row['Tickerz']]) <= lncap_UL
                    #prob += lpSum([(port_wts[row['Tickerz']]+port_wts[date1['Tickerz'][index+fs_size]])*row['lncap'] for index,row in date1.iterrows() if row['Ticker'] == row['Tickerz']]) >= lncap_LL

                    #----------------------------------------------------------------------
                    # Solving for the objective statement
                    # IF PROB1.SOLVE() FAILS: Check for blanks cells in the data.
                    # Status of the optimizer can be checked by LpStatus[prob1.status]
                    # Final Value of objective function can be checked by value(prob1.objective)
                    #----------------------------------------------------------------------
                    #pdb.set_trace()
                    prob1.writeLP("OptimizeStocks1.lp")
                    prob1.solve()

                    #----------------------------------------------------------------------
                    # Assigning the optimized values to the Portfolio Weight column

                    for v in prob1.variables():
                        date1.loc[date1['Tickerz'] == v.name[5:], 'Port_Wt'] = v.varValue

                    ########### Calculating key stats ###################
                    date1['Cons_Port_Wt'] = np.nan
                    date1['Active_Bet'] = np.nan
                    date1['Abs Change'] = np.nan
                    date1['Active Contribution'] = np.nan
                    
                    #*** Check if we can make these loops more efficient
                    upper_half = np.array(date1['Port_Wt'][:fs_size])
                    lower_half = np.array(date1['Port_Wt'][fs_size:])
                    date1['Cons_Port_Wt'][:fs_size] = upper_half + lower_half
                    date1['Abs Change'][:fs_size] = (np.array(date1['Cons_Port_Wt'][:fs_size])) - (np.array(date1['Begin_PW'][fs_size:]))
                    date1['Active_Bet'][:fs_size] = (np.array(date1['Cons_Port_Wt'][:fs_size])) - (np.array(date1['bench_wt'][:fs_size]))
                    date1['Active Contribution'][:fs_size] = (np.array(date1['Active_Bet'][:fs_size])) * (np.array(date1['FWDR'][:fs_size]))
                    
                    final_port = date1[:fs_size]
                    final_port.sort(['ALPHA'], ascending = [False], inplace = True)
                    winner_count = (final_port['Active Contribution']>0).sum()
                    loser_count = (final_port['Active Contribution']<0).sum()
                    ###########  ###################

                    turnover = final_port['Abs Change'].sum()/2
                    final_port['Portfolio_Return'] = final_port['FWDR']*final_port['Cons_Port_Wt']
                    Port_Return = final_port['Portfolio_Return'].sum()
                    #Port_Return = sum(row['FWDR']*row['Cons_Port_Wt'] for index, row in final_port.iterrows())
                    final_port['Bench_Return'] = final_port['FWDR']*final_port['bench_wt']
                    bench_Return = final_port['Bench_Return'].sum()
                    #bench_Return = sum(row['FWDR']*row['bench_wt'] for index, row in final_port.iterrows())
                    active_return = Port_Return - bench_Return
                    #final_port['Active Contribution'] = final_port['Active_Bet']*final_port['FWDR']
                    #winner_count = final_port[final_port['Active Contribution'] >0].count()
                    #loser_count = final_port[final_port['Active Contribution'] <0].count()
                    stats = pd.Series([m, security_off, sector_off, alphaPick, turnover, Port_Return, bench_Return, active_return, winner_count, loser_count])
                    key_stats= pd.DataFrame([list(stats)], columns= ['Date','Security Offset','Sector Offset','Alpha Pickup','Turnover','Port Return', 'Bench Return', 'Active Return', 'Winners', 'Losers'])

                    backtest_results = backtest_results.append(key_stats)
                    #final_sheet1 = final_sheet1.append(final_port)
                    #final_data1 = final_data1.append(date1)
                    print 'Completed '+str(m)
                    if(count > 1):
                        avgTurnover += turnover

                avgActiveReturn = backtest_results['Active Return'].mean()
                avgTE = backtest_results['Active Return'].std()
                avgWin = backtest_results['Winners'].mean()
                avgLoss = backtest_results['Losers'].mean()
                IR = avgActiveReturn/avgTE
                Annual_IR = IR*math.sqrt(12)
                avgTurnover = avgTurnover/(months_select-1)
                summary = pd.Series([alphaPick, security_off, avgActiveReturn, avgTE, IR, Annual_IR, avgWin, abs(avgLoss), avgTurnover])
                sumStats = pd.DataFrame([list(summary)], columns = ['Alpha Pickup', 'Security Offset','Avg Active Return', 'Average TE', 'IR', 'Annualized IR', 'Average Winners', 'Average Losers', 'Average Turnover'])
                #backtest_results.to_excel(writer, sheet_name='BacktestStats_'+str(security_off)+'_'+str(sector_off)+'_'+str(alphaPick))
                #final_sheet1.to_excel(writer, sheet_name='finalsheet_'+str(security_off)+'_'+str(sector_off)+'_'+str(alphaPick))
                #final_data1.to_excel(writer, sheet_name='finaldata'+str(security_off)+'_'+str(sector_off)+'_'+str(alphaPick))
                final_stats = final_stats.append(sumStats)
            final_stats.to_excel(writer, sheet_name = 'Summary Stats'+str(security_off)+'_'+str(sector_off)+'_'+str(alphaPick))
        
        
  
    
    '''Checking for Significance. Needs more work
    rolling_ic['Sig'] = 0

    #rolling_ic['Sig'] = np.where((rolling_ic.pval < 0.05 & rolling_ic.IC >0, "Up", np.where(rolling_ic.pval <0.05 & rolling_ic.IC <0, "Down", "Insig")))


    for index1, row in rolling_ic.iterrows():
        print index1
        print 'Pval:',row['pval'], 'IC:', row['IC']
        if (row['pval'] < 0.05 and row['IC']>0.0):
            row.set_value(index1, 'Sig', 1)
        elif (row['pval'] < 0.05 and row['IC']<0.0):
            row.set_value(index1, 'Sig', 0)
        else:
            row.set_value(index1, 'Sig', -1)

    '''

    '''
    '''
    '''Need to look into how to check for Null values in the quantile column
    for i in dates:
        for j in range(1,3):
            date1 = first_sheet.loc[first_sheet.Date == i]
            date2 = date1.loc[first_sheet.E_P_q == nan]
            RetMean = date2['fret'+str(j)].mean()
            print 'Shape:',date2.shape
            print i, 'fret'+str(j),'Q5', 'Return Mean:', RetMean

    '''
