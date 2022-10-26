import pandas as pd
import pandas_datareader as pdr
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import wget
import os
import statsmodels.formula.api as sm

#inputs
tickers = ['AAPL','DD','PG','KO','MSFT','JPM','XOM', 'GE']
weights = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
lookback = 10

if sum(weights) != 1:
    print('Portfolio size error')
    if sum(weights) > 1:
        print('Too much weight')
    if sum(weights) < 1:
        print('Not enough weight')
    exit()
if len(tickers) != len(weights):
    print('Portfolio mismatch error')
    if len(tickers) > len(weights):
        print('Too many tickers; too few weights')
    if len(tickers) < len(weights):
        print('Too few tickers; too many weights')
    exit()

def end_of_month(dates):
    delta_m = relativedelta(months=1)
    delta_d = timedelta(days=1)
    next_month = dates + delta_m
    end_of_month = date(next_month.year, next_month.month, 1) - delta_d
    return end_of_month

start_date = end_of_month(datetime.today()) - relativedelta(years=lookback) - relativedelta(months=1)
end_date = end_of_month(datetime.today()) - relativedelta(months=1)

factors = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3',start=start_date, end=end_date)[0]
factors = factors.join(pdr.get_data_famafrench('F-F_Momentum_Factor',start=start_date, end=end_date)[0])
factors = factors.drop(columns='RF')
factors.rename(columns = {'Mom   ':'UMD'}, inplace = True)
factors.rename(columns = {'Mkt-RF':'MKT'}, inplace = True)
factors.index = factors.index.to_timestamp(freq='M')
factors = factors/100


response = wget.download('https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx', out='/Users/User/Desktop/Practice Python')
response = wget.download('https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx', out='/Users/User/Desktop/Practice Python')

bab = pd.read_excel('/Users/User/Desktop/Practice Python/Betting Against Beta Equity Factors Monthly.xlsx', 'BAB Factors', skiprows=17, parse_dates=True, index_col=0, header=1)
qmj = pd.read_excel('/Users/User/Desktop/Practice Python/Quality Minus Junk Factors Monthly.xlsx', 'QMJ Factors', skiprows=17, parse_dates=True, index_col=0, header=1)

for file_path in ['/Users/User/Desktop/Practice Python/Betting Against Beta Equity Factors Monthly.xlsx','/Users/User/Desktop/Practice Python/Quality Minus Junk Factors Monthly.xlsx']:
    if os.path.isfile(file_path):
        os.remove(file_path)

aqr_factors = pd.DataFrame(data={'BAB':bab['USA'],'QMJ':qmj['USA']})
aqr_factors = aqr_factors.dropna()
factors = factors.join(aqr_factors)

price_df = pdr.get_data_yahoo(symbols=tickers,start=start_date - relativedelta(months=1), end=factors.index[-1], ret_index=True)
price_df = price_df.resample('M').last()
ret_df = price_df['Ret_Index'].pct_change(1).dropna()

df = factors.join(ret_df)
window_len = 60

b_all = []
for ticker in tickers:
    b = pd.DataFrame(columns=list(factors.columns))
    b.insert(0,'Intercept',[])
    for x in range(len(df)-window_len):
        result = sm.ols(formula=f"{ticker} ~ MKT + SMB + HML + RMW + CMA + UMD + BAB + QMJ", data=df[x:60+x]).fit()
        b = pd.concat([b,result.params.to_frame().swapaxes("index", "columns")])
        #print(result.summary())
    b.index = df[len(df)-len(b):].index
    b_all.append(b)

betas = pd.DataFrame(columns=list(factors.columns))
betas.insert(0,'Intercept',[])
for ticker in range(len(tickers)):
    betas = pd.concat([betas,b_all[ticker].iloc[-1:]])
betas.index = pd.Index(tickers)
betas['Weights'] = weights
betas.rename(columns = {'Intercept':'Alpha'}, inplace = True)

y=[]
for beta in range(len(betas.columns)-1):
    y.append(sum(betas[betas.columns[beta]]*betas['Weights']))
y.append(betas['Weights'].sum())
x = pd.DataFrame(y).transpose()
x.index = ['Port']
x.columns = betas.columns
betas = betas.append(x)

ax = betas.iloc[-1:,1:9].transpose().plot.bar()
ax.legend().remove()
ax.axhline(0, color='grey', linewidth=0.8)


spy_df = pdr.get_data_yahoo(symbols='SPY',start=start_date - relativedelta(months=1), end=factors.index[-1], ret_index=True)
spy_df = spy_df.resample('M').last()
spy_df = spy_df['Ret_Index'].pct_change(1).dropna()
spy_df = pd.DataFrame(spy_df)
spy_df.rename({'Ret_Index':'SPY'}, axis='columns', inplace=True)

s_df = factors.join(spy_df).dropna()
s = pd.DataFrame(columns=list(factors.columns))
s.insert(0,'Intercept',[])
for x in range(len(s_df)-window_len):
    result = sm.ols(formula="SPY ~ MKT + SMB + HML + RMW + CMA + UMD + BAB + QMJ", data=s_df[x:60+x]).fit() #
    s = pd.concat([s,result.params.to_frame().swapaxes("index", "columns")])
    #print(result.summary())
s.index = s_df[len(s_df)-len(s):].index
b_spy=s
b_spy.rename(columns = {'Intercept':'Alpha'}, inplace = True)

ax.plot(b_spy.iloc[-1:,1:9].transpose(),'*r')
plt.show()
