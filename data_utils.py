import yfinance as yf
import pandas as pd

def get_financials_from_ticker(ticker):

    company = yf.Ticker(ticker)
    return company.history(period="max")

def yfinance_get_data(tickers, period):

    data = 0
    columns = ['Date', 'High', 'Low', 'Close']
    for ticker in tickers:
        df = get_financials_from_ticker(ticker).reset_index()
        start, end = list(map(lambda x: pd.to_datetime(x), period))
        df = df.loc[(df['Date'] >= start) & (df['Date'] <= end)][columns].reset_index(drop = True)
        df.columns = [columns[0]] + list(ticker + "_" + df.columns[1:]) 
        try:
            data = pd.merge(
                left = data,
                right = df,
                how = 'outer',
                on = 'Date'
            )
        except:
            data = df

    close_columns = [c for c in data.columns if 'close' in c.lower()]
    low_columns = [c for c in data.columns if 'low' in c.lower()]
    high_columns = [c for c in data.columns if 'high' in c.lower()]

    data = data[close_columns + low_columns + high_columns]

    return data