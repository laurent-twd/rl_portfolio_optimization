import yfinance as yf

def get_financials_from_ticker(ticker):

    company = yf.Ticker(ticker)
    return company

