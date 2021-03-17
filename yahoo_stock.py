"""
Created on Sun Oct 25 19:38:10 2020

@author: Tristan

This script
    a. pulls stock info, option info, and market info
    b. plots historical price of one or multiple tickers

"""


import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
from yahoo_fin.stock_info import *
from yahoo_fin.options import *
import requests_html
import requests
from tabulate import tabulate


"""
Pandas DataReader:
e.g.
apple = web.DataReader("AAPL", "yahoo", start, end)
google=web.get_data_yahoo('GOOG',start,end)
google.head()


yahoo_fin API:
http://theautomatic.net/yahoo_fin-documentation/#get_day_gainers
"""


def set_date(y,m,d):
    if y==0 and m==0 and d==0:
        return datetime.date.today()
    return datetime.datetime(y,m,d)


def symbol_to_name(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']


def plot_price(company, start, end, mode= 'candle', mav=(5,20,90),volume=True):
    """
    Plot the adjusted closing price of company

    type: ohlc, line, candle
    mav -- moving averge, >= 2; multiple mac is allowed; default: 5,20,90 days
    volumn: True or False

    """

    data= web.get_data_yahoo(company,start,end)
    mpf.plot(data, type=mode, mav=mav, volume= volume)


def get_yesterday_data(company):
    """
    get the most recent day data including:
        ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
    """
    start = datetime.date.today() + datetime.timedelta(-1)
    end = datetime.date.today()
    # Slow
    data= web.get_data_yahoo(company,start,end)

    return data.tail(1)


def get_high(company):
    return get_yesterday_data(company)["High"]

def get_low(company):
    return get_yesterday_data(company)["Low"]

def get_open(company):
    return get_yesterday_data(company)["Open"]

def get_volume(company):
    return get_yesterday_data(company)["Volume"]

def get_close(company):
    return get_yesterday_data(company)["Adj Close"]

def get_change(company, percent=False):
    current = get_realtime_price(company)
    close = get_open(company)[0]
    change = (current - close) * 100 / close if percent else current - close
    return round(change, 2)


def get_multiple_comps_data(company_list, start,end):
    """
    get data of multiple companys from strat to end
    """
    container = []
    for i in company_list:

        data= web.get_data_yahoo(i,start,end)
        container.append(data)

    df = {}
    for c in range(len(company_list)):
        df[company_list[c]] = container[c]["Adj Close"]

    stocks = pd.DataFrame(df)

    return stocks


def plot_multiple_comps_price(company_list, start, end):
    """
    Plot the adjusted closing price of multiple companys
    """
    stocks = get_multiple_comps_data(company_list, start, end)
    fig = stocks.plot(grid = True).get_figure()
    fig.savefig("aaa.png")


def return_analysis(file_name, company_list, start, end):
    """
    Input: a list of companies, start date, end date
    return(t,t-1) = return(t)/return(t-1)
    """
    stocks = get_multiple_comps_data(company_list, start, end)
    stock_return = stocks.apply(lambda x: x / x[0])
    fig = stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2).get_figure()
    fig.savefig('/tmp/' + file_name)


def volatility_analysis(company_list, start, end):
    """
    Input: a list of companies, start date, end date
    increase(t) = (price(t) - price(t-1)) / price(t)
    change(t) = log(price(t)) - log(price(t-1))

    """
    stocks = get_multiple_comps_data(company_list, start, end)

    # shift moves dates back by 1.
    stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
    fig = stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2).get_figure()
    fig.savefig('bbb.png')


def get_realtime_price(company):
    """
    Get the real-time price of a stock
    Using yahoo_fin API
    Input: ticker
    ticker is case insensitive
    """
    return get_live_price(company)

def get_quote(company):
    """
    Input: ticker

    dict_result: Default is True.
    If True, the function returns the results in a dict format.  Otherwise, the
    results are returned in a data frame.

    Output:
        1y Target Est
        52 Week Range
        Ask
        Volume
        Beta
        Bid
        Days Range
        Dividend & Yield
        EPS (TTM)
        Earnings Date
        Ex-Dividend Date
        Market Cap
        Open
        PE Ratio (TTM)
        Previous Close
        Quote Price
        Volume
    """
    return get_quote_table(company, dict_result = False)

def get_valuation(company):
    """
    Input: ticker

    Scrapes the “Valuation Measures” data off the statistics page for the input
    ticker, which includes information on Price / Sales, P/E, and market cap.

    """
    return get_stats_valuation(company)

def get_gainers(N):
    """
    Using yahoo_fin API

    Scrapes the top N (at most) stocks with the largest gains (on the given
    trading day) from Yahoo Finance

    Required N <= 100

    """
    return get_day_gainers().head(N)

def get_most_active(N):
    """
    Using yahoo_fin API

    Scrapes the top N most active stocks (on the given trading day)
    from Yahoo Finance

    Required N <= 100
    """
    return get_day_most_active().head(N)

def get_losers(N):
    """
    Using yahoo_fin API

    Scrapes the top N (at most) worst performing stocks (on the given
    trading day) from Yahoo Finance

    Required N <= 100

    """
    return get_day_losers().head(N)


def get_stock_analysis(company):
    """
    Input: ticker

    """
    return get_analysts_info(company)


def get_option_table(ticker, date = None):
    """
    Input: ticker

    Returns a dictionary with two data frames. The keys of the dictionary are
    labeled calls (which maps to the calls data table) and puts (which maps to
    the puts data table).

    """
    return get_options_chain(ticker, date)


if __name__=='__main__':
    company = "AAPL"
    start_y, start_m, start_d = 2016,1,1
    end_y, end_m, end_d = 2020,1,1
    start = set_date(start_y, start_m, start_d)
    end = set_date(end_y, end_m, end_d)
    start_date = "2018-11-12"
    end_date = "2018-12-12"
    print(str(start))
    start = datetime.datetime.strptime(start_date.split('T')[0], '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date.split('T')[0], '%Y-%m-%d')

    company_list = ["AAPL", "BA", "BABA"]

    #print(get_losers(3))

    #plot_multiple_comps_price(company_list, start, end)
    #volatility_analysis(company_list, start, end)
    print(tabulate(get_stock_analysis("AAPL")))

    #plot_multiple_comps_price(company_list, start,end)


