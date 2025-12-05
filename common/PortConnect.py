
import requests
import pandas as pd
import numpy as np
import time
import certifi
import json

class Port_Connect(object):
    
    def __init__(self,api_key:str):
        self._api_key = api_key
        
    def set_apikey(self,new_apikey):
        self._api_key = new_apikey
    
    def get_apikey(self) -> str:
        return self._api_key

    def _merge_dfs(first_df:pd.DataFrame, second_df:pd.DataFrame, how:str = 'left'):

        cols_to_use = second_df.columns.difference(first_df.columns)

        new_df = pd.merge(first_df, second_df[cols_to_use], left_index=True, right_index=True, how=how)

        return new_df

    def _get_df(self,url:str,is_historical:bool = False) -> pd.DataFrame:

        try:
            # For Python 3.0 and later
            from urllib.request import urlopen
        except ImportError:
            # Fall back to Python 2's urllib2
            from urllib2 import urlopen
        
        def get_jsonparsed_data(url):
            response = urlopen(url,cafile=certifi.where())
            data = response.read().decode("utf-8")
            return json.loads(data)
        
        parsed_url = get_jsonparsed_data(url)

        df = pd.DataFrame(parsed_url)

        if is_historical == True:
            
            df["date"] = pd.to_datetime(df["date"])

            df.sort_values(by='date',ascending=True,inplace=True)

            df.set_index('date',inplace=True)

            df.set_index = pd.to_datetime(df.index)

        else:
            if "symbol" in df.columns:
                df.set_index('symbol',inplace=True)

        return df
    


    def historical_price_by_interval(self,ticker:str,interval:str='1d') -> pd.DataFrame:

        """
        Retrieve historical price data from various time granularities

        Parameters
        ----------
        ticker:str :
            The ticker of the financial instrument to retrieve historical price data. 

        
        api_key:str :
            your FMP API Key
        
        interval: {1min,5min,15min,30min,1hour,4hour,1d,1w,1m,1q,1y} :
            The granularity of how often the price historical data must be retrieved
             (Default value = '1d')

        Returns
        -------

        pd.DataFrame

        """

        url = None

        # Retrieve Historical info from 1 min to 4 hours

        if interval in ['4hour','1hour','30min','15min','5min','1min']:
            url = f'https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{ticker}?apikey={self._api_key}'

            historical_df = self._get_df(url)
            historical_df.insert(0,'symbol',ticker)

            if 'close' and 'date' in list(historical_df.columns):

            

                historical_df.sort_values(by='date',ascending=True,inplace=True)

                historical_df.set_index('date',inplace=True)

                historical_df.index = pd.to_datetime(historical_df.index)

                historical_df['change'] = historical_df['close'].pct_change()   

                historical_df['realOpen'] = historical_df['close'].shift(1)         
            

            

            return historical_df

        # Retrieve Daily Info

        elif interval == '1d':

            url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={self._api_key}'

            historical_df = self._get_df(url,True)

            historical_df['change'] = historical_df['close'].pct_change()

            historical_df['realOpen'] = historical_df['close'].shift(1)

            return historical_df

        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={self._api_key}'

        historical_df = self._get_df(url,True)

    

        historical_df['daily'] = pd.to_datetime(historical_df.index)

        # Retrieve Weekly, Monthly, Quarterly and Yearly Price Data

        if interval == '1w':
        
            historical_df['week'] = historical_df['daily'].dt.to_period('w').apply(lambda r: r.start_time)

            df = historical_df.drop_duplicates(subset=['week'],keep='first')

            df['change'] = df['close'].pct_change()

            df['realOpen'] = df['close'].shift(1)


            return df

        elif interval == '1m':

            historical_df['monthly'] = historical_df['daily'].astype('datetime64[M]')

            df = historical_df.drop_duplicates(subset=['monthly'],keep='first')

            df['change'] = df['close'].pct_change()

            df['realOpen'] = df['close'].shift(1)

            return df

        elif interval == '1q':

            historical_df['quarter'] = historical_df['daily'].dt.to_period('q')

            df = historical_df.drop_duplicates(subset=['quarter'], keep='first')

            df['change'] = df['close'].pct_change()

            df['realOpen'] = df['close'].shift(1)

            return df

        elif interval == '1y':

            historical_df['year'] = historical_df['daily'].dt.year

            df = historical_df.drop_duplicates(subset=['year'],keep='first')

            df['change'] = df['close'].pct_change()

            df['realOpen'] = df['close'].shift(1)

            return df
    
        else:

            raise ValueError('unsupported interval for ',interval,'check your spelling')
        
    def resample_prices(self,df:pd.DataFrame, frequency:str):

    
        # Define frequency mapping
        freq_map = {
            '1d': 'D',
            '1w': 'W',
            '1m': 'MS',  # MS - Month Start
            '1q': 'QS',  # QS - Quarter Start
            '1y': 'AS'  # AS - Year Start
        }
        
        # Get the pandas frequency
        pandas_frequency = freq_map.get(frequency.lower())
        if pandas_frequency is None:
            raise ValueError("Unsupported frequency. Supported frequencies are: daily, weekly, monthly, quarterly, annually.")
        
        # Resample the data according to the specified frequency
        # and take the first occurrence after resampling
        resampled_df = df.resample(pandas_frequency).last().dropna()
        
        # Reset index to turn the date index back into a column
        resampled_df.reset_index(inplace=True)

        return resampled_df
            
            
    def historical_closing_price(self,ticker:str,interval:str = '1d'):
        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?serietype=line&apikey={self._api_key}'
    
        df = self._get_df(url,True)
    
        if df is None:
            return None
        
        resampled_df = self.resample_prices(df=df,frequency=interval)

        resampled_df.set_index('date',inplace=True)

        return resampled_df
    
    
    def get_closing_prices(self,tickers:[str], interval:str = '1d', from_date:str = None,to_date:str=None):
    
        if isinstance(tickers,str):
        
            df = self.historical_closing_price(tickers,interval)
        
            closing_df = pd.pivot_table(data=df,index=df.index,columns='symbol',values='close',aggfunc='mean')
            closing_df.index = pd.to_datetime(closing_df.index)

        else:
        
            dfs = []
    
            for ticker in tickers:
                df = self.historical_closing_price(ticker,interval)
                dfs.append(df)
    
            x = pd.concat(dfs)
    
            closing_df = pd.pivot_table(data=x, index=x.index, columns='symbol',values='close',aggfunc='mean')
            closing_df.index = pd.to_datetime(closing_df.index)

        from_d = from_date if from_date != None else closing_df.index.min()

        to_d = to_date if to_date != None else closing_df.index.max()

        return closing_df[from_d:to_d]



    ## CRYPTO CURRENCIES RELATED

    def get_crypto_quote(self,ticker):

      if isinstance(ticker,str):

        url = f'https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={self.get_apikey()}'

        df = self._get_df(url)

        return df
      elif isinstance(ticker,list):

        dfs = []

        for tick in ticker:
          url = f'https://financialmodelingprep.com/api/v3/quote/{tick}?apikey={self.get_apikey()}'

          df = self._get_df(url)

          dfs.append(df)
        
        cryptos = pd.concat(dfs)
        cryptos.set_index('symbol',inplace=True)

        return cryptos

    def get_available_cryptos(self,min_marketcap=None):
        url = f'https://financialmodelingprep.com/api/v3/quotes/crypto?apikey={self._api_key}'

        df = self._get_df(url)

        df['Is Above 50Avg'] = np.where(df['price'] > df['priceAvg50'],1,0)
        df['Is Above 200Avg'] = np.where(df['price'] > df['priceAvg200'],1,0)

        df['Off Year High'] = df['price'] / df['yearHigh'] - 1

        
        return df


    ## GET FUNDAMENTALS

    def get_financial_statements(self,ticker:str, interval:str = 'annual',type_statement:str='income-statement'):
    
        url = f'https://financialmodelingprep.com/api/v3/{type_statement}/{ticker}?period={interval}&apikey={self.get_apikey()}'
    
        df = self._get_df(url)
    
        df['date'] = pd.to_datetime(df['date'])
    
        df['ym'] = df['date'].astype('datetime64[M]')
    
        return df

    def get_full_financial_statements(self,ticker:str,interval:str='annual'):
        url = f'https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/{ticker}?apikey={self.get_apikey()}'
    
        df = self._get_df(url)
    
        df['date'] = pd.to_datetime(df['date'])
    
        return df

    def get_enterprise_value(self,ticker:str,interval:str='annual'):

        url = f'https://financialmodelingprep.com/api/v3/enterprise-values/{ticker}?&apikey={self.get_apikey()}'
    
        df = self._get_df(url)
    
        df['date'] = pd.to_datetime(df['date'])
    
        return df

    def get_insider_trading(self,ticker:str):
        
        url = f'https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&apikey={self.get_apikey()}'

        df = self._get_df(url)

        df['filingDate'] = pd.to_datetime(df['filingDate'])

        df['transactionDate'] = pd.to_datetime(df['transactionDate'])

        df.set_index('filingDate',inplace=True)

        return df

    def get_filing_dates(self,cik:str):

        url = f'https://financialmodelingprep.com/api/v3/form-thirteen-date/{cik}?apikey={self.get_apikey()}'

        df = self._get_df(url)

        df.columns = ['date']

        return df

    def get_institutional_13f(self,cik:str,date:str):
        url = f'https://financialmodelingprep.com/api/v3/form-thirteen/{cik}?date={date}&apikey={self.get_apikey()}'

        df = self._get_df(url)

        df['date'] = pd.to_datetime(df['date'])

        df['fillingDate'] = pd.to_datetime(df['fillingDate'])

        df.set_index('date',inplace=True)

        return df

    def get_index_tickers(self,index='dowjones'):

        """
        Dow Jones: dowjones
        S&P 500: sp500
        Nasdaq: nasdaq 
        
        """

        url = f'https://financialmodelingprep.com/api/v3/{index}_constituent?apikey={self.get_apikey()}'
        
        df = self._get_df(url)

        return df


    def get_financial_ratios(self,ticker:str,interval='quarter'):
        url = f'https://financialmodelingprep.com/api/v3/ratios/{ticker}?period={interval}&apikey={self.get_apikey()}'

        df = self._get_df(url)

        return df

    def get_key_metrics(self,ticker:str, interval='quarter'):

        url = f'https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period={interval}&apikey={self.get_apikey()}'

        df = self._get_df(url)

        year_period = df['date'].apply(lambda s: s[:4]).astype(str) + ' ' + df['period']

        df.insert(3,'year_period',year_period)

        df['date'] = pd.to_datetime(df['date'])

        df.sort_values(by='date',ascending=True,inplace=True)

        df.set_index('year_period',inplace=True)

        return df


    ## GET ECONOMIC INDICATORS

    def get_economic_indicator(self,indicator:str):
    
        url = f'https://financialmodelingprep.com/api/v4/economic?name={indicator}&apikey={self.get_apikey()}'
    
        df = self._get_df(url)
    
        df['date'] = pd.to_datetime(df['date'])

        df['Year'] = df['date'].dt.year
        df['Month'] = df['date'].dt.month

        df = df.sort_values(by='date',ascending=True)
    
        df.set_index('date',inplace=True)
    
        df.rename(columns={
        'value':indicator
        },inplace=True)
    
        return df

    def get_interest_rates(self,from_date:str,to_date:str):
        url = f'https://financialmodelingprep.com/api/v4/treasury?from={from_date}&to={to_date}&apikey={self.get_apikey()}'

        df = self._get_df(url)

        df['date'] = pd.to_datetime(df['date'])

        df.set_index('date',inplace=True)

        return df

    def get_available_etfs(self):
        url = f'https://financialmodelingprep.com/api/v3/symbol/available-etfs?apikey={self.get_apikey()}'

        df = self._get_df(url)

        return df

    def get_etfs_list(self):

        url = f'https://financialmodelingprep.com/api/v3/etf/list?apikey={self.get_apikey()}'

        df = self._get_df(url)

        return df

    def get_commitment_report_tickers(self):

        url = f'https://financialmodelingprep.com/api/v4/commitment_of_traders_report/list?apikey={self.get_apikey()}'

        df = self._get_df(url)

        return df

    def get_commitment_report(self,ticker):

        url = f'https://financialmodelingprep.com/api/v4/commitment_of_traders_report_analysis/{ticker}?apikey={self.get_apikey()}'

        df = self._get_df(url)

        df['date'] = pd.to_datetime(df['date'])

        df.set_index('date',inplace=True)

        return df

    def company_quote(self, ticker):

        if isinstance(ticker,str):

            url = f'https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={self.get_apikey()}'

            df = self._get_df(url)

            df.set_index('symbol',inplace=True)

            return df
        elif isinstance(ticker, list):

            text = ''.join(i + ',' for i in ticker) 

            text = text[:-1]

            url = f'https://financialmodelingprep.com/api/v3/quote/{text}?apikey={self.get_apikey()}'

            df = self._get_df(url)

            df.set_index('symbol',inplace=True)

            return df

        else:
            return None

    def get_etf_holdings(self,ticker:str):

        url = f'https://financialmodelingprep.com/api/v3/etf-holder/{ticker}?apikey={self.get_apikey()}'

        df = self._get_df(url)

        return df

    def get_profile(self, ticker:str):

        url = f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={self.get_apikey()}'

        profile_df = self._get_df(url)

        profile_df.set_index('symbol',inplace=True)

        return profile_df

    def get_all_tickers(self):

        url = f'https://financialmodelingprep.com/api/v3/available-traded/list?apikey={self.get_apikey()}'

        tickers_df = self._get_df(url)

        return tickers_df

    def get_financial_score(self,ticker:str):

        url = f'https://financialmodelingprep.com/api/v4/score?symbol={ticker}&apikey={self.get_apikey()}'

        score = self._get_df(url)

        # score.set_index('symbol',inplace=True)

        return score

    def get_stock_overview(self,ticker:str):

        company = self.company_quote(ticker=ticker)
        profile = self.get_profile(ticker=ticker)

        df = self._merge_dfs(company,profile)

        return df

    def get_upgrades(self,ticker:str):

        url = f'https://financialmodelingprep.com/api/v4/upgrades-downgrades?symbol={ticker}&apikey={self.get_apikey()}'

        upgrade = self._get_df(url)

        return upgrade

    def get_consensus(self,ticker:str):

        url = f'https://financialmodelingprep.com/api/v4/upgrades-downgrades-consensus?symbol={ticker}&apikey={self.get_apikey()}'

        consensus = self._get_df(url)

        return consensus

    def get_senate_trading(self,ticker):

        url = f'https://financialmodelingprep.com/api/v4/senate-trading?symbol={ticker}&apikey={self.get_apikey()}'

        senate = self._get_df(url)

        return senate
    
    def get_senate_disclosure(self, ticker):

        url = f'https://financialmodelingprep.com/api/v4/senate-disclosure?symbol={ticker}&apikey={self.get_apikey()}'

        senate_disclosure = self._get_df(url)

        return senate_disclosure

    def get_price_change(self,ticker):

        url = f'https://financialmodelingprep.com/api/v3/stock-price-change/{ticker}?apikey={self.get_apikey()}'

        json_df = requests.get(url).json()

        return self._get_df(url)
    
    def get_etf_composition(self,etf_ticker:str):

        url = f'https://financialmodelingprep.com/api/v3/etf-holder/{etf_ticker}?apikey={self.get_apikey()}'

        return self._get_df(url)
    
    def get_all_tickers(self):

        url = f'https://financialmodelingprep.com/api/v3/available-traded/list?apikey={self.get_apikey()}'

        return self._get_df(url)
    
    def get_historical_dividends(self,ticker,from_date=None,to_date=None,tax=0):

        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}?apikey={self.get_apikey()}'

        df = self._get_df(url,is_historical=True)

        if df.empty == True:
            return df
        else: 
            from_d = from_date if from_date != None else df.index.min()

            to_d = to_date if to_date != None else df.index.max()

            df = df[from_d:to_d]
            df['adjDividend'] = df['adjDividend'] * (1-tax)

            return df
    
    def get_price_and_dividends(self,ticker,from_date=None,to_date=None,tax=0):

        dividends = self.get_historical_dividends(ticker=ticker,from_date=from_date,to_date=to_date,tax=tax)
        price = self.get_closing_prices(tickers=ticker,from_date=from_date,to_date=to_date)

        df = pd.merge(price,dividends[["adjDividend"]],left_index=True,right_index=True,how='left').fillna(0)

        df.attrs['title'] = ticker

        return df


        """
        price_return = df[ticker].pct_change().fillna(0)

        yield_return = (df['adjDividend'] / df[ticker].shift(1)).fillna(0)

        total_return = price_return + yield_return 
        
        """



    
    def get_historical_etf_available_dates(self,ticker):

        url = f'https://financialmodelingprep.com/api/v4/etf-holdings/portfolio-date?symbol={ticker}&apikey={self.get_apikey()}'

        return self._get_df(url)
    
    def get_historical_earnings(self,ticker,from_date=None,to_date=None,by_col = 'fiscalDateEnding'):
        url = f'https://financialmodelingprep.com/api/v3/historical/earning_calendar/{ticker}?apikey={self.get_apikey()}'

        df = self._get_df(url)

        df['date'] = pd.to_datetime(df['date'])

        df['updatedFromDate'] = pd.to_datetime(df['updatedFromDate'])

        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])

        df.sort_values(by=by_col,ascending=True,inplace=True)

        df.set_index(by_col,inplace=True)

        from_d = from_date if from_date != None else df.index.min()

        to_d = to_date if to_date != None else df.index.max()

        return df[from_d:to_d]
    

