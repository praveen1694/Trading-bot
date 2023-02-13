
## Reading necessary Python Libraries
import pandas as pd
import yfinance as yf
import datetime
import pmdarima as pm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import joblib
import pickle

## Setting training period and prediction period
start_training = datetime.date(1990,1,1)
end_training = datetime.date(2021,12,31)
start_testing = datetime.date(2022,1,1)
end_testing = datetime.datetime.today()

print("Enter the Stock Code :")
ticker = str(input())

## Getting the training data
df_training = yf.download(ticker, start = start_training, end=end_training, progress = False)
print(f"Downloaded training data with {df_training.shape[0]} rows and {df_training.shape[1]} columns of {ticker} data")

## Formatting the time series data into Weekly Aggregated Stock Price
df_training = df_training.resample('W').agg({'Open':'first','High':'max','Low':'min','Close':'last','Adj Close':'last'})

## Model Selection
print("Training Auto ARIMA model")

arima_fit = pm.auto_arima(df_training['Close'], error_action = 'ignore',suppress_warnings = True, stepwise = True, approximation= False, seasonal = True)
arima_fit.summary()

print(f"Trained Auto ARIMA model for {ticker} stock prices")

## Saving trained Auto Arima model as a serialized Pickle Object
with open(f'./Model_obj/arima_{ticker}.pkl', 'wb') as pkl:
    pickle.dump(arima_fit, pkl)

print(f"{ticker} trained model saved")

## Testing data
df_testing = yf.download(ticker, start = start_testing, end = end_testing, progress = False)
print(f"Downloaded {df_testing.shape[0]} rows and {df_testing.shape[1]} columns of {ticker} data")
df_testing.drop(columns = ["Open", "High","Adj Close","Low"],inplace = True)
df_testing.shape

## Setting Forecast Horizon
n_fcast1 = len(df_testing)

## Making Forecasts & retrieving confidence Intervals
arima_fcast = arima_fit.predict(n_periods = n_fcast1, return_conf_int = True, alpha = 0.05)
pred = pd.DataFrame(arima_fcast[0],columns = ['predictions']).set_index(df_testing.index)
conf_int = pd.DataFrame(arima_fcast[1],columns = ['lower_95','upper_95']).set_index(df_testing.index)

arima_fcast = pd.concat([pred,conf_int], axis=1)
arima_fcast.head()

print("Predictions produced")


## Plotting the Forecasts vs Actuals
fig, ax = plt.subplots(1,1, figsize = (12,8))

ax = sns.lineplot(data = df_testing['Close'],color = 'black',label = 'Actual')
ax.plot(arima_fcast.predictions, color = 'red', label = 'Forecast with Confidence Interval')

ax.fill_between(arima_fcast.index,arima_fcast.lower_95,arima_fcast.upper_95,alpha = 0.2,facecolor = 'red')
ax.set(title=f"{ticker} stock price - actual vs predicted", xlabel = 'Date', ylabel = 'Close Price(US$)')
ax.legend(loc = 'upper left')

plt.tight_layout()
plt.show()
plt.savefig(f'./Plots/Predictions_{ticker}')


