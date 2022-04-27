### This repository contains the code for the personal project completed as part of the Codeup Data Science curriculum

## Repo contents:
### 1. This Readme:
- Project description with goals
- Inital hypothesis/questions on data, ideas
- Data dictionary
- Project planning
- Instructions for reproducing this project and findings
- Key findings and recommendations for this project
- Conclusion
### 2. Final report (predict_crypto.ipynb)
### 3. Acquire and Prepare modules (acquire.py, prepare.py)
### 4. Exploration & modeling notebooks (explore.ipynb, model.ipynb)
### 5. Functions to support exploration and modeling work (model.py)

### Project Description and Goals

The goal of this project was to compare the forecasting ability of machine learning and ARIMA models in predicting crytocurrency returns. The profitability of trading strategies built from the results was also evaluated. Cryptocurrency is a relatively new market that has seen explosive growth in the past 5 years. Owing to their tremendous volatility Bitcoin and other cryptocurrencies have gained a reputation for being more speculative assets. The ability to predict cryptocurrency prices is useful for developing profitable trading strategies and understanding the risk they add to a portfolio.

### Initial Questions and Hypotheses

1. Are past returns predictive of future returns for cryptocurrencies?
2. Is there a relationship between volatility and returns?

### Data Dictionary

| Variable    | Meaning     |
| ----------- | ----------- |
| btc   |  Bitcoin       |
| sigma |  Parkinson range volatility estimator (lag values 1- 7)     |
| RR    |  Relative price range (first lag)   |
| fwd_log_ret   |  the log return for one day in the future (regression target)   |
| fwd_close_positive    |  whether the next day close is higher than today's close (classification target)  |


### Project Plan

For this project I followed the data science pipeline:

Planning: I wanted to roughly follow the methodology used by Helder Sebastiao and Pedro Godinho in their article "Forecasting and trading cryptocurrencies with machine learning under changing market conditions," published 06 Jan 2021 (https://rdcu.be/cMaLB). The authors examined the predictability of three major crytocurrencies - Bitcoin, Ethereum, and Litecoin - using machine learning techniques for the period April 15, 2015 - March 03, 2019. Due to time constraints I focused solely on Bitcoin and limited the number of models tested. By no means is this project a faithful replication of their work but does provide insight into potential uses of machine learning to predict Bitcoin returns and profitability of trading strategies built from model results. This work also covers a wider range in data and compares the machine learning models to an ARIMA model.

Acquire: The data for this project consists of daily open, high, low, and close prices as well as volume data for Bitcoin from 2016-08-24 - 2022-04-21 and was acquired using the Coinbase Pro API. An account and API key are required for access. Scripts to acquire this data are included in acquire.py.

Prepare: The prepare.py module cleans the data and also contains a function to add features and targets (regression and classification) to the dataframes. For Bitcoin some 

Explore: The questions established in planning were analyzed using statistical tests including correlation and t-tests to confirm hypotheses about the data. Relationships between predictors and the target were explored. 

Model: Four different regression algorithms were investigated to determine if log returns could be predicted

Delivery: This is in the form of this github repository. I am happy to talk through my results with anyone interested and collaborate on any projects related to trading.

### Steps to Reproduce
1. You will need an env.py file that contains the passphrase, secret_key and api_key of your Coinbase PRO account. Store that env file locally in the repository. 
2. Clone my repository. Confirm .gitignore is hiding your env.py file.
3. Libraries used are pandas, matplotlib, scipy, sklearn, seaborn, and numpy.
4. You should be able to run predict_crypto.ipynb.

### Key Findings 


### Conclusion and Recommendations

### Future work

- Explore other features and feature combinations that may be predictive of returns. The original paper also included blockchain information (such as on-chain volume, active addresses, and block sizes) as inputs, though for most of the most successful models only returns, volatility, and daily dummies were actually used. 
- Test additional hyperparameters for the models and different algorithms. Based on time and resource constraints some models used in the paper could not be tested, particularly for the single-step prediction method. 
- Test Ethereum and Litecoin. This data was original downloaded and intended to be tested but due to time constraints was not.
- Generate additional trading statistics for strategies based on the model results. Here I only included average trade but other metrics such as win rate, Sharpe ratio, and max drawdown are important to know prior to implementing in live trading. 
- Test ensemble methods of determining whether a trade should be taken, per the paper. These were shown to be more successful than using a single model alone to make trading decisions. 
- Test models with higher frequency data
