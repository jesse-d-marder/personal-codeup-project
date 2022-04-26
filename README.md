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
### 2. Final report (improve_zillow.ipynb)
### 3. Acquire and Prepare modules (acquire.py, prepare.py)
### 4. Exploration & modeling notebooks (explore.ipynb, model.ipynb)
### 5. Functions to support exploration and modeling work (model.py)

### Project Description and Goals

The goal of this project was to compare the forecasting ability of machine learning and ARIMA models in predicting crytocurrency returns. The profitability of trading strategies built from the results was also evaluated. Cryptocurrency is a relatively new market that has seen explosive growth in the past 5 years. Owing to their tremendous volatility Bitcoin and other cryptocurrencies have gained a reputation for being more speculative assets. The ability to predict cryptocurrency prices is useful for developing profitable trading strategies.

### Initial Questions and Hypotheses

1. Are there any 


### Data Dictionary

| Variable    | Meaning     |
| ----------- | ----------- |
| bedroom    |  number of bedrooms in home         |
| bathroom           |  number of bathrooms in home          |
| bathroom_bin    |  number of bathrooms split into 3 categories     |
| bedroom_bin   |  number of bedrooms split into 3 categories     |
| age    |  age of the home   |


### Project Plan

For this project I followed the data science pipeline:

Planning: I established the goals for this project and the relevant questions I wanted to answer. I used the results from my exploration to guide completion of this project. I followed similar steps as previous projects and used the Trello board from the Zillow regression project as a guide.

Acquire: The data for this project is from a SQL Database called 'zillow' located on a cloud server. The wrangle.py script is used to query the database for the required data tables and returns the data in a Pandas DataFrame. This script also saves the DataFrame to a .csv file for faster subsequent loads. The script will check if the zillow_2017.csv file exists in the current directory and if so will load it into memory, skipping the SQL query.

Prepare: The wrangle.py script uses the same wrangle_zillow function from the acquire step to prepare the data for exploration and modeling. Steps here include removing or filling in  null values (NaN), generating additional features, and encoding categorical variables. This script also contains a split_data function to split the dataset into train, validate, and test sets cleanly. Additional functions to remove outliers and scale the data are included in this file. The model.py file includes a function add_features to create additional features used in exploration and modeling.

Explore: The questions established in planning were analyzed using statistical tests including correlation and t-tests to confirm hypotheses about the data. This work was completed in the explore_zillow.ipynb file and relevant portions were moved to the improve_zillow.ipynb final deliverable. A visualization illustrating the results of the tests and answering each question is included. 

Model: Four different regression algorithms were investigated to determine if log errors could be predicted using features identified during exploration. A select set of hyperparameters were tested against train and validate data to determine which demonstrated the best performance. The final model was selected based on RMSE score on validate (after checking for overfitting) and used to make predictions on the withheld test data.

Delivery: This is in the form of this github repository as well as a presentation of my final notebook to the stakeholders.

### Steps to Reproduce

1. You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the zillow database. Store that env file locally in the repository. 
2. Clone my repository (including the wrangle_zillow.py and model.py modules). Confirm .gitignore is hiding your env.py file.
3. Libraries used are pandas, matplotlib, scipy, sklearn, seaborn, and numpy.
4. You should be able to run improve_zillow.ipynb.

### Key Findings 


### Conclusion and Recommendations

### Future work

- Explore other factors that may affect log errors
- Create additional features and clusters for modeling
- Test additional hyperparameters for the models