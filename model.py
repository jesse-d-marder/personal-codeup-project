
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def evaluate_arima_model(train, test, target, arima_order):
    """ Evaluates an ARIMA model based on arima_order argument, train set, test set, and target. 
    Outputs error, actual test values, and predictions for every timestep in test"""
    train_target = train[target]
    test_target = test[target]
    history = [x for x in train_target]

    # Make predictions
    predictions = []
    for t in range(len(test_target)):
        print(f"\tTesting {arima_order} {t}/{len(test_target)}", end="\r")
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit()
        # Forecast returns forecast value, standard error, and confidence interval - only need forecast value ([0])
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        # Adds the latest test value to history so it can be used to train
        history.append(test_target[t])
    error = mean_squared_error(test_target, predictions)
    print("\n")
    return error, test_target, predictions

def evaluate_models(train, test, target, p_values, d_values, q_values):
    """ Evaluates an ARIMA model per the inputted p, d, and q values. Returns a pandas dataframe with the results from the model"""
    mses=[]
    prediction_list=[]
    actual_test = []
    orders = []
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                orders.append(order)
                try:
                    mse, test_target, predictions = evaluate_arima_model(train, test, target, order)
                    mses.append(mse)
                    prediction_list.append(predictions)
                    actual_test.append(test_target)
                except KeyboardInterrupt:
                    print("Keyboard interrupt")
                    raise
                except:
                    print(f"{order} didn't work, continuing with next order")
                    continue
    results_df = pd.DataFrame.from_records(orders, columns = ['p','d','q'])
    results_df["mse"] = mses
    results_df["test_predictions"] = prediction_list
    results_df["test_actual"] = actual_test
    return results_df

def scale_datasets(train, validate, test, target, features_to_use, features_to_scale):
    """Returns split dataframes with scaled features (and those features that did not need scaling)"""
    
    # Segment out features into individual dataframes
    X_train = train[features_to_use]
    X_validate = validate[features_to_use]
    X_test = test[features_to_use]

    # Segment out target into individual dataframe
    y_train = train[[target]]
    y_validate = validate[[target]]
    y_test = test[[target]]

    # Will scale features using StandardScaler as sigmas are orders of magnitude different from log_ret
    scaler = StandardScaler()

    # Fit scaler to train. Transform validate and test based on fitted scaler. Concatenate to df with non-scaled features.
    X_train_scaled = pd.concat([X_train.drop(columns = features_to_scale),
                                pd.DataFrame(data = scaler.fit_transform(X_train[features_to_scale]), 
                                             columns = features_to_scale, index = X_train.index)], 
                               axis=1)
    X_validate_scaled = pd.concat([X_validate.drop(columns = features_to_scale),
                                pd.DataFrame(data = scaler.transform(X_validate[features_to_scale]), 
                                             columns = features_to_scale, index = X_validate.index)], 
                               axis=1)
    X_test_scaled = pd.concat([X_test.drop(columns = features_to_scale),
                                pd.DataFrame(data = scaler.transform(X_test[features_to_scale]), 
                                             columns = features_to_scale, index = X_test.index)], 
                               axis=1)
                               
    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test

def get_top_features(X_train_scaled, y_train, model, target, n_features):
    """ Performs recursive feature elimination using the inputted model. Returns the top n_features"""
    lm = model
    
    rfe = RFE(lm, n_features_to_select= n_features)

    rfe.fit(X_train_scaled, y_train[[target]])

    # Get mask of the columns selected
    feature_mask = rfe.support_

    # Get list of column names
    rfe_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()

    # view list of columns and their ranking

    # get the ranks
    var_ranks = rfe.ranking_
    # get the variable names
    var_names = X_train_scaled.columns.tolist()
    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    # sort the df by rank
    rfe_ranks_df.sort_values('Rank')
    
    return rfe_feature

def predict_regression(models, X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, perform_feature_selection):
    """Fits and predicts using inputted list of models. Outputs RMSE results, y_train, and y_validate with individual model results"""
    
    rmses_validate={}
    
    # iterate through each model
    for model in models:

        regression_model = model
        
        # Gets a string name for the model
        model_name = model.__repr__().split('()')[0]

        print(model_name)
        
        # Whether to use recursive feature elimination
        if perform_feature_selection:

            top_features = get_top_features(X_train_scaled, y_train, model,target, 16)

            X_train_scaled_featured = X_train_scaled[top_features]
            X_validate_scaled_featured  = X_validate_scaled[top_features]
            X_test_scaled_featured = X_test_scaled[top_features]
            
        else:
            
            X_train_scaled_featured = X_train_scaled.copy()
            X_validate_scaled_featured = X_validate_scaled.copy()
            X_test_scaled_featured = X_test_scaled.copy()
        
        # Fit model to the training data
        regression_model.fit(X_train_scaled_featured, y_train.fwd_log_ret)
        
        # Predict on train and add results to y_train
        y_train[model_name] = regression_model.predict(X_train_scaled_featured)
        
        # Get RMSE metric for train
        rmse_train = mean_squared_error(y_train.fwd_log_ret, y_train[model_name], squared=False)
        
        # Predict on validate
        y_validate[model_name] = regression_model.predict(X_validate_scaled_featured)

        # Get RMSE metric for validate
        rmse_validate = mean_squared_error(y_validate.fwd_log_ret, y_validate[model_name], squared=False)

        # Print RMSE results for train and validate
        # print(f"RMSE for {model_name}\nTraining/In-Sample: ", rmse_train, 
              # "\nValidation/Out-of-Sample: ", rmse_validate)

        rmses_validate[model_name] = rmse_validate
        
    return rmses_validate, y_train, y_validate

def calculate_regression_results(models, rmses_validate, validate, y_validate):
    """Generates average trade and RMSE dataframe from results of regression modeling"""
    
    # Get names of each model
    model_names = [m.__repr__().split('()')[0] for m in models]
    # Add close prices to y_validate to enable calculated trade return
    y_validate["close"] = validate.close
    y_validate["next_day_close"] = validate.close.shift(-1)

    model_rmse_validate = []
    model_average_trade_returns = []

    # Iterate through each model
    for mod in model_names:
        # Create a column saying whether we would go long or not (short)
        y_validate[mod+"_long"] = y_validate[mod]>0
        # Calculate the return that day (assumes always goes long or short every day)
        y_validate[mod+"_ret"] = np.where(y_validate[mod+"_long"], y_validate.next_day_close-y_validate.close, y_validate.close-y_validate.next_day_close)
        print(mod,round(y_validate[mod+"_ret"].mean(),2))
        model_average_trade_returns.append(y_validate[mod+"_ret"].mean())
        model_rmse_validate.append(rmses_validate[mod])
        
    # Add forward return column to y_validate for baseline comparison
    y_validate['daily_return'] = validate.fwd_ret

    # Create dataframe of avg trade and rmse values
    avg_trade_model = pd.DataFrame(data = {'avg_trade':model_average_trade_returns,'rmse':model_rmse_validate}, index = model_names)

    # Concatenate to the avg_trade_model df the avg trade we'd get if just bought every day and sold next (close to close)
    # This is a form of baseline
    avg_trade_model= pd.concat([avg_trade_model,pd.DataFrame({'avg_trade':y_validate.daily_return.mean(),'rmse':np.nan}, index = ['buy_everyday'])])

    return avg_trade_model.sort_values(by='avg_trade',ascending=False)

def get_rolling_predictions(X_train_scaled_featured, X_validate_scaled_featured, y_train, y_validate, model_under_test, target):
    """Predicts target for each day in validate based on rolling window of previous n days"""
    
    # Create copies of dataframes to avoid editing originals
    X_train_scaled_featured_rolled = X_train_scaled_featured.copy()
    X_validate_scaled_featured_rolled = X_validate_scaled_featured.copy()
    y_train_rolled = y_train.copy()              
    
    # Create empty lists to hold predictions. Actuals included here for easier bookkeeping
    train_rolling_predictions = []
    train_rolling_actuals = []
    validate_rolling_predictions = []
    validate_rolling_actuals = []
    
    # Iterate through each row in validate
    for validate_row in range(len(X_validate_scaled_featured)):
        
        # Print out which row we're on 
        print(f"{model_under_test} {validate_row+1}/{len(X_validate_scaled_featured)} Train X range: {X_train_scaled_featured_rolled.index.min().date()} - {X_train_scaled_featured_rolled.index.max().date()}",end="\r")
        # print(f"\nTrain X range: {X_train_scaled_featured_rolled.index.min().date()} - {X_train_scaled_featured_rolled.index.max().date()}",end="\r")

        # Fit the model to the training data
        # print(f"Fitting to train, X: {X_train_scaled_featured_rolled.index.min().date()} - {X_train_scaled_featured_rolled.index.max().date()}, y: {y_train_rolled.index.min().date()} - {y_train_rolled.index.max().date()}") 
        model_under_test.fit(X_train_scaled_featured_rolled, y_train_rolled[target])

        # Predict on Train
        train_prediction = model_under_test.predict(X_train_scaled_featured_rolled)
        train_actual = y_train_rolled[target]

        # Append train results to list
        # print(f"First train prediction {train_prediction[0]} vs actual {train_actual[0]}")
        train_rolling_predictions.append(train_prediction)
        train_rolling_actuals.append(train_actual)

        # Predict on validate
        validate_rolling_predictions.append(model_under_test.predict(X_validate_scaled_featured_rolled.iloc[validate_row].array.reshape(1,-1)))
        validate_rolling_actuals.append(y_validate.iloc[validate_row][target])

        # Remove the first row from X train and y train
        X_train_scaled_featured_rolled = X_train_scaled_featured_rolled.iloc[1:]
        y_train_rolled= y_train_rolled.iloc[1:]

        # Add the latest row from X validate and y validate
        X_train_scaled_featured_rolled = X_train_scaled_featured_rolled.append(X_validate_scaled_featured.iloc[validate_row])
        y_train_rolled = y_train_rolled.append(y_validate.iloc[validate_row])
    
    return train_rolling_predictions, train_rolling_actuals, validate_rolling_predictions, validate_rolling_actuals

