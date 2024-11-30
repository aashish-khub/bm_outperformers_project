#TODO: COPY OVER BIG FUNCTIONS FROM VARIOUS NOTEBOOKS!
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def featurize_time_series(df, period_chosen, num_trailing_periods, reduce_rows=True, set_threshold_for_target_var_bps=None):
    '''
    Create features for forecasting active returns based on specified periods.

    This function generates a new DataFrame containing lagged features of active returns
    for a specified period. It can also reduce the number of rows based on the chosen period 
    to retain only the first business day of each week, month, quarter, or year.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing at least the columns 'Ticker', 'Date', and 
        'active_returns_*' for various periods.

    period_chosen : str
        The period for which to create features. Must be one of:
        - '1b' : 1 business day
        - '1w' : 1 week
        - '1m' : 1 month
        - '1q' : 1 quarter
        - '1y' : 1 year

    num_trailing_periods : int
        The number of trailing periods to create lagged features for.

    reduce_rows : bool, optional
        If True, reduces the DataFrame to keep only the first business day of the week,
        month, quarter, or year based on the selected period. False means it keeps all rows.
        Defaults to True.
    
    set_threshold_for_target_var_bps : float, optional
        If not None, the function will set the target variable to 1 if it is greater than 
        the threshold and 0 otherwise. Defaults to None

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame containing:
        - Ticker
        - Date
        - ar_{period_chosen}_t (the value to be predicted)
        - ar_{period_chosen}_t_minus_{i} (for i in range(1, num_trailing_periods + 1))

    Notes:
    ------
    - The function handles missing data by dropping rows where there are NaN values 
      after creating lagged features.
    '''
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    col_name = f'active_returns_{period_chosen}'    
    # Create a new dataframe with only the required columns
    new_df = df[['Ticker', 'Date', col_name]].copy()
    new_df = new_df.rename(columns={col_name: f'ar_{period_chosen}_t'})
    if reduce_rows:
        if period_chosen == '1w':
            new_df = new_df.groupby(['Ticker', pd.Grouper(key='Date', freq='W-MON')]).first().reset_index()
        elif period_chosen == '1m':
            new_df = new_df.groupby(['Ticker', pd.Grouper(key='Date', freq='MS')]).first().reset_index()
        elif period_chosen == '1q':
            new_df = new_df.groupby(['Ticker', pd.Grouper(key='Date', freq='QS')]).first().reset_index()
        elif period_chosen == '1y':
            new_df = new_df.groupby(['Ticker', pd.Grouper(key='Date', freq='AS')]).first().reset_index()
    #create lagged features 
    for i in range(1, num_trailing_periods + 1):
        new_df[f'ar_{period_chosen}_t_minus_{i}'] = new_df.groupby('Ticker')[f'ar_{period_chosen}_t'].shift(i)
    new_df = new_df.dropna() #insufficient trailing periods
    if set_threshold_for_target_var_bps is not None:
        threshold_float = set_threshold_for_target_var_bps / 10000
        new_df[f'ar_{period_chosen}_t'] = (new_df[f'ar_{period_chosen}_t'] > threshold_float).astype(int)
    return new_df    

def evaluate_model_performance(y_eval, y_pred, y_pred_proba, PREDICTION_PERIOD, NUM_FEATURES, plot_confusion_matrix=True):
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)
    roc_auc = roc_auc_score(y_eval, y_pred_proba[:, 1])
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC AUC:", roc_auc)

    if plot_confusion_matrix:
        # Show the confusion matrix in a plot using seaborn
        cm = confusion_matrix(y_eval, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # Print title that mentions the prediction period and num features
        plt.title('Confusion Matrix for ' + PREDICTION_PERIOD + " prediction using " + str(NUM_FEATURES) + " Features")
        plt.show()    