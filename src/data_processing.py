import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a specified CSV file path.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
    
    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
    """
    try:
        logging.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file was not found at {filepath}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic preprocessing on the transaction DataFrame.

    - Converts 'TransactionStartTime' to datetime objects.
    - Handles potential inconsistencies in 'Amount' and 'Value'.

    Args:
        df (pd.DataFrame): The raw transaction DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logging.info("Starting data preprocessing...")
    # Work on a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # Convert to datetime
    df_processed['TransactionStartTime'] = pd.to_datetime(df_processed['TransactionStartTime'])

    # Ensure 'Value' is positive as it represents the transaction's magnitude
    if 'Value' not in df_processed.columns:
        logging.warning("'Value' column not found. Creating from 'Amount'.")
        df_processed['Value'] = df_processed['Amount'].abs()
    
    logging.info("Data preprocessing complete.")
    return df_processed

def generate_rfms_features(df: pd.DataFrame, snapshot_date: datetime) -> pd.DataFrame:
    """
    Generates Recency, Frequency, Monetary, and Std Dev (Volatility) features.

    Args:
        df (pd.DataFrame): The preprocessed transaction DataFrame.
        snapshot_date (datetime): The reference date for calculating recency.

    Returns:
        pd.DataFrame: A DataFrame with CustomerId and RFMS features.
    """
    logging.info("Generating RFMS features...")
    
    # Aggregate transaction data at the customer level
    agg_df = df.groupby('CustomerId').agg(
        Last_Transaction_Date=('TransactionStartTime', 'max'),
        Frequency=('TransactionId', 'count'),
        Monetary=('Value', 'sum'),
        Std_Dev_Amount=('Value', 'std')
    ).reset_index()

    # Calculate Recency
    agg_df['Recency'] = (snapshot_date - agg_df['Last_Transaction_Date']).dt.days

    # Fill NaN in Std_Dev_Amount for customers with a single transaction
    agg_df['Std_Dev_Amount'] = agg_df['Std_Dev_Amount'].fillna(0)
    
    # Select and rename columns for clarity
    rfms_df = agg_df[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Std_Dev_Amount']]
    
    logging.info("RFMS feature generation complete.")
    return rfms_df

def create_risk_proxy(rfms_df: pd.DataFrame, fraud_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a binary risk proxy variable based on RFM scores and fraud history.
    
    Proxy logic: A customer is 'high_risk' (1) if they have any fraudulent 
    transactions OR if their combined RFM score is in the bottom quintile (20%). 
    Otherwise, they are 'low_risk' (0).

    Args:
        rfms_df (pd.DataFrame): DataFrame with RFMS features.
        fraud_df (pd.DataFrame): DataFrame with CustomerId and FraudResult.

    Returns:
        pd.DataFrame: The RFMS DataFrame with an added 'high_risk' proxy column.
    """
    logging.info("Creating risk proxy variable...")
    
    # 1. Handle Fraud Data
    fraud_summary = fraud_df.groupby('CustomerId')['FraudResult'].sum().reset_index()
    fraud_summary.rename(columns={'FraudResult': 'TotalFraudTransactions'}, inplace=True)
    
    # 2. Calculate RFM Scores
    data = rfms_df.copy()
    data['R_Score'] = pd.qcut(data['Recency'], 5, labels=False, duplicates='drop') + 1
    data['F_Score'] = pd.qcut(data['Frequency'].rank(method='first'), 5, labels=False) + 1
    data['M_Score'] = pd.qcut(data['Monetary'].rank(method='first'), 5, labels=False) + 1
    
    # Recency is inverted: lower is better
    data['R_Score'] = 6 - data['R_Score']
    
    data['RFM_Score'] = data['R_Score'] + data['F_Score'] + data['M_Score']
    
    # 3. Merge fraud data with RFM data
    data = pd.merge(data, fraud_summary, on='CustomerId', how='left')
    data['TotalFraudTransactions'].fillna(0, inplace=True)

    # 4. Define the proxy
    # Define risk based on RFM score quantile
    rfm_risk_threshold = data['RFM_Score'].quantile(0.20)
    
    # A customer is high risk if they have any fraud history OR a very low RFM score
    data['high_risk'] = np.where(
        (data['TotalFraudTransactions'] > 0) | (data['RFM_Score'] <= rfm_risk_threshold), 
        1, 
        0
    )
    
    logging.info(f"Risk proxy created. High-risk proportion: {data['high_risk'].mean():.2%}")
    return data


def get_feature_correlations(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Calculates the correlation of all numerical features with the target variable.

    Args:
        df (pd.DataFrame): The DataFrame containing features and the target.
        target_col (str): The name of the target variable column.

    Returns:
        pd.DataFrame: A DataFrame showing features and their correlation with the target.
    """
    logging.info(f"Calculating feature correlations with target '{target_col}'...")
    
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found in DataFrame.")
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    corr_matrix = df.corr()
    corr_target = corr_matrix[[target_col]].sort_values(by=target_col, ascending=False)
    
    logging.info("Correlation calculation complete.")
    return corr_target.drop(target_col) # Drop the target's self-correlation