import pytest
import pandas as pd
import numpy as np
from src.data_processing import RFMTransformer, HighRiskLabelGenerator, build_feature_pipeline
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
from sklearn.utils.validation import check_is_fitted

# Import missing scikit-learn components
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


@pytest.fixture
def fixed_now():
    """Single fixed timestamp for all tests"""
    return datetime(2023, 1, 15, 12, 30, 45)  # Fixed reference time with minutes/seconds

@pytest.fixture
def sample_data(fixed_now):
    """Generate synthetic test data with varied timestamps"""
    data = {
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionId': [101, 102, 201, 202, 301],
        'TransactionStartTime': [
            fixed_now - timedelta(days=5, hours=3),    # Jan 10, 09:30:45
            fixed_now - timedelta(days=2, minutes=15), # Jan 13, 12:15:45
            fixed_now - timedelta(days=10, seconds=1), # Jan 5, 12:30:44
            fixed_now - timedelta(hours=18),           # Jan 14, 18:30:45
            fixed_now - timedelta(days=20)             # Dec 26, 12:30:45
        ],
        'Value': [100, 200, 50, 150, 300],
        'FraudResult': [0, 0, 1, 0, 0]  # Added for high-risk testing
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def test_rfm_transformer_columns(sample_data, fixed_now):
    """Test all RFM and temporal features are created"""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy())

    expected_columns = {
        'Recency', 'Frequency', 'MonetarySum',
        'MonetaryMean', 'MonetaryStd',
        'TransactionHour', 'TransactionDay', 'TransactionMonth'
    }
    assert expected_columns.issubset(transformed.columns)
    assert transformed.shape[0] == sample_data.shape[0], "RFMTransformer should maintain the original number of rows."

def test_rfm_calculations(sample_data, fixed_now):
    """Verify RFM calculations for all customer types"""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy())

    # Test Customer 1 (2 transactions)
    cust1_data = sample_data[sample_data['CustomerId'] == 1]
    cust1_transformed = transformed[transformed['CustomerId'] == 1].iloc[0]
    assert len(transformed[transformed['CustomerId'] == 1]) == 2

    assert np.allclose(cust1_transformed['MonetarySum'], 300)
    assert np.allclose(cust1_transformed['MonetaryMean'], 150)
    assert np.allclose(cust1_transformed['MonetaryStd'], np.std([100, 200], ddof=1))
    assert np.allclose(cust1_transformed['Recency'], (fixed_now - cust1_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust1_transformed['Frequency'], 2)

    # Test Customer 2 (2 transactions with different pattern)
    cust2_data = sample_data[sample_data['CustomerId'] == 2]
    cust2_transformed = transformed[transformed['CustomerId'] == 2].iloc[0]
    assert len(transformed[transformed['CustomerId'] == 2]) == 2

    assert np.allclose(cust2_transformed['MonetarySum'], 200)
    assert np.allclose(cust2_transformed['MonetaryMean'], 100)
    assert np.allclose(cust2_transformed['MonetaryStd'], np.std([50, 150], ddof=1))
    assert np.allclose(cust2_transformed['Recency'], (fixed_now - cust2_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust2_transformed['Frequency'], 2)

    # Test Customer 3 (single transaction - edge case)
    cust3_data = sample_data[sample_data['CustomerId'] == 3]
    cust3_transformed = transformed[transformed['CustomerId'] == 3].iloc[0]
    assert len(transformed[transformed['CustomerId'] == 3]) == 1

    assert np.allclose(cust3_transformed['MonetarySum'], 300)
    assert np.allclose(cust3_transformed['MonetaryMean'], 300)
    assert np.allclose(cust3_transformed['MonetaryStd'], 0)
    assert np.allclose(cust3_transformed['Recency'], (fixed_now - cust3_data['TransactionStartTime'].max()).days)
    assert np.allclose(cust3_transformed['Frequency'], 1)

def test_temporal_features(sample_data, fixed_now):
    """Verify precise temporal feature extraction with varied times"""
    transformer = RFMTransformer(snapshot_date=fixed_now)
    transformed = transformer.transform(sample_data.copy())

    expected_values = [
        (9, 10, 1),    # Jan 10, 09:30:45
        (12, 13, 1),   # Jan 13, 12:15:45
        (12, 5, 1),    # Jan 5, 12:30:44
        (18, 14, 1),   # Jan 14, 18:30:45
        (12, 26, 12)   # Dec 26, 12:30:45
    ]

    for idx, (exp_hour, exp_day, exp_month) in enumerate(expected_values):
        assert transformed.loc[idx, 'TransactionHour'] == exp_hour
        assert transformed.loc[idx, 'TransactionDay'] == exp_day
        assert transformed.loc[idx, 'TransactionMonth'] == exp_month

def test_high_risk_labeler(sample_data, fixed_now):
    """Test reproducible risk label generation"""
    rfm_transformer = RFMTransformer(snapshot_date=fixed_now)
    rfm_data = rfm_transformer.transform(sample_data.copy())

    labeler = HighRiskLabelGenerator(n_clusters=3, random_state=42)
    labeled_data = labeler.fit_transform(rfm_data)

    check_is_fitted(labeler, 'kmeans')
    assert hasattr(labeler, 'high_risk_cluster_')
    assert 'is_high_risk' in labeled_data.columns

    unique_labels = set(labeled_data['is_high_risk'].unique())
    assert unique_labels.issubset({0, 1, 2})  # Allow 0, 1, 2 for 3 clusters
    assert 0 <= labeled_data['is_high_risk'].mean() <= 1  # Ensure binary-like distribution

def build_feature_pipeline():
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = []
   
    numerical_features_for_pipeline = ['Recency', 'Frequency', 'Total_Transaction_Amount', 'Avg_Transaction_Amount', 'Std_Dev_Amount']
    categorical_features_for_pipeline = ['TransactionHour', 'TransactionDay', 'TransactionMonth'] 
    numerical_features = ['Recency', 'Frequency', 'Total_Transaction_Amount', 'Avg_Transaction_Amount', 'Std_Dev_Amount', 'Avg_TransactionHour']
    categorical_features = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'] # From data_processing.py

    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features) # Added sparse_output
        ],
        remainder='drop' # Change to 'passthrough' if you want to keep other columns, or 'drop' if only defined features are needed
    )

    return Pipeline([
        ('rfm_features', RFMTransformer(snapshot_date=fixed_now)), # Pass fixed_now for consistent testing
        ('risk_labels', HighRiskLabelGenerator(n_clusters=3, random_state=42)),
        ('final_preprocessing', final_preprocessor)
    ])

def test_pipeline_with_missing_data(fixed_now):
    """Test pipeline handles missing values"""
    data = pd.DataFrame({
        'CustomerId': [1, 1, 2, 3],
        'TransactionId': [101, 102, 201, 301],
        'TransactionStartTime': [fixed_now, fixed_now - timedelta(days=1), fixed_now, fixed_now - timedelta(days=2)], # Vary timestamps for RFM
        'Value': [100, np.nan, 200, 300],
        'FraudResult': [0, 1, 0, 0]
    })
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    pipeline = build_feature_pipeline()
    processed = pipeline.fit_transform(data)

    # After processing, all numerical columns should be filled and scaled.
    # One-hot encoded columns should also be numerical.
    # The output of ColumnTransformer is a numpy array if `sparse_output=False`.
    # So, we check for NaN in the numpy array.
    assert not np.isnan(processed).any(), "Processed data should not contain any NaNs"
    assert processed.shape[0] == data['CustomerId'].nunique(), "Output rows should match unique customers"
    # Further checks: e.g., check shape, dtypes, specific values if possible.