"""
Data Preparation Module for ICU Requirement Prediction

This module handles:
- Data ingestion from S3
- Schema validation
- Data profiling and reporting
- Preprocessing (with leakage-safe practices)
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
import boto3
from io import StringIO

# Configuration
S3_BUCKET = "icu-required"
S3_KEY = "cleaned_data.csv"
AWS_REGION = "us-east-1"

# Actual column names from the dataset (already pre-encoded)
ACTUAL_COLUMNS = [
    "Type",                    # hematologic vs solid (encoded)
    "Neutropenia",             # binary
    "Hypotension_Level",       # ordinal/categorical
    "Focus_PneumResp",         # one-hot encoded
    "UTI",                     # one-hot encoded
    "Focus_Bloodstream",       # one-hot encoded
    "Focus_GI_Hepatobiliary",  # one-hot encoded
    "Focus_SoftTissue",        # one-hot encoded
    "Focus_NoneUnknown",       # one-hot encoded
    "Comorb",                  # comorbidity score/flag
    "Mets_Binary",             # metastasis binary
    "Mets_Missing",            # metastasis missing indicator
    "Line_Rx",                 # line of treatment
    "MASCC",                   # MASCC score (numeric)
    "qSOFA",                   # qSOFA score (numeric)
    "Gender",                  # encoded gender
    "Age_Group",               # age group (ordinal)
]

# Target column
TARGET_COLUMN = "ICU_Requirement"

# Feature columns (all except target)
FEATURE_COLUMNS = ACTUAL_COLUMNS.copy()

# Numeric features (continuous or ordinal scores)
NUMERIC_FEATURES = ["MASCC", "qSOFA", "Hypotension_Level", "Line_Rx", "Age_Group", "Comorb"]

# Binary/already-encoded features (no further encoding needed)
BINARY_FEATURES = [f for f in FEATURE_COLUMNS if f not in NUMERIC_FEATURES]


def load_data_from_s3(bucket: str, key: str, region: str) -> pd.DataFrame:
    """Load CSV data from S3 bucket."""
    print(f"Loading data from s3://{bucket}/{key} (region: {region})...")
    
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        print(f"Successfully loaded {len(df)} rows from S3.")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data from S3: {e}")


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a comprehensive profile of the dataset."""
    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "value_counts": {}
    }
    
    # Add value counts for each column
    for col in df.columns:
        profile["value_counts"][col] = df[col].value_counts().to_dict()
    
    return profile


def validate_schema(df: pd.DataFrame, expected_features: list, target_col: str) -> tuple:
    """
    Validate that required columns exist in the dataset.
    Returns (valid: bool, missing_features: list, extra_columns: list)
    """
    actual_columns = set(df.columns)
    expected_columns = set(expected_features + [target_col])
    
    missing_features = list(expected_columns - actual_columns)
    extra_columns = list(actual_columns - expected_columns)
    
    is_valid = len(missing_features) == 0 and target_col in actual_columns
    
    return is_valid, missing_features, extra_columns


def analyze_target(df: pd.DataFrame, target_col: str) -> dict:
    """
    Analyze the target variable distribution.
    """
    target = df[target_col]
    
    analysis = {
        "column_name": target_col,
        "dtype": str(target.dtype),
        "unique_values": list(target.unique()),
        "value_counts": target.value_counts().to_dict(),
        "class_balance": (target.value_counts(normalize=True) * 100).round(2).to_dict(),
        "missing_count": target.isnull().sum(),
        "mapping": "Binary: 1 = ICU Required, 0 = Not ICU Required"
    }
    
    return analysis


def describe_features(df: pd.DataFrame, feature_cols: list) -> dict:
    """Generate descriptive statistics for features."""
    descriptions = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        desc = {
            "dtype": str(df[col].dtype),
            "unique_count": df[col].nunique(),
            "missing_count": df[col].isnull().sum(),
        }
        
        if df[col].dtype in ['int64', 'float64']:
            desc["min"] = float(df[col].min())
            desc["max"] = float(df[col].max())
            desc["mean"] = float(df[col].mean())
            desc["std"] = float(df[col].std())
            desc["median"] = float(df[col].median())
        
        desc["value_counts"] = df[col].value_counts().head(10).to_dict()
        descriptions[col] = desc
    
    return descriptions


def main():
    """Main data preparation function."""
    parser = argparse.ArgumentParser(description="Data Preparation for ICU Prediction")
    parser.add_argument("--s3-bucket", type=str, default=S3_BUCKET)
    parser.add_argument("--s3-key", type=str, default=S3_KEY)
    parser.add_argument("--region", type=str, default=AWS_REGION)
    parser.add_argument("--output-dir", type=str, default="./data")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = load_data_from_s3(args.s3_bucket, args.s3_key, args.region)
    
    # Profile dataset
    print("\n" + "="*60)
    print("DATASET PROFILE")
    print("="*60)
    profile = profile_dataset(df)
    
    print(f"\nRow count: {profile['row_count']}")
    print(f"Column count: {profile['column_count']}")
    print(f"\nColumns: {profile['columns']}")
    print(f"\nData types:")
    for col, dtype in profile['dtypes'].items():
        print(f"  {col}: {dtype}")
    
    print(f"\nMissingness (%):")
    any_missing = False
    for col, pct in profile['missing_percentages'].items():
        if pct > 0:
            print(f"  {col}: {pct}%")
            any_missing = True
    if not any_missing:
        print("  No missing values detected!")
    
    # Validate schema
    print("\n" + "="*60)
    print("SCHEMA VALIDATION")
    print("="*60)
    
    is_valid, missing_features, extra_columns = validate_schema(
        df, FEATURE_COLUMNS, TARGET_COLUMN
    )
    
    if TARGET_COLUMN in df.columns:
        print(f"\n✓ Target column found: '{TARGET_COLUMN}'")
    else:
        print(f"\n✗ Target column NOT found: '{TARGET_COLUMN}'")
    
    if missing_features:
        print(f"\n✗ Missing expected features: {missing_features}")
    else:
        print(f"\n✓ All expected features found")
    
    if extra_columns:
        print(f"\nℹ Extra columns in dataset (not in expected list): {extra_columns}")
    
    if not is_valid:
        print("\n⚠ Schema validation had issues - proceeding with available columns")
    
    # Analyze target variable
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS")
    print("="*60)
    
    target_analysis = analyze_target(df, TARGET_COLUMN)
    print(f"\nColumn: {target_analysis['column_name']}")
    print(f"Data type: {target_analysis['dtype']}")
    print(f"Unique values: {target_analysis['unique_values']}")
    print(f"\nClass distribution:")
    for val, count in target_analysis['value_counts'].items():
        pct = target_analysis['class_balance'].get(val, 0)
        label = "ICU Required" if val == 1 else "Not ICU Required"
        print(f"  {val} ({label}): {count} ({pct}%)")
    print(f"\nTarget mapping: {target_analysis['mapping']}")
    
    # Feature descriptions
    print("\n" + "="*60)
    print("FEATURE DESCRIPTIONS")
    print("="*60)
    
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    feature_descriptions = describe_features(df, available_features)
    
    for feat, desc in feature_descriptions.items():
        print(f"\n{feat}:")
        print(f"  Type: {desc['dtype']}, Unique: {desc['unique_count']}, Missing: {desc['missing_count']}")
        if 'mean' in desc:
            print(f"  Range: [{desc['min']}, {desc['max']}], Mean: {desc['mean']:.2f}, Std: {desc['std']:.2f}")
    
    # Save validation report
    validation_report = {
        "profile": profile,
        "target_analysis": target_analysis,
        "feature_descriptions": feature_descriptions,
        "is_valid": is_valid,
        "feature_columns": available_features,
        "target_column": TARGET_COLUMN
    }
    
    report_path = os.path.join(args.output_dir, "data_validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    print(f"\n\nValidation report saved to: {report_path}")
    
    # Save data locally for further processing
    data_path = os.path.join(args.output_dir, "processed_data.csv")
    df.to_csv(data_path, index=False)
    print(f"Processed data saved to: {data_path}")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    
    return df, validation_report


if __name__ == "__main__":
    main()
