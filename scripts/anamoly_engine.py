"""
Project Sentinel – Anomaly Detection Engine
"""

import pandas as pd
import numpy as np


# ============================================
# STEP 1: DATA PREPARATION
# ============================================

def prepare_data(df):
    """
    Input:
        Raw dataset with columns:
        - state
        - district
        - time_period
        - enrolment_count
        - update_count

    Output:
        DataFrame with derived metrics
    """
    required_cols = [
        'state',
        'district',
        'time_period',
        'enrolment_count',
        'update_count'
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df['enrolment_count'] = pd.to_numeric(df['enrolment_count'], errors='coerce')
    df['update_count'] = pd.to_numeric(df['update_count'], errors='coerce')

    df.dropna(subset=['enrolment_count', 'update_count'], inplace=True)

    df['total_activity'] = df['enrolment_count'] + df['update_count']
    return df


# ============================================
# STEP 2: Z-SCORE DETECTION
# ============================================

def calculate_zscore_flags(group_df, metric_column):
    mean = group_df[metric_column].mean()
    std = group_df[metric_column].std()

    group_df = group_df.copy()

    if std == 0 or np.isnan(std):
        group_df['z_score'] = 0.0
    else:
        group_df['z_score'] = (group_df[metric_column] - mean) / std

    group_df['z_flag'] = 'Normal'
    group_df.loc[group_df['z_score'] >= 3, 'z_flag'] = 'Hot_High'
    group_df.loc[(group_df['z_score'] >= 2) & (group_df['z_score'] < 3), 'z_flag'] = 'Hot_Medium'
    group_df.loc[group_df['z_score'] <= -3, 'z_flag'] = 'Dark_High'
    group_df.loc[(group_df['z_score'] <= -2) & (group_df['z_score'] > -3), 'z_flag'] = 'Dark_Medium'

    return group_df


# ============================================
# STEP 3: IQR DETECTION
# ============================================

def calculate_iqr_flags(group_df, metric_column):
    Q1 = group_df[metric_column].quantile(0.25)
    Q3 = group_df[metric_column].quantile(0.75)
    IQR = Q3 - Q1

    group_df = group_df.copy()

    lower_medium = Q1 - 1.5 * IQR
    lower_high = Q1 - 3.0 * IQR
    upper_medium = Q3 + 1.5 * IQR
    upper_high = Q3 + 3.0 * IQR

    group_df['iqr_flag'] = 'Normal'
    group_df.loc[group_df[metric_column] >= upper_high, 'iqr_flag'] = 'Hot_High'
    group_df.loc[(group_df[metric_column] >= upper_medium) & (group_df[metric_column] < upper_high), 'iqr_flag'] = 'Hot_Medium'
    group_df.loc[group_df[metric_column] <= lower_high, 'iqr_flag'] = 'Dark_High'
    group_df.loc[(group_df[metric_column] <= lower_medium) & (group_df[metric_column] > lower_high), 'iqr_flag'] = 'Dark_Medium'

    return group_df


# ============================================
# STEP 4: COMBINE FLAGS
# ============================================

def combine_flags(df):
    def decide(row):
        z = row['z_flag']
        i = row['iqr_flag']

        if 'Hot_High' in (z, i) and 'Hot' in z and 'Hot' in i:
            return 'HOT_SPOT_HIGH'
        if 'Dark_High' in (z, i) and 'Dark' in z and 'Dark' in i:
            return 'DARK_SPOT_HIGH'
        if 'Hot' in z or 'Hot' in i:
            return 'HOT_SPOT_MEDIUM'
        if 'Dark' in z or 'Dark' in i:
            return 'DARK_SPOT_MEDIUM'
        return 'NORMAL'

    df = df.copy()
    df['final_flag'] = df.apply(decide, axis=1)

    df['risk_severity'] = df['final_flag'].apply(
        lambda x: 'High' if 'HIGH' in x else ('Medium' if 'MEDIUM' in x else 'Normal')
    )

    df['anomaly_type'] = df['final_flag'].apply(
        lambda x: 'Hot Spot' if 'HOT' in x else ('Dark Spot' if 'DARK' in x else 'Normal')
    )

    return df


# ============================================
# STEP 5: MAIN PIPELINE
# ============================================

def detect_anomalies(df, groupby_column='state', metric='total_activity'):
    results = []

    for _, group in df.groupby(groupby_column):
        g = calculate_zscore_flags(group, metric)
        g = calculate_iqr_flags(g, metric)
        g = combine_flags(g)
        results.append(g)

    return pd.concat(results, ignore_index=True)


# ============================================
# STEP 6: FINAL REPORT
# ============================================

def generate_flagged_report(df):
    flagged = df[df['final_flag'] != 'NORMAL'].copy()

    def recommendation(row):
        if row['anomaly_type'] == 'Hot Spot':
            return f"Activity {abs(row['z_score']):.2f} SD above state average – Review for abnormal spikes"
        if row['anomaly_type'] == 'Dark Spot':
            return f"Activity {abs(row['z_score']):.2f} SD below state average – Check service availability"
        return ""

    flagged['recommendation'] = flagged.apply(recommendation, axis=1)

    return flagged[[
        'state',
        'district',
        'time_period',
        'enrolment_count',
        'update_count',
        'total_activity',
        'z_score',
        'anomaly_type',
        'risk_severity',
        'final_flag',
        'recommendation'
    ]].sort_values(['risk_severity', 'state', 'district'])


# ============================================
# STEP 7: EXECUTION ENTRY POINT
# ============================================

# ============================================
# STEP 7: EXECUTION ENTRY POINT (STATE-WISE)
# ============================================

if __name__ == "__main__":
    try:
        # Load ALL-STATE cleaned dataset
        data = pd.read_csv("/content/drive/MyDrive/Data_Hackathon/Cleaned_Dataset/assam_master_cleaned1.csv")

        # Rename columns to match engine expectations
        data = data.rename(columns={
            'date': 'time_period',
            'total_enrolments': 'enrolment_count',
            'total_updates': 'update_count'
        })

        # Prepare data
        data = prepare_data(data)

        # 🔥 STATE-WISE anomaly detection
        results = detect_anomalies(
            data,
            groupby_column='state',      # KEY CHANGE
            metric='total_activity'
        )

        # Generate final report
        report = generate_flagged_report(results)

        # Export results
        report.to_csv(
            "data/processed/statewise_anomaly_alerts.csv",
            index=False
        )

        print("State-wise anomaly detection completed successfully.")
        print(f"Total flagged records: {len(report)}")
        print(f"States covered: {report['state'].nunique()}")

    except FileNotFoundError:
        print("Input file not found. Please place india_master_cleaned.csv in data/processed/")
    except Exception as e:
        print(f"Error: {e}")

