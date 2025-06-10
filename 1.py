# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('test_BdBKkAj.csv.xls', parse_dates=['TIME'])
print(f"Dataset shape: {df.shape}")
df.head()

# Check for missing values
missing = df.isnull().sum()
print("Missing values:\n", missing)

# Handle missing values
df['SPEED'].fillna(df['SPEED'].median(), inplace=True)  # Numerical
df['SEGMENT_ID'].fillna('UNKNOWN', inplace=True)        # Categorical

# Remove duplicates
df.drop_duplicates(inplace=True)

# Validate
assert df.duplicated().sum() == 0, "Duplicates still exist!"

# Extract time features
df['HOUR'] = df['TIME'].dt.hour
df['DAY_OF_WEEK'] = df['TIME'].dt.day_name()
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin(['Saturday', 'Sunday'])

# Create traffic categories
df['TRAFFIC_LEVEL'] = pd.cut(df['SPEED'],
                            bins=[0, 20, 40, np.inf],
                            labels=['Heavy', 'Medium', 'Light'])

# Add peak hour flag
df['PEAK_HOUR'] = df['HOUR'].apply(lambda x: 1 if (7 <= x <= 9) | (16 <= x <= 18) else 0)

# Validation checks
speed_check = df['SPEED'].between(0, 100).all()
time_check = (df['TIME'] >= pd.Timestamp('2023-01-01')).all()  # Adjust based on your data

print(f"Speed values valid: {speed_check}")
print(f"Timestamps valid: {time_check}")

# Save cleaned data
df.to_csv('cleaned_traffic.csv', index=False)

# Generate comprehensive statistics
stats = df.describe(include='all').round(2)
stats.to_markdown("summary_stats.md")  # For presentation

# Key insights
peak_hour_stats = df.groupby('PEAK_HOUR')['SPEED'].mean()
print(f"Average speed during peak hours: {peak_hour_stats[1]:.1f} mph vs non-peak: {peak_hour_stats[0]:.1f} mph")

plt.figure(figsize=(10,6))
sns.histplot(df['SPEED'], bins=30, kde=True)
plt.title("Distribution of Vehicle Speeds")
plt.xlabel("Speed (mph)")
plt.ylabel("Frequency")
plt.savefig('speed_distribution.png', dpi=300, bbox_inches='tight')

hourly = df.groupby('HOUR')['SPEED'].mean().reset_index()

plt.figure(figsize=(12,6))
sns.lineplot(data=hourly, x='HOUR', y='SPEED', marker='o')
plt.title("Average Speed by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Speed (mph)")
plt.axvspan(7, 9, color='red', alpha=0.1, label='Morning Peak')
plt.axvspan(16, 18, color='blue', alpha=0.1, label='Evening Peak')
plt.legend()
plt.savefig('hourly_traffic.png', dpi=300)

# Identify unusual slow periods
Q1 = df['SPEED'].quantile(0.25)
Q3 = df['SPEED'].quantile(0.75)
IQR = Q3 - Q1
slow_threshold = Q1 - 1.5*IQR

anomalies = df[df['SPEED'] < slow_threshold]
print(f"Found {len(anomalies)} anomalous slow periods")

# Visualize one anomaly
sample_anomaly = anomalies.iloc[0]['TIME']
anomaly_window = df[(df['TIME'] >= sample_anomaly - pd.Timedelta(hours=1)) &
                    (df['TIME'] <= sample_anomaly + pd.Timedelta(hours=1))]

plt.figure(figsize=(12,6))
sns.lineplot(data=anomaly_window, x='TIME', y='SPEED')
plt.title(f"Traffic Anomaly Detected at {sample_anomaly}")
plt.savefig('anomaly.png', dpi=300)