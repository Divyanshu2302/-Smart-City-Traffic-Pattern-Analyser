# -Smart-City-Traffic-Pattern-Analyser
- Downloads the Chicago traffic dataset using `wget` and saves it as `chicago_traffic.csv`.
- Imports essential Python libraries:
  - `pandas` for data manipulation.
  - `numpy` for numerical operations.
  - `matplotlib.pyplot` and `seaborn` for data visualization.
- Loads the CSV file into a DataFrame using `pd.read_csv()`, with the `TIME` column parsed as datetime format.
- Prints the shape of the dataset to show the number of rows and columns.
- Displays the first 5 rows of the dataset using `df.head()` to preview the structure and content.

# Direct download link (alternative: download manually)
!wget https://data.cityofchicago.org/api/views/77hq-huss/rows.csv?accessType=DOWNLOAD -O chicago_traffic.csv
#  Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('chicago_traffic.csv', parse_dates=['TIME'])
print(f"Dataset shape: {df.shape}")
df.head()
- Checks for missing values in each column using `df.isnull().sum()` and prints the result.
- Handles missing data:
  - Fills missing values in the `SPEED` column with the median value (suitable for numerical data).
  - Replaces missing values in the `SEGMENT_ID` column with the placeholder `'UNKNOWN'` (suitable for categorical data).
- Removes duplicate rows from the dataset using `df.drop_duplicates(inplace=True)`.
- Validates that no duplicate rows remain using an `assert` statement.

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
- Extracts time-related features from the `TIME` column:
  - `HOUR`: Extracts the hour of the day (0–23).
  - `DAY_OF_WEEK`: Extracts the day name (e.g., Monday, Tuesday).
  - `IS_WEEKEND`: Flags whether the day is a weekend (Saturday or Sunday).

- Categorizes traffic conditions based on vehicle speed:
  - Uses `pd.cut()` to group `SPEED` into bins:
    - `0–20` mph as **Heavy**
    - `20–40` mph as **Medium**
    - `40+` mph as **Light**
  - Stores the result in a new column `TRAFFIC_LEVEL`.

- Flags peak traffic hours:
  - Adds a `PEAK_HOUR` column where the value is `1` if the hour is between **7–9 AM** or **4–6 PM**, otherwise `0`.

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
- Performs data validation checks:
  - `speed_check`: Ensures all speed values fall within the range of 0 to 100 mph.
  - `time_check`: Verifies that all timestamps are on or after January 1, 2023 (can be adjusted as needed).

- Prints the results of both validation checks to confirm data integrity.

- Saves the cleaned and processed DataFrame to a new CSV file named `cleaned_traffic.csv` without the index column.

# Validation checks
speed_check = df['SPEED'].between(0, 100).all()
time_check = (df['TIME'] >= pd.Timestamp('2023-01-01')).all()  # Adjust based on your data

print(f"Speed values valid: {speed_check}")
print(f"Timestamps valid: {time_check}")

# Save cleaned data
df.to_csv('cleaned_traffic.csv', index=False)
- Generates a comprehensive statistical summary of the dataset using `describe(include='all')`, which includes both numerical and categorical columns, and rounds the results to two decimal places.
- Saves the summary statistics in Markdown format to a file named `summary_stats.md` for easy reporting or presentation.
- Calculates the average vehicle speed during peak (`PEAK_HOUR = 1`) and non-peak (`PEAK_HOUR = 0`) hours by grouping the data and computing the mean speed.
- Prints a comparison of the average speeds to gain insight into traffic flow patterns during different times of the day.

# Generate comprehensive statistics
stats = df.describe(include='all').round(2)
stats.to_markdown("summary_stats.md")  # For presentation

# Key insights
peak_hour_stats = df.groupby('PEAK_HOUR')['SPEED'].mean()
print(f"Average speed during peak hours: {peak_hour_stats[1]:.1f} mph vs non-peak: {peak_hour_stats[0]:.1f} mph")
- Creates a histogram of vehicle speeds using Seaborn’s `histplot()`:
  - Divides the speed data into 30 bins for detailed distribution.
  - Includes a KDE (Kernel Density Estimate) curve to show the probability density.

- Sets the figure size to 10×6 inches for better visibility.

- Adds informative chart elements:
  - Title: "Distribution of Vehicle Speeds"
  - X-axis label: "Speed (mph)"
  - Y-axis label: "Frequency"

- Saves the resulting plot as a high-resolution PNG file named `speed_distribution.png` with tight bounding box.

plt.figure(figsize=(10,6))
sns.histplot(df['SPEED'], bins=30, kde=True)
plt.title("Distribution of Vehicle Speeds")
plt.xlabel("Speed (mph)")
plt.ylabel("Frequency")
plt.savefig('speed_distribution.png', dpi=300, bbox_inches='tight')
- Groups the dataset by hour of the day and calculates the average vehicle speed for each hour.
- Resets the index to prepare the grouped data (`hourly`) for plotting.

- Creates a line plot using Seaborn:
  - X-axis represents each hour of the day (0–23).
  - Y-axis shows the corresponding average speed.
  - Markers are added to each data point for clarity.

- Adds visual highlights for peak traffic hours:
  - Red shaded region for morning peak hours (7–9 AM).
  - Blue shaded region for evening peak hours (4–6 PM).

- Adds chart title and axis labels for clarity.
- Displays a legend to label the peak hour zones.
- Saves the plot as a high-resolution PNG file named `hourly_traffic.png`.

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
- Calculates the interquartile range (IQR) of vehicle speeds to identify statistical outliers:
  - Q1: 25th percentile of speed values.
  - Q3: 75th percentile of speed values.
  - IQR: Difference between Q3 and Q1.
  - `slow_threshold`: Any speed below `Q1 - 1.5 * IQR` is considered an unusually slow traffic event.

- Filters and stores all such outlier records (anomalies) where the speed is significantly lower than normal.

- Prints the total count of anomalous slow periods found.

- Selects the first anomaly’s timestamp and creates a time window of 2 hours around it (1 hour before and after).

- Plots a line chart of speed within this window to visually inspect the anomaly:
  - X-axis shows time.
  - Y-axis shows speed.
  - Title includes the timestamp of the anomaly for reference.

- Saves the visualization as a PNG file named `anomaly.png`.

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
