import json
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
from dateutil.parser import parse as parse_time

# === CONFIG ===
TARGET_USER_ID = 1
START_DATE = "2023-10-01"
END_DATE = "2023-10-03"

# Convert date strings to timestamps
start_ts = time.mktime(datetime.strptime(START_DATE, "%Y-%m-%d").timetuple())
end_ts = time.mktime(datetime.strptime(END_DATE + " 23:59:59", "%Y-%m-%d %H:%M:%S").timetuple())

# Load anomalies
with open("sentiment_anomalies.json", "r") as f:
    all_anomalies = json.load(f)

# Filter anomalies by user and date
filtered = []
for anomaly in all_anomalies:
    if anomaly["user_id"] != TARGET_USER_ID:
        continue
    ts = parse_time(anomaly["timestamp"]).timestamp() if isinstance(anomaly["timestamp"], str) else float(anomaly["timestamp"])
    if start_ts <= ts <= end_ts:
        anomaly["timestamp_parsed"] = ts
        filtered.append(anomaly)

# Export to CSV
csv_filename = f"anomaly_table_user_{TARGET_USER_ID}_{START_DATE}_to_{END_DATE}.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Timestamp", "New Sentiment", "Detection Method", "current_message", "previous_message"])
    for a in filtered:
        ts_readable = datetime.fromtimestamp(a["timestamp_parsed"]).strftime('%Y-%m-%d %H:%M:%S')
        details = a["detected_by"]
        if details.get("rule_based") and not (details.get("ml_based_svm") or details.get("ml_based_iforest")):
            method = "Rule"
        elif details.get("ml_based_svm") and details.get("ml_based_iforest"):
            method = "SVM+IForest"
        elif details.get("ml_based_svm"):
            method = "SVM"
        elif details.get("ml_based_iforest"):
            method = "IForest"
        elif details.get("rule_based"):
            method = "Rule+ML"
        else:
            method = "Unknown"
        writer.writerow([ts_readable, round(float(a["new_sentiment"]), 2), method, a["current_message"], a["previous_message"]])

print(f"CSV export completed: {csv_filename}")

# Bar Plot
if filtered:
    timestamps = [datetime.fromtimestamp(a["timestamp_parsed"]).strftime('%m-%d %H:%M') for a in filtered]
    sentiments = [float(a["new_sentiment"]) for a in filtered]

    # Color map by detection method
    color_map = {
        "Rule": "red",
        "IForest": "blue",
        "SVM": "green",
        "SVM+IForest": "purple",
        "Rule+ML": "orange",
        "Unknown": "gray"
    }

    # Determine label and assign colors
    colors = []
    for a in filtered:
        details = a["detected_by"]
        if details.get("rule_based") and not (details.get("ml_based_svm") or details.get("ml_based_iforest")):
            label = "Rule"
        elif details.get("ml_based_svm") and details.get("ml_based_iforest"):
            label = "SVM+IForest"
        elif details.get("ml_based_svm"):
            label = "SVM"
        elif details.get("ml_based_iforest"):
            label = "IForest"
        elif details.get("rule_based"):
            label = "Rule+ML"
        else:
            label = "Unknown"
        colors.append(color_map.get(label, "gray"))

    # Plot bars
    plt.figure(figsize=(12, 6))
    bars = plt.bar(timestamps, sentiments, color=colors)

    # Axis formatting
    plt.xlabel("Timestamp")
    plt.ylabel("Sentiment Polarity")
    plt.title(f"Anomaly Sentiment Trend (Bar)\nUser {TARGET_USER_ID} ({START_DATE} to {END_DATE})")
    plt.xticks(rotation=45, ha='right')

    # Add legend
    legend_elements = [
        Patch(facecolor="red", label="Rule"),
        Patch(facecolor="blue", label="IForest"),
        Patch(facecolor="green", label="SVM"),
        Patch(facecolor="purple", label="SVM+IForest"),
        Patch(facecolor="orange", label="Rule+ML"),
        Patch(facecolor="gray", label="Unknown")
    ]
    plt.legend(handles=legend_elements, title="Detection Method")

    plt.tight_layout()
    plot_filename = f"anomaly_bar_plot_user_{TARGET_USER_ID}_{START_DATE}_to_{END_DATE}.png"
    plt.savefig(plot_filename)
    print(f"Bar plot saved to {plot_filename}")
else:
    print(f"No anomalies to plot for User {TARGET_USER_ID} between {START_DATE} and {END_DATE}.")
