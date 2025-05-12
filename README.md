# Real-Time Sentiment Anomaly Detection from StoryBot Conversations

## Overview

This project implements a real-time machine learning pipeline that monitors one-on-one conversations between users and an AI agent (StoryBot) to detect anomalous sentiment shifts, such as sudden mood swings or behavioral changes.

Anomalies are detected using:

* Rule-based logic (sentiment delta threshold)
* Machine learning (Isolation Forest and One-Class SVM)
* Anomalies can be exported, visualized, and analyzed per user

## Dataset

**File:** `conversations.json`
Each record contains:

* `messages_list`: list of chat messages with timestamp and user ID
* `ref_user_id`: user ID
* `message`: text content
* `transaction_datetime_utc`: ISO or Unix-formatted timestamp

## Anomaly Detection Methods

### 1. Rule-Based

Flags sentiment jumps where the delta exceeds a threshold (default: 0.4)

### 2. Isolation Forest

Unsupervised model trained on sentiment deltas to detect outliers

### 3. One-Class SVM

Alternative anomaly detector for modeling abnormal shifts

### 4. Hybrid

An anomaly is logged if any method fires

## Key Components

* `streaming_pipeline.py`: Core engine that processes streamed messages
* `detector.py`: Implements all detection logic
* `sentiment_tracker.py`: Runs full pipeline and create anomalies JSON file
* `export_anomaly_table.py`: Exports and plots anomaly data per user from JSON

## How to Run

### 1. Stream and Detect

```bash
python sentiment_tracker.py
```

* Generates anomaly log: `sentiment_anomalies.json`
* Saves a sentiment plot: `sentiment_plot_user_<ID>.png`

### 2. Export Table and Plot for a User

```bash
python export_anomaly_table.py
```

* Customize `TARGET_USER_ID` inside the script
* Outputs:

  * `anomaly_table_user_<ID>.csv`
  * `anomaly_plot_user_<ID>.png`

## Evaluation Summary (Auto-Printed)

After detection:

* View counts per method
* View anomaly timestamps, previous and new sentiment

## Example Output (CSV)

| Timestamp           | Previous Sentiment | New Sentiment | Detection Method                |
| ------------------- | ------------------ | ------------- | ------------------------------- |
| 2023-10-01 10:50:00 | -0.75              | 0.85          | ml\_based\_svm                  |
| 2023-10-01 11:00:00 | 0.85               | 0.20          | rule\_based, ml\_based\_iforest |

## Use Cases

* Mood tracking in health/therapy bots
* Monitoring agent performance or conversation quality
* Early detection of disengaged or distressed users
