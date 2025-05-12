import json
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from dateutil.parser import parse as parse_time
from streaming_pipeline import StreamingSentimentMonitor

# Data structure to collect sentiment per user
time_series = defaultdict(list)
sentiment_values = defaultdict(list)
last_message = dict()  # Only keep the latest message per user
anomaly_points = defaultdict(list)
all_anomalies = []
sentiment_key = 'combined_sentiment'

# Monitor instance
monitor = StreamingSentimentMonitor()

with open("data/data/conversations.json", "r") as f:
    conversations = json.load(f)

print("Tracking sentiment shifts over time")
for conv in conversations:
    for msg in conv.get("messages_list", []):
        user_id = msg.get("ref_user_id")
        if user_id is None:
            continue

        timestamp = msg["transaction_datetime_utc"]
        metadata = {"timestamp": timestamp, "user_id": user_id}

        monitor.process(msg["message"], metadata)
        features = monitor.extract_features(msg["message"], metadata)

        if sentiment_key not in features:
            print(f"Missing {sentiment_key} for user {user_id}, message: {msg['message']}")
            continue

        time_series[user_id].append(timestamp)
        sentiment_values[user_id].append(features[sentiment_key])

        previous = sentiment_values[user_id][-2] if len(sentiment_values[user_id]) > 1 else 0
        current = sentiment_values[user_id][-1]
        delta = abs(current - previous)
        previous_message = last_message.get(user_id)

        is_anomaly, details = monitor.user_sentiments[user_id].is_anomalous()
        if is_anomaly:
            anomaly = {
                "user_id": user_id,
                "timestamp": timestamp,
                "delta": delta,
                "new_sentiment": current,
                "current_message": msg["message"],
                "previous_message": previous_message,
                "detected_by": details
            }
            all_anomalies.append(anomaly)

            if details["rule_based"] and not (details["ml_based_svm"] or details["ml_based_iforest"]):
                label = "Rule"
            elif details["ml_based_svm"] and details["ml_based_iforest"]:
                label = "SVM+IForest"
            elif details["ml_based_svm"]:
                label = "SVM"
            elif details["ml_based_iforest"]:
                label = "IForest"
            elif details["rule_based"]:
                label = "Rule+ML"
            else:
                label = "Unknown"

            anomaly_points[user_id].append((timestamp, current, label))

        # Update most recent message for the user
        last_message[user_id] = msg["message"]

        time.sleep(0.1)

# Save anomalies to file
with open("sentiment_anomalies.json", "w") as out_file:
    json.dump(all_anomalies, out_file, indent=2, default=str)
