# streaming_pipeline.py
import time
from typing import Dict
from features import extract_features
from detector import SentimentWindow
import json
from dateutil.parser import parse as parse_time

class StreamingSentimentMonitor:
    def __init__(self):
        self.user_sentiments = {}

    def extract_features(self, message: str, metadata: Dict) -> Dict:
        return extract_features(message)

    def process(self, message: str, metadata: Dict):
        user_id = metadata["user_id"]
        timestamp = metadata["timestamp"]
        features = self.extract_features(message, metadata)

        if user_id not in self.user_sentiments:
            self.user_sentiments[user_id] = SentimentWindow()

        window = self.user_sentiments[user_id]
        window.update(features["sentiment"], features["combined_sentiment"])

        is_anomaly, details = window.is_anomalous()
        if is_anomaly:
            self.alert(user_id, timestamp, message, features, details)

    def alert(self, user_id, timestamp, message, features, details):
        print(f"\n Anomaly Detected for User {user_id} at {timestamp}")
        print(f"Message: {message}")
        print(f"Sentiment Score: {features['sentiment']:.2f}, "
              f"Lexicon Score: {features['lexicon_sentiment']}, "
              f"Combined: {features['combined_sentiment']:.2f}")
        print(f"Detected by: {', '.join([k for k, v in details.items() if v])}\n")


if __name__ == "__main__":
    monitor = StreamingSentimentMonitor()

    # Load and stream conversations.json
    with open("data/data/conversations.json", "r") as f:
        conversations = json.load(f)

    print("\n Streaming messages from conversations.json")
    for conv in conversations:
        for msg in conv["messages_list"]:
            metadata = {
                "timestamp": time.mktime(parse_time(msg["transaction_datetime_utc"]).timetuple()),
                "user_id": msg["ref_user_id"],
                "screen_name": msg.get("screen_name", "unknown"),
                "conversation_id": msg.get("ref_conversation_id"),
                "source": "conversation"
            }
            monitor.process(msg["message"], metadata)
            time.sleep(0.2)
