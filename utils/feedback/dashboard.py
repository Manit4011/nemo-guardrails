import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

# Load the chat log CSV
df = pd.read_excel("utils/feedback/Data/conversation_data.xlsx", parse_dates=["time"])
# df['time'] = pd.to_datetime(df['time'], errors='coerce', dayfirst=False)
# df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')

# --- Basic Metrics ---
total_messages = len(df)
unique_users = df['conv_id'].nunique()
total_conversations = df['conversation_id'].nunique() if 'conversation_id' in df.columns else None
start_date = df['time'].min()
end_date = df['time'].max()

print(f"Total messages: {total_messages}")
print(f"Unique users: {unique_users}")
if total_conversations:
    print(f"Total conversations: {total_conversations}")
print(f"Date range: {start_date} to {end_date}")

# --- Messages Per User ---
messages_per_user = df.groupby('conv_id').size()
print("\nAverage messages per user:", messages_per_user.mean())

def unique_user_day():
    # Make sure 'time' is in datetime format
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Filter only user messages
    user_msgs = df[df['role'] == 'user'].copy()

    # Extract just the date (not time)
    user_msgs['date'] = user_msgs['time'].dt.date

    # Group by date and count unique conv_ids
    daily_unique_users = user_msgs.groupby('date')['conv_id'].nunique()

    print(daily_unique_users.tail())  # Show recent days
    return daily_unique_users

def plot_daily_usage_stats(daily_unique_users):
    daily_unique_users.plot(kind='line', marker='o', figsize=(12, 6))
    plt.title("Daily Unique Users")
    plt.xlabel("Date")
    plt.ylabel("Unique Users")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("utils/feedback/Data/daily_usage.png")

def week_month_usage():
    # Assuming df['time'] is already datetime
    today = pd.Timestamp.today()
    one_week_ago = today - timedelta(days=7)
    one_month_ago = today - timedelta(days=30)

    # Only consider user messages
    user_msgs = df[df['role'] == 'user']

    # Unique users in the last 7 days
    users_last_week = user_msgs[user_msgs['time'] >= one_week_ago]['conv_id'].nunique()

    # Unique users in the last 30 days
    users_last_month = user_msgs[user_msgs['time'] >= one_month_ago]['conv_id'].nunique()

    print(f"Unique Users in Last 7 Days: {users_last_week}")
    print(f"Unique Users in Last 30 Days: {users_last_month}")

week_month_usage()
daily_unique_users = unique_user_day()
plot_daily_usage_stats(daily_unique_users)

# --- Daily Usage ---
df['date'] = df['time'].dt.date
daily_usage = df.groupby('date')['conv_id'].nunique()
daily_usage.plot(kind='line', title='Daily Active Users', xlabel='Date', ylabel='Unique Users')
plt.tight_layout()
plt.savefig("utils/feedback/Data/daily_users.png")

# --- Hourly Heatmap ---
df['hour'] = df['time'].dt.hour
df['weekday'] = df['time'].dt.day_name()
heatmap_data = df[df['role'] == 'user'].groupby(['weekday', 'hour']).size().unstack(fill_value=0)

# Sort weekdays
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(ordered_days)

plt.figure(figsize=(12,6))
heatmap_data = heatmap_data.fillna(0).astype(int)
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d")
plt.title("User Message Volume (Heatmap by Day and Hour)")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.tight_layout()
plt.savefig("utils/feedback/Data/usage_heatmap.png")

# --- Optional: Intent Analysis ---
if 'intent' in df.columns:
    top_intents = df['intent'].value_counts().head(10)
    top_intents.plot(kind='bar', title='Top Intents', xlabel='Intent', ylabel='Count')
    plt.tight_layout()
    plt.savefig("utils/feedback/Data/top_intents.png")

# --- Optional: Feedback Rating ---
if 'feedback_rating' in df.columns:
    print("\nAverage feedback rating:", df['feedback_rating'].mean())
    df['feedback_rating'].hist(bins=5)
    plt.title("User Feedback Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("utils/feedback/Data/feedback_rating.png")
