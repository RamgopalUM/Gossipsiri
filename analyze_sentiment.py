import json
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
RANDOM_SEED = 46


def main():
    df = pd.read_csv("twitter_api/Tesla_pred.csv")
    df = df.sort_values('created_at')
    print(df.tail())
    current_date = datetime(2020, 11, 23)
    bearish_counts = {}
    neutral_counts = {}
    bullish_counts = {}
    total_counts = {}
    bull_divide_bear = {}
    bear_ratio = {}
    bull_ratio = {}
    curr_bearish = 0
    curr_neutral = 0
    curr_bullish = 0
    curr_total = 0
    for i in df.index:
        date_time_obj = datetime.strptime(str(df['created_at'][i]), '%Y-%m-%d %H:%M:%S')
        if date_time_obj.day != current_date.day:
            print(date_time_obj)
            bearish_counts[str(current_date.date())] = curr_bearish
            neutral_counts[str(current_date.date())] = curr_neutral
            bullish_counts[str(current_date.date())] = curr_bullish
            total_counts[str(current_date.date())] = curr_total
            bull_divide_bear[str(current_date.date())] = curr_bullish / curr_bearish
            bear_ratio[str(current_date.date())] = curr_bearish / curr_total
            bull_ratio[str(current_date.date())] = curr_bullish / curr_total
            current_date = date_time_obj
            curr_bearish = 0
            curr_neutral = 0
            curr_bullish = 0
            curr_total = 0
        if df['pred_sentiment'][i] == 0:
            curr_bearish += 1
        elif df['pred_sentiment'][i] == 1:
            curr_neutral += 1
        elif df['pred_sentiment'][i] == 2:
            curr_bullish += 1
        curr_total += 1
    #bearish_counts[str(current_date.date())] = curr_bearish
    #neutral_counts[str(current_date.date())] = curr_neutral
    #bullish_counts[str(current_date.date())] = curr_bullish
    #total_counts[str(current_date.date())] = curr_total

    print(bearish_counts)
    print(neutral_counts)
    print(bullish_counts)
    print(total_counts)

    plt.xlabel('Date')
    plt.ylabel('Num_tweets')
    total_plt, = plt.plot(*zip(*sorted(total_counts.items())), label=['Total'])
    bear_plt, = plt.plot(*zip(*sorted(bearish_counts.items())), label=['Bearish'])
    neutral_plt, = plt.plot(*zip(*sorted(neutral_counts.items())), label=['Neutral'])
    bull_plt, = plt.plot(*zip(*sorted(bullish_counts.items())), label=['Bullish'])
    plt.legend(handles=[total_plt, bear_plt, neutral_plt, bull_plt])
    plt.show()

    bearish_ratio_plt, = plt.plot(*zip(*sorted(bear_ratio.items())), label=['Bearish_ratio'])
    bullish_ratio_plt, = plt.plot(*zip(*sorted(bull_ratio.items())), label=['Bullish_ratio'])
    plt.legend(handles=[bearish_ratio_plt, bullish_ratio_plt])
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.show()


if __name__ == "__main__":
    main()
