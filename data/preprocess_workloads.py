import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Only Azure



def convert_to_timestamp(date_str):
    # Parse the string into a datetime object
    dt = datetime.fromisoformat(date_str)
    # Convert the datetime object to a timestamp
    timestamp = dt.timestamp()
    return timestamp


def filter_data(df, start_fraction, end_fraction, time_window = 5 * 60):
    df.columns = ['Timestamp', 'Request tokens', 'Response tokens']
    df['Timestamp'] = df['Timestamp'].apply(convert_to_timestamp)
    begin_timestamp = df.iloc[0]['Timestamp']
    end_timestamp = df.iloc[-1]['Timestamp']
    start_timestamp = begin_timestamp + start_fraction * (end_timestamp - begin_timestamp)
    start_timestamp = (start_timestamp // time_window) * time_window
    end_timestamp = begin_timestamp + end_fraction * (end_timestamp - begin_timestamp)
    end_timestamp = (end_timestamp // time_window) * time_window
    df = df[(df['Timestamp'] >= start_timestamp) & (df['Timestamp'] <= end_timestamp)]
    return df


def filter_token_len(df, total_min_length, total_max_length, 
                     input_min_length, input_max_length, 
                     output_min_length, output_max_length):
    df = df[df['Request tokens'] + df['Response tokens'] <= total_max_length]
    df = df[df['Request tokens'] + df['Response tokens'] >= total_min_length]
    df = df[df['Request tokens'] >= input_min_length]
    df = df[df['Request tokens'] <= input_max_length]
    df = df[df['Response tokens'] >= output_min_length]
    df = df[df['Response tokens'] <= output_max_length]
    return df

# def split_data(df, time_window=5*60, train_fraction=0.6):
#     df['TimeWindow'] = df['Timestamp'] // time_window
#     unique_windows = sorted(df['TimeWindow'].unique())
#     split_index = int(len(unique_windows) * 0.6)
#     train_windows = unique_windows[:split_index]
#     test_windows = unique_windows[split_index:]
    
#     train_df = df[df['TimeWindow'].isin(train_windows)].drop(columns=['TimeWindow'])
#     test_df = df[df['TimeWindow'].isin(test_windows)].drop(columns=['TimeWindow'])
#     return train_df, test_df


def process_azure(azure_type, start_fraction, end_fraction, time_window=5*60, train_fraction=0.6):
    df = pd.read_csv(f'./workloads/Azure/Azure_{azure_type}.csv').dropna()
    print(f"{azure_type} #req = {len(df)}")
    
    df.columns = ['Timestamp', 'Request tokens', 'Response tokens']
    df['Timestamp'] = df['Timestamp'].apply(convert_to_timestamp)

    df = filter_time_range(df, start_fraction, end_fraction, time_window)
    df = filter_token_len(df, 32, 4096, 16, 4096, 16, 4096)
    print(f"{azure_type} #filtered_req = {len(df)}")

    print(f"Time range = [{df['Timestamp'].min()}, {df['Timestamp'].max()}]")
    print(f'Request tokens range = [{df["Request tokens"].min()}, {df["Request tokens"].max()}]')
    print(f'Request tokens mean  = {df["Request tokens"].mean()}')
    print(f'Response tokens range = [{df["Response tokens"].min()}, {df["Response tokens"].max()}]')
    print(f'Response tokens mean  = {df["Response tokens"].mean()}')
    
    df.to_csv(f'./workloads/Azure_{azure_type}/cleaned.csv', index=False)
    # train_df, test_df = split_data(df, time_window, train_fraction)
    # train_df.to_csv(f'./workloads/Azure/Azure_{azure_type}_train.csv', index=False)
    # test_df.to_csv(f'./workloads/Azure/Azure_{azure_type}_test.csv', index=False)

    plot_requests_rate(df, f"Azure_{azure_type}", time_window=180)
    plot_avg_token_len_by_time_window(df, f"Azure_{azure_type}")
    print()



if __name__ == "__main__":
    process_azure("code", 0.5, 1)  # 3.5 days
    process_azure("conv", 0.2, 0.8)  # 4.2 days
    # process_azure("code", 0, 1)
    # process_azure("conv", 0, 1)
    # process_azure("code", start_fraction=0.75, end_fraction=1)
    # process_azure("conv", start_fraction=0.5, end_fraction=0.8)