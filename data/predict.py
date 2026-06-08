from prophet import Prophet
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


class request_predictor:
    def __init__(self):
        self.prompt_model = Prophet(changepoint_prior_scale=0.001)
        self.response_model = Prophet(changepoint_prior_scale=0.001)
        self.time_window = 5 * 60
        train_data = self.train_model("./workloads/Azure/Azure_conv_train.csv")
        self.predict_and_evaluate("./workloads/Azure/Azure_conv_test.csv", train_data, True)
        
    def train_model(self, train_file):
        df = pd.read_csv(train_file)
        df['TimeWindow'] = (df['Timestamp'] // self.time_window) * self.time_window
        grouped_data = df.groupby('TimeWindow').agg({
            'Request tokens': 'sum',
            'Response tokens': 'sum'
        }).reset_index()
        
        prompt_data = grouped_data[['TimeWindow', 'Request tokens']]
        prompt_data['TimeWindow'] = prompt_data['TimeWindow'].apply(self.convert_to_ds)
        prompt_data.columns = ['ds', 'y']
        
        response_data = grouped_data[['TimeWindow', 'Response tokens']]
        response_data['TimeWindow'] = response_data['TimeWindow'].apply(self.convert_to_ds)
        response_data.columns = ['ds', 'y']
        
        self.prompt_model.fit(prompt_data)
        self.response_model.fit(response_data)
        return {
            "prompt_train_data": prompt_data,
            "response_train_data": response_data
        }
        
    
    def predict_and_evaluate(self, test_file, train_data=None, plot=False):
        df = pd.read_csv(test_file)
        df['TimeWindow'] = (df['Timestamp'] // self.time_window) * self.time_window
        grouped_data = df.groupby('TimeWindow').agg({
            'Request tokens': 'sum',
            'Response tokens': 'sum'
        }).reset_index()

        test_prompt_data = grouped_data[['TimeWindow', 'Request tokens']]
        test_prompt_data['TimeWindow'] = test_prompt_data['TimeWindow'].apply(self.convert_to_ds)
        test_prompt_data.columns = ['ds', 'y']
        print(f"Mean Request tokens of test_data: {test_prompt_data['y'].mean()}")
        
        test_response_data = grouped_data[['TimeWindow', 'Response tokens']]
        test_response_data['TimeWindow'] = test_response_data['TimeWindow'].apply(self.convert_to_ds)
        test_response_data.columns = ['ds', 'y']
        print(f"Mean Response tokens of test_data: {test_response_data['y'].mean()}")
        
        future = self.prompt_model.make_future_dataframe(periods=len(test_prompt_data), freq='5T')
        forecast = self.prompt_model.predict(future)
        prompt_forecast = forecast.iloc[-len(test_prompt_data):]
        prompt_forecast = prompt_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        if plot and train_data is not None:
            prompt_mae = self.evaluate(test_prompt_data, prompt_forecast, train_data['prompt_train_data'], True, "prompt")
        else:
            prompt_mae = self.evaluate(test_prompt_data, prompt_forecast)
        print(f"Fraction of MAE of request tokens: {prompt_mae / test_prompt_data['y'].mean()}")
        
        future = self.response_model.make_future_dataframe(periods=len(test_response_data), freq='5T')
        forecast = self.response_model.predict(future)
        response_forecast = forecast.iloc[-len(test_response_data):]
        response_forecast = response_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        if plot and train_data is not None:
            response_mae = self.evaluate(test_response_data, response_forecast, train_data['response_train_data'], True, "response")
        else:
            response_mae = self.evaluate(test_response_data, response_forecast)
        print(f"Fraction of MAE of response tokens: {response_mae / test_response_data['y'].mean()}")
    
    
    def evaluate(self, test_data, forecast, train_data=None, plot=False, data_type=None):
        comparison = test_data.copy()
        comparison['ds'] = pd.to_datetime(comparison['ds'])
        train_data['ds'] = pd.to_datetime(train_data['ds'])
        comparison = comparison.merge(forecast, on='ds', how='left')
        mae = mean_absolute_error(comparison['y'], comparison['yhat'])
        print(f"Mean Absolute Error (MAE): {mae}")
        
        if plot and train_data is not None:
            plt.figure(figsize=(14, 8))
            plt.plot(train_data['ds'], train_data['y'], label='Train Data (Actual)', marker='o', linestyle='-', color='blue')
            plt.plot(comparison['ds'], comparison['y'], label='Test Data (Actual)', marker='o', linestyle='-', color='green')
            plt.plot(comparison['ds'], comparison['yhat'], label='Test Data (Forecast)', marker='x', linestyle='--', color='red')
            plt.fill_between(comparison['ds'], comparison['yhat_lower'], comparison['yhat_upper'], color='black', alpha=1, label='Forecast Uncertainty')
            plt.xlabel('Timestamp')
            plt.ylabel(f'{data_type} Tokens')
            plt.savefig(f"./{data_type}_prediction.png")
        return mae
        
        
    
    def convert_to_ds(self, timestamp):
        dt = datetime.fromtimestamp(int(timestamp))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    

if __name__ == "__main__":
    predictor = request_predictor()
    