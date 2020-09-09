import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

class weather_wrapper:
    
    def read(input_file):
        input_path = Path(input_file)
        df_weather = pd.read_csv(input_path)
    
    def describe(start_time, end_time, target_var):
        target_interval = (df_weather['local_eastern_time'] >= start_time) & (df_weather['local_eastern_time'] <= end_time)
        out_df = df_weather.loc[target_interval]
        out_df = out_df.groupby('day')[target_var].agg([pd.np.min, pd.np.max, pd.np.mean, pd.np.std])
        return out_df
    
    def plot(start_time, end_time, target_var):
        target_interval = (df_weather['local_eastern_time'] >= start_time) & (df_weather['local_eastern_time'] <= end_time)
        out_df = df_weather.loc[target_interval]
        plot_df = out_df[['day', target_var]]
        plot_df.plot(x="day")
        plt.xticks(rotation=45)