# -*- coding: utf-8 -*-
"""
Title: pandas_dates.py
Author: F. Maier
Date: Sept 1, 2020

Scrap code for creating timestamps in pandas or otherwise manipulating columns
"""

import pandas as pd
from pathlib import Path


def go_ver_1(infile):
    """Read in a csv file, extract data from date strings (but keep as 
    strings). This is simple and sufficient for grouping of data by days, 
    months, etc."""
    path_to_data = Path(infile)
    
    # read in data from csv file; create Pandas DataFrame object from it.
    data = pd.read_csv(path_to_data)
    
    data = data[data['StationID'] == 410]
    data['day'] = data['local_eastern_time'].apply(extract_day)
    data['hour'] = data['local_eastern_time'].apply(extract_hour)        
    data['minute'] = data['local_eastern_time'].apply(extract_minute)        
    data['month'] = data['local_eastern_time'].apply(extract_month)        
    data['year'] = data['local_eastern_time'].apply(extract_year)        

    grouped_temp = data.groupby('month')['temp_air_60cm_C']
    
    mean_temp_by_month = grouped_temp.mean()
    #grouped_temp.agg([...])
    #mean_temp_by_month.columns = [...]
    mean_temp_by_month.name = 'mean_' +  mean_temp_by_month.name
    mean_temp_by_month.plot.line(legend=True)
    print(mean_temp_by_month)
    print(type(mean_temp_by_month))
    return data, mean_temp_by_month



def go_ver_2(infile):
    """Read in a csv file, extract data from date strings (convert to 
    Timestamps). Conversion is very slow (it's done on all rows). """
    path_to_data = Path(infile)
    
    # read in data from csv file; use a converter for dates (takes a long time)
    data = pd.read_csv(path_to_data, converters={'local_eastern_time':crude_parse_datetime})
    
    data = data[data['StationID'] == 410]

    #data[data['Local_eastern_time'] < pd.to_datetime("2018-01-01 23:45:00-0500")]

    data['day'] = data['local_eastern_time'].apply(lambda x: x.day)
    data['hour'] = data['local_eastern_time'].apply(lambda x: x.hour)
    data['minute'] = data['local_eastern_time'].apply(lambda x: x.minute)
    data['month'] = data['local_eastern_time'].apply(lambda x: x.month)     
    data['year'] = data['local_eastern_time'].apply(lambda x: x.year)    

    grouped_temp = data.groupby('month')['temp_air_60cm_C']
    
    mean_temp_by_month = grouped_temp.mean()
    #grouped_temp.agg([...])
    #mean_temp_by_month.columns = [...]
    mean_temp_by_month.name = 'mean_' +  mean_temp_by_month.name
    mean_temp_by_month.plot.line(legend=True)
    print(mean_temp_by_month)
    print(type(mean_temp_by_month))
    return data, mean_temp_by_month



def go_ver_3(infile):
    """Read in a csv file, extract data from date strings (convert to 
    Timestamps). This will be faster than ver_2, since many rows are first eliminated."""
    path_to_data = Path(infile)
    
    # read in data from csv file; use a converter for dates (takes a long time)
    data = pd.read_csv(path_to_data)
    
    data = data[data['StationID'] == 410]
    data['local_eastern_time'] = data['local_eastern_time'].apply(crude_parse_datetime)
    data['day'] = data['local_eastern_time'].apply(lambda x: x.day)
    data['hour'] = data['local_eastern_time'].apply(lambda x: x.hour)
    data['minute'] = data['local_eastern_time'].apply(lambda x: x.minute)
    data['month'] = data['local_eastern_time'].apply(lambda x: x.month)     
    data['year'] = data['local_eastern_time'].apply(lambda x: x.year)    

    grouped_temp = data.groupby('month')['temp_air_60cm_C']
    
    mean_temp_by_month = grouped_temp.mean()
    #grouped_temp.agg([...])
    #mean_temp_by_month.columns = [...]
    mean_temp_by_month.name = 'mean_' +  mean_temp_by_month.name
    mean_temp_by_month.plot.line(legend=True)
    print(mean_temp_by_month)
    print(type(mean_temp_by_month))
    return data, mean_temp_by_month



def extract_day(timestamp):    
    """01-Jan-2018 23:45:00 --> 01"""    
    return  timestamp[:2]

def extract_month(timestamp):
    """01-Feb-2018 23:45:00 --> 02"""
    return  convert_month(timestamp[3:6])

def extract_year(timestamp):
    """01-Jan-2018 23:45:00 --> 2018"""
    return   timestamp[7:11]

def extract_hour(timestamp):
    """01-Jan-2018 23:45:00 --> 23"""
    return   timestamp[-8:-6]

def extract_minute(timestamp):
    """01-Jan-2018 23:45:00 --> 45"""
    return    timestamp[-5:-3]


def parse_datetime(t, date_format='%Y-%m-%d', tz='UTC'):
    # Can also use pd.Timestamp('2019-04-01T23:45:01-0500') to create a timestamp. 
    # Use tz_convert to convert between zone aware timestamps.
    # Use tz_localize to add timezone to a zone unaware timestamp.
    # Use pd.Timedelta(days=1, hours=4, minutes=10, seconds=55) etc to add/subtract time.
    #tz='US/Eastern'
    
    return pd.to_datetime(t, format=date_format).tz_localize(tz=tz)


def crude_parse_datetime(t,tz='US/Eastern'):
    """format of input datetime strings: 01-Jan-2018 00:00:00"""
    day = t[0:2]
    month = convert_month(t[3:6])
    year = t[7:11]
    time = t[12:]
    date_format='%Y-%m-%d %H:%M:%S'
    return pd.to_datetime(f"{year}-{month}-{day} {time}", format=date_format).tz_localize(tz=tz)

def convert_month(m):
    d = {
        "Jan" : "01",
        "Feb" : "02",
        "Mar" : "03",
        "Apr" : "04",
        "May" : "05",
        "Jun" : "06",
        "Jul" : "07",
        "Aug" : "08",
        "Sep" : "09",
        "Oct" : "10",
        "Nov" : "10",
        "Dec" : "12"}
    if m in d:
        return d[m]
    return m
