import numpy as np
import pandas
from ggplot import *
import sys

# Ridership by time of day or day of week
def ridership_time(rider_df):
	#use ENTRIESn_hourly,hour,day_week
	plot = ggplot(rider_df, aes(x=reorder('factor(hour)'),y='ENTRIESn_hourly')) + geom_point()+geom_line()
	print plot
	
	
if __name__ == '__main__':
    
    input_file = "data/turnstile_weather_v2.csv"
    weather_turnstile = pandas.read_csv(input_file)

    # list of features
    print list(weather_turnstile)
    # print weather_turnstile.head(2)
    rider_time = weather_turnstile.groupby(by=['hour'])['ENTRIESn_hourly'].sum()
    rider_df = rider_time.to_frame().reset_index()
    print rider_df
    print list(rider_df)
    ridership_time(rider_df)