import numpy as np
import pandas
from ggplot import *
import sys

# Ridership by time of day or day of week
def ridership_time(weather_turnstile):
	#use ENTRIESn_hourly,hour,day_week
	pass

if __name__ == '__main__':
    
    input_file = "data/turnstile_weather_v2.csv"
    weather_turnstile = pandas.read_csv(input_file)

    # list of features
    print list(weather_turnstile)
    print weather_turnstile.head(2)