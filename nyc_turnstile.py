import numpy as np
import pandas
from ggplot import *
import sys
import statsmodels.api as sm


def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()

    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, 
    and the values for our thetas.
    """
    
    predicted = np.dot(features,theta)
    sum_of_sq = np.square(predicted - values).sum()
    cost =  1.0/(2.0*len(values)) * (sum_of_sq)
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        predicted = np.dot(features,theta)
        pred = np.dot((values-predicted),features)
        theta = theta + (alpha/m)*pred
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
    return theta, pandas.Series(cost_history)

def predictions(dataframe):
    '''
    let's predict the ridership of the NYC subway using linear regression with gradient descent.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv     
      
   '''
    # Select Features (try different features!)
    features = dataframe[['rain', 'precipi', 'hour', 'meantempi']]
    
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    

    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)

    # Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to change this value
    num_iterations = 75 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)

    # print len(theta_gradient_descent)
    
    # linear regression using OLS model
    model = sm.OLS(values,features)
    result = model.fit()
    theta_ols = result.params
    # sys.exit()    

    plot = None
    # -------------------------------------------------
    # Uncomment the next line to see your cost history
    # -------------------------------------------------
    # plot = plot_cost_history(alpha, cost_history)
    
    predictions = np.dot(features_array, theta_gradient_descent)
    r_sq = r_squared(values_array,predictions)
    ols_predictions = np.dot(features_array, theta_ols)
    ols_r_sq = r_squared(values, ols_predictions)
    print ols_r_sq
    return predictions, plot, r_sq


def plot_cost_history(alpha, cost_history):
   """This function is for viewing the plot of your cost history.
   """
   cost_df = pandas.DataFrame({
      'Cost_History': cost_history,
      'Iteration': range(len(cost_history))
   })
   return ggplot(cost_df, aes('Iteration', 'Cost_History')) + \
      geom_point() + ggtitle('Cost History for alpha = %.3f' % alpha )


def r_squared(values, predicted):
    #compute r_squared value

    avg_val = np.mean(values)
    num = np.square(predicted-avg_val).sum()
    den = np.square(values-avg_val).sum()
    r_sq = num/den
    return r_sq



if __name__ == '__main__':
    
    input_file = "data/turnstile_weather_v2.csv"
    weather_turnstile = pandas.read_csv(input_file)

    # list of features
    print list(weather_turnstile)
    predictions, plot, r_sq = predictions(weather_turnstile)
    print r_sq
    print predictions
    print plot
