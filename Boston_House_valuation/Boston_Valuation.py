import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error

#Gather data
load_boston = fetch_openml(data_id=531)
boston_dataset = load_boston
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
data['CHAS'] = data['CHAS'].astype(float)
data['RAD'] = data['RAD'].astype(float)
features = data.drop(['INDUS','AGE'], axis=1)
log_prices = np.log(boston_dataset.target)
target = log_prices.to_frame('PRICE')
features.head()

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8
Median_Price = np.median(boston_dataset.target)
ZILLOW_PRICE_MEDIAN = 583.3
SCALE_FACTOR = ZILLOW_PRICE_MEDIAN / Median_Price

property_stats = features.mean().values.reshape(1,11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)
MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)
# print(f'MSE: {round(MSE,3)} & RMSE: {round(RMSE,3)}')


def get_current_price_estimate(nr_rooms, students_per_classroom, next_to_river = 0.0, high_confidence = True):
    #Configure Property:
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][CHAS_IDX] = next_to_river
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    #Make Prdiction:
    log_estimate = regr.predict(property_stats)[0][0]
    
    #Calculate Range:
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    return np.e**log_estimate*SCALE_FACTOR*1000, np.e**upper_bound*SCALE_FACTOR*1000, np.e**lower_bound*SCALE_FACTOR*1000, interval

def get_dollar_estimate(rm, ptratio,chas=0.0,large_range=True):
    """Estimate the price of a property in Boston
    Keyword arguments:
    rm - number of rooms in the property
    ptratio - Student to teacher ratio
    chas - 1.0 if property next to Chas river
    large_range - True by default, giving 95% confidence interval, else gives 68% interval
    
    """
    if rm < 1 or ptratio < 1:
        print('That is unrealistic. Try again!')
        return
    current_estimate, upper, lower, conf = get_current_price_estimate(rm,ptratio,chas, large_range)
    rounded_est = np.around(current_estimate,-3)
    rounded_hi = np.around(upper, -3)
    rounded_low = np.around(lower, -3)
    print(f'The Estimated property value is ${rounded_est}')
    print(f'At {conf}% confidence level, the valuation range is')
    print(f'${rounded_low} at the lower end and ${rounded_hi} at the higher end.')
