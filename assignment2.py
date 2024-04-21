import numpy as np
import pandas as pd
import scipy 
from scipy.stats import norm, multivariate_normal

def unconditional_ar_mean_variance(c, phis, sigma2):

    ## The length of phis is p
    p = len(phis)
    A = np.zeros((p, p))
    A[0, :] = phis
    A[1:, 0:(p-1)] = np.eye(p-1)
    ## Check for stationarity
    eigA = np.linalg.eig(A)
    if all(np.abs(eigA.eigenvalues)<1):
        stationary = True
    else:
        stationary = False
    # Create the vector b
    b = np.zeros((p, 1))
    b[0, 0] = c
    
    # Compute the mean using matrix algebra
    I = np.eye(p)
    mu = np.linalg.inv(I - A) @ b
    
    # Solve the discrete Lyapunov equation
    Q = np.zeros((p, p))
    Q[0, 0] = sigma2
    #Sigma = np.linalg.solve(I - np.kron(A, A), Q.flatten()).reshape(7, 7)
    Sigma = scipy.linalg.solve_discrete_lyapunov(A, Q)
    
    return mu.ravel(), Sigma, stationary


def lagged_matrix(Y, max_lag=7):
    n = len(Y)
    lagged_matrix = np.full((n, max_lag), np.nan)    
    # Fill each column with the appropriately lagged data
    for lag in range(1, max_lag + 1):
        lagged_matrix[lag:, lag - 1] = Y[:-lag]
    return lagged_matrix


def cond_loglikelihood_ar7(params, y):
    c = params[0] 
    phi = params[1:8]
    sigma2 = params[8]
    mu, Sigma, stationary = unconditional_ar_mean_variance(c, phi, sigma2)
    ## We could check that at phis the process is stationary and return -Inf if it is not
    if not(stationary):
        return -np.inf
    ## The distribution of 
    # y_t|y_{t-1}, ..., y_{t-7} ~ N(c+\phi_{1}*y_{t-1}+...+\phi_{7}y_{t-7}, sigma2)
    ## Create lagged matrix
    X = lagged_matrix(y, 7)
    yf = y[7:]
    Xf = X[7:,:]
    loglik = np.sum(norm.logpdf(yf, loc=(c + Xf@phi), scale=np.sqrt(sigma2)))
    return loglik

def uncond_loglikelihood_ar7(params, y):
    ## The unconditional loglikelihood
    ## is the unconditional "plus" the density of the
    ## first p (7 in our case) observations
    cloglik = cond_loglikelihood_ar7(params, y)

    ## Calculate initial
    # y_1, ..., y_7 ~ N(mu, sigma_y)
    c = params[0] 
    phi = params[1:8]
    sigma2 = params[8]
    mu, Sigma, stationary = unconditional_ar_mean_variance(c, phi, sigma2)
    if not(stationary):
        return -np.inf
    mvn = multivariate_normal(mean=mu, cov=Sigma, allow_singular=True)
    uloglik = cloglik + mvn.logpdf(y[0:7])
    return uloglik
    

## Unconditional - define the negative loglikelihood

## Starting value. 
## These estimates should be close to the OLS

# Importing data and applying transformation to get log differences

df = pd.read_csv("/Users/lavin/Downloads/current (1).csv")
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')

transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned = df_cleaned[2:]
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.head()


X = lagged_matrix(df_cleaned['INDPRO'], 7)
yf = df_cleaned['INDPRO'][7:]
Xf = np.hstack((np.ones((772,1)), X[7:,:]))
beta = np.linalg.solve(Xf.T@Xf, Xf.T@yf)
sigma2_hat = np.mean((yf - Xf@beta)**2)

params = np.hstack((beta, sigma2_hat))

def cobj(params, y): 
    return - cond_loglikelihood_ar7(params, y)

results_cond = scipy.optimize.minimize(cobj, params, args = df_cleaned['INDPRO'], method='L-BFGS-B')

def uobj(params, y): 
    return - uncond_loglikelihood_ar7(params,y)

bounds_constant = tuple((-np.inf, np.inf) for _ in range(1))
bounds_phi = tuple((-1, 1) for _ in range(7))
bounds_sigma = tuple((0,np.inf) for _ in range(1))
bounds = bounds_constant + bounds_phi + bounds_sigma

## L-BFGS-B support bounds
results_uncond = scipy.optimize.minimize(uobj, results_cond.x, args = df_cleaned['INDPRO'], method='L-BFGS-B', bounds = bounds)

# results_cond.x are the parameters obtained through conditional likelihood maximisation.
# They are the same as the OLS parameters, which are stored in params. 
# results_uncond.x are the parameters obtained through unconditional likelihood maximisation.

# Forecasting future values for the next 8 periods

def ar_forecast (data, phi, p, h):
    forecasted_values = np.zeros(h)
    for i in range(h):
        # Construct the lagged values for the autoregressive model
        lagged_values = np.flip(data[-p:])
        # Calculate the forecasted value using the autoregressive model
        forecasted_values[i] = np.dot(phi, np.concatenate(([1], lagged_values)))
        # Append the forecasted value to the time series for the next forecast iteration
        data = np.append(data, forecasted_values[i])
    
    return forecasted_values

forecasted_values_cond = ar_forecast(df_cleaned['INDPRO'], phi = results_cond.x[0:8], p = 7, h = 8)
forecasted_values_uncond = ar_forecast(df_cleaned['INDPRO'], phi = results_uncond.x[0:8], p = 7, h = 8)
print(f'The values forecasted through conditional likelihood maximization are: {forecasted_values_cond}\nThe values forecasted through unconditional likelihood maximization are: {forecasted_values_uncond}')

# Comparing the accuracy of the forecasts

series = df_cleaned['INDPRO'][:600]
results_cond = scipy.optimize.minimize(cobj, params, args = series, method='L-BFGS-B')
results_uncond = scipy.optimize.minimize(uobj, results_cond.x, args = series, method='L-BFGS-B', bounds = bounds)

def evalforecast (phis, i, n):
    e = []
    Y_hat = []
    Y_actual = []
    for j in range(0, i):
        yhat = ar_forecast(data = df_cleaned['INDPRO'][:n+j], phi = phis, p = 7, h = 1)
        Y_hat.append(yhat.flatten())
        yactual=df_cleaned['INDPRO'][n+1+j]
        Y_actual.append(yactual.flatten())
        ehat = yactual - yhat
        e.append(ehat.flatten())
    return (Y_actual, Y_hat, e)

condforecast = evalforecast(phis = results_cond.x[0:8], i = 178, n = 600)
uncondforecast = evalforecast(phis = results_uncond.x[0:8],i = 178, n = 600)

# Plotting actual data vs data forecasted with conditional and unconditional MLE

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

dates= pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
titles = ['Conditional Likelihood Maximisation', 'Unconditional Likelihood Maximisation']
forecasts = [condforecast, uncondforecast]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for i, ax in enumerate(axs):
    ax.plot(dates[601:], forecasts[i][1], label='Forecasted data')
    ax.plot(dates[601:], forecasts[i][0], label='Actual data') 
    ax.set_title(titles[i])
    ax.legend()
plt.tight_layout()
plt.show()

# Comparing mean squared forecast errors

e_cond, e_uncond = pd.DataFrame(condforecast[2]), pd.DataFrame(uncondforecast[2])
msfe_cond, msfe_uncond = (e_cond.apply(np.square).mean()), (e_uncond.apply(np.square).mean())
msfe_cond, msfe_uncond = float(msfe_cond.iloc[0]), float(msfe_uncond.iloc[0])

print(f'The MSFE of the forecast obtained through conditional MLE is: {msfe_cond:.10f}\nThe MSFE of the forecast obtained through unconditional MLE is: {msfe_uncond:.10f}')

