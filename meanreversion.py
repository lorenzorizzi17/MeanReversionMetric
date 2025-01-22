import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import norm, anderson

import mplfinance as mp
import yfinance as yf

def download_asset(ticker_symbol : str, start : str, end : str) -> pd.DataFrame:
    """
    Downloads historical stock data for a given ticker symbol from Yahoo Finance within a specified date range.
    
    Parameters:
     - ticker_symbol (str): The ticker symbol of the asset to download.
     - start (str): The start date for the data in the format 'YYYY-MM-DD'.
     - end (str): The end date for the data in the format 'YYYY-MM-DD'.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data with columns 'Open', 'High', 'Low', 'Close', 'Vol', and 'AbsTime'.
    """
    # download the data from Yahoo Finance
    download_asset = yf.download(ticker_symbol, start=start, end=end)

    # adapt the dataframe to the previous format
    len_dataset = len(download_asset)
    deltaT = (download_asset.index[1] - download_asset.index[0]).total_seconds() / 3600   #in hours, for homogeneity
    data = pd.DataFrame({
        "Open": download_asset["Open"].values.reshape(len_dataset),
        "High": download_asset["High"].values.reshape(len_dataset),
        "Low": download_asset["Low"].values.reshape(len_dataset),
        "Close": download_asset["Close"].values.reshape(len_dataset),
        "Vol": download_asset["Volume"].values.reshape(len_dataset),
        "AbsTime" : download_asset.index
    })
    data.index = map(int,np.arange(0, deltaT*len(data), deltaT))
    return data

def load_asset(name: str) -> pd.DataFrame:
    """
    Load asset data from a CSV file and preprocess it.

    Parameters:
    name (str): The name of the CSV file to load.

    Returns:
    pd.DataFrame: A DataFrame containing the preprocessed asset data.

    The CSV file is expected to have the following columns:
    - <DATE>: The date of the data point.
    - <TIME>: The time of the data point.
    - <OPEN>: The opening price.
    - <CLOSE>: The closing price.
    - <HIGH>: The highest price.
    - <LOW>: The lowest price.
    - <VOL>: The volume.
    - <TICKVOL>: The tick volume (will be dropped).
    - <SPREAD>: The spread (will be dropped).

    The function performs the following preprocessing steps:
    1. Reads the CSV file with tab-separated values.
    2. Sets the DataFrame index to be a time coordinate measured in hours, with points separated by 0.25 hours (15 minutes).
    3. Creates a new column "AbsTime" by combining the <DATE> and <TIME> columns and converting them to datetime.
    4. Drops the <TIME>, <DATE>, <TICKVOL>, and <SPREAD> columns.
    5. Renames the remaining columns to more readable names: "Open", "Close", "High", "Low", and "Vol".
    """
    data = pd.read_csv(name, sep = "	")
    # data.index is to be interpreted as the time coordinate, measured in hours (hence points are separated by 0.25 h = 15 min)
    # this way, we eliminate the discontinuities in the time coordinate due to the weekend breaks
    data.index = np.arange(0, 0.25*len(data), 0.25)
    data["AbsTime"] = pd.to_datetime(data["<DATE>"] + " " + data["<TIME>"])
    data = data.drop("<TIME>",axis = 1)
    data = data.drop("<DATE>", axis = 1)
    data = data.drop("<TICKVOL>", axis = 1)
    data = data.drop("<SPREAD>", axis = 1)
    data = data.rename(columns = {"<OPEN>": "Open", "<CLOSE>": "Close", "<HIGH>": "High", "<LOW>":"Low", "<VOL>":"Vol" })
    return data

def removeTrend(dataset : pd.DataFrame, mesh : np.array, ax = None) -> np.array:
    """
    Removes the linear trend from segments of a dataset based on provided mesh points.

    Parameters:
     - dataset (pd.DataFrame): The input dataset containing a 'Close' column.
     - mesh (np.array): An array of indices defining the segments of the dataset.
     - ax (matplotlib.axes._axes.Axes, optional): Matplotlib Axes object for plotting the linear fit. Defaults to None.

    Returns:
     - np.array: An array of cleaned data segments with the linear trend removed.
    """
    l = np.zeros(len(mesh), dtype = object)
    for i in range(len(mesh)-1):
        r = (int(mesh[i]), int(mesh[i+1]))
        # resized dataset
        curr_dataset = dataset.iloc[r[0]:r[1]]["Close"].copy() # do we need to copy? To be addressed later
        # performing the linear fit
        def linear_fit(x, a, b):
            return a*x + b
        popt, pcov = curve_fit(linear_fit, curr_dataset.index, curr_dataset.values, p0 = [0,0])  # work on the initial guess !
        #arr[i] = popt[0]
        if (ax is not None):
            ax.plot(np.linspace(curr_dataset.index[0],curr_dataset.index[-1], 1000 ), linear_fit(np.linspace(curr_dataset.index[0],curr_dataset.index[-1], 1000 ), *popt), "r")
        cleaned_data = curr_dataset - linear_fit(curr_dataset.index, *popt)
        l[i] = cleaned_data
    return l


def draw_info(data : pd.DataFrame, start = None, end = None):
    """
    Visualizes the time series of closing prices from a given DataFrame.

    Parameters:
     - data (pd.DataFrame): The input data containing at least 'AbsTime' and 'Close' columns.
     - start (datetime, optional): The start time for the plot. If None, the plot starts from the first timestamp in 'AbsTime'.
     - end (datetime, optional): The end time for the plot. If None, the plot ends at the last timestamp in 'AbsTime'.

    Returns:
    None: This function does not return any value. It displays a plot of the closing prices over time.
    """
    # One can visualize the time series in a naive way by simply plotting the closing price as a function of time
    fig, ax = plt.subplots(figsize=(10,8))

    # A zoomed view of the same dataset reveals the small differences between closing and opening prices
    # N.B. the plot shows discontinuities due to the weekend breaks, but that's only cause I'm plotting against the columns "AbsTime"
    # which is a datetime object. The actual time coordinate is the index of the DataFrame, which is continuous (and that's what we'll use)
    # Just for this picture, I thought it would have been more meaningful to plot against the datetime object
    if(start is None):
        start = (data["AbsTime"].values)[0]
    if (end is None):
        end = (data["AbsTime"].values)[-1]
    filtered_data = data[(data["AbsTime"] > start) & (data["AbsTime"] < end)]
    ax.plot(filtered_data["AbsTime"].values, filtered_data["Close"].values, "b", label = "Closing price")
    ax.set_title("Time series of closing prices")
    ax.legend(loc = "best")
    ax.set_xlabel("Time (datetime)")
    ax.set_ylabel("Price (a.u.)")

    fig.tight_layout()
    plt.show()


## WIP !!! what if we chose a gaussian sampling? (The distance between two points is gaussian distributed)
def assess_normality(dataset : pd.Series, lambda_ : int, n : int, ax = None) -> float:
    """
    Assess the normality of a dataset by sampling it at intervals determined by a Poisson distribution.

    Parameters:
     - dataset (pd.Series): The input dataset to be assessed.
     - lambda_ (int): The average interval between points to sample, based on a Poisson distribution.
     - n (int): The number of times to repeat the sampling process.
     - ax (matplotlib.axes._subplots.AxesSubplot or None): The axis on which to plot the histogram of the sampled data. If None, no plot is generated.

    Returns:
     - float: The Anderson-Darling test statistic for the sampled data, which indicates the degree of normality.
    """
    x = np.array([0])
    # repeat the process n times
    for i in range(n):
        # we will sample the dataset choosing a value every lambda points on average (poisson distr)
        choice = [np.random.randint(lambda_)]  #choice is the vector of indices of chosen values
        for i in range(1, int(len(dataset)/lambda_) ):
            l = np.random.poisson(lambda_)
            if (int(choice[-1]+l) > len(dataset)-1):
                break
            choice.append(int(choice[-1]+l))
        #filtered data, chosen approx every lambda points
        chosen_data = (dataset.values)[choice]
        x = np.concatenate((x,chosen_data), axis = 0)
    if (ax is not None):
        bin_edges = np.arange(np.min(x), np.max(x), 50)
        ax.hist(x, bins = "auto", density=True, edgecolor = "black")
        ax.set_title(f"Input dataset has {len(dataset)} points"+ '\n'+f"Sampling every ~ {lambda_} points, repeating {n} times " + '\n' + f"Total number of points in the distr: {len(x)}" )


    return anderson(x).statistic


###########################################################################

def find_zeros(t : np.array, x : np.array) -> np.array:
    """
    Find the zeros of a function based on its sampled values.
    This function takes two numpy arrays, `t` and `x`, where `t` represents the
    time or independent variable and `x` represents the dependent variable. It
    returns the values of `t` where `x` crosses zero.
    
    Parameters:
     - t (np.array): An array of time or independent variable values.
     - x (np.array): An array of dependent variable values corresponding to `t`.

    Returns:
     - (np.array) An array of `t` values where `x` crosses zero.
    """
    assert len(t) == len(x), "t and x must have the same length"
    zeros = []
    for i in range(len(x)-1):
        if x[i]*x[i+1] < 0:
            zeros.append(t[i])
    return np.array(zeros, dtype=float)

def find_periods(t : np.array, x : np.array) -> np.array:
    """
    Calculate the periods between zero crossings in the given data.

    Parameters:
     - t (np.array): Array of time values.
     - x (np.array): Array of corresponding data values.

    Returns:
     - np.array: Array of periods between zero crossings, each period is calculated as twice the difference between consecutive zero crossings.
    """
    zeros = find_zeros(t, x)
    periods = []
    for i in range(len(zeros)-1):
        periods.append((zeros[i+1] - zeros[i])*2)
    periods = np.array(periods, dtype=float)
    return periods


def find_amplitudes(t : np.array, x : np.array, factor = 0.3) -> np.array:
    from scipy.signal import find_peaks
    """
    Find the amplitudes of the peaks in the given time series data.

    Parameters:
     - t (np.array): Array of time values.
     - x (np.array): Array of corresponding data values.
     - factor (float, optional): Factor to adjust the tollerance distance between peaks. Default is 0.3.

    Returns:
     - np.array: Array of amplitudes of the detected peaks.
    """
    dist = factor*len(x)*np.mean(find_periods(t, x))/(t.max()-t.min())
    max_peaks, _ = find_peaks(x, distance=dist, height=0.0)
    min_peaks, _ = find_peaks(-x, distance=dist, height=0.0)
    return np.concatenate((x[max_peaks], -x[min_peaks]))




def find_period_fft(t : np.array, x : np.array) -> tuple:
    from scipy.fftpack import fft, fftfreq, ifft
    """
    Calculate the period and phase of a signal using Fast Fourier Transform (FFT).

    Parameters:
     - t (np.array): Array of time values.
     - x (np.array): Array of signal values corresponding to the time values.

    Returns:
     - tuple: A tuple containing the period of the signal (1/max_freq) and the phase of the signal.
    """
    fourier = np.abs(fft(x))
    freqs = fftfreq(len(t), t[1] - t[0])
    mask = np.where(freqs > 0)
    fourier = fourier[mask]
    freqs = freqs[mask]
    max_freq = freqs[np.argmax(fourier)]
    phase = np.angle(fft(x)[np.argmax(fourier)])
    return (1/max_freq, phase)