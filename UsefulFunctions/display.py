import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from UsefulFunctions import data
import cufflinks as cf
import plotly.offline as pyo
from plotly.subplots import make_subplots
import plotly.graph_objs as go



def plot_test_returns(returns, legend=True, name=" "):
    """
    Plots the cumulative percentage returns from a trading strategy.

    This function takes a series or dataframe of trading strategy returns, computes the cumulative sum,
    and plots it as a percentage. The plot visualizes the profit and loss (P&L) over time.

    Args:
        returns_serie (pandas.Series): A series containing the returns from the trading strategy.

    Returns:
        None: The function generates and displays a plot.
    """
    
    # Plot cumulative returns as a percentage
    (np.cumsum(returns) * 100).plot(figsize=(15, 5), alpha=0.65)
    
    # Draw a red horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='-', linewidth=1)
    
    # Set labels and title
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('P&L in %', fontsize=20)
    plt.title(f'Cumulative Returns {name}', fontsize=20)
    plt.legend().set_visible(legend)
    print(f"Profits : {'%.2f' % (returns.cumsum().iloc[-1].sum() * 100)}%")

    # Display the plot
    plt.show()



def plot_drawdown(return_series, name=" "):
    """
    Computes and visualizes the drawdown of a strategy based on its return series.

    Parameters:
    return_series (pd.Series): A pandas Series containing the return series of the strategy. 
                               Each value represents the return for a specific period.

    Displays:
    - A plot showing the drawdown over time as a filled area chart.
    - The maximum drawdown percentage is printed to the console.

    Notes:
    - The function assumes the return series is cumulative and starts at zero.
    - NaN values in the return series are dropped before computation.
    - If the return series is empty or contains only NaN values, no plot will be generated.
    """
    
    if return_series.dropna().empty:
        print("The return series is empty or contains only NaN values.")
        return

    # Compute cumulative return
    cumulative_return = return_series.dropna().cumsum() + 1

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_return)

    # Computing the drawdown
    drawdown = cumulative_return / running_max - 1

    plt.figure(figsize=(15, 4))
    plt.fill_between(drawdown.index, drawdown * 100, 0, drawdown, color="red", alpha=0.70)
    plt.title(f"Strategy Drawdown {name}", fontsize=20)
    plt.ylabel("Drawdown %", fontsize=15)
    plt.xlabel("Time")
    plt.show()

    maximum_drawdown = np.min(drawdown) * 100
    print(f"Max Drawdown: {'%.2f' % maximum_drawdown}%")


