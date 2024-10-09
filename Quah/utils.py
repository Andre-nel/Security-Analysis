import os
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader

import seaborn as sns
from sklearn.metrics import confusion_matrix

from . import logger


def load_and_clean_excel(file_path: Union[str, Path],
                         sheets: list[str] = [
                             'Income-Quarterly',
                             'Balance-Sheet-Quarterly',
                             'Cash-Flow-Quarterly',
                             'Ratios-Quarterly']) -> pd.DataFrame:
    """Load and clean data from an Excel file, focusing on quarterly financial data.

    Args:
        file_path (Union[str, Path]): Path to the Excel file.
        sheets (list[str]): List of sheet names to load from the Excel file.
    Returns:
        pd.DataFrame: Cleaned and merged quarterly data from all sheets.
    """
    excel: dict[str, pd.DataFrame] = pd.read_excel(file_path, sheet_name=sheets)
    stock_df: Union[None, pd.DataFrame] = None

    for sheet_name, data in excel.items():
        # Optimize DataFrame operations
        data = data.transpose(copy=False)
        data.columns = data.iloc[0]
        data = data.iloc[1:]
        data.index = pd.to_datetime(data.index, errors='coerce')
        data.dropna(how='all', inplace=True)
        data.sort_index(ascending=True, inplace=True)
        data = data.iloc[8:]

        if stock_df is None:
            stock_df = data
        else:
            stock_df = stock_df.merge(data, left_index=True, right_index=True, how='inner', suffixes=('', '_drop'))

    # Drop duplicated columns from merges
    stock_df.drop(columns=[col for col in stock_df.columns if col.endswith('_drop')], inplace=True)
    return stock_df


def fetch_yahoo_finance_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch stock price data from Yahoo Finance for the given date range.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for the data in 'YYYY-MM-DD' format.
        end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing Yahoo Finance stock price history.
    """
    try:
        stock_yf = yf.Ticker(ticker)
        prices_yf = stock_yf.history(start=start_date, end=end_date, interval='3mo')
        if prices_yf.empty:
            raise ValueError(f'No data returned for ticker {ticker}')
    except Exception:
        stock_yf = yf.Ticker(ticker.replace('.', '-'))
        prices_yf = stock_yf.history(start=start_date, end=end_date, interval='3mo')

    return prices_yf


def process_stock_file(file_path: Union[str, Path], features: list[str],
                       features_pct_change: bool = False, prev_return: bool = False) -> pd.DataFrame:
    """Process an individual stock file and merge its financial data with Yahoo Finance stock prices.

    Args:
        file_path (Union[str, Path]): Path to the stock file.
        features (list[str]): List of features to include in the final DataFrame.
        features_pct_change (bool): Whether to include the percentage change of features as the
                                    values of the features, instead of the raw values.
        prev_return (bool): Whether to include the previous stock price return as a feature.
    Returns:
        pd.DataFrame: DataFrame containing merged stock data and prices.
    """
    stock_df = load_and_clean_excel(file_path)
    if len(stock_df) < 40:
        raise ValueError(f'Insufficient data: len: {len(stock_df)} for {file_path.stem}')
    ticker = file_path.stem.split('-')[0].upper()

    start_date = stock_df.index.min().strftime('%Y-%m-%d')
    end_date = stock_df.index.max().strftime('%Y-%m-%d')

    prices_yf = fetch_yahoo_finance_data(ticker, start_date, end_date)

    # Ensure matching lengths between stock data and prices
    min_length = min(len(stock_df), len(prices_yf))
    stock_df = stock_df.iloc[-min_length:]
    prices_yf = prices_yf.iloc[-min_length:]

    stock_df['Close'] = prices_yf['Close'].values
    stock_df['Ticker'] = ticker

    # process each dataFrame before returning
    stock_df = stock_df.loc[:, features + ['Ticker', 'Close']]
    numeric_cols = features + ['Close']
    stock_df.loc[:, numeric_cols] = stock_df.loc[:, numeric_cols].apply(pd.to_numeric)
    stock_df = stock_df.interpolate(limit_direction='both')

    price_df = stock_df[features + ['Ticker', 'Close']]
    price_df = price_df.rename(columns={'Close': 'Open'})
    # Open column
    price_df = price_df.assign(Close=price_df['Open'].shift(-1))
    price_df = price_df.assign(Date=price_df.index)
    price_df = price_df.iloc[1:-1]

    # Optional to use the pct_change for the features
    if features_pct_change:
        stock_df.loc[:, features] = stock_df.loc[:, features].pct_change()

    stock_df.loc[:, 'Return'] = stock_df.loc[:, 'Close'].pct_change()

    # OPTIONAL: Include the previous stock price movement as a feature
    if prev_return:
        stock_df.loc[:, 'Prev Return'] = stock_df.loc[:, 'Return']

    stock_df.loc[:, 'Return'] = stock_df.loc[:, 'Return'].shift(-1)
    stock_df = stock_df.iloc[1:-1]
    assert stock_df.shape[0] == price_df.shape[0]
    return stock_df, price_df


def process_all_stocks(file_paths: list[Union[str, Path]], features: list[str],
                       features_pct_change: bool = False, prev_return: bool = False,
                       num_stocks: int = 10_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process multiple stock files and return a concatenated DataFrame with all data.

    Args:
        file_paths (list[Union[str, Path]]): List of file paths to process.
        features (list[str]): List of features to include in the final DataFrame.
        features_pct_change (bool): Whether to include the percentage change of features as the
                                    values of the features, instead of the raw values.
        prev_return (bool): Whether to include the previous stock price return as a feature.
        num_stocks (int): Number of stocks to process.
    Returns:
        pd.DataFrame: Concatenated DataFrame containing all stock data.
    """
    stocks = []
    price_dfs = []
    max_workers = min(32, (os.cpu_count() or 1) * 5)

    # for file_path in file_paths[:num_stocks]:
    #     try:
    #         stock_df, price_df = process_stock_file(file_path)
    #         stocks.append(stock_df)
    #         price_dfs.append(price_df)
    #     except Exception as e:
    #         ticker = file_path.stem.split('-')[0].upper()
    #         logger.error(f'Error processing {ticker}: {e}')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_stock_file, file_path, features,
                            features_pct_change=features_pct_change,
                            prev_return=prev_return): file_path
            for file_path in file_paths[:num_stocks]
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            ticker = file_path.stem.split('-')[0].upper()
            try:
                stock_df, price_df = future.result()
                stocks.append(stock_df)
                price_dfs.append(price_df)
            except Exception as e:
                logger.error(f'Error processing {ticker}: {e}')

    combined_df = pd.concat(stocks, axis=0, ignore_index=True)
    price_df = pd.concat(price_dfs, axis=0, ignore_index=True)
    return combined_df, price_df


def split_train_val_test(stocks_df: pd.DataFrame, prices_df: pd.DataFrame,
                         features: list[str],
                         target_col='Close',
                         n_val=12,
                         n_test=16,):
    """
    Split the data into training, validation, and test sets.

    Args:
        stocks_df (pd.DataFrame): The DataFrame containing stock data.
        prices_df (pd.DataFrame): The DataFrame containing stock prices.
        features (list[str]): The list of feature columns to use.
        target_col (str, optional): The target column. Defaults to 'Close'.
        n_val (int, optional): The number of validation samples. Defaults to 12.
        n_test (int, optional): The number of test samples. Defaults to 16.

    Returns:
        tuple: Tuple containing the training, validation, test sets and the test price DataFrame.
    """
    # NOTE The data must already sorted by Ticker and Date

    # Compute cumulative count per Ticker
    stocks_df['cum_count'] = stocks_df.groupby('Ticker').cumcount()
    prices_df['cum_count'] = prices_df.groupby('Ticker').cumcount()

    # Compute total count per Ticker
    group_sizes = stocks_df.groupby('Ticker')['Ticker'].transform('size')
    group_sizes_price = prices_df.groupby('Ticker')['Ticker'].transform('size')

    # Compute 'n_train' per row
    stocks_df['n_train'] = group_sizes - n_val - n_test
    prices_df['n_train'] = group_sizes_price - n_val - n_test

    # Assign split labels
    stocks_df['split'] = np.where(
        stocks_df['cum_count'] < stocks_df['n_train'],
        'train',
        np.where(
            stocks_df['cum_count'] < stocks_df['n_train'] + n_val,
            'val',
            'test'
        )
    )

    # Split the data
    train_df = stocks_df[stocks_df['split'] == 'train']
    val_df = stocks_df[stocks_df['split'] == 'val']
    test_df = stocks_df[stocks_df['split'] == 'test'].reset_index(drop=True)

    # Extract features and targets
    X_train = train_df[features].values
    Y_train = train_df[[target_col]].values
    train_tickers = train_df[['Ticker']]

    X_val = val_df[features].values
    Y_val = val_df[[target_col]].values
    val_tickers = val_df[['Ticker']]

    X_test = test_df[features].values
    Y_test = test_df[[target_col]].values
    test_tickers = test_df[['Ticker']]

    # For 'test_price_df', repeat the same process for 'prices_df'

    prices_df.loc[:, 'split'] = np.where(
        prices_df['cum_count'] < prices_df['n_train'],
        'train',
        np.where(
            prices_df['cum_count'] < prices_df['n_train'] + n_val,
            'val',
            'test'
        )
    )
    # Extract 'test_price_df'
    test_price_df = prices_df[prices_df['split'] == 'test'].reset_index(drop=True)

    # Ensure that the indices and ordering match between test_df and test_price_df
    assert test_df['Ticker'].equals(test_price_df['Ticker']), "Tickers do not match between test_df and test_price_df"

    if not (np.sign(stocks_df['Return']) == np.sign(prices_df['Close'] - prices_df['Open'])).all():
        logger.error("ERROR: The stock df and prices df does not correspond.")

        logger.error(np.sign(stocks_df['Return']).sum(), np.sign(prices_df['Close'] - prices_df['Open']).sum())

        # find the indices where the sign is different
        indices = np.where(np.sign(stocks_df['Return']) != np.sign(prices_df['Close'] - prices_df['Open']))[0]
        logger.error(indices)

        # logger.error both df's at the indices
        logger.error(stocks_df.iloc[indices])
        logger.error(prices_df.iloc[indices])
        raise ValueError("The stock df and prices df does not correspond.")

    return (train_df, val_df, test_df,
            X_train, Y_train, train_tickers,
            X_val, Y_val, val_tickers,
            X_test, Y_test, test_tickers,
            test_price_df)


def batch_gd(model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
             train_loader: DataLoader, val_loader: DataLoader,
             epochs: int = 200, print_losses: bool = True,
             save_path: str = './models/best_multi_class_accuracy.pth') -> tuple[np.ndarray, np.ndarray]:

    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)

    train_accuracies = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)

    best_val_accuracy = 0.0

    for it in range(1, epochs + 1):
        model.train()
        t0 = datetime.now()
        train_loss = []
        train_accuracy = []
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            if inputs.shape[0] == 1:
                continue
            # print(inputs.shape, targets.shape)
            outputs: torch.Tensor = model(inputs)
            # print(outputs.shape)
            loss: torch.Tensor = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Record batch loss
            train_loss.append(loss.item())

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            accuracy = torch.sum(preds == targets).item() / len(preds)
            train_accuracy.append(accuracy)

        # Compute mean train loss and accuracy for the epoch
        train_loss = np.mean(train_loss)
        train_losses[it - 1] = train_loss

        train_accuracy = np.mean(train_accuracy)
        train_accuracies[it - 1] = train_accuracy

        # Validation phase
        model.eval()
        val_loss = []
        val_accuracy = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())

                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                accuracy = torch.sum(preds == targets).item() / len(preds)
                val_accuracy.append(accuracy)

        # Compute mean val loss and accuracy for the epoch
        val_loss = np.mean(val_loss)
        val_losses[it - 1] = val_loss

        val_accuracy = np.mean(val_accuracy)
        val_accuracies[it - 1] = val_accuracy

        # Check if this is the best model so far based on val loss
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save the model state dict and optimizer state dict
            torch.save({
                'epoch': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, save_path)
            if print_losses:
                print(f'Epoch {it}: New best val accuracy: {val_accuracy:.3f}. Model saved.')

        dt = datetime.now() - t0
        if print_losses:
            print(f'Epoch: {it}/{epochs}, '
                        f'Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, '
                        f'Train Accuracy: {train_accuracy:.3f}, Val Accuracy: {val_accuracy:.3f}, '
                        f'Duration: {dt}')

    return train_losses, val_losses


def plot_losses(ticker: str, train_losses: list, test_losses: list,
                save: bool = True, path: str = 'train_losses.png') -> None:
    """Plot the training and test losses for a ticker."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'{ticker} - Train vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if save:
        plt.savefig(path)
    else:
        plt.show()


def plot_confusion_matrix(ticker: str, Y_true: np.ndarray, Y_pred: np.ndarray,
                          save: bool = False, path: str = 'confusion_matrix.png') -> None:
    """Plot the confusion matrix for the predicted values.

    Args:
        ticker (str): The stock ticker symbol.
        Y_true (np.ndarray): The true labels.
        Y_pred (np.ndarray): The predicted labels.
        save (bool, optional): Whether to save the plot to a file. Defaults to False.
        path (str, optional): The file path to save the plot. Defaults to 'confusion_matrix.png'.
    """
    cm = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{ticker} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save:
        plt.savefig(path)
    else:
        plt.show()


def extractTestPeriodLists(model_test_df: pd.DataFrame, price_data_df: pd.DataFrame,
                           X_test_t: torch.Tensor, Y_test_t: np.ndarray, n_val: int, n_test: int):
    # Compute 'period' labels
    model_test_df['period'] = model_test_df['cum_count'] - model_test_df['n_train'] - n_val
    price_data_df['period'] = price_data_df['cum_count'] - price_data_df['n_train'] - n_val

    # Initialize lists to hold the resulting DataFrames and tensors
    price_periods = []
    X_test_periods = []
    Y_test_periods = []

    # Loop over each period
    for period in range(n_test):
        print(f"Processing period {period + 1}")

        # Create a mask for the current period
        mask = model_test_df['period'] == period

        # Check if there is data for the current period
        if not mask.any():
            print(f"No data for period {period + 1}")
            continue

        # Extract indices for the current period
        indices = model_test_df.index[mask]

        # Extract data for the current period
        X_q = X_test_t[indices]
        Y_q = Y_test_t[indices]
        price_q = price_data_df.loc[mask].reset_index(drop=True)

        # Append to the lists
        X_test_periods.append(X_q)
        Y_test_periods.append(Y_q)
        price_periods.append(price_q)

    return X_test_periods, Y_test_periods, price_periods


def simulateModelsTradingStrategy(model: nn.Module,
                                  X_test_periods: list[torch.Tensor],
                                  Y_test_periods: list[torch.Tensor],
                                  price_periods: list[pd.DataFrame],
                                  bins: np.ndarray,
                                  cash_balance: float = 500000,
                                  num_months_period: int = 3,
                                  apply_screening: bool = True,
                                  screening_criteria: dict = None):
    """
    Tests a trading strategy using a given model on provided quarterly data.

    Args:
        model (nn.Module): The trained model used for predicting stock classes.
        X_test_periods (list[torch.Tensor]): Test features for each period.
        Y_test_periods (list[torch.Tensor]): Actual labels for each period.
        price_periods (list[pd.DataFrame]): Price data for each period.
        bins (np.ndarray): Classification bins used by the model.
        cash_balance (float): Starting cash balance.
        num_months_period (int): Number of months per period.
        apply_screening (bool): Whether to apply screening criteria.
        screening_criteria (list[tuple[str, function]]): Custom screening criteria provided by the user.

    Returns:
        tuple: Periodic returns, cumulative returns, and final cash balance.
    """
    # Default screening criteria if none is provided
    if screening_criteria is None and apply_screening:
        screening_criteria = [
            ('Profit Margin', lambda x: x > 0.05),
            ('PE Ratio', lambda x: 0 < x),
            ('PE Ratio', lambda x: x < 20),
            ('PB Ratio', lambda x: 0 < x),
            ('PB Ratio', lambda x: x < 2.5),
            ('Current Ratio', lambda x: x > 1),
            ('EPS (Basic)', lambda x: x > 0),
            ('Total Current Assets', lambda x: x > 0),
            ('Book Value Per Share', lambda x: x > 0)
        ]
    elif not screening_criteria or screening_criteria is None or not apply_screening:
        screening_criteria = []

    portfolio = {}  # {ticker: shares}
    periodic_returns = []  # returns for each period
    cumulative_returns = []  # cumulative returns

    # Ensure the model is in evaluation mode
    model.eval()

    for period in range(len(X_test_periods)):
        # Get the data for the current period
        X_test_q = X_test_periods[period]
        Y_test_q = Y_test_periods[period]
        prices_q = price_periods[period].reset_index(drop=True)

        # Print the date range for the period and the period number
        period_start_date = prices_q['Date'].iloc[0]
        period_end_date = pd.Timestamp(period_start_date) + pd.DateOffset(months=num_months_period)
        print(f"\n--- Period {period + 1}: {period_start_date} - {period_end_date.date().strftime('%Y-%m-%d')} ---")

        # Sell all holdings at the beginning of the period
        if portfolio:
            print("Selling all holdings from previous period...")
            for stock, num_shares in portfolio.items():
                # Get the opening price of the stock in the current period
                opening_price = prices_q.loc[prices_q['Ticker'] == stock, 'Open'].values[0]
                sale_value = num_shares * opening_price
                cash_balance += sale_value
            portfolio = {}  # Reset portfolio

        # Use the model to predict classes for each stock
        with torch.no_grad():
            outputs = model(X_test_q)
            _, predicted_classes = torch.max(outputs, 1)

        # Identify the highest class
        highest_class = len(bins)
        selected_indices = (predicted_classes == highest_class).nonzero(as_tuple=True)[0]

        if len(selected_indices) == 0:
            print("No stocks classified in the highest class. Holding cash this period.")
        else:
            # Screen the selected stocks using the screening criteria
            selected_stocks = prices_q.iloc[selected_indices.cpu().numpy()].copy()

            to_be_screened_tickers = selected_stocks['Ticker'].tolist()
            num_to_be_screened = len(to_be_screened_tickers)
            # print(f"Model selected: {num_to_be_screened} Stocks selected for screening: ", to_be_screened_tickers)
            print(f"Model selected: {num_to_be_screened} Stocks selected for screening")

            # Apply screening criteria
            for column, condition in screening_criteria:
                selected_stocks = selected_stocks[condition(selected_stocks[column])]

            # Get tickers of passed and failed stocks
            if apply_screening:
                pass
                passed_tickers = selected_stocks['Ticker'].tolist()
                print(f"{len(passed_tickers)} Stocks passed the screening: ", passed_tickers)
            else:
                print("No screening criteria applied.")

            # Only buy the stocks that passed the screening
            num_stocks = len(selected_stocks)
            if num_stocks == 0:
                print("No stocks passed the screening criteria. Holding cash this period.")
            else:
                investment_per_stock = cash_balance / num_stocks
                date_str = prices_q['Date'].iloc[0]
                print(f"Buying {num_stocks} stocks that passed the screening on {date_str}...")
                
                # raise exception if a ticker occurs more than one in the selected_stocks
                if len(selected_stocks['Ticker'].unique()) != len(selected_stocks):
                    raise ValueError("A ticker occurs more than once in the selected_stocks")
                for idx, stock in selected_stocks.iterrows():
                    ticker = stock['Ticker']
                    opening_price = stock['Open']
                    num_shares = investment_per_stock / opening_price
                    portfolio[ticker] = num_shares

                cash_balance = 0  # All cash invested

        # At the end of the period, sell all stocks at closing price
        period_return = 0
        total_investment = 0
        total_value = 0

        if portfolio:
            print("Selling all holdings at the end of the period...")
            for stock, num_shares in portfolio.items():
                opening_price = prices_q.loc[prices_q['Ticker'] == stock, 'Open'].values[0]
                closing_price = prices_q.loc[prices_q['Ticker'] == stock, 'Close'].values[0]
                investment = num_shares * opening_price
                sale_value = num_shares * closing_price
                profit = sale_value - investment
                total_investment += investment
                total_value += sale_value
                period_return += profit / investment  # Return per stock

            # Average return per stock
            period_return = period_return / len(portfolio)
            # Update cash balance
            cash_balance += total_value
        else:
            period_return = 0  # No investments made

        periodic_returns.append(period_return)
        print(f"Period {period + 1} Return: {period_return * 100:.2f}%")

        # Update cumulative returns
        if cumulative_returns:
            cumulative_return = (1 + cumulative_returns[-1]) * (1 + period_return) - 1
        else:
            cumulative_return = period_return
        cumulative_returns.append(cumulative_return)

        # Reset portfolio for the next period
        portfolio = {}

    return periodic_returns, cumulative_returns, cash_balance


def modelVsBenchmark(prices_df: pd.DataFrame, models_cumulative_returns: list[float],
                     benchmark_tickers: list[str] = ['VOOG', 'OEF'],
                     start_date: str = None, end_date: str = None,
                     save_path: str = None, num_months_period: int = 3,
                     title: str = 'Model vs. Benchmark Returns') -> dict[str, np.ndarray]:
    """
    Compare the model's performance against benchmark indices.

    Args:
        prices_df (pd.DataFrame): Price data containing stock tickers and dates.
        models_cumulative_returns (list[float]): Cumulative returns from the model.
        benchmark_tickers (list[str]): List of benchmark tickers to compare against.
        start_date (str): Start date for the comparison period.
        end_date (str): End date for the comparison period.
        save_path (str): Optional file path to save the plot.
        num_months_period (int): Number of months per period for fetching data.

    Returns:
        dict: Periodic returns for all the benchmark tickers.
    """
    # Ensure the number of months per the model's period is a multiple of 3
    if num_months_period % 3 != 0:
        raise ValueError("The number of months per period must be a multiple of 3.")

    # Determine the start and end dates if not provided
    if end_date is None:
        end_date = pd.Timestamp(prices_df['Date'].max()).date().strftime('%Y-%m-%d')
    if start_date is None:
        num_periods = len(models_cumulative_returns)
        start_date = (pd.Timestamp(end_date) - pd.DateOffset(months=(num_periods - 1) * num_months_period)
                      ).date().strftime('%Y-%m-%d')

    # Fetch data for the benchmark tickers
    benchmark_data: dict[str, pd.DataFrame] = {}
    for ticker in benchmark_tickers:
        ticker_data = yf.Ticker(ticker).history(start=start_date,
                                                end=end_date,
                                                interval='3mo')
        benchmark_data[ticker] = ticker_data

    # If the number of months per period is not 3, convert the benchmark data to the same period
    if num_months_period != 3:
        for ticker, data in benchmark_data.items():
            # Resample the data to the desired period
            data = data.resample(f'{num_months_period}M').agg({'Close': 'last'})
            benchmark_data[ticker] = data

    # Plot benchmark and model returns
    plt.figure(figsize=(10, 6))
    for ticker, data in benchmark_data.items():
        start_price = data['Close'].iloc[0]
        plt.plot(data['Close'] / start_price - 1, label=ticker)

    # Convert models_cumulative_returns to a Pandas Series and assign the correct index
    models_cumulative_returns = pd.Series(models_cumulative_returns, index=next(iter(benchmark_data.values())).index)

    # Plot cumulative returns
    plt.plot(models_cumulative_returns, label="Model's Returns")

    # Add labels, title, and legend
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()

    # Optionally save the plot to a file
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    # Print end returns for benchmarks
    periodic_returns = {}
    for ticker, data in benchmark_data.items():
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        end_return = end_price / start_price - 1
        print(f"{ticker} End return: {end_return:.2%}")
        periodic_returns[ticker] = data['Close'].pct_change().values

    # Print model end return
    print(f"Model End return: {models_cumulative_returns[-1]:.2%}")

    return periodic_returns
