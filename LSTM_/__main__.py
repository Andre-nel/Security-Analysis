import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .utils import process_all_stocks
from .models import LSTM, train_model, evaluate_model


def parseArgs():
    parser = argparse.ArgumentParser(description='LSTM')
    # tickers
    parser.add_argument("--tickers", nargs="+", help="Tickers to train on", default=[], required=False)
    # data folder, where the excel files are stored
    parser.add_argument("--data_folder", type=str,
                        default="C:/Users/candr/Documents/learning/Udemy/lazyprogrammer/notebooks/TimeSeries/10_data_collection/downloads",
                        help="Data folder")
    parser.add_argument("--features", nargs="+", type=str, help="Features to use",
                        default=['PE Ratio', 'PB Ratio', 'Current Ratio', 'EPS (Basic)',
                                 'Total Current Assets', 'Book Value Per Share'])
    parser.add_argument("--features_to_normalize", nargs="+", type=str,
                        help="Features to normalize, divide by `Shares Outstanding (Basic)`",
                        default=[])
    # stocks csv path
    parser.add_argument("--stocks_csv_path", type=str,
                        help="Path to save/load stocks csv")
    # prices csv path
    parser.add_argument("--prices_csv_path", type=str,
                        help="Path to save/load prices csv")
    return parser.parse_args()


def main():
    args = parseArgs()

    tickers: list = args.tickers
    data_folder: Path = Path(args.data_folder)
    features: list = args.features
    features_to_normalize: list = args.features_to_normalize
    stocks_csv_path: Path = Path(args.stocks_csv_path)
    prices_csv_path: Path = Path(args.prices_csv_path)

    if stocks_csv_path and prices_csv_path:
        # load the stocks and prices csv files
        stocks_df = pd.read_csv(stocks_csv_path)
        prices_df = pd.read_csv(prices_csv_path)

        features = [feature.replace('/', 'Per') for feature in features]
        features_to_normalize = [feature.replace('/', 'Per') for feature in features_to_normalize]

        tickers = stocks_df['Ticker'].unique().tolist()
    else:
        file_paths: list[Path] = []
        for file_path in data_folder.iterdir():
            if not file_path.stem.startswith('~') and file_path.is_file():
                if tickers:
                    if file_path.stem.split('-')[0].upper() in tickers:
                        file_paths.append(file_path)
                else:
                    file_paths.append(file_path)

        stocks_df, prices_df = process_all_stocks(file_paths,
                                                features=features,
                                                features_pct_change=False,
                                                prev_return=False,
                                                min_num_quarters=100,
                                                )

        # replace all the `/` in the column names with `Per` to avoid issues with the file names
        stocks_df.columns = [col.replace('/', 'Per') for col in stocks_df.columns]
        prices_df.columns = [col.replace('/', 'Per') for col in prices_df.columns]

        features = [feature.replace('/', 'Per') for feature in features]
        features_to_normalize = [feature.replace('/', 'Per') for feature in features_to_normalize]

        tickers = stocks_df['Ticker'].unique().tolist()

        if features_to_normalize:
            # which of the features to normalize occur multiple times in stocks_df's columns?
            duplicate_features = [feature for feature in features_to_normalize
                                if (stocks_df.columns == feature).sum() > 1]

            if duplicate_features:
                raise ValueError(f"The following features occur multiple times in stocks_df: {duplicate_features}")

            stocks_df.loc[:, features_to_normalize] = stocks_df.loc[:, features_to_normalize].div(
                stocks_df['Shares Outstanding (Basic)'], axis=0)
        if 'Shares Outstanding (Basic)' in features:
            features.remove('Shares Outstanding (Basic)')   # remove the Shares Outstanding (Basic) from the features list

        # drop all rows where all the cells have the value 0.0 excluding the 'Ticker' column
        # not nan, but 0.0
        stocks_df = stocks_df.replace(0.0, pd.NA)
        stocks_df = stocks_df.dropna(how='all', subset=features)
        stocks_df = stocks_df.fillna(0.0)
        # stocks_df.loc[:, features] = stocks_df.loc[:, features].clip(lower=-100, upper=100)
        # replace all inf with 1, when something is divided by 0, went from 0 to > 0
        stocks_df = stocks_df.replace([float('inf'), float('-inf')], [1, -1])

        assert stocks_df.shape[0] == prices_df.shape[0]

        # Ensure that the indices and ordering match between test_df and test_price_df
        assert stocks_df['Ticker'].equals(prices_df['Ticker']), "Tickers do not match between test_df and test_price_df"

        stocks_df.to_csv('stocks.csv', index=False)
        prices_df.to_csv('prices.csv', index=False)

    # create the LSTM data
    T = 10
    D = len(features)
    n_val = 20
    n_test = 50

    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []

    for ticker in tickers:
        stock_data = stocks_df[stocks_df['Ticker'] == ticker].reset_index(drop=True)

        len_train = len(stock_data) - n_val - n_test - T
        X_train_ticker = []
        Y_train_ticker = []
        X_val_ticker = []
        Y_val_ticker = []
        X_test_ticker = []
        Y_test_ticker = []

        for t in range(len(stock_data) - T - 1):
            if t < len_train:
                x = stock_data.loc[:, features][t:t+T]
                y = stock_data['Return'][t+T-1]
                X_train_ticker.append(x)
                Y_train_ticker.append(y)
            elif t < len_train + n_val:
                x = stock_data.loc[:, features][t:t+T]
                y = stock_data['Return'][t+T-1]
                X_val_ticker.append(x)
                Y_val_ticker.append(y)
            else:
                x = stock_data.loc[:, features][t:t+T]
                y = stock_data['Return'][t+T-1]
                X_test_ticker.append(x)
                Y_test_ticker.append(y)

        X_train_ticker = np.array(X_train_ticker).reshape(-1, T, D)  # make it N x T x D
        Y_train_ticker = np.array(Y_train_ticker).reshape(-1, 1)
        # print(X_train_ticker.shape, Y_train_ticker.shape)
        X_train.append(X_train_ticker)
        Y_train.append(Y_train_ticker)

        X_val_ticker = np.array(X_val_ticker).reshape(-1, T, D)  # make it N x T x D
        Y_val_ticker = np.array(Y_val_ticker).reshape(-1, 1)
        # print(X_val_ticker.shape, Y_val_ticker.shape)
        X_val.append(X_val_ticker)
        Y_val.append(Y_val_ticker)

        X_test_ticker = np.array(X_test_ticker).reshape(-1, T, D)  # make it N x T x D
        Y_test_ticker = np.array(Y_test_ticker).reshape(-1, 1)
        # print(X_test_ticker.shape, Y_test_ticker.shape)
        X_test.append(X_test_ticker)
        Y_test.append(Y_test_ticker)

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    Y_val = np.concatenate(Y_val, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)

    print(X_train.shape, Y_train.shape)
    print(X_val.shape, Y_val.shape)
    print(X_test.shape, Y_test.shape)

    # convert the returns to binary values
    Y_train = (Y_train > 0.2).astype(int)
    Y_val = (Y_val > 0.2).astype(int)
    Y_test = (Y_test > 0.2).astype(int)

    # convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.int64).squeeze()
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.int64).squeeze()
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.int64).squeeze()

    # make datasets
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)

    train_class_counts = np.bincount(Y_train)
    train_class_weights = 1. / train_class_counts
    train_class_weights = torch.tensor(train_class_weights, dtype=torch.float32)

    val_class_counts = np.bincount(Y_val)
    val_class_weights = 1. / val_class_counts
    val_class_weights = torch.tensor(val_class_weights, dtype=torch.float32)
    # Assign weight to each sample in the dataset
    train_sample_weights = train_class_weights[Y_train]
    val_sample_weights = val_class_weights[Y_val]
    # create a sampler
    from torch.utils.data.sampler import WeightedRandomSampler

    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )
    val_sampler = WeightedRandomSampler(
        weights=val_sample_weights,
        num_samples=len(val_sample_weights),
        replacement=True
    )

    # make data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model_flag = True
    if train_model_flag:
        model = LSTM(n_inputs=len(features), n_hidden=3*len(features), n_rnn_layers=3, n_outputs=2)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        (trained_model,
         train_losses, val_losses,
         train_f1_scores, val_f1_scores) = train_model(model, train_loader,
                                                            val_loader=val_loader,
                                                            optimizer=optimizer,
                                                            criterion=criterion,
                                                            max_epochs=40,
                                                            save_path='C:/Users/candr/Documents/masters/Security-Analysis/LSTM_/best_lstm_model.pt')
    else:
        # load the model
        trained_model = LSTM(n_inputs=len(features), n_hidden=2*len(features), n_rnn_layers=1, n_outputs=2)
        trained_model.to(device)
        trained_model.load_state_dict(torch.load(r'C:/Users/candr/Documents/masters/Security-Analysis/LSTM_/best_ass2_model_1.pt',
                                                 map_location=device))

    accuracy = evaluate_model(trained_model, test_loader, ticker='All Stocks')
    print(f"Accuracy: {accuracy}")


if __name__ == '__main__':
    main()
