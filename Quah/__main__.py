import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import warnings

from . import logger
from .utils import (process_all_stocks,
                    split_train_val_test,
                    batch_gd,
                    plot_losses,
                    plot_confusion_matrix,
                    extractTestPeriodLists,
                    simulateModelsTradingStrategy,
                    modelVsBenchmark)

from .nns import ANN
from .argParser import get_args


warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    timestamp_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    args = get_args()

    data_folder = Path(args.data_folder)
    model_name = args.model_name
    model_folder = Path(args.model_folder)
    figures_folder = Path(args.figures_folder)

    if not model_folder.exists():
        model_folder.mkdir(parents=True)
    if not data_folder.exists():
        data_folder.mkdir(parents=True)

    # The flag to train the model, if not set the model will be loaded.
    train_flag = args.train

    tickers = args.tickers
    load_stock_data = args.load_stock_data
    stocks_csv_path = args.stocks_csv_path
    prices_csv_path = args.prices_csv_path
    features = args.features
    n_val = args.n_val
    n_test = args.n_test
    features_pct_change = args.features_pct_change
    prev_return = args.prev_return
    features_to_normalize = args.features_to_normalize

    file_paths: list[Path] = []
    for file_path in data_folder.iterdir():
        if not file_path.stem.startswith('~') and file_path.is_file():
            if tickers:
                if file_path.stem.split('-')[0].upper() in tickers:
                    file_paths.append(file_path)
            else:
                file_paths.append(file_path)

    if not load_stock_data:
        stocks_df, prices_df = process_all_stocks(file_paths,
                                                  features=features,
                                                  features_pct_change=features_pct_change,
                                                  prev_return=prev_return,
                                                  )

        # replace all the `/` in the column names with `Per` to avoid issues with the file names
        stocks_df.columns = [col.replace('/', 'Per') for col in stocks_df.columns]
        prices_df.columns = [col.replace('/', 'Per') for col in prices_df.columns]

        features = [feature.replace('/', 'Per') for feature in features]
        features_to_normalize = [feature.replace('/', 'Per') for feature in features_to_normalize]

        if stocks_csv_path:
            stocks_df.to_csv(stocks_csv_path, index=False)
        else:
            # better name for this file, incorporate date
            name = f"./Quah/data/stocks_feat{len(features)}_{len(tickers)}_{timestamp_str}.csv"
            stocks_df.to_csv(name, index=False)
        if prices_csv_path:
            prices_df.to_csv(prices_csv_path, index=False)
        else:
            # better name for this file, incorporate date
            name = f"./Quah/data/prices_feat{len(features)}_{len(tickers)}_{timestamp_str}.csv"
            prices_df.to_csv(name, index=False)
    else:
        stocks_df = pd.read_csv(stocks_csv_path)
        prices_df = pd.read_csv(prices_csv_path)

        # replace all the `/` in the column names with `Per` to avoid issues with the file names
        stocks_df.columns = [col.replace('/', 'Per') for col in stocks_df.columns]
        prices_df.columns = [col.replace('/', 'Per') for col in prices_df.columns]

        features = [feature.replace('/', 'Per') for feature in features]
        features_to_normalize = [feature.replace('/', 'Per') for feature in features_to_normalize]

    if features_to_normalize:
        stocks_df.loc[:, features_to_normalize] = stocks_df.loc[:, features_to_normalize].div(
            stocks_df['Shares Outstanding (Basic)'], axis=0)
        features.remove('Shares Outstanding (Basic)')   # remove the Shares Outstanding (Basic) from the features list

    # drop invalid ticker rows
    # all tickers in prices_df where there are -ve values in the Open or close coluns
    invalid_tickers = prices_df[(prices_df['Open'] < 0) | (prices_df['Close'] < 0)]['Ticker'].unique().tolist()

    # get tickers with a Return > 45,
    unstable_tickers = stocks_df[stocks_df['Return'] > 45]['Ticker'].unique().tolist()

    if len(invalid_tickers) > 0 or len(unstable_tickers) > 0:
        logger.warning('invalid_tickers: ', invalid_tickers, ' unstable_tickers: ', unstable_tickers)
        # drop all the rows with invalid tickers
        prices_df = prices_df[~prices_df['Ticker'].isin(list(invalid_tickers) + list(unstable_tickers))]
        stocks_df = stocks_df[~stocks_df['Ticker'].isin(list(invalid_tickers) + list(unstable_tickers))]

    # Rename columns in the DataFrame
    stocks_df.columns = [col.replace('/', 'Per') for col in stocks_df.columns]

    # Rename values in the features list
    features = [feature.replace('/', 'Per') for feature in features]

    # drop all rows where all the cells have the value 0.0 excluding the 'Ticker' column
    # not nan, but 0.0
    stocks_df = stocks_df.replace(0.0, pd.NA)
    stocks_df = stocks_df.dropna(how='all', subset=features)
    stocks_df = stocks_df.fillna(0.0)
    # stocks_df.loc[:, features] = stocks_df.loc[:, features].clip(lower=-100, upper=100)
    # replace all inf with 1, when something is divided by 0, went from 0 to > 0
    stocks_df = stocks_df.replace([float('inf'), float('-inf')], [1, -1])

    assert stocks_df.shape[0] == prices_df.shape[0]

    for col in stocks_df.columns:
        if col != 'Ticker':
            print(f'{col}: min: {stocks_df[col].min()}, mean: {stocks_df[col].mean()}, max: {stocks_df[col].max()}')

    # Ensure that the indices and ordering match between test_df and test_price_df
    assert stocks_df['Ticker'].equals(prices_df['Ticker']), "Tickers do not match between test_df and test_price_df"

    (train_df, val_df, test_df,
     X_train, Y_train, train_tickers,
     X_val, Y_val, val_tickers,
     X_test, Y_test, test_tickers,
        test_price_df) = split_train_val_test(stocks_df,
                                              prices_df,
                                              features,
                                              target_col='Return',
                                              n_val=n_val,
                                              n_test=n_test)

    # print the shapes of the train, val and test datasets
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    # print the tickers in the train, val and test datasets
    print(f"{len(train_tickers['Ticker'].unique())} Tickers: {train_tickers['Ticker'].unique()}")

    # scale the data
    from sklearn.preprocessing import StandardScaler
    # For training data
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)

    X_val_standardized = scaler.transform(X_val)

    # For test data
    X_test_standardized = scaler.transform(X_test)

    bins: list[Union[float, int]] = args.bins
    Y_train_binned = np.digitize(Y_train, bins)
    Y_val_binned = np.digitize(Y_val, bins)
    Y_test_binned = np.digitize(Y_test, bins)

    train_class_counts = np.bincount(Y_train_binned.flatten())
    train_class_weights = 1. / train_class_counts
    train_class_weights = torch.tensor(train_class_weights, dtype=torch.float32)

    val_class_counts = np.bincount(Y_val_binned.flatten())
    val_class_weights = 1. / val_class_counts
    val_class_weights = torch.tensor(val_class_weights, dtype=torch.float32)

    # print the number of samples in each class
    # improve the logging of the class counts
    for i in np.unique(Y_train_binned):
        print(f'Class {i}: {np.sum(Y_train_binned == i)}')

    for i in np.unique(Y_test_binned):
        print(f'Class {i}: {np.sum(Y_test_binned == i)}')

    for i in np.unique(Y_val_binned):
        print(f'Class {i}: {np.sum(Y_val_binned == i)}')

    # convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train_standardized.astype(np.float32))
    Y_train_t = torch.from_numpy(Y_train_binned.flatten())

    X_val_t = torch.from_numpy(X_val_standardized.astype(np.float32))
    Y_val_t = torch.from_numpy(Y_val_binned.flatten())

    X_test_t = torch.from_numpy(X_test_standardized.astype(np.float32))
    Y_test_t = torch.from_numpy(Y_test_binned.flatten())

    # Assign weight to each sample in the dataset
    train_sample_weights = train_class_weights[Y_train_t]
    val_sample_weights = val_class_weights[Y_val_t]

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

    # datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    # test_dataset = TensorDataset(X_test_t, Y_test_t)

    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if train_flag:
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

        input_size = X_train_t.shape[1]
        hidden_size = int(input_size * 2)
        output_size = len(np.unique(Y_train_t))

        # Create the model
        model = ANN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        model.apply(init_weights)

        # class_weights = torch.tensor([1.0, 5.0])  # Increase weight for class 1
        criterion = nn.CrossEntropyLoss()   # weight=class_weights
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # model name
        model_name = (f"{model.__class__.__name__}_"
                      f"in{input_size}_hid{hidden_size}_out{output_size}_"
                      f"batch{batch_size}_lr{optimizer.param_groups[0]['lr']}_"
                      f"bins{''.join(map(str, bins)).replace(".", "p")}_"
                      f"{timestamp_str}")
        # model output path
        model_output_path = model_folder / f'{model_name}.pth'
        train_losses, test_losses = batch_gd(model, criterion, optimizer,
                                             train_loader, val_loader=val_loader,
                                             epochs=args.epochs,
                                             save_path=model_output_path)

        model_figures_folder = figures_folder / model_name

        if not model_figures_folder.exists():
            model_figures_folder.mkdir(parents=True)

        # losses_plot name: model_name + losses.png
        losses_path = model_figures_folder / f'{model_name}_losses.png'
        plot_losses('All Stocks', train_losses, test_losses, save=True,
                    path=losses_path)

        # Load the model that was saved
        checkpoint = torch.load(model_output_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        best_val_accuracy = checkpoint['val_f1']
    else:
        if not model_name:
            raise ValueError('Model name must be provided to load the model')
        if not model_folder.exists():
            raise ValueError('Model folder does not exist')

        model_figures_folder: Path = figures_folder / model_name

        if not model_figures_folder.exists():
            model_figures_folder.mkdir(parents=True)

        checkpoint_path = model_folder / f'{model_name}.pth'
        # Scrape the model parameters from the model name
        # Get the input size from the model name
        input_size_start_idx = model_name.find('_in') + 3
        input_size_end_idx = model_name.find('_hid')
        input_size = int(model_name[input_size_start_idx:input_size_end_idx])
        # Get the hidden size from the model name
        hidden_size_start_idx = model_name.find('_hid') + 4
        hidden_size_end_idx = model_name.find('_out')
        hidden_size = int(model_name[hidden_size_start_idx:hidden_size_end_idx])
        # Get the output size from the model name
        output_size_start_idx = model_name.find('_out') + 4
        output_size_end_idx = model_name.find('_batch')
        output_size = int(model_name[output_size_start_idx:output_size_end_idx])
        model = ANN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # load the model
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        best_val_accuracy = checkpoint['val_f1']

    model.eval()  # Set the model to evaluation mode

    #  Log the epoch, best_val_loss, best_val_accuracy
    print(f'Epoch: {epoch}, Best Val Loss: {best_val_loss}, '
          f'Best Val Accuracy: {best_val_accuracy}')

    # Get the accuracies
    # make predictions
    with torch.no_grad():
        Y_train_pred: torch.Tensor = model(X_train_t)
        Y_val_pred: torch.Tensor = model(X_val_t)
        Y_test_pred: torch.Tensor = model(X_test_t)

        # get the actual values
        Y_train_pred = torch.argmax(Y_train_pred, dim=1).numpy()
        Y_val_pred = torch.argmax(Y_val_pred, dim=1).numpy()
        Y_test_pred = torch.argmax(Y_test_pred, dim=1).numpy()

        # get the boolean accuracy, if both are positive or negative
        train_accuracy = np.mean(Y_train_pred == Y_train_t.numpy())
        val_accuracy = np.mean(Y_val_pred == Y_val_t.numpy())
        test_accuracy = np.mean(Y_test_pred == Y_test_t.numpy())

        print(f'Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}, Test Accuracy: {test_accuracy}')

    print(f'Model: {model_name}')
    # plot and save the confusion matrix
    cm_path = model_figures_folder / f'{model_name}_cm.png'
    plot_confusion_matrix('All Stocks', Y_test_t.numpy(), Y_test_pred,
                          save=True, path=cm_path)

    # log the classification accuracy report
    from sklearn.metrics import classification_report
    # Generate target names dynamically according to the bins
    bin_edges = [-float('inf')] + bins + [float('inf')]
    target_names = [f'{bin_edges[i]} to {bin_edges[i+1]}' for i in range(len(bin_edges) - 1)]
    print(classification_report(Y_test_t.numpy(), Y_test_pred, target_names=target_names))

    # save the classification report to a text file
    report_path = model_figures_folder / f'{model_name}_classification_report.txt'
    with open(report_path, 'w') as report_file:
        report_file.write(classification_report(Y_test_t.numpy(), Y_test_pred, target_names=target_names))

    # Get the test period lists
    X_test_periods, Y_test_periods, price_periods = extractTestPeriodLists(test_df, test_price_df,
                                                                           X_test_t=X_test_t,
                                                                           Y_test_t=Y_test_t,
                                                                           n_val=n_val,
                                                                           n_test=n_test)

    # simulate the model's trading strategy on the test data
    criterias = [
        # [
        #     ('Profit Margin', lambda x: x > 0.05),
        #     ('PE Ratio', lambda x: 0 < x),
        #     ('PE Ratio', lambda x: x < 20),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: 0 < x),
        #     ('PE Ratio', lambda x: x < 20),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: x > 5),
        #     ('PE Ratio', lambda x: x < 20),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: x > 5),
        #     ('PE Ratio', lambda x: x < 15),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: x > 5),
        #     ('PE Ratio', lambda x: x < 15),
        #     ('PB Ratio', lambda x: x > 0.5),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: x > 5),
        #     ('PE Ratio', lambda x: x < 15),
        #     ('PB Ratio', lambda x: x > 0.5),
        #     ('PB Ratio', lambda x: x < 2),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: x > 5),
        #     ('PE Ratio', lambda x: x < 15),
        #     ('PB Ratio', lambda x: x > 0.5),
        #     ('PB Ratio', lambda x: x < 2),
        #     ('Current Ratio', lambda x: x > 1.5),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0),
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: x > 5),
        #     ('PE Ratio', lambda x: x < 15),
        #     ('PB Ratio', lambda x: x > 0.5),
        #     ('PB Ratio', lambda x: x < 2),
        #     ('Current Ratio', lambda x: x > 1.5),
        #     ('Current Ratio', lambda x: x < 3),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0),
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: x > 5),
        #     ('PE Ratio', lambda x: x < 15),
        #     ('PB Ratio', lambda x: x > 0.5),
        #     ('PB Ratio', lambda x: x < 2),
        #     ('Current Ratio', lambda x: x > 1.5),
        #     ('Current Ratio', lambda x: x < 3),
        #     ('EPS (Basic)', lambda x: x > 1),
        #     ('Total Current Assets', lambda x: x >= 100),
        #     ('Book Value Per Share', lambda x: x > 0),
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.10),
        #     ('PE Ratio', lambda x: 5 < x),
        #     ('PE Ratio', lambda x: x < 15),
        #     ('PB Ratio', lambda x: 0.5 < x),
        #     ('PB Ratio', lambda x: x < 2),
        #     ('Current Ratio', lambda x: 1.5 < x),
        #     ('Current Ratio', lambda x: x < 3),
        #     ('EPS (Basic)', lambda x: x > 1),
        #     ('Total Current Assets', lambda x: x >= 100),
        #     ('Book Value Per Share', lambda x: x > 10),
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.05),
        #     ('PE Ratio', lambda x: 5 < x),
        #     ('PE Ratio', lambda x: x < 15),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.05),
        #     ('PE Ratio', lambda x: 0 < x),
        #     ('PE Ratio', lambda x: x < 20),
        #     ('PB Ratio', lambda x: 0.5 < x),
        #     ('PB Ratio', lambda x: x < 2),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.05),
        #     ('PE Ratio', lambda x: 0 < x),
        #     ('PE Ratio', lambda x: x < 20),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: 1.5 < x),
        #     ('Current Ratio', lambda x: x < 3),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.05),
        #     ('PE Ratio', lambda x: 0 < x),
        #     ('PE Ratio', lambda x: x < 20),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x >= 100),
        #     ('Book Value Per Share', lambda x: x > 0)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.05),
        #     ('PE Ratio', lambda x: 0 < x),
        #     ('PE Ratio', lambda x: x < 20),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x > 0),
        #     ('Book Value Per Share', lambda x: x > 10)
        # ],
        # [
        #     ('Profit Margin', lambda x: x > 0.05),
        #     ('PE Ratio', lambda x: 0 < x),
        #     ('PE Ratio', lambda x: x < 20),
        #     ('PB Ratio', lambda x: 0 < x),
        #     ('PB Ratio', lambda x: x < 2.5),
        #     ('Current Ratio', lambda x: x > 1),
        #     ('EPS (Basic)', lambda x: x > 0),
        #     ('Total Current Assets', lambda x: x >= 100),
        #     ('Book Value Per Share', lambda x: x > 10)
        # ],
        [
            ('Profit Margin', lambda x: x > 0),
            ('PE Ratio', lambda x: 0 < x),
            ('PE Ratio', lambda x: x < 15),
            ('PB Ratio', lambda x: 0 < x),
            ('PB Ratio', lambda x: x < 2),
            ('Current Ratio', lambda x: x > 1),
            ('EPS (Basic)', lambda x: x > 0),
            ('Total Current Assets', lambda x: x >= 100),
            ('Book Value Per Share', lambda x: x > 10)
        ],
    ]
    apply_screening = False
    for i, criteria in enumerate(criterias):
        print(f'Applying Screening: {apply_screening}')
        (periodic_returns,
         cumulative_returns,
            cash_balance) = simulateModelsTradingStrategy(model,
                                                          X_test_periods,
                                                          Y_test_periods,
                                                          price_periods,
                                                          bins,
                                                          cash_balance=100_000,
                                                          num_months_period=3,
                                                          apply_screening=apply_screening,
                                                          screening_criteria=criteria)

        # Log the effective annual return, total return and cash balance
        total_return = cumulative_returns[-1]
        num_years = len(cumulative_returns) / 4
        effective_annual_return = (1 + total_return) ** (1/num_years) - 1
        print(f'Effective Annual Return: {effective_annual_return}')
        print(f'Total Return: {total_return}')
        print(f'Cash Balance: {cash_balance}')

        # compare the model's trading strategy with the benchmark
        if apply_screening:
            benchmark_path = model_figures_folder / f'{model_name}_vs_benchmark_screened_{i}.png'
            title = 'Model vs Benchmark with Screening'
        else:
            benchmark_path = model_figures_folder / f'{model_name}_vs_benchmark_no_screening_{i}.png'
            title = 'Model vs Benchmark without Screening'

        benchmark_returns: dict[str, np.ndarray] = modelVsBenchmark(
            prices_df,
            cumulative_returns,
            benchmark_tickers=['VOOG', 'SPY', 'OEF'],
            start_date=None,
            end_date=None,
            save_path=benchmark_path,
            num_months_period=3,
            title=title,
        )


if __name__ == '__main__':
    main()
