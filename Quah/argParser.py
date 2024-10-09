import argparse


def parse_bin(value: str) -> float:
    """
    Parse input as either an integer or a float.

    Args:
        value (str): Input value from argparse.

    Returns:
        float: The parsed value as either an integer or float.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be parsed as int or float.
    """
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid bin value: {value}")


def get_args():
    parser = argparse.ArgumentParser(description="Quah")
    parser.add_argument("--train", action="store_true",
                        help="Train the model")
    parser.add_argument("--data_folder", type=str,
                        default="./Quah/data",
                        help="Data folder")
    parser.add_argument("--tickers", nargs="+", help="Tickers to train on", default=[], required=False)

    parser.add_argument("--model_folder", type=str, default="./Quah/models", help="Model folder")
    parser.add_argument("--figures_folder", type=str, default="./Quah/figures", help="Model folder")
    parser.add_argument("--model_name", type=str,
                        required=False,
                        help="Model name, to load the model")
    parser.add_argument("--load_stock_data", action="store_true",
                        help="Load stock data from a saved csv file")
    parser.add_argument("--stocks_csv_path", type=str,
                        help="Path to save/load stocks csv")
    parser.add_argument("--prices_csv_path", type=str,
                        help="Path to save/load prices csv")
    parser.add_argument("--n_val", type=int, default=12, help="Number of validation samples")
    parser.add_argument("--n_test", type=int, default=16, help="Number of test samples")
    parser.add_argument("--bins", nargs="+", type=parse_bin, help="Bins for the target", default=[0.3])
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--features", nargs="+", type=str, help="Features to use",
                        default=['PE Ratio', 'PB Ratio', 'Current Ratio', 'EPS (Basic)',
                                 'Total Current Assets', 'Book Value Per Share'])
    parser.add_argument("--features_pct_change", action="store_true",
                        help="Include percentage change of features as values")
    parser.add_argument("--prev_return", action="store_true", help="Include previous stock price return as a feature")
    return parser.parse_args()
