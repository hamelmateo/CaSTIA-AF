import pandas as pd


def load_and_merge_datasets(control_paths: dict[str, str], file_name: str) -> pd.DataFrame:
    """
    Load and merge datasets from multiple control paths.

    Args:
        control_paths (dict[str, str]): Dictionary mapping dataset labels to their file paths.
        file_name (str): Name of the CSV file to load from each path.
        
    Returns:
        pd.DataFrame: Merged DataFrame with an additional 'dataset' column.
    """
    dfs = []
    for label, path in control_paths.items():
        df = pd.read_csv(f"{path}/{file_name}")
        df["dataset"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_dataset(path: str, file_name: str) -> pd.DataFrame:
    """
    Load a dataset from a specified path.

    Args:
        path (str): Path to the directory containing the dataset file.
        file_name (str): Name of the CSV file to load.

    Returns:
        pd.DataFrame: DataFrame containing the loaded dataset.
    """
    return pd.read_csv(f"{path}/{file_name}")
