from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """
    Resolve the project root directory.

    Returns
    -------
    pathlib.Path
        Project root directory.
    """
    return Path.cwd().resolve().parents[0]


def load_csv(relative_path: str) -> pd.DataFrame:
    """
    Load a CSV file relative to the project root.

    Parameters
    ----------
    relative_path
        Path relative to the project root.

    Returns
    -------
    pandas.DataFrame
        Loaded data.
    """
    project_root = get_project_root()
    return pd.read_csv(project_root / relative_path)


def save_csv(
    df: pd.DataFrame,
    relative_path: str,
    *,
    suffix: str | None = None,
    index: bool = False,
) -> Path:
    """
    Save a DataFrame as a CSV file.

    Parameters
    ----------
    df
        DataFrame to save.
    relative_path
        Original CSV path relative to the project root.
    suffix
        Optional suffix to append to the original filename stem
        (e.g. ``"_preprocessed"``).
    index
        Whether to write row indices.

    Returns
    -------
    pathlib.Path
        Path of the saved CSV file.
    """
    project_root = get_project_root()
    original_path = project_root / relative_path

    if suffix is None:
        output_path = original_path
    else:
        output_path = original_path.with_name(
            f"{original_path.stem}{suffix}{original_path.suffix}"
        )

    df.to_csv(output_path, index=index)
    return output_path
