import pandas as pd
import numpy as np
from typing import NamedTuple

class MyDataset(NamedTuple):
    data: np.ndarray
    target: np.ndarray
    target_names: np.ndarray
    feature_names: list
    DESCR: str

def load_csv(filepath: str, target_column: str = 'species') -> MyDataset:
    """
    Wczytuje plik CSV do formatu kompatybilnego ze sklearn.datasets.load_iris()
    
    Parameters:
    -----------
    filepath : str
        Ścieżka do pliku CSV
    target_column : str
        Nazwa kolumny zawierającej etykiety klas (domyślnie 'species')
    
    Returns:
    --------
    IrisDataset : NamedTuple
        Obiekt zawierający data, target, target_names, feature_names, DESCR
    
    Example:
    --------
    >>> iris = load_iris_csv('iris.csv')
    >>> print(iris.data.shape)
    >>> print(iris.target_names)
    """
    
    df = pd.read_csv(filepath)
    
    # Wyodrębnij cechy i etykiety
    feature_names = [col for col in df.columns if col != target_column]
    X = df[feature_names].values
    
    # Konwertuj etykiety tekstowe na liczby
    y_text = df[target_column].values
    unique_targets = np.unique(y_text)
    target_names = unique_targets
    
    # Mapuj etykiety tekstowe na liczby
    target_mapping = {name: idx for idx, name in enumerate(unique_targets)}
    y = np.array([target_mapping[label] for label in y_text], dtype=np.int64)
    
    # Opis datasetu
    DESCR = f"""Dataset loaded from {filepath}
Features: {feature_names}
Target: {target_column}
Target classes: {target_names}
Samples: {len(X)}"""
    
    return MyDataset(
        data=X,
        target=y,
        target_names=target_names,
        feature_names=feature_names,
        DESCR=DESCR
    )


# Przykład użycia
if __name__ == "__main__":    
    dataset_custom = load_csv('pirvision_office_dataset1.csv', "Label")
    
    # Porównaj
    print("=== PORÓWNANIE ===")
    print(f"Kształt danych: {dataset_custom.data.shape}")
    print(f"Kształt targetów: {dataset_custom.target.shape}")
    print(f"Nazwy cech: {dataset_custom.feature_names}")
    print(f"Nazwy klas: {dataset_custom.target_names}")
    print(f"\nPierwsze 5 próbek:\n{dataset_custom.data[:5]}")
    print(f"\nPierwsze 5 targetów: {dataset_custom.target[:5]}")
