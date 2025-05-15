import pandas as pd
from pathlib import Path
from typing import Optional
from app.config import settings
from app.logger import logger

class _DataSingleton :
    """
    Singleton class for loading and storing the dataset.
    Ensures that the dataset is loaded only once during the application's lifecycle.
    """
    _instance = None
    _dataframe : Optional[pd.DataFrame] = None

    def __new__ (cls) :
        if cls._instance is None :
            cls._instance = super(_DataSingleton, cls).__new__(cls)
        
        return cls._instance
    
    def load_data(self,filepath : Path = settings.data_dir / "adult.csv") -> pd.DataFrame :
        if self._dataframe is None:
            logger.info(f"Loading dataset from {filepath}")
            try:
                df = pd.read_csv(filepath)
                logger.success("Dataset loaded successfully.")
            except Exception as e:
                logger.critical(f"Failed to load dataset: {e}")
                raise

            df.columns = [col.strip().lower().replace("-", "_") for col in df.columns]
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            self._dataframe = df
            logger.debug(f"Dataset shape: {df.shape}")
        else:
            logger.info("Dataset already loaded. Using cached version.")
        return self._dataframe
    
class DataAdapter :
    """
    Adapter class to provide a standard interface for data retrieval.
    """
    def __init__(self):
        self.df = None
        self.feature_names = None
        self.loader = _DataSingleton()
        logger.debug("DataAdapter initialized.")

    def get_dataset(self) -> pd.DataFrame:
        logger.info("Retrieving dataset through DataAdapter.")
        self.df = self.loader.load_data()

        target_column = "income"
        if self.feature_names is None:
            self.feature_names = [col for col in self.df.columns if col != target_column]
            logger.debug(f"Feature names set: {self.feature_names[:5]}... ({len(self.feature_names)} features)")

        return self.df

    def get_feature_names(self, target_column: str) -> list:
        if self.feature_names is None:
            self.get_dataset()
        return self.feature_names



