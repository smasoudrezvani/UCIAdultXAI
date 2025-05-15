from typing import Dict, Union
import numpy as np
from app.data import DataAdapter
from app.model import (
    preprocess_data,
    ModelContext,
    LogisticRegressionStrategy,
    RandomForestStrategy,
    XGBoostStrategy,
    CatBoostStrategy,
)
from app.logger import logger

class MLPipeline :
    """
    Facade class for managing end-to-end ML tasks.
    It encapsulates data loading, preprocessing, training, and evaluation.
    """

    def __init__ (self, model_type: str = "logistic") :
        self.adapter = DataAdapter()
        self.context = ModelContext(self._select_strategy(model_type))
        logger.debug(f"MLPipeline initialized with model: {model_type}")
    
    def _select_strategy (self , model_type: str) :
        model_type = model_type.lower()
        if model_type == "logistic":
            return LogisticRegressionStrategy()
        elif model_type == "randomforest":
            return RandomForestStrategy()
        elif model_type == "xgboost":
            return XGBoostStrategy()
        elif model_type == "catboost":
            return CatBoostStrategy()
        else:
            logger.error(f"Unknown model type '{model_type}'. Defaulting to LogisticRegression.")
            return LogisticRegressionStrategy()

    def run(self, target_column: str = "income") -> Dict[str, Union[str, float]]:
        """
        Executes the full ML pipeline.
        Returns
        -------
        Dict[str, Union[str, float]]
            A dictionary with model type, accuracy score, and classification report.
        """
        logger.info("Starting ML pipeline execution.")
    
        df = self.adapter.get_dataset()
        logger.info(f"Loaded dataset with shape: {df.shape}")

        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df, target_column)
        logger.info(f"Preprocessed data: X_train {X_train.shape}, X_test {X_test.shape}")

        results = self.context.execute(X_train, y_train, X_test, y_test)
        logger.info(f"Model evaluation complete. Accuracy: {results['accuracy']}")

        # Get feature names
        try:
            # feature_names = self.adapter.get_feature_names(target_column)
            logger.info(f"Feature names extracted: {len(feature_names)} features")
        except Exception as e:
            feature_names = []
            logger.error(f"Failed to extract feature names: {e}")

        # Feature importance + SHAP
        try:
            logger.info("Attempting to get feature importance and SHAP values.")
            
            fi = self.context.get_feature_importance()
            logger.info(f"Feature importance: {type(fi)} of length {len(fi)}")

            shap_vals = self.context.get_shap_summary()
            if hasattr(shap_vals, "values"):
                shap_array = shap_vals.values
                shap_array = shap_array[:100]
                logger.info(f"SHAP values retrieved. Shape: {shap_array.shape}")
            else:
                shap_array = shap_vals
                logger.info(f"SHAP values retrieved (no .values attr). Type: {type(shap_array)}")

            # Validate length match
            if len(fi) != len(feature_names):
                logger.warning(f"Length mismatch: {len(fi)} features vs {len(feature_names)} feature names.")
            if len(shap_array[0]) != len(feature_names):
                logger.warning(f"SHAP shape mismatch: shap_array.shape={shap_array.shape} vs feature_names={len(feature_names)}")

            # Save to results
            results["feature_importance"] = [float(val) for val in fi]  # ensure float
            results["shap_values"] = shap_array.tolist()  # ndarray to list
            results["feature_names"] = list(map(str, feature_names))  # ensure str
            logger.info(f"Feature names length: {len(results['feature_names'])}")
            logger.info(f"Feature importance length: {len(results['feature_importance'])}")
            logger.info(f"SHAP values shape: {np.array(results['shap_values']).shape}")
        except Exception as e:
            results["feature_importance"] = []
            results["shap_values"] = []
            results["feature_names"] = []
            logger.warning(f"Could not generate FI or SHAP: {e}")

        logger.success(f"Pipeline completed. Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Final keys in result: {list(results.keys())}")

        return results
        
