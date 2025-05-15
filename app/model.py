import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import shap
from app.logger import logger

class ModelStrategy() :
    """Interface for all model strategies."""

    def train (self, X_train: pd.DataFrame, y_train: pd.Series) :
        raise NotImplementedError
    
    def evaluate (self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Union[str, float]] :
        raise NotImplementedError

class LogisticRegressionStrategy(ModelStrategy) :

    def __init__ (self) :
        self.model = LogisticRegression(max_iter=500)
        self.feature_names = None
        self.X_sample = None
    
    def train (self , X_train, y_train) :
        logger.info("Training Logistic Regression model.")
        self.model.fit(X_train, y_train)
        self.X_sample = X_train[:100].toarray()  # for SHAP summary
        logger.info(f"SHAP sample shape: {self.X_sample.shape}")
        self.feature_names = X_train.shape[1]
        logger.info("Logistic Regression model trained.")

    def evaluate(self, X_test, y_test):
        logger.info("Evaluating Logistic Regression model.")
        predictions = self.model.predict(X_test)
        return {
            "model" : "Logistic Regression",
            "accuracy" : accuracy_score (y_test, predictions),
            "report" : classification_report(y_test, predictions)
        }
    
    def get_feature_importance(self):
        if hasattr(self.model, 'coef_'):
            return np.mean(np.abs(self.model.coef_), axis=0).tolist()
        return []
    
    def get_shap_summary(self):
        if self.X_sample is None:
            return []
        explainer = shap.LinearExplainer(self.model, self.X_sample)
        return explainer.shap_values(self.X_sample)

class RandomForestStrategy (ModelStrategy) :

    def __init__ (self) :
        self.model = RandomForestClassifier()
        self.feature_names = None
        self.X_sample = None

    def train (self , X_train, y_train) :
        logger.info("Training Random Forest model.")
        self.model.fit(X_train, y_train)
        self.X_sample = X_train[:100].toarray()  # for SHAP summary
        self.feature_names = X_train.shape[1]
        logger.info(f"SHAP sample shape: {self.X_sample.shape}")
        logger.success("Random Forest model trained.")

    def evaluate(self, X_test, y_test):
        logger.info("Evaluating Random Forest model.")
        predictions = self.model.predict(X_test)
        return {
            "model": "Random Forest",
            "accuracy": accuracy_score(y_test, predictions),
            "report": classification_report(y_test, predictions)
        }
    
    def get_feature_importance(self):
        return self.model.feature_importances_.tolist()
    
    def get_shap_summary(self):
        if self.X_sample is None:
            return []
        explainer = shap.Explainer(self.model)
        return explainer(self.X_sample)

class XGBoostStrategy (ModelStrategy) :

    def __init__ (self) :
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.feature_names = None
        self.X_sample = None

    def train (self , X_train, y_train) :
        logger.info("Training XGBoost model.")
        self.model.fit(X_train, y_train)
        self.X_sample = X_train[:100].toarray()  # for SHAP summary
        self.feature_names = X_train.shape[1]
        logger.info(f"SHAP sample shape: {self.X_sample.shape}")
        logger.success("XGBoost model trained.")

    def evaluate(self, X_test, y_test):
        logger.info("Evaluating XGBoost model.")
        predictions = self.model.predict(X_test)
        return {
            "model": "XGBoost",
            "accuracy": accuracy_score(y_test, predictions),
            "report": classification_report(y_test, predictions)
        }
    
    def get_feature_importance(self):
        return self.model.feature_importances_.tolist()
    
    def get_shap_summary(self):
        if self.X_sample is None:
            return []
        explainer = shap.Explainer(self.model)
        return explainer(self.X_sample)

class CatBoostStrategy (ModelStrategy) :

    def __init__ (self) :
        self.model = CatBoostClassifier(verbose=0)
        self.feature_names = None
        self.X_sample = None

    def train (self , X_train, y_train) :
        logger.info("Training CatBoost model.")
        self.model.fit(X_train, y_train)
        self.X_sample = X_train[:100].toarray()  # for SHAP summary
        logger.info(f"SHAP sample shape: {self.X_sample.shape}")
        self.feature_names = X_train.shape[1]
        logger.success("CatBoost model trained.")

    def evaluate(self, X_test, y_test):
        logger.info("Evaluating CatBoost model.")
        predictions = self.model.predict(X_test)
        return {
            "model": "CatBoost",
            "accuracy": accuracy_score(y_test, predictions),
            "report": classification_report(y_test, predictions)
        }

    def get_feature_importance(self):
        return self.model.feature_importances_.tolist()
    
    def get_shap_summary(self):
        if self.X_sample is None:
            return []
        explainer = shap.Explainer(self.model)
        return explainer(self.X_sample)

class ModelContext:
    """Context class to switch between model strategies."""

    def __init__ (self, strategy: ModelStrategy) :
        self._strategy = strategy

    def set_strategy (self, strategy: ModelStrategy) :
        self._strategy = strategy
        logger.debug(f"Switched strategy to {strategy.__class__.__name__}")
    
    def execute (self, X_train, y_train, X_test, y_test) -> Dict[str, Union[str, float]] :
        self._strategy.train(X_train, y_train)
        return self._strategy.evaluate(X_test, y_test)
    
    def get_feature_importance(self):
        return self._strategy.get_feature_importance()

    def get_shap_summary(self):
        return self._strategy.get_shap_summary()

def preprocess_data (df: pd.DataFrame, target_column: str) -> tuple :

    logger.info("Preprocessing dataset.")
    y = df[target_column].apply(lambda x: 1 if x.strip() == ">50K" else 0)
    X = df.drop(columns=[target_column])

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_transformed = pipeline.fit_transform(X)

    logger.debug(f"Feature matrix shape after preprocessing: {X_transformed.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42
    )
    logger.success("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test, get_feature_names(preprocessor)

def get_feature_names(column_transformer):
    output_features = []
    for name, transformer, features in column_transformer.transformers_:
        if name != "remainder":
            try:
                if hasattr(transformer, "get_feature_names_out"):
                    output_features.extend(transformer.get_feature_names_out(features))
                else:
                    output_features.extend(features)
            except Exception:
                output_features.extend(features)
    return output_features