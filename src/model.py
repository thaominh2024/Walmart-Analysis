import pandas as pd
import numpy as np
import logging
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator

# XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# Tạo folder output để lưu kết quả
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Class tạo file Log
class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            if line.strip():
                self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

# Cấu hình Log chuẩn
if not os.path.exists('logs'):
    os.makedirs('logs')

# Tạo tên file log
log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

# Ghi vào File
file_handler = logging.FileHandler(log_filename, mode='w')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# In ra màn hình
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)


class ModelTrainer:
    def __init__(self, random_seed: int = 42):
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models: Dict[str, BaseEstimator] = {}
        self.best_model = None
        self.best_model_name = ""
        self.results_history = []

        logger.info(f"KHỞI TẠO SESSION: Random Seed {self.random_seed}")

    @property
    def feature_importances(self):
        if hasattr(self.best_model, 'feature_importances_'):
            return self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            return self.best_model.coef_
        return None

    def load_data(self, file_path: str):
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Load Data: {file_path} | Shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Lỗi Load Data: {e}")
            raise

    def split_data(self, target_col: str, test_size: float = 0.2, shuffle: bool = True):
        if self.df is None: raise ValueError("Chưa nạp dữ liệu!")
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=self.random_seed
        )
        logger.info(f"Split Data: Train={self.X_train.shape}, Test={self.X_test.shape}")

    def _get_model_instance(self, model_name: str) -> BaseEstimator:
        if model_name == 'linear': return LinearRegression()
        elif model_name == 'rf': return RandomForestRegressor(random_state=self.random_seed, n_jobs=-1)
        elif model_name == 'xgboost':
            if XGBRegressor: return XGBRegressor(objective='reg:squarederror', random_state=self.random_seed, n_jobs=-1)
            else: return None
        return None

    def _get_param_grid(self, model_name: str) -> Dict:
        if model_name == 'rf':
            return {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': [100, 500],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        return {}

    def optimize_params(self, model_name: str, n_iter: int = 5):
        base_model = self._get_model_instance(model_name)
        if base_model is None: return

        param_grid = self._get_param_grid(model_name)
        if not param_grid:
            self.train_model(model_name)
            return

        logger.info(f"--- BẮT ĐẦU TỐI ƯU: {model_name.upper()} ---")

        # Chọn K-Fold = 3
        cv = 3
        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=n_iter,
            scoring='neg_root_mean_squared_error', cv=cv,
            random_state=self.random_seed, n_jobs=-1, verbose=1
        )

        # Bắt đầu Train
        search.fit(self.X_train, self.y_train)

        # LOGGING TỪNG VÒNG LẶP
        results = search.cv_results_
        logger.info(f"Chi tiết kết quả Finetuning ({n_iter} iterations):")

        for i in range(len(results['params'])):
            # Loss ở đây là RMSE
            loss_score = -results['mean_test_score'][i]
            params_str = str(results['params'][i])
            rank = results['rank_test_score'][i]

            # Ghi vào log file
            logger.info(f"Iter {i+1}: Loss (RMSE)={loss_score:,.2f} | Rank={rank} | Params={params_str}")

        self.models[model_name] = search.best_estimator_
        logger.info(f"DONE {model_name}. Best RMSE: {-search.best_score_:,.2f}")
        logger.info(f"Best Params: {search.best_params_}")
        return search.best_params_

    def train_model(self, model_name: str):
        # Hàm train thường (không tối ưu)
        model = self._get_model_instance(model_name)
        logger.info(f"Start Training Default: {model_name}")
        model.fit(self.X_train, self.y_train)
        self.models[model_name] = model
        logger.info(f"Training Finished: {model_name}")

    def evaluate(self, model_name: str) -> Dict:
        if model_name not in self.models: return {}
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        metrics = {"Model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}

        self.results_history.append(metrics)

        # Log kết quả cuối cùng
        logger.info(f"EVALUATE {model_name}: R2={r2:.4f} | RMSE={rmse:,.2f} | MAE={mae:,.2f}")
        return metrics

    def auto_run(self, models_to_try: list, do_optimize: bool = False):
        logger.info("--- START AUTO RUN ---")
        for m in models_to_try:
            if do_optimize:
                self.optimize_params(m)
            else:
                self.train_model(m)
            self.evaluate(m)

        # Chọn model tốt nhất
        if self.results_history:
            best_run = max(self.results_history, key=lambda x: x['R2'])
            self.best_model_name = best_run['Model']
            self.best_model = self.models[self.best_model_name]
            logger.info(f"=== WINNER: {self.best_model_name} (R2={best_run['R2']:.4f}) ===")

            # Lưu JSON
            json_path = os.path.join(self.output_dir, 'experiment_results.json')
            with open(json_path, 'w') as f:
                json.dump(self.results_history, f, indent=4)
            self._plot_comparison()

    def _plot_comparison(self):
        df = pd.DataFrame(self.results_history)
        if df.empty: return

        plt.figure(figsize=(10, 6))

        # Vẽ biểu đồ
        ax = sns.barplot(x='Model', y='R2', data=df, palette='viridis', hue='Model', legend=False)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3, fontweight='bold')
        plt.title('Model R2 Score Comparison')
        plt.ylabel('R2 Score')
        min_r2 = df['R2'].min()
        max_r2 = df['R2'].max()
        lower_limit = 0 if min_r2 > 0 else min_r2 - 0.1
        upper_limit = max(1.0, max_r2 + 0.1)
        plt.ylim(lower_limit, upper_limit)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(save_path)
        logger.info("Saved plot: model_comparison.png")

    def save_model(self):
        """ Lưu model tốt nhất """
        if self.best_model:
            filename = f"best_model_{self.best_model_name}.pkl"
            filename = os.path.basename(filename)
            save_path = os.path.join(self.output_dir, filename)
            joblib.dump(self.best_model, save_path)
            logger.info(f"Saved Model: {save_path}")

    def plot_feature_importance(self, top_n=20):
        if not self.best_model: return
        importances = self.feature_importances
        if importances is None: return

        if self.best_model_name == 'linear': importances = np.abs(importances)

        fi_df = pd.DataFrame({'Feature': self.X_train.columns, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title(f'Feature Importance ({self.best_model_name})')
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(save_path)
        logger.info("Saved Feature Importance plot")
