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
from sklearn.preprocessing import StandardScaler
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

# Logging
output_dir = 'output'
log_dir = os.path.join(output_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:

    def __init__(self, random_seed: int = 42):
        """
        Khởi tạo ModelTrainer.
        random_seed (int): Seed cố định để đảm bảo tính tái lập (reproducibility).
        """
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

        logger.info(f"Khởi tạo ModelTrainer với random_seed={self.random_seed}")

    @property
    def feature_importances(self):
      # Lấy những đặc trưng quan trọng của mô hình
        if hasattr(self.best_model, 'feature_importances_'):
            return self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            return self.best_model.coef_
        return None

    def plot_feature_importance(self, top_n: int = 20):
        # Vẽ biểu đồ tầm quan trọng của các đặc trưng (Feature Importance).
        if self.best_model is None:
            logger.warning("Chưa có model tốt nhất để vẽ Feature Importance.")
            return
        # Lấy giá trị importance/coefficient
        importances = self.feature_importances

        if importances is None:
            logger.warning(f"Model {self.best_model_name} không hỗ trợ trích xuất Feature Importance.")
            return

        # Lấy tên cột
        feature_names = self.X_train.columns

        # Tạo DataFrame
        if self.best_model_name == 'linear':
            importances = np.abs(importances)

        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sắp xếp và lấy Top N
        fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_n)

        # Vẽ biểu đồ
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
        plt.title(f'Top {top_n} Feature Importance ({self.best_model_name})')
        plt.xlabel('Mức độ ảnh hưởng')
        plt.ylabel('Đặc trưng')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

        save_path = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(save_path)
        plt.show() 
        plt.close()
        logger.info("Đã lưu biểu đồ Feature Importance: feature_importance.png")

    def load_data(self, file_path: str):
        """Nạp dữ liệu từ file CSV đã pre-processing và Scaling."""
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Đã load dữ liệu từ {file_path}. Kích thước: {self.df.shape}")
        except Exception as e:
            logger.error(f"Lỗi khi load dữ liệu: {e}")
            raise

    def split_data(self, target_col: str, test_size: float = 0.2, shuffle: bool = True):
        if self.df is None:
            raise ValueError("Chưa có dữ liệu!")

        # Tách Feature và Target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        # Chia Train/Test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=self.random_seed
        )
        logger.info(f"Chia dữ liệu thành công. Train: {self.X_train.shape}, Test: {self.X_test.shape}")

        # SCALING
        cols_to_scale = [
            'temperature', 'fuel_price', 'cpi', 'unemployment',
            'month_sin', 'month_cos', 'day_sin', 'day_cos'
        ]

        valid_scale_cols = [c for c in cols_to_scale if c in self.X_train.columns]

        if valid_scale_cols:
            self.scaler = StandardScaler()

            # fit_transform trên tập TRAIN
            # transform trên tập TEST
            self.X_train[valid_scale_cols] = self.scaler.fit_transform(self.X_train[valid_scale_cols])
            self.X_test[valid_scale_cols] = self.scaler.transform(self.X_test[valid_scale_cols])

            logger.info(f"Đã scale cho {len(valid_scale_cols)} cột: {valid_scale_cols}")
        else:
            logger.warning("Không tìm thấy cột nào để scale.")

    def _get_model_instance(self, model_name: str) -> BaseEstimator:
        #Hàm local để khởi tạo model dựa trên tên model
        if model_name == 'linear':
            return LinearRegression()
        elif model_name == 'rf':
            return RandomForestRegressor(random_state=self.random_seed, n_jobs=-1)
        elif model_name == 'xgboost':
            if XGBRegressor:
                return XGBRegressor(objective='reg:squarederror', random_state=self.random_seed, n_jobs=-1)
            else:
                logger.warning("XGBoost chưa được cài đặt.")
                return None
        else:
            raise ValueError(f"Mô hình {model_name} chưa được hỗ trợ.")

    def _get_param_grid(self, model_name: str) -> Dict:
        # Hàm local trả về không gian tham số để tối ưu
        if model_name == 'rf':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10]
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': [100, 500, 1000],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        return {} # Linear Regression không cần tune

    def train_model(self, model_name: str, params: Dict = None):
        # Huấn luyện một mô hình cụ thể
        model = self._get_model_instance(model_name)
        if model is None: return

        if params:
            model.set_params(**params)

        logger.info(f"Bắt đầu huấn luyện {model_name}...")
        model.fit(self.X_train, self.y_train)

        self.models[model_name] = model
        logger.info(f"Đã huấn luyện xong {model_name}.")

    def optimize_params(self, model_name: str, n_iter: int = 10):
        # Tối ưu siêu tham số sử dụng RandomizedSearchCV.
        base_model = self._get_model_instance(model_name)
        if base_model is None: return

        param_grid = self._get_param_grid(model_name)
        if not param_grid:
            logger.info(f"{model_name} không có tham số để tối ưu. Train mặc định.")
            self.train_model(model_name)
            return

        logger.info(f"Đang tối ưu tham số cho {model_name} (RandomizedSearch)...")

        cv = 3 # Chọn K-Fold

        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=n_iter,
            scoring='neg_root_mean_squared_error', cv=cv,
            random_state=self.random_seed, n_jobs=-1, verbose=1
        )

        search.fit(self.X_train, self.y_train)

        self.models[model_name] = search.best_estimator_
        logger.info(f"Tối ưu xong {model_name}. Best params: {search.best_params_}")
        return search.best_params_

    def evaluate(self, model_name: str) -> Dict:
        # Đánh giá mô hình trên tập Test và lưu kết quả.
        if model_name not in self.models:
            logger.error(f"Model {model_name} chưa được train.")
            return {}

        model = self.models[model_name]
        y_pred = model.predict(self.X_test)

        # Tính metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)

        metrics = {
            "Model": model_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }

        self.results_history.append(metrics)
        logger.info(f"Đánh giá {model_name}: {metrics}")

        # Cập nhật Best Model
        if self.best_model is None or r2 > self.results_history[0]['R2']:
            pass

        return metrics

    def auto_run(self, models_to_try: list = ['linear', 'rf'], is_optimize : bool=False):
        # Tự động chạy thử, tối ưu và so sánh các mô hình
        logger.info("--- BẮT ĐẦU CHẾ ĐỘ AUTO RUN ---")

        for m in models_to_try:
            if is_optimize:
              self.optimize_params(m)
            else:
              self.train_model(m)
            self.evaluate(m)

        # Chọn model tốt nhất (dựa trên R2)
        best_run = max(self.results_history, key=lambda x: x['R2'])
        self.best_model_name = best_run['Model']
        self.best_model = self.models[self.best_model_name]

        logger.info(f"--- KẾT THÚC AUTO RUN. MODEL TỐT NHẤT: {self.best_model_name} (R2={best_run['R2']}) ---")

        # Lưu kết quả ra file JSON
        json_path = os.path.join(self.output_dir, 'experiment_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results_history, f, indent=4)
        logger.info(f"Đã lưu kết quả JSON: {json_path}")

        # Vẽ biểu đồ so sánh
        self._plot_comparison()
        

    def _plot_comparison(self):
        # Vẽ biểu đồ so sánh hiệu năng các mô hình
        results_df = pd.DataFrame(self.results_history)

        if results_df.empty:
            logger.warning("Chưa có kết quả nào để vẽ biểu đồ.")
            return

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Model', y='R2', data=results_df, palette='viridis', hue='Model', legend=False)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3, fontsize=10, fontweight='bold')

        plt.title('So sánh R2 Score giữa các Mô hình', fontsize=14)
        plt.ylabel('R2 Score')
        plt.ylim(0, 1.15)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(save_path)
        plt.close()
        logger.info("Đã lưu biểu đồ so sánh: model_comparison.png")

    def save_model(self, filename: str = None):
        """Lưu model tốt nhất."""
        if self.best_model is None:
            logger.warning("Chưa có model tốt nhất để lưu.")
            return

        if filename is None:
            filename = f"best_model_{self.best_model_name}.pkl"

        filename = os.path.basename(filename) 
        save_path = os.path.join(self.output_dir, filename)

        joblib.dump(self.best_model, save_path)
        logger.info(f"Đã lưu model tốt nhất vào: {save_path}")

    def load_model(self, filename: str):
        """Nạp model từ file."""
        try:
            self.best_model = joblib.load(filename)
            logger.info(f"Đã nạp model từ {filename}")
        except Exception as e:
            logger.error(f"Lỗi nạp model: {e}")


            