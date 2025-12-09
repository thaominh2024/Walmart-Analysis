import pandas as pd
import numpy as np
import re
import os
import logging
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.scalers = {}

        # Khai báo cột cần onehot
        self.one_hot_cols = ['season','store']

        # Khai báo cột cần scaling
        self.scale_cols = [
            'temperature', 'fuel_price', 'cpi', 'unemployment',
            'month_sin', 'month_cos', 'day_sin', 'day_cos'
        ]

    def __repr__(self):
        return f"DataPreprocessor(file='{os.path.basename(self.file_path)}')"

    @staticmethod
    def to_snake_case(name: str) -> str:
        """Chuyển tên cột sang snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        return s2.replace('__', '_')

    def read_data(self) -> pd.DataFrame:
        try:
            _, ext = os.path.splitext(self.file_path)
            if ext == '.csv':
                self.df = pd.read_csv(self.file_path)
            elif ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError(f"Không hỗ trợ định dạng {ext}")

            self.df.columns = [self.to_snake_case(col) for col in self.df.columns]
            logger.info(f"Đọc dữ liệu thành công. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Lỗi đọc file: {e}")
            raise

    def handle_missing_values(self):
        """
        Xử lý giá trị thiếu theo logic:
        - Temperature: Mean
        - CPI, Unemployment: Forward Fill (vì là chỉ số kinh tế theo thời gian)
        """
        if self.df is None: return
        try:
            # Temperature -> Mean
            if 'temperature' in self.df.columns:
                mean_temp = self.df['temperature'].mean()
                self.df['temperature'] = self.df['temperature'].fillna(mean_temp)

            # CPI & Unemployment -> Forward Fill
            cols_ffill = ['cpi', 'unemployment', 'fuel_price']
            for col in cols_ffill:
                if col in self.df.columns:
                    self.df[col] = self.df[col].ffill()

            logger.info("Đã xử lý missing values (Temp: Mean; CPI/Unemp: Ffill).")
        except Exception as e:
            logger.error(f"Lỗi xử lý missing values: {e}")

    def handle_outliers(self, method: str = 'iqr'):
        if self.df is None: return

        # Các cột cần loại Outlier
        target_cols = ['weekly_sales', 'temperature', 'unemployment']
        cols = [c for c in target_cols if c in self.df.columns]

        try:
            if method == 'iqr':
                for col in cols:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR

                    # Capping (Kẹp biên)
                    self.df[col] = np.where(self.df[col] < lower, lower, self.df[col])
                    self.df[col] = np.where(self.df[col] > upper, upper, self.df[col])

            elif method == 'isolation_forest':
                iso_cols = [c for c in cols if c != 'weekly_sales']
                if iso_cols:
                    iso = IsolationForest(contamination=0.01, random_state=42)
                    preds = iso.fit_predict(self.df[iso_cols])
                    self.df = self.df[preds == 1]

            logger.info(f"Đã xử lý ngoại lai ({method}).")
        except Exception as e:
            logger.error(f"Lỗi xử lý ngoại lai: {e}")

    def feature_engineering(self):
        """Feature Engineering chuyên biệt cho Sales & Time Series."""
        if self.df is None: return
        try:
            # chuyển cột 'date' về đúng dạng datetime
            if 'date' in self.df.columns:
                # convert mixed format, dayfirst
                self.df['date'] = pd.to_datetime(self.df['date'], format='mixed', dayfirst=True)

                # Trích xuất ra day, month, year từ date
                self.df['day'] = self.df['date'].dt.day
                self.df['month'] = self.df['date'].dt.month
                self.df['year'] = self.df['date'].dt.year

                # Tạo Season
                def get_season(m):
                    if 3 <= m <= 5: return 'Spring'
                    elif 6 <= m <= 8: return 'Summer'
                    elif 9 <= m <= 11: return 'Autumn'
                    else: return 'Winter'
                self.df['season'] = self.df['month'].apply(get_season)

                """Mã hóa tuần hoàn
                (do tháng cuối năm này gần với tháng đầu năm sau,
                ngày cuối tháng này gần với ngày đầu tháng sau)
                """
                # Month
                self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
                self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
                # Day
                self.df['day_sin'] = np.sin(2 * np.pi * self.df['day'] / 31)
                self.df['day_cos'] = np.cos(2 * np.pi * self.df['day'] / 31)

            logger.info("Hoàn thành Feature Engineering (Date Cyclical + Season).")
        except Exception as e:
            logger.error(f"Lỗi Feature Engineering: {e}")

    def encode(self):
        if self.df is None: return
        try:
            cols_oh = [c for c in self.one_hot_cols if c in self.df.columns]
            if cols_oh:
                self.df = pd.get_dummies(self.df, columns=cols_oh, dtype=int)
                logger.info(f"Đã One-Hot Encoding: {cols_oh}")
            # Xóa các cột không cần thiết nữa
            drop_cols = ['date', 'day', 'month'] # Giữ lại year, sin/cos
            self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], inplace=True)

        except Exception as e:
            logger.error(f"Lỗi Encode: {e}")

    def process(self, output_path: str = None) -> pd.DataFrame:
        """Pipeline chạy toàn bộ."""
        self.read_data()
        self.handle_missing_values()
        self.handle_outliers()
        self.feature_engineering()
        self.encode()

        if output_path:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Đã lưu file: {output_path}")
        return self.df