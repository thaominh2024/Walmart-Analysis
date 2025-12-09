import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import zscore
import logging


logger = logging.getLogger(__name__)

class EDAAnalyzer:
    def __init__(self, data, output_dir='EDA_analysis'):
        """
        Khởi tạo EDAAnalyzer.
        """
        self.data = data.copy()
        self.output_dir = output_dir
        
        sns.set(style="whitegrid")
        self.numerical_cols = self.data.select_dtypes(include=['number']).columns.tolist()

    def _save_plot(self, filename):
        """Hàm nội bộ để lưu ảnh"""
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Đã lưu biểu đồ: {path}")

    def print_summary(self, filename='data_summary.txt'):
        """
        Lưu các thông tin cơ bản (Info, Describe, Null) vào file txt trong thư mục output
        """
        file_path = os.path.join(self.output_dir, filename)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # 1. Ghi thông tin Shape và Columns
                f.write("========== BÁO CÁO TỔNG QUAN DỮ LIỆU ==========\n\n")
                f.write("--- 1. THÔNG TIN CƠ BẢN (INFO) ---\n")
                f.write(f"Kích thước (Shape): {self.data.shape}\n")
                f.write(f"Danh sách cột: {self.data.columns.tolist()}\n\n")
                
                # 2. Ghi bảng thống kê mô tả (Describe)
                f.write("--- 2. THỐNG KÊ MÔ TẢ (DESCRIBE) ---\n")
                f.write(self.data.describe().T.to_string())
                f.write("\n\n")
                
                # 3. Ghi thông tin giá trị khuyết (Null)
                f.write("--- 3. GIÁ TRỊ KHUYẾT (NULL) ---\n")
                null_counts = self.data.isnull().sum()
                if null_counts.sum() > 0:
                    f.write(null_counts[null_counts > 0].to_string())
                else:
                    f.write("Dữ liệu sạch, không có giá trị khuyết (Null).")
                f.write("\n")

            logger.info(f"Đã lưu file báo cáo tóm tắt tại: {file_path}")
            
        except Exception as e:
            logger.error(f"Lỗi khi ghi file summary: {e}")
            

    def plot_histograms(self, output_name='numerical_histograms.png'):
        """Vẽ Histogram cho tất cả các biến số """
        if not self.numerical_cols:
            return

        self.data[self.numerical_cols].hist(figsize=(12, 12), bins=20, edgecolor='black')
        plt.suptitle('Distribution of Numerical Features', fontsize=20, fontweight="bold", color='red')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_plot(output_name)

    def plot_density(self, output_name='numerical_density.png'):
        """Vẽ biểu đồ mật độ (Density plot)"""
        if not self.numerical_cols:
            return

        n_cols = 3
        n_rows = (len(self.numerical_cols) - 1) // n_cols + 1
        
        axes = self.data[self.numerical_cols].plot(
            kind='density',
            figsize=(12, 4 * n_rows),
            subplots=True,
            layout=(n_rows, n_cols),
            sharex=False,
            fontsize=12
        )

        for ax in axes.flatten():
            if ax is not None:
                ax.set_ylabel("Density", fontsize=12, fontweight='bold')
                ax.tick_params(axis='y', labelsize=10)

        plt.suptitle('Density plot of Numerical Features', fontsize=18, fontweight="bold", color='red')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_plot(output_name)

    def analyze_zscore(self):
        """Tính và in ra Z-score min/max"""
        if not self.numerical_cols:
            return

        logger.info("--- PHÂN TÍCH Z-SCORE ---")
        try:
            df_zscore = self.data[self.numerical_cols].apply(zscore, nan_policy='omit')
            for col in df_zscore.columns:
                min_val = df_zscore[col].min()
                max_val = df_zscore[col].max()
                logger.info(f"{col}: [{round(min_val, 2)} , {round(max_val, 2)}]")
        except Exception as e:
            logger.warning(f"Không thể tính Z-score (có thể do dữ liệu constant hoặc lỗi khác): {e}")

    def plot_boxplots_4_features(self, output_name='selected_boxplots.png'):
        """
        Vẽ Boxplot cho 4 cột đặc thù: weekly_sales, holiday_flag, temperature, unemployment
        """
        target_cols = ['weekly_sales', 'holiday_flag', 'temperature', 'unemployment']
        
        # Kiểm tra xem data có đủ 4 cột này không
        missing_cols = [c for c in target_cols if c not in self.data.columns]
        if missing_cols:
            logger.warning(f"Thiếu các cột sau để vẽ Boxplot 4 ô: {missing_cols}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i, col in enumerate(target_cols):
            sns.boxplot(x=self.data[col], ax=axes[i])
            axes[i].set_title(col.replace('_', ' ').title()) 

        plt.suptitle('Boxplots of Selected Numerical Features', fontsize=20, fontweight="bold", color='red')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_plot(output_name)

    def plot_correlation_heatmap(self, output_name='correlation_heatmap.png'):
        """Vẽ Heatmap tương quan """
        if not self.numerical_cols:
            return

        plt.figure(figsize=(10, 8))
        correlation = self.data[self.numerical_cols].corr(method='pearson')
        sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        self._save_plot(output_name)