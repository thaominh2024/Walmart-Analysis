from src.preprocessing import DataPreprocessor
from src.model import ModelTrainer
from src.eda import EDAAnalyzer 

def main():
    print("============== BẮT ĐẦU CHƯƠNG TRÌNH ==============")
    print("=== BƯỚC 1: ĐỌC DỮ LIỆU GỐC ===")
    
    raw_processor = DataPreprocessor(file_path='data/walmart2010.csv')
    raw_data = raw_processor.read_data() 

    print("=== BƯỚC 2: THỰC HIỆN EDA (KHÁM PHÁ DỮ LIỆU) ===")
    eda = EDAAnalyzer(raw_data, output_dir='EDA_analysis')
    eda.print_summary()        
    eda.plot_histograms()      
    eda.plot_density()           
    eda.analyze_zscore()         
    eda.plot_boxplots_4_features()
    eda.plot_correlation_heatmap('correlation.png')
    
    print("=== BƯỚC 3: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING) ===")
    raw_processor.process(output_path='data/processed_data.csv')
    
    print("=== BƯỚC 4: HUẤN LUYỆN MÔ HÌNH ===")
    trainer = ModelTrainer()
    trainer.load_data(file_path='data/processed_data.csv')
    trainer.split_data(target_col='weekly_sales') 
    trainer.auto_run(models_to_try=['linear', 'rf', 'xgboost'], is_optimize=True)
    trainer.save_model('best_sales_model.pkl')
    print("============== KẾT THÚC CHƯƠNG TRÌNH ==============")
if __name__ == "__main__":
    main()