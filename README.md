# 📊 Walmart Sales Analysis 
Dự án Machine Learning dự báo doanh số bán hàng hàng tuần cho chuỗi cửa hàng Walmart, giúp tối ưu hóa hàng tồn kho và chiến lược kinh doanh.

## 📖 Giới thiệu (Overview)
### **DATASET**: https://www.kaggle.com/datasets/mikhail1681/walmart-sales

📈 Dự báo Doanh số (Sales Prediction)
Sử dụng các mô hình Machine Learning  (**Linear Regression, Random Forest, XGBoost**) để dự đoán `Weekly_Sales` dựa trên các yếu tố vĩ mô:
* 🌡️ **Nhiệt độ (Temperature)**
* ⛽ **Giá nhiên liệu (Fuel Price)**
* 📉 **Chỉ số CPI & Thất nghiệp**
* 🎉 **Các ngày lễ lớn (Holiday Flag)**


---

## 📂 Cấu trúc dự án (Project Structure)

```text
Walmart-Analysis/
│
├── data/                          # DỮ LIỆU ĐẦU VÀO
│   ├── walmart2010.csv            # Dữ liệu doanh số bán hàng
│   └── processed_data.csv         # Dữ liệu sau khi được xử lý 
│
├── EDA_analysis/                  # Thư mục chứa hình vẽ và summary cho quá trình EDA
│
├── src/                           # MÃ NGUỒN XỬ LÝ (CORE MODULES)
│   ├── preprocessing.py           # Tiền xử lý dữ liệu Sales
│   ├── model.py                   # Huấn luyện mô hình Machine Learning
│   ├── eda.py                     # Vẽ biểu đồ phân tích (EDA)
│
├── output/                        # KẾT QUẢ ĐẦU RA (AUTO-GENERATED)
│   ├── experiment_results.json    # Kết quả so sánh độ chính xác Model
│   └── best_sales_model.pkl       # File model tốt nhất đã lưu
│  
├── logs/                          # File nhật ký chạy (Logs)
│  
├── requirements.txt               # Các thư viện cần thiết
│  
└── README.md                      # Tài liệu hướng dẫn
```

## 🛠️ Cài đặt & Chạy (Installation & Usage)
**Bước 1: Cài đặt môi trường**
```bash
# Clone dự án
git clone https://github.com/thaominh2024/Walmart-Analysis.git
```
```bash
cd Walmart-Analysis
```
**Bước 2: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```
**Bước 3: Thực thi chương trình**
```bash
python main.py
```