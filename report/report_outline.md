# BÁO CÁO ĐỒ ÁN
## Xây Dựng Hệ Thống Dự Báo Thời Tiết Sử Dụng Apache Spark

**Môn học:** Các công cụ và nền tảng Trí tuệ nhân tạo
**Sinh viên:** [Tên sinh viên]
**MSSV:** [Mã số sinh viên]
**GVHD:** [Tên giảng viên]

---

## MỤC LỤC

1. Giới thiệu
2. Cơ sở lý thuyết
3. Phân tích bài toán
4. Thiết kế hệ thống
5. Cài đặt và thực nghiệm
6. Kết quả và đánh giá
7. Kết luận
8. Tài liệu tham khảo

---

## CHƯƠNG 1: GIỚI THIỆU

### 1.1 Đặt vấn đề
- Tầm quan trọng của dự báo thời tiết trong đời sống (nông nghiệp, giao thông, phòng chống thiên tai)
- Lượng dữ liệu thời tiết toàn cầu tăng nhanh → cần công cụ Big Data
- Ứng dụng Machine Learning để tự động hóa và nâng cao độ chính xác dự báo

### 1.2 Mục tiêu đồ án
- Xây dựng hệ thống dự báo thời tiết sử dụng PySpark và Spark MLlib
- Thực hiện 2 bài toán:
  - **Classification:** Dự đoán giờ tới có mưa không (Rain_next)
  - **Regression:** Dự đoán nhiệt độ giờ tới (Temp_next, °C)
- Xây dựng pipeline hoàn chỉnh từ dữ liệu thô → feature engineering → training → evaluation → dashboard

### 1.3 Phạm vi đồ án
- **Dataset:** WEATHER-5K — 5,672 trạm thời tiết toàn cầu, dữ liệu hourly 2014–2023
- **Sampling:** 200 trạm (cấu hình `NUM_STATIONS_SAMPLE`), ~10.8M dòng
- **Công cụ:** Apache Spark 3.5, PySpark, Spark MLlib, Streamlit
- **Models:** Logistic Regression, Random Forest, Gradient Boosted Trees (Classification + Regression)

### 1.4 Bố cục báo cáo
- Chương 1: Giới thiệu, mục tiêu, phạm vi
- Chương 2: Cơ sở lý thuyết (Spark, MLlib, thuật toán)
- Chương 3: Phân tích bài toán và dataset
- Chương 4: Thiết kế hệ thống (pipeline architecture)
- Chương 5: Cài đặt và thực nghiệm
- Chương 6: Kết quả và đánh giá
- Chương 7: Kết luận và hướng phát triển

---

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

### 2.1 Big Data
- Định nghĩa Big Data (5V: Volume, Velocity, Variety, Veracity, Value)
- Thách thức xử lý dữ liệu lớn trong lĩnh vực khí tượng

### 2.2 Apache Spark
- Kiến trúc Spark (Driver, Executor, Cluster Manager)
- So sánh Spark vs MapReduce (in-memory processing, DAG execution)
- Spark SQL và DataFrame API
- PySpark: Python API cho Spark

### 2.3 Spark MLlib
- Giới thiệu thư viện ML của Spark
- Pipeline API: Transformer → Estimator → Pipeline
- Các giai đoạn: StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

### 2.4 Các thuật toán Machine Learning

#### 2.4.1 Logistic Regression
- Nguyên lý: hàm sigmoid để ước lượng xác suất
- Ưu điểm: đơn giản, nhanh, dễ giải thích
- Nhược điểm: giả định quan hệ tuyến tính giữa features và log-odds

#### 2.4.2 Random Forest
- Nguyên lý: tập hợp nhiều Decision Tree (ensemble)
- Bagging (Bootstrap Aggregating) để giảm variance
- Ưu điểm: giảm overfitting, xử lý non-linear, feature importance
- Nhược điểm: chậm hơn, model lớn, khó giải thích

#### 2.4.3 Gradient Boosted Trees (GBT)
- Nguyên lý: boosting tuần tự — mỗi cây học từ sai số cây trước
- So sánh bagging (RF) vs boosting (GBT)
- Ưu điểm: hiệu suất cao, xử lý tốt dữ liệu phức tạp
- Nhược điểm: dễ overfit nếu không tune, chậm hơn RF

#### 2.4.4 Linear Regression
- Nguyên lý: tối thiểu hóa hàm loss (MSE) bằng gradient descent
- Ứng dụng: dự đoán giá trị liên tục (nhiệt độ)

### 2.5 Các metric đánh giá

#### Classification
| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Tỷ lệ dự đoán đúng tổng thể |
| Precision | TP/(TP+FP) | Trong các dự đoán positive, bao nhiêu đúng |
| Recall | TP/(TP+FN) | Trong các actual positive, bao nhiêu được phát hiện |
| F1-Score | 2·P·R/(P+R) | Trung bình điều hòa của Precision và Recall |
| AUC-ROC | Diện tích dưới đường ROC | Khả năng phân biệt hai lớp |

#### Regression
| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| RMSE | √(Σ(ŷ-y)²/n) | Sai số trung bình (cùng đơn vị với target) |
| MAE | Σ|ŷ-y|/n | Sai số tuyệt đối trung bình |
| R² | 1 - SS_res/SS_tot | Tỷ lệ variance được giải thích bởi model |

---

## CHƯƠNG 3: PHÂN TÍCH BÀI TOÁN

### 3.1 Mô tả dataset WEATHER-5K
- **Nguồn:** WEATHER-5K (dữ liệu khí tượng toàn cầu)
- **Quy mô:** 5,672 trạm × ~87,000 dòng/trạm ≈ 493 triệu quan sát (full)
- **Thời gian:** 2014–2023, tần suất 1 giờ (hourly)
- **Sampling:** Lấy mẫu 200 trạm có `valid_percent >= 0.8` → ~10.8M dòng
- **Metadata:** `meta_info.json` chứa latitude, longitude, ELEVATION, valid_percent

### 3.2 Mô tả các biến dữ liệu

| Cột gốc (CSV) | Tên sau rename | Kiểu | Mô tả |
|----------------|---------------|------|-------|
| DATE | datetime | Timestamp | Thời điểm đo (hourly) |
| TMP | Temperature | Double | Nhiệt độ (°C) |
| DEW | DewPoint | Double | Điểm sương (°C) |
| WND_ANGLE | WindDirection | Double | Hướng gió (°) |
| WND_RATE | WindSpeed | Double | Tốc độ gió (m/s) |
| SLP | Pressure | Double | Áp suất mực nước biển (hPa) |
| MASK | — | String | Validity mask `[1 1 1 1 1]` = hợp lệ |

### 3.3 Xác định bài toán

**Bài toán 1 — Classification (Rain_next):**
- **Input:** Các đặc trưng thời tiết tại thời điểm t
- **Output:** Giờ t+1 có mưa không? (0/1)
- **Label:** Dew Point Depression (DPD) < 2.5°C → Rain = 1
  - DPD = Temperature - DewPoint
  - Cơ sở khí tượng: DPD nhỏ → không khí bão hòa → ngưng tụ hơi nước → mưa

**Bài toán 2 — Regression (Temp_next):**
- **Input:** Các đặc trưng thời tiết tại thời điểm t
- **Output:** Nhiệt độ (°C) tại thời điểm t+1
- **Label:** `lead(Temperature, 1)` — shift 1 giờ theo StationID

### 3.4 Phân tích tính phù hợp với Big Data và ML
- Dữ liệu lớn (10.8M+ dòng sampled, 493M full) → Spark xử lý phân tán hiệu quả
- 5,672 trạm toàn cầu → đa dạng pattern khí hậu
- Time series hourly → cần lag/rolling features phức tạp
- Checkpointing cho phép tái sử dụng kết quả trung gian

---

## CHƯƠNG 4: THIẾT KẾ HỆ THỐNG

### 4.1 Kiến trúc tổng quan
```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│  WEATHER-5K: 5,672 station CSVs (2014–2023, hourly)                │
│  Columns: DATE, TMP, DEW, WND_ANGLE, WND_RATE, SLP, MASK          │
│                    ↓ Sample 200 stations                            │
│                    ↓ Filter MASK = [1 1 1 1 1]                     │
│                    ↓ Union all → ~10.8M rows × 7 columns           │
└─────────────────────────┬───────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                                  │
│  ✓ Rain label (DewPoint Depression < 2.5°C → Rain = 1)            │
│  ✓ Targets: Rain_next, Temp_next (lead 1 hour per station)        │
│  ✓ Fill missing (median per column)                                │
└─────────────────────────┬───────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                               │
│  ✓ Time: Hour, Month, DayOfWeek, DayOfYear + sin/cos encoding    │
│  ✓ Lag: Temp 1/3/6/24h, DewPoint/Pressure/Wind 1h, Rain 1/3h    │
│  ✓ Rolling: avg 6h/24h, Rain_sum 6h/24h                          │
│  ✓ Derived: TempChange, DewPointChange, PressureChange            │
│  Total: 35 feature columns                                        │
└─────────────────────────┬───────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE (Spark MLlib)                       │
│  StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler │
│                              ↓                                     │
│  Classification          Regression                                │
│  ├─ LogisticRegression   ├─ LinearRegression                      │
│  ├─ RandomForest         ├─ RandomForest                          │
│  └─ GBTClassifier        └─ GBTRegressor                          │
└─────────────────────────┬───────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION + DASHBOARD                         │
│  Classification: Accuracy, Precision, Recall, F1, AUC-ROC         │
│  Regression:     RMSE, MAE, R²                                    │
│  Dashboard:      Streamlit + Plotly (interactive map, charts)      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Cấu trúc project
```
weather_forecast/
├── data/WEATHER-5K/
│   ├── global_weather_stations/     # 5,672 CSV files
│   ├── meta_info.json               # Metadata (lat, lon, elevation, valid%)
│   └── percentile.json              # Phân vị thống kê
├── src/
│   ├── config.py                    # Config (paths, sampling, thresholds)
│   ├── data_loader.py               # Load + sample stations
│   ├── preprocessing.py             # Clean, label, fill missing
│   ├── feature_engineering.py       # Time, lag, rolling, derived features
│   ├── model_training.py            # Train, evaluate, checkpoint
│   └── pipeline.py                  # Orchestrator chạy toàn bộ
├── dashboard/app.py                 # Streamlit dashboard
├── notebooks/weather_forecast.ipynb # Notebook all-in-one
├── checkpoints_weather5k/           # Parquet checkpoints + models + metrics
└── report/                          # Báo cáo
```

### 4.3 Data Ingestion
- Đọc `meta_info.json` → lọc trạm `valid_percent >= 0.8`
- Random sample 200 trạm → load từng CSV → union tất cả
- Rename cột: TMP→Temperature, DEW→DewPoint, WND_ANGLE→WindDirection, WND_RATE→WindSpeed, SLP→Pressure

### 4.4 Preprocessing
- Parse `DATE` thành timestamp
- Tạo nhãn mưa: `Rain = 1 if (Temperature - DewPoint) < 2.5 else 0`
- Tạo targets: `Rain_next = lead(Rain, 1)`, `Temp_next = lead(Temperature, 1)` — partition by StationID, order by datetime
- Fill null bằng median cho 5 cột numeric

### 4.5 Feature Engineering (35 features)

| Loại | Features | Số lượng |
|------|----------|----------|
| Numeric | Temperature, DewPoint, Pressure, WindSpeed, WindDirection | 5 |
| Time | Hour, Month, DayOfWeek, DayOfYear + sin/cos encoding | 8 |
| Lag | Temp 1/3/6/24h, DewPoint/Pressure/Wind 1h, Rain 1/3h | 9 |
| Rolling | Temp/DewPoint/Pressure avg 6h/24h, Rain_sum 6h/24h | 6 |
| Derived | TempChange, DewPointChange, PressureChange, TempDewPoint | 4 |
| Categorical | StationID (StringIndexer → OneHotEncoder) | 1 |
| **Tổng** | | **35** |

### 4.6 ML Pipeline (Spark MLlib)
- **Stages:** StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler → Model
- **Train/Test split:** 80/20 random split
- **Checkpointing:** Lưu data/model/metrics dưới dạng Parquet → tái sử dụng lần chạy sau

---

## CHƯƠNG 5: CÀI ĐẶT VÀ THỰC NGHIỆM

### 5.1 Môi trường
- **Local:** Python 3.10, PySpark 3.5.4, Java 8/11, MSI Thin GF63 12UC
- **Cloud:** Google Colab (GPU T4, ~15GB RAM) — chạy nhiều trạm hơn
- **Dashboard:** Streamlit, Plotly, Matplotlib, Pandas

### 5.2 Cài đặt
- `pip install pyspark streamlit plotly pandas matplotlib`
- Hadoop winutils (required cho Windows): đặt tại `C:\hadoop\bin\winutils.exe`
- Spark driver memory: 8GB (`SPARK_DRIVER_MEMORY = "8g"`)

### 5.3 Cấu hình chính (`config.py`)
```python
NUM_STATIONS_SAMPLE = 200     # Số trạm lấy mẫu
MIN_VALID_PERCENT = 0.8       # Lọc trạm >= 80% valid
SPARK_DRIVER_MEMORY = "8g"    # Bộ nhớ driver
DPD_RAIN_THRESHOLD = 2.5      # Ngưỡng DPD cho nhãn mưa
```

### 5.4 Chạy Pipeline
```bash
# Chạy pipeline (tự checkpoint)
python src/pipeline.py

# Chạy lại từ đầu (xóa checkpoints)
python src/pipeline.py --force

# Chạy dashboard
streamlit run dashboard/app.py
```

### 5.5 Notebook
- `notebooks/weather_forecast.ipynb`: chạy all-in-one trên Colab hoặc local
- Bao gồm EDA, preprocessing, FE, training, evaluation, visualization

---

## CHƯƠNG 6: KẾT QUẢ VÀ ĐÁNH GIÁ

### 6.1 Thống kê dữ liệu sau xử lý
- **Tổng dòng:** ~2.67M (50 trạm) / ~10.8M (200 trạm)
- **Rain positive ratio:** 29.9% (DPD < 2.5°C)
- **Features:** 35 cột → 40 cột sau FE (drop NaN giảm ~1,200 dòng)
- **Train/Test:** 80%/20%

### 6.2 Kết quả Classification (50 trạm)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | 0.9032 | 0.9031 | 0.9032 | 0.9031 | 0.9552 |
| Random Forest | 0.9093 | 0.9090 | 0.9093 | 0.9091 | 0.9633 |
| **GBT Classifier** | **0.9122** | **0.9120** | **0.9122** | **0.9121** | **0.9652** |

- **Best:** GBT Classifier (AUC = 0.9652)
- Confusion Matrix (GBT): TP=135,457 | FP=22,910 | FN=23,987 | TN=351,560

### 6.3 Kết quả Regression (50 trạm)

| Model | RMSE (°C) | MAE (°C) | R² |
|-------|-----------|----------|-----|
| Linear Regression | 1.6979 | 1.0733 | 0.9803 |
| Random Forest | 1.7021 | 1.0749 | 0.9802 |
| **GBT Regressor** | **1.6429** | **1.0687** | **0.9815** |

- **Best:** GBT Regressor (RMSE = 1.6429°C, R² = 0.9815)
- Sai số trung bình chỉ ~1.07°C — chính xác cao

### 6.4 So sánh và phân tích
- **GBT > RF > LR** cho cả 2 bài toán → boosting ensemble vượt trội
- Rain detection đạt 91.2% accuracy — DPD là proxy hiệu quả
- R² = 0.9815 → model giải thích được 98.15% variance của nhiệt độ
- RMSE ~1.64°C — sai số nhỏ, phù hợp ứng dụng thực tế

### 6.5 Hạn chế
- Nhãn Rain dựa trên DPD (proxy), chưa phải ground truth precipitation
- Chưa tối ưu hyperparameter (default PySpark params)
- Sampling 200/5672 trạm → chưa tận dụng hết dữ liệu
- Chỉ dự đoán 1 giờ tới → chưa dự đoán dài hạn

---

## CHƯƠNG 7: KẾT LUẬN

### 7.1 Kết quả đạt được
- Xây dựng pipeline dự báo thời tiết hoàn chỉnh với PySpark + Spark MLlib
- Sử dụng dataset WEATHER-5K (5,672 trạm toàn cầu, 2014–2023)
- So sánh 3 thuật toán cho Classification (best AUC=0.9652) và Regression (best R²=0.9815)
- Feature engineering 35 features (time, lag, rolling, derived) cải thiện kết quả đáng kể
- Hệ thống checkpointing giúp tái sử dụng kết quả trung gian
- Dashboard Streamlit trực quan với bản đồ interactive, biểu đồ so sánh models

### 7.2 Hướng phát triển
- Áp dụng Spark Structured Streaming cho dữ liệu thời gian thực
- Tích hợp API thời tiết (OpenWeatherMap, WeatherAPI)
- Thử nghiệm Deep Learning: LSTM/Transformer cho time series
- Hyperparameter tuning với CrossValidator
- Mở rộng sang dự đoán nhiều giờ tới (multi-step forecasting)
- Triển khai web service (FastAPI + Docker)
- Sử dụng toàn bộ 5,672 trạm trên Spark cluster

---

## TÀI LIỆU THAM KHẢO

1. Apache Spark Documentation — https://spark.apache.org/docs/latest/
2. PySpark MLlib Guide — https://spark.apache.org/docs/latest/ml-guide.html
3. WEATHER-5K Dataset — Dữ liệu khí tượng toàn cầu, 5672 trạm
4. Zaharia, M. et al. "Apache Spark: A Unified Engine for Big Data Processing." Communications of the ACM, 2016.
5. Hastie, T., Tibshirani, R., Friedman, J. "The Elements of Statistical Learning." Springer, 2009.
6. Lawrence, M.G. "The Relationship between Relative Humidity and the Dewpoint Temperature." BAMS, 2005. (Cơ sở DPD → Rain detection)
7. Streamlit Documentation — https://docs.streamlit.io/
