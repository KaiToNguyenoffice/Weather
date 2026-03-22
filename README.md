# Hệ Thống Dự Báo Thời Tiết — Apache Spark

## Dataset

**WEATHER-5K** — 5,672 trạm thời tiết toàn cầu, dữ liệu theo giờ từ 2014–2023.

| Cột | Nội dung | Đơn vị |
|-----|----------|--------|
| `DATE` | Thời điểm đo | datetime |
| `TMP` | Nhiệt độ | °C |
| `DEW` | Điểm sương | °C |
| `WND_ANGLE` | Hướng gió | degrees |
| `WND_RATE` | Tốc độ gió | m/s |
| `SLP` | Áp suất mực nước biển | hPa |
| `MASK` | Dữ liệu hợp lệ | [1 1 1 1 1] |
| `LONGITUDE/LATITUDE` | Tọa độ trạm | degrees |

- **~87,000 dòng/trạm × 5,672 trạm = ~493 triệu quan sát**
- Pipeline mặc định sample **50 trạm** (~4.3 triệu dòng) để chạy local
- Metadata: `meta_info.json` (vị trí, elevation, valid %), `percentile.json` (phân phối thống kê)

## Bài toán

| Task | Target | Mô tả |
|------|--------|-------|
| Classification | `Rain_next` | Giờ tiếp theo có mưa không? (Dew Point Depression < 2.5°C) |
| Regression | `Temp_next` | Nhiệt độ giờ tiếp theo (°C) |

## Pipeline

```
WEATHER-5K CSVs → Sample N stations → Filter MASK → Union
→ Preprocessing → Feature Engineering → Train/Test Split
→ Model Training → Evaluation → metrics.json
```

**Checkpoint:** Lần 1 train + lưu. Lần 2+ tự load, không train lại.

## Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu: Python 3.8+, Java 8/11 (cho PySpark).

## Chạy

### Command line

```bash
cd weather_forecast
python src/pipeline.py          # chạy pipeline (50 trạm)
python src/pipeline.py --force  # xóa checkpoint, train lại
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

### Cấu hình

Sửa `src/config.py`:
- `NUM_STATIONS_SAMPLE = 50` — số trạm sample (tăng nếu có GPU/cluster)
- `MIN_VALID_PERCENT = 0.8` — ngưỡng dữ liệu hợp lệ
- `DEW_POINT_DEPRESSION_THRESHOLD = 2.5` — ngưỡng phân loại mưa

## Cấu trúc

```
weather_forecast/
├── data/
│   └── WEATHER-5K/
│       ├── global_weather_stations/   # 5,672 CSV files
│       ├── meta_info.json             # Metadata trạm
│       └── percentile.json            # Phân phối thống kê
├── checkpoints_weather5k/             # Tự tạo khi chạy
│   ├── processed_data.parquet/
│   ├── train.parquet/
│   ├── test.parquet/
│   ├── models/
│   └── metrics.json
├── src/
│   ├── config.py
│   ├── data_loader.py                 # Load WEATHER-5K, sample, union
│   ├── preprocessing.py               # Rain label (DPD), targets
│   ├── feature_engineering.py         # Lag/rolling/time/derived
│   ├── model_training.py              # 6 models + checkpoint
│   ├── evaluation.py                  # Metrics
│   └── pipeline.py                    # Main entry point
├── dashboard/
│   └── app.py
├── DEV_LOG.md
└── requirements.txt
```

## Models

**Classification (Rain_next):**
- Logistic Regression
- Random Forest Classifier
- GBT Classifier

**Regression (Temp_next):**
- Linear Regression
- Random Forest Regressor
- GBT Regressor
