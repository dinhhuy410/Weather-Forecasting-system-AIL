# Weather Forecasting — Hà Nội

Dự án thu thập dữ liệu thời tiết theo giờ từ Open-Meteo, phân tích EDA, chọn feature và dự báo nhiệt độ một giờ tiếp theo bằng Random Forest, XGBoost, LSTM và GRU.

## Cấu trúc và vai trò

```text
.
├── README.md
├── requirements.txt
├── requirements-dl.txt
├── crawl_data/
│   ├── crawl_hanoi_enriched_weather.py
│   └── crawl_test_set.py
├── src/
│   ├── inspect_data.py
│   └── weather_data.py
├── data/
│   ├── raw/
│   │   ├── Hanoi.csv
│   │   └── test/Hanoi_test_set.csv
│   └── processed/
│       ├── Hanoi_final.csv
│       └── Hanoi_test_final.csv
├── data_processing/
│   ├── clean_raw.ipynb
│   ├── prepare_test_set.ipynb
│   └── feature_selection.json
├── eda/
│   ├── data_analyze.ipynb
│   └── eda_visualization.ipynb
├── tests/
│   └── test_pipeline.py
└── train_model/
    ├── ml_models.ipynb
    ├── dl_models.ipynb
    ├── evaluate_2026_test_set.ipynb
    └── artifacts/                  # model, metrics và predictions theo lượt chạy
```

`.gitignore` và `.gitattributes` giữ cấu hình bỏ qua file sinh tự động và Git LFS cho CSV raw.

## Bốn file trong data

| File | Kích thước hiện tại | Mục đích |
|---|---|---|
| `data/raw/Hanoi.csv` | 140.256 dòng × 34 cột | Dữ liệu gốc 2010–2025; đầu vào EDA và chọn feature. |
| `data/raw/test/Hanoi_test_set.csv` | 6.528 dòng × 34 cột | Dữ liệu gốc 2026-01-01 đến 2026-09-29; dành cho đánh giá ngoài tập huấn luyện. |
| `data/processed/Hanoi_final.csv` | 140.256 dòng × 13 cột | Dữ liệu sạch đã chọn feature, dùng để chuẩn bị tập train. |
| `data/processed/Hanoi_test_final.csv` | 6.528 dòng × 13 cột | Dữ liệu test được xử lý theo đúng schema train. |

Hai file processed chứa **9 feature và 4 cột thông tin** (`time`, `city`, `latitude`, `longitude`). Các notebook tạo nhãn `temperature_2m` của giờ tiếp theo khi chuẩn bị dữ liệu huấn luyện. Nhiệt độ hiện tại là một trong các đầu vào.

Dữ liệu raw giữ đầy đủ biến để có thể phân tích và chạy lại selection. Dữ liệu processed giữ các biến được chọn. Hai giai đoạn train và test nằm trong các file riêng.

## Một báo cáo chọn feature

`data_processing/feature_selection.json` chứa toàn bộ kết quả chọn feature:

- `selected_features`: danh sách đầu vào theo đúng thứ tự mà cả ba notebook model đọc.
- `feature_scores`: importance trung bình, độ lệch chuẩn, thứ hạng và cờ được chọn của từng biến.
- `subset_cv_scores`: so sánh các bộ top-k bằng cross-validation.
- `holdout_scores`: đánh giá bộ đã chọn và toàn bộ biến ứng viên trên holdout.
- Các tham số: target, horizon, số fold, tỷ lệ holdout, random seed và ngưỡng MAE.

`clean_raw.ipynb` làm sạch raw, chỉ forward-fill feature tối đa 3 giờ từ quá khứ và báo lỗi nếu thiếu nhiệt độ mục tiêu hoặc mất giờ. Notebook dùng `cloud_cover` đại diện cho nhóm mây, loại cột hằng trên phần train, rồi dùng Random Forest và 5 fold `TimeSeriesSplit` để xếp hạng biến và so sánh các bộ top-k. Nó chọn số biến nhỏ nhất có MAE CV trong 1% mức thấp nhất; 20% thời gian cuối dành cho holdout. Notebook ghi thẳng dataset processed và báo cáo JSON, đồng thời hiển thị các bảng đánh giá trong notebook. Báo cáo lưu mốc holdout và SHA-256; model sẽ báo lỗi nếu dataset không khớp lượt chọn feature.

## Cách chạy

Chạy các lệnh từ thư mục gốc dự án:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Môi trường đã kiểm tra là Python 3.12; requirements ghim các phiên bản đang chạy được. Nếu chạy LSTM/GRU, cài thêm:

```bash
pip install -r requirements-dl.txt
python -m ipykernel install --user --name weather-dl --display-name weather-dl
```

Chọn kernel của môi trường vừa cài trong VS Code/Jupyter. Nếu đã có dữ liệu raw thì có thể bỏ qua bước tải:

```bash
python3 crawl_data/crawl_hanoi_enriched_weather.py
python3 crawl_data/crawl_test_set.py
python3 src/inspect_data.py
```

Cả hai script tải ERA5 qua archive API, với cùng 30 biến, tọa độ Hà Nội, múi giờ UTC+7 và đơn vị °C/km/h/mm. Script chia yêu cầu theo năm, kiểm tra đủ giờ và đủ giá trị, rồi mới thay CSV bằng thao tác đổi tên file tạm. Nếu API thiếu dữ liệu, script báo lỗi và giữ file cũ.

Bản raw test đã được tải lại hoàn toàn từ ERA5 ngày 06/10/2026. So với file cũ, 116 giờ trong giai đoạn 25–29/09 có thay đổi, gồm 111 giá trị nhiệt độ; chênh lệch nhiệt độ tối đa 2,9°C. Các chỉ số test cũ cần được thay bằng lượt đánh giá lại này. Không dùng forecast API để điền nhãn test.

| Bước | File | Công việc |
|---|---|---|
| 1. Khảo sát raw | `eda/data_analyze.ipynb` | Thống kê, missing values, tương quan và tổng hợp theo giờ/tháng. |
| 2. Trực quan raw | `eda/eda_visualization.ipynb` | Biểu đồ chuỗi thời gian, phân phối, mùa vụ, heatmap và quan hệ giữa các biến. |
| 3. Làm sạch và chọn feature | `data_processing/clean_raw.ipynb` | Tạo `Hanoi_final.csv` và `feature_selection.json`. |
| 4. Chuẩn bị test | `data_processing/prepare_test_set.ipynb` | Tạo `Hanoi_test_final.csv` theo schema train. |
| 5. Train ML | `train_model/ml_models.ipynb` | So sánh Random Forest/XGBoost, cross-validation, tuning và phân tích lỗi. |
| 6. Train DL | `train_model/dl_models.ipynb` | Chuẩn hóa đầu vào, tạo chuỗi thời gian và train LSTM/GRU. |
| 7. Đánh giá 2026 | `train_model/evaluate_2026_test_set.ipynb` | Đọc cấu hình đã chọn ở bước 5, refit trên 2010–2025 và đánh giá 2026; có tùy chọn đánh giá DL đã lưu. |

Hai notebook EDA đọc raw và có thể chạy độc lập sau khi có raw. `src/inspect_data.py` là cách kiểm tra nhanh chất lượng raw bằng terminal. Các notebook model chia theo **thời gian của nhãn**, dùng đúng mốc holdout từ feature selection. ML chọn cấu hình/model bằng CV trên phần train, với khoảng cách giữa các fold bằng horizon. Model được chốt trước khi xem test. DL dùng cùng danh sách feature trong chuỗi 24 giờ, chia khoảng 72% train / 8% validation / 20% test; scaler chỉ fit 72% train. Model DL được chọn theo validation RMSE, mỗi lần fit có bộ callback riêng.

`ml_models.ipynb` lưu các cấu hình được chọn cho hai họ model; notebook 2026 refit đúng các cấu hình đó trên toàn bộ train. File metrics DL ghi rõ thời gian train/validation; DL đã lưu có lịch sử huấn luyện ngắn hơn ML được refit toàn bộ, nên cần lưu ý khi so sánh.

## Kết quả model đang lưu

`train_model/artifacts/` chứa model đã fit, chỉ số đánh giá và dự đoán từng dòng. Ba loại file phục vụ ba mục đích riêng. Chạy lại notebook train sẽ tạo kết quả mới theo dấu thời gian; có thể bỏ các lượt cũ khi đã xác định bộ cần giữ.

Bộ ML đang lưu có dấu thời gian `20261005_110858` và dùng 9 feature:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.4655 | 0.6962 | 0.9830 |
| XGBoost | 0.4768 | 0.6986 | 0.9829 |
| Persistence baseline | 0.5828 | 0.9343 | 0.9694 |

Các notebook còn báo Median AE và MAPE. Cần chạy `dl_models.ipynb` để tạo LSTM/GRU theo schema hiện tại trước khi bật đánh giá DL trong notebook test 2026.

## Giới hạn khi diễn giải kết quả

- 9 feature là bộ gọn theo tiêu chí MAE trong 1% mức tốt nhất của các bộ top-k đã thử; không chứng minh tối ưu toàn bộ tổ hợp feature hay mọi model.
- CV/tuning và validation DL thuộc giai đoạn phát triển, có dùng danh sách feature chọn từ phần phát triển. Đây không phải nested CV. Holdout cuối kỳ và test 2026 được báo riêng; không dùng điểm test để tự động chọn model thắng.
- ERA5 là dữ liệu tái phân tích, cập nhật trễ khoảng 5 ngày theo [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api#data-sources). Kết quả ở đây là backtest dự báo trước một giờ trên dữ liệu lịch sử; chưa chứng minh chất lượng khi triển khai realtime.
- MAE và RMSE theo °C là chỉ số chính. MAPE tính trên °C phụ thuộc mốc của thang nhiệt độ nên chỉ là số tham khảo.

Quy tắc fit preprocessing chỉ trên train được đối chiếu với [scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage); cách tách validation trước fit được đối chiếu với [Keras](https://keras.io/api/models/model_training_apis/).

## Kiểm tra hồi quy

```bash
python3 -m unittest discover -s tests -v
```

Các kiểm tra bao gồm thời gian thiếu/trùng, nhãn đúng horizon, ranh giới holdout, dataset và selection không khớp, không nội suy target, scaler không nhìn validation, chọn model trước test và tải API lỗi không ghi đè dữ liệu cũ.
