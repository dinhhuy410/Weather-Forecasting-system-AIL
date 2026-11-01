"""Regression checks for time boundaries, preprocessing leakage and archive writes."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main, mock
import contextlib
import hashlib
import io
import json
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.weather_data import (
    WEATHER_COLUMNS, METADATA_COLUMNS, validate_hourly_frame,
    build_forecast_dataset, load_feature_selection,
)
from crawl_data.crawl_hanoi_enriched_weather import download_archive


def cell(path, index):
    return ''.join(json.loads((ROOT / path).read_text())['cells'][index]['source'])


def hourly(n=24):
    return pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n, freq='h'),
        'temperature_2m': np.arange(n, dtype=float),
        'cloud_cover': np.zeros(n),
    })


class DataContractTests(TestCase):
    def test_exact_hour_labels_and_trailing_rows(self):
        frame = hourly()
        for horizon in [1, 3]:
            result, label = build_forecast_dataset(frame, ['temperature_2m'], 'temperature_2m', horizon)
            self.assertEqual(len(result), len(frame) - horizon)
            self.assertTrue((result.label_time - result.time).eq(pd.Timedelta(hours=horizon)).all())
            np.testing.assert_array_equal(result[label], frame.temperature_2m.iloc[horizon:])

    def test_rejects_gaps_duplicates_invalid_and_nonfinite(self):
        examples = [hourly().drop(index=2), pd.concat([hourly(), hourly().iloc[[0]]]), hourly().iloc[::-1]]
        invalid = hourly(); invalid.loc[2, 'time'] = pd.NaT; examples.append(invalid)
        invalid = hourly(); invalid.loc[2, 'temperature_2m'] = np.inf; examples.append(invalid)
        invalid = hourly(); invalid.loc[2, 'temperature_2m'] = np.nan; examples.append(invalid)
        for frame in examples:
            with self.subTest(frame=frame.shape), self.assertRaises(ValueError):
                build_forecast_dataset(frame, ['temperature_2m'], 'temperature_2m', 1)

    def test_schema_missing_column(self):
        with self.assertRaises(ValueError):
            validate_hourly_frame(hourly(), ['rain'])

    def test_horizon_validation(self):
        for horizon in [0, -1, 1.5, 24]:
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                build_forecast_dataset(hourly(), ['temperature_2m'], 'temperature_2m', horizon)

    def test_no_label_overlap_at_holdout(self):
        result, _ = build_forecast_dataset(hourly(), ['temperature_2m'], 'temperature_2m', 1)
        cutoff = hourly().time.iloc[19]
        train, test = result[result.label_time < cutoff], result[result.label_time >= cutoff]
        self.assertLess(train.label_time.max(), test.label_time.min())
        self.assertEqual(test.iloc[0].time, train.label_time.max())
        self.assertEqual(test.iloc[0].label_time, cutoff)

    def test_report_rejects_stale_dataset_or_horizon(self):
        with tempfile.TemporaryDirectory() as folder:
            data, report = Path(folder)/'data.csv', Path(folder)/'selection.json'
            hourly().to_csv(data, index=False)
            report.write_text(json.dumps({'target':'temperature_2m', 'horizon_hours':1,
                'selected_features':['temperature_2m'], 'processed_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}))
            self.assertEqual(load_feature_selection(report, 'temperature_2m', 1, data), ['temperature_2m'])
            with self.assertRaises(ValueError): load_feature_selection(report, 'temperature_2m', 2, data)
            data.write_text('changed')
            with self.assertRaises(ValueError): load_feature_selection(report, 'temperature_2m', 1, data)


class LeakageTests(TestCase):
    def clean_frame(self):
        frame=hourly()
        for col in WEATHER_COLUMNS:
            if col not in frame: frame[col]=1.0
        frame['city']='Hanoi';frame['latitude']=21.0285;frame['longitude']=105.8542
        return frame

    def cleaning_namespace(self,frame):
        return dict(df=frame, TARGET_COLUMN='temperature_2m', WEATHER_COLUMNS=WEATHER_COLUMNS,
                    METADATA_COLUMNS=METADATA_COLUMNS, validate_hourly_frame=validate_hourly_frame)

    def test_missing_feature_uses_only_past(self):
        frame=self.clean_frame();frame.loc[1,'cloud_cover']=10
        frame.loc[2,'cloud_cover']=np.nan;frame.loc[3,'cloud_cover']=90
        ns=self.cleaning_namespace(frame)
        with contextlib.redirect_stdout(io.StringIO()): exec(cell('data_processing/clean_raw.ipynb',11),ns)
        self.assertEqual(ns['df'].loc[2,'cloud_cover'],10)

    def test_missing_target_is_not_fabricated(self):
        frame=self.clean_frame();frame.loc[2,'temperature_2m']=np.nan
        with self.assertRaises(ValueError): exec(cell('data_processing/clean_raw.ipynb',11),self.cleaning_namespace(frame))

    def test_scaler_excludes_validation_extremes(self):
        frame=hourly(100);frame.loc[72:,'temperature_2m']=10000
        with tempfile.TemporaryDirectory() as folder:
            data, report=Path(folder)/'data.csv',Path(folder)/'selection.json'
            frame.to_csv(data,index=False)
            report.write_text(json.dumps({'target':'temperature_2m','horizon_hours':1,
                'selected_features':['temperature_2m'],'holdout_start':str(frame.time.iloc[80]),
                'processed_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}))
            ns=dict(df=frame,DATA_PATH=data,FEATURE_SELECTION_PATH=report,TARGET_COLUMN='temperature_2m',
                    HORIZON_HOURS=1,SEQUENCE_LENGTH=24,load_feature_selection=load_feature_selection,
                    validate_hourly_frame=validate_hourly_frame,json=json,pd=pd,MinMaxScaler=MinMaxScaler)
            with contextlib.redirect_stdout(io.StringIO()):exec(cell('train_model/dl_models.ipynb',6),ns)
            self.assertEqual(ns['validation_index'],72)
            self.assertEqual(ns['target_scaler'].data_max_[0],71)
            self.assertGreater(ns['target_scaled'][72],1)

    def test_model_selection_does_not_use_test_ranking(self):
        class Model:
            def __init__(self,error):self.error=error
            def fit(self,*args):return self
        rf,xgb=Model(10),Model(0)
        metric=lambda y,p: {'MAE':20.,'RMSE':20.}
        ns=dict(models={'Random Forest':rf,'XGBoost':xgb},
            rf_search=SimpleNamespace(best_estimator_=Model(11),best_score_=-3.),
            xgb_search=SimpleNamespace(best_estimator_=Model(1),best_score_=-4.),
            cv_fold_mae={'Random Forest':np.array([1.,1.,1.]),'XGBoost':np.array([2.,2.,2.])},
            X_train=hourly(),y_train=pd.Series([0]),X_test=hourly(),y_test=pd.Series([0]),
            TARGET_COLUMN='temperature_2m',pd=pd,
            evaluate_model=lambda m,x,y:{'MAE':m.error,'RMSE':m.error},metrics_from_prediction=metric,
            format_metrics=lambda frame:frame)
        with contextlib.redirect_stdout(io.StringIO()):exec(cell('train_model/ml_models.ipynb',16),ns)
        self.assertEqual(ns['best_model_name'],'Random Forest')
        self.assertEqual(ns['test_results'].index[0],'XGBoost')


class CollectionTests(TestCase):
    def test_incomplete_archive_cannot_replace_existing_dataset(self):
        incomplete=hourly()
        for col in WEATHER_COLUMNS:
            if col not in incomplete:incomplete[col]=1.0
        incomplete.loc[2,'temperature_2m']=np.nan
        response=mock.Mock();response.json.return_value={'hourly':incomplete.to_dict('list')}
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'existing.csv';output.write_text('original data')
            with mock.patch('crawl_data.crawl_hanoi_enriched_weather.requests.get',return_value=response) as get:
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    self.assertFalse(download_archive('2020-01-01','2020-01-01',output))
            self.assertEqual(output.read_text(),'original data')
            self.assertEqual(get.call_count,1)
            self.assertEqual(get.call_args.kwargs['params']['models'],'era5')
            self.assertNotIn('forecast-api',get.call_args.args[0])

    def test_complete_archive_writes_requested_hours(self):
        complete=hourly()
        for col in WEATHER_COLUMNS:
            if col not in complete:complete[col]=1.0
        response=mock.Mock();response.json.return_value={'hourly':complete.to_dict('list')}
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'data.csv'
            with mock.patch('crawl_data.crawl_hanoi_enriched_weather.requests.get',return_value=response):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertTrue(download_archive('2020-01-01','2020-01-01',output))
            self.assertEqual(pd.read_csv(output).shape,(24,34))
            self.assertFalse(output.with_suffix('.csv.tmp').exists())


if __name__=='__main__':main()
