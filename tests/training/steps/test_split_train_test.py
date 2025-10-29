import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from summit.training.steps.split_train_test import split_train_test


class TestSplitTrainTest:

    @patch('summit.training.steps.split_train_test.os.remove')
    @patch('summit.training.steps.split_train_test.mlflow')
    @patch('summit.training.steps.split_train_test.sklearn.model_selection.train_test_split')
    @patch('summit.training.steps.split_train_test.pd.read_csv')
    @patch('summit.training.steps.split_train_test.client')
    def test_split_train_test_success(self, mock_client, mock_read_csv, mock_train_test_split,
                                     mock_mlflow, mock_remove):
        data_path = "path_output/data.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_client.download_artifacts.return_value = "/downloaded/data.csv"

        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        mock_read_csv.return_value = mock_df

        mock_x_train = pd.DataFrame({'feature1': [1, 2], 'feature2': [10, 20]})
        mock_x_test = pd.DataFrame({'feature1': [3, 4], 'feature2': [30, 40]})
        mock_y_train = pd.Series([0, 1])
        mock_y_test = pd.Series([0, 1])

        mock_train_test_split.return_value = (mock_x_train, mock_x_test, mock_y_train, mock_y_test)

        with patch('builtins.open', create=True), \
             patch.object(pd.DataFrame, 'to_csv') as mock_to_csv_df, \
             patch.object(pd.Series, 'to_csv') as mock_to_csv_series:

            result = split_train_test(data_path)

            mock_client.download_artifacts.assert_called_once_with(
                run_id="test_run_id",
                path=data_path
            )
            mock_read_csv.assert_called_once_with("/downloaded/data.csv", index_col=False)
            mock_train_test_split.assert_called_once()

            assert mock_mlflow.log_artifact.call_count == 4
            assert mock_remove.call_count == 4

        assert result == ("xtrain/xtrain.csv", "xtest/xtest.csv", "ytrain/ytrain.csv", "ytest/ytest.csv")

    @patch('summit.training.steps.split_train_test.os.remove')
    @patch('summit.training.steps.split_train_test.mlflow')
    @patch('summit.training.steps.split_train_test.sklearn.model_selection.train_test_split')
    @patch('summit.training.steps.split_train_test.pd.read_csv')
    @patch('summit.training.steps.split_train_test.client')
    def test_split_train_test_train_test_split_parameters(self, mock_client, mock_read_csv,
                                                         mock_train_test_split, mock_mlflow, mock_remove):
        data_path = "path_output/data.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_client.download_artifacts.return_value = "/downloaded/data.csv"

        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        mock_read_csv.return_value = mock_df

        expected_y = mock_df["target"]
        expected_x = mock_df.drop(columns="target")

        mock_train_test_split.return_value = (
            pd.DataFrame({'feature1': [1]}),
            pd.DataFrame({'feature1': [2]}),
            pd.Series([0]),
            pd.Series([1])
        )

        with patch('builtins.open', create=True), \
             patch.object(pd.DataFrame, 'to_csv'), \
             patch.object(pd.Series, 'to_csv'):

            split_train_test(data_path)

            # Vérifier que train_test_split a été appelé avec les bons paramètres
            call_args = mock_train_test_split.call_args
            assert call_args[1]['test_size'] == 0.3
            assert call_args[1]['random_state'] == 42

    @patch('summit.training.steps.split_train_test.os.remove')
    @patch('summit.training.steps.split_train_test.mlflow')
    @patch('summit.training.steps.split_train_test.sklearn.model_selection.train_test_split')
    @patch('summit.training.steps.split_train_test.pd.read_csv')
    @patch('summit.training.steps.split_train_test.client')
    def test_split_train_test_file_operations(self, mock_client, mock_read_csv,
                                             mock_train_test_split, mock_mlflow, mock_remove):
        data_path = "path_output/data.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_client.download_artifacts.return_value = "/downloaded/data.csv"

        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })
        mock_read_csv.return_value = mock_df

        mock_x_train = pd.DataFrame({'feature1': [1, 2]})
        mock_x_test = pd.DataFrame({'feature1': [3, 4]})
        mock_y_train = pd.Series([0, 1])
        mock_y_test = pd.Series([0, 1])

        mock_train_test_split.return_value = (mock_x_train, mock_x_test, mock_y_train, mock_y_test)

        with patch('builtins.open', create=True), \
             patch.object(pd.DataFrame, 'to_csv') as mock_to_csv_df, \
             patch.object(pd.Series, 'to_csv') as mock_to_csv_series:

            split_train_test(data_path)

            expected_calls_df = [
                ('xtrain.csv', {'index': False}),
                ('xtest.csv', {'index': False})
            ]
            expected_calls_series = [
                ('ytrain.csv', {'index': False}),
                ('ytest.csv', {'index': False})
            ]

            assert mock_to_csv_df.call_count == 2
            assert mock_to_csv_series.call_count == 2

            expected_log_artifact_calls = [
                (('xtrain.csv', 'xtrain'),),
                (('xtest.csv', 'xtest'),),
                (('ytrain.csv', 'ytrain'),),
                (('ytest.csv', 'ytest'),)
            ]

            assert mock_mlflow.log_artifact.call_count == 4

            expected_remove_calls = [
                'xtrain.csv', 'xtest.csv', 'ytrain.csv', 'ytest.csv'
            ]
            assert mock_remove.call_count == 4

    @patch('summit.training.steps.split_train_test.client')
    def test_split_train_test_mlflow_client_usage(self, mock_client):
        data_path = "path_output/data.csv"

        with patch('summit.training.steps.split_train_test.mlflow') as mock_mlflow, \
             patch('summit.training.steps.split_train_test.pd.read_csv') as mock_read_csv, \
             patch('summit.training.steps.split_train_test.sklearn.model_selection.train_test_split') as mock_train_test_split, \
             patch('summit.training.steps.split_train_test.os.remove'), \
             patch.object(pd.DataFrame, 'to_csv'), \
             patch.object(pd.Series, 'to_csv'):

            mock_run = MagicMock()
            mock_run.info.run_id = "test_run_id"
            mock_mlflow.active_run.return_value = mock_run

            mock_client.download_artifacts.return_value = "/downloaded/data.csv"

            mock_df = pd.DataFrame({
                'feature1': [1, 2],
                'target': [0, 1]
            })

            with patch('summit.training.steps.split_train_test.pd.read_csv', return_value=mock_df):
                with patch('summit.training.steps.split_train_test.sklearn.model_selection.train_test_split',
                          return_value=(pd.DataFrame({'feature1': [1]}), pd.DataFrame({'feature1': [2]}),
                                       pd.Series([0]), pd.Series([1]))):
                    split_train_test(data_path)

            mock_client.download_artifacts.assert_called_once_with(
                run_id="test_run_id",
                path=data_path
            )

    def test_split_train_test_function_exists(self):
        assert callable(split_train_test)

    @patch('summit.training.steps.split_train_test.mlflow')
    @patch('summit.training.steps.split_train_test.pd.read_csv')
    @patch('summit.training.steps.split_train_test.client')
    def test_split_train_test_missing_target_column(self, mock_client, mock_read_csv, mock_mlflow):
        data_path = "path_output/data.csv"

        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.active_run.return_value = mock_run

        mock_client.download_artifacts.return_value = "/downloaded/data.csv"

        mock_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40]
        })
        mock_read_csv.return_value = mock_df

        with pytest.raises(KeyError):
            split_train_test(data_path)
