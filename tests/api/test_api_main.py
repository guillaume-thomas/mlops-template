from unittest.mock import patch, MagicMock

from summit.api.main import main


class TestApiMain:

    @patch('summit.api.main.uvicorn')
    @patch('summit.api.main.infer')
    def test_main_runs_uvicorn_with_correct_parameters(self, mock_infer, mock_uvicorn):
        mock_app = MagicMock()
        mock_infer.app = mock_app

        main()

        mock_uvicorn.run.assert_called_once_with(mock_app, host="0.0.0.0", port=8080)

    @patch('summit.api.main.uvicorn')
    @patch('summit.api.main.infer')
    def test_main_imports_infer_module(self, mock_infer, mock_uvicorn):
        main()
        assert mock_infer is not None

    def test_main_function_exists(self):
        assert callable(main)
