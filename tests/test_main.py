from unittest.mock import patch
from io import StringIO

from summit.main import main


class TestMain:

    def test_main_prints_hello_message(self):
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            main()
            assert "Hello from mlops-template-ses-test!" in mock_stdout.getvalue()

    def test_main_function_exists(self):
        assert callable(main)

    def test_main_function_returns_none(self):
        with patch('sys.stdout', new_callable=StringIO):
            result = main()
            assert result is None
