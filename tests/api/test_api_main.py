from unittest.mock import patch, Mock, mock_open
import numpy as np


mock_model = Mock()
mock_model.predict.return_value = np.array([1])

with patch("builtins.open", mock_open(read_data=b"mock")), patch("pickle.load", return_value=mock_model):
    from summit.api.infer import Pclass, Sex, Passenger
    from summit.api.main import main
    from summit.api import infer


def test_api_main_is_runnable():
    """Test que main peut être appelé (sans vraiment démarrer le serveur)."""
    with patch("uvicorn.run") as mock_run:
        main()

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["host"] == "0.0.0.0"
        assert call_args[1]["port"] == 8080


def test_api_infer_module_is_importable():
    """Test que le module infer peut être importé sans erreur."""

    assert hasattr(infer, "app")
    assert hasattr(infer, "infer")
    assert hasattr(infer, "health")


def test_api_has_required_enums():
    """Test que les enums Pclass et Sex sont bien définis."""
    assert hasattr(Pclass, "UPPER")
    assert hasattr(Pclass, "MIDDLE")
    assert hasattr(Pclass, "LOW")

    assert Pclass.UPPER.value == 1
    assert Pclass.MIDDLE.value == 2
    assert Pclass.LOW.value == 3

    assert Sex.MALE.value == "male"
    assert Sex.FEMALE.value == "female"


def test_api_passenger_dataclass():
    """Test que la dataclass Passenger fonctionne correctement."""

    passenger = Passenger(pclass=Pclass.UPPER, sex=Sex.FEMALE, sibSp=1, parch=2)

    passenger_dict = passenger.to_dict()

    assert passenger_dict["Pclass"] == 1
    assert passenger_dict["Sex"] == "female"
    assert passenger_dict["SibSp"] == 1
    assert passenger_dict["Parch"] == 2
