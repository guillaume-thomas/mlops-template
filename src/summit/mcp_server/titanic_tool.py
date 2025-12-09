import httpx
from typing import Dict, Any


class TitanicInferenceTool:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    def predict_survival(self, pclass: int, sex: str, sibsp: int, parch: int) -> Dict[str, Any]:
        url = f"{self.api_url}/infer"
        payload = {
            "pclass": pclass,
            "sex": sex,
            "sibSp": sibsp,
            "parch": parch
        }

        try:
            response = httpx.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
            survival = result[0] if isinstance(result, list) else result
            return {
                "survived": bool(survival),
                "prediction": int(survival),
                "passenger": payload
            }
        except Exception as e:
            return {
                "error": str(e),
                "passenger": payload
            }

