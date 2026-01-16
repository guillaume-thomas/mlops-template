import pytest
import jwt
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from summit.api.auth import verify_token


private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())

private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)

public_key = private_key.public_key()
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
)


def create_jwt(payload: dict, private_key_pem: bytes = private_pem, algorithm: str = "RS256") -> str:
    """Helper pour créer un JWT de test avec RSA."""
    return jwt.encode(payload, private_key_pem, algorithm=algorithm)


@pytest.mark.asyncio
async def test_verify_token_without_oauth2_domain():
    """Test que verify_token accepte n'importe quel token si OAUTH2_DOMAIN n'est pas défini."""
    with patch("os.getenv", return_value=None):
        validator = verify_token("api:read")
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = "any-token"

        token = await validator(credentials)
        assert token == "any-token"


@pytest.mark.asyncio
async def test_verify_token_with_valid_jwt():
    """Test que verify_token accepte un JWT valide avec le bon scope."""
    payload = {
        "sub": "user123",
        "scope": "api:read api:write",
        "aud": "titanic-api",
        "iss": "https://test-tenant.eu.auth0.com/",
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    token = create_jwt(payload)

    def mock_getenv(key, default=None):
        if key == "OAUTH2_DOMAIN":
            return "test-tenant.eu.auth0.com"
        if key == "OAUTH2_JWT_AUDIENCE":
            return "titanic-api"
        return default

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_pem

    with patch("os.getenv", side_effect=mock_getenv), patch("summit.api.auth.PyJWKClient") as mock_jwks_client:
        mock_client_instance = MagicMock()
        mock_client_instance.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client.return_value = mock_client_instance

        validator = verify_token("api:read")
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token

        result = await validator(credentials)
        assert result == token


@pytest.mark.asyncio
async def test_verify_token_with_expired_jwt():
    """Test que verify_token rejette un JWT expiré."""
    payload = {
        "sub": "user123",
        "scope": "api:read",
        "aud": "titanic-api",
        "iss": "https://test-tenant.eu.auth0.com/",
        "exp": datetime.now(UTC) - timedelta(hours=1),
    }
    token = create_jwt(payload)

    def mock_getenv(key, default=None):
        if key == "OAUTH2_DOMAIN":
            return "test-tenant.eu.auth0.com"
        if key == "OAUTH2_JWT_AUDIENCE":
            return "titanic-api"
        return default

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_pem

    with patch("os.getenv", side_effect=mock_getenv), patch("summit.api.auth.PyJWKClient") as mock_jwks_client:
        mock_client_instance = MagicMock()
        mock_client_instance.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client.return_value = mock_client_instance

        validator = verify_token("api:read")
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token

        with pytest.raises(HTTPException) as exc_info:
            await validator(credentials)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_verify_token_with_invalid_audience():
    """Test que verify_token rejette un JWT avec une mauvaise audience."""
    payload = {
        "sub": "user123",
        "scope": "api:read",
        "aud": "wrong-api",
        "iss": "https://test-tenant.eu.auth0.com/",
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    token = create_jwt(payload)

    def mock_getenv(key, default=None):
        if key == "OAUTH2_DOMAIN":
            return "test-tenant.eu.auth0.com"
        if key == "OAUTH2_JWT_AUDIENCE":
            return "titanic-api"
        return default

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_pem

    with patch("os.getenv", side_effect=mock_getenv), patch("summit.api.auth.PyJWKClient") as mock_jwks_client:
        mock_client_instance = MagicMock()
        mock_client_instance.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client.return_value = mock_client_instance

        validator = verify_token("api:read")
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token

        with pytest.raises(HTTPException) as exc_info:
            await validator(credentials)

        assert exc_info.value.status_code == 401
        assert "audience" in exc_info.value.detail.lower()


@pytest.mark.asyncio
async def test_verify_token_with_insufficient_scope():
    """Test que verify_token rejette un JWT avec un scope insuffisant."""
    payload = {
        "sub": "user123",
        "scope": "api:read",
        "aud": "titanic-api",
        "iss": "https://test-tenant.eu.auth0.com/",
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    token = create_jwt(payload)

    def mock_getenv(key, default=None):
        if key == "OAUTH2_DOMAIN":
            return "test-tenant.eu.auth0.com"
        if key == "OAUTH2_JWT_AUDIENCE":
            return "titanic-api"
        return default

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_pem

    with patch("os.getenv", side_effect=mock_getenv), patch("summit.api.auth.PyJWKClient") as mock_jwks_client:
        mock_client_instance = MagicMock()
        mock_client_instance.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client.return_value = mock_client_instance

        validator = verify_token("api:write")
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token

        with pytest.raises(HTTPException) as exc_info:
            await validator(credentials)

        assert exc_info.value.status_code == 403
        assert "Insufficient permissions" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_token_with_invalid_signature():
    """Test que verify_token rejette un JWT avec une mauvaise signature."""
    wrong_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    wrong_private_pem = wrong_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    payload = {
        "sub": "user123",
        "scope": "api:read",
        "aud": "titanic-api",
        "iss": "https://test-tenant.eu.auth0.com/",
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    token = create_jwt(payload, private_key_pem=wrong_private_pem)

    def mock_getenv(key, default=None):
        if key == "OAUTH2_DOMAIN":
            return "test-tenant.eu.auth0.com"
        if key == "OAUTH2_JWT_AUDIENCE":
            return "titanic-api"
        return default

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_pem

    with patch("os.getenv", side_effect=mock_getenv), patch("summit.api.auth.PyJWKClient") as mock_jwks_client:
        mock_client_instance = MagicMock()
        mock_client_instance.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client.return_value = mock_client_instance

        validator = verify_token("api:read")
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token

        with pytest.raises(HTTPException) as exc_info:
            await validator(credentials)

        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_token_with_multiple_scopes():
    """Test que verify_token valide correctement avec plusieurs scopes."""
    payload = {
        "sub": "user123",
        "scope": "api:read api:write api:admin",
        "aud": "titanic-api",
        "iss": "https://test-tenant.eu.auth0.com/",
        "exp": datetime.now(UTC) + timedelta(hours=1),
    }
    token = create_jwt(payload)

    def mock_getenv(key, default=None):
        if key == "OAUTH2_DOMAIN":
            return "test-tenant.eu.auth0.com"
        if key == "OAUTH2_JWT_AUDIENCE":
            return "titanic-api"
        return default

    mock_signing_key = MagicMock()
    mock_signing_key.key = public_pem

    with patch("os.getenv", side_effect=mock_getenv), patch("summit.api.auth.PyJWKClient") as mock_jwks_client:
        mock_client_instance = MagicMock()
        mock_client_instance.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwks_client.return_value = mock_client_instance

        validator = verify_token("api:write")
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = token

        result = await validator(credentials)
        assert result == token
