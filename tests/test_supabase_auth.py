from __future__ import annotations

import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth import signup_user, login_user
from app.supabase_client import get_supabase_admin_client, get_supabase_client
from app.config import SUPABASE_SECRET_KEY


@pytest.fixture
def test_user_credentials():
    """Generates unique test credentials and guarantees cleanup."""
    unique_id = uuid.uuid4().hex[:8]
    email = f"analyst_test_{unique_id}@gmail.com"
    password = f"AegisFinPass_{unique_id}!123"
    full_name = f"Test Analyst {unique_id}"
    created_user_id = None

    admin = get_supabase_admin_client()

    yield {
        "email": email,
        "password": password,
        "full_name": full_name,
    }

    # Teardown: ensure user and profile are purged
    try:
        # Check if user was registered
        users_res = admin.auth.admin.list_users()
        for u in users_res:
            if u.email == email:
                created_user_id = u.id
                break
    except Exception:
        pass

    if created_user_id:
        try:
            admin.table("profiles").delete().eq("id", created_user_id).execute()
        except Exception:
            pass
        try:
            admin.auth.admin.delete_user(created_user_id)
        except Exception:
            pass


def test_signup_creates_user_and_profile(test_user_credentials):
    """Verify signup flow creates user in Supabase Auth and corresponding profile in public.profiles."""
    creds = test_user_credentials
    result = signup_user(
        email=creds["email"],
        password=creds["password"],
        full_name=creds["full_name"],
    )

    assert "user_id" in result
    user_id = result["user_id"]
    assert result["email"] == creds["email"]
    assert result["full_name"] == creds["full_name"]

    # Verify profile exists in public.profiles with profiles.id = auth.users.id
    admin = get_supabase_admin_client()
    p_res = admin.table("profiles").select("*").eq("id", user_id).execute()
    assert len(p_res.data) == 1, "Profile row was not created in public.profiles!"
    profile = p_res.data[0]
    assert profile["id"] == user_id
    assert profile["full_name"] == creds["full_name"]
    assert profile["role"] == "analyst"

    # Security check: Password must never be stored in profiles
    assert "password" not in profile, "Security breach: password stored in profile!"


def test_login_flow(test_user_credentials):
    """Verify login flow returns a valid access token and session."""
    creds = test_user_credentials
    signup_res = signup_user(
        email=creds["email"],
        password=creds["password"],
        full_name=creds["full_name"],
    )
    user_id = signup_res["user_id"]

    # Attempt login
    login_res = login_user(email=creds["email"], password=creds["password"])
    assert "access_token" in login_res
    assert bool(login_res["access_token"])
    assert login_res["user_id"] == user_id
    assert login_res["email"] == creds["email"]
    assert login_res["authenticated"] is True


def test_login_invalid_credentials(test_user_credentials):
    """Verify login rejects invalid password."""
    creds = test_user_credentials
    signup_user(
        email=creds["email"],
        password=creds["password"],
        full_name=creds["full_name"],
    )

    with pytest.raises(ValueError) as excinfo:
        login_user(email=creds["email"], password="WrongPassword123!")
    assert "Invalid email or password" in str(excinfo.value) or "failed" in str(excinfo.value).lower()


def test_protected_auth_me_missing_token():
    """Verify GET /api/v1/auth/me rejects requests without a token with HTTP 401."""
    client = TestClient(app)
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert "Missing authentication credentials" in response.text or "detail" in response.json()


def test_protected_auth_me_invalid_token():
    """Verify GET /api/v1/auth/me rejects invalid bearer token with HTTP 401."""
    client = TestClient(app)
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid_jwt_token_payload_xyz"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] is not None


def test_protected_auth_me_valid_token(test_user_credentials):
    """Verify GET /api/v1/auth/me returns authenticated user identity with valid token."""
    creds = test_user_credentials
    signup_res = signup_user(
        email=creds["email"],
        password=creds["password"],
        full_name=creds["full_name"],
    )
    user_id = signup_res["user_id"]

    login_res = login_user(email=creds["email"], password=creds["password"])
    token = login_res["access_token"]

    client = TestClient(app)
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["authenticated"] is True
    assert data["user_id"] == user_id
    assert data["email"] == creds["email"]
    assert data["full_name"] == creds["full_name"]

    # Security check: SUPABASE_SECRET_KEY must never leak in API responses
    assert SUPABASE_SECRET_KEY not in str(data), "SUPABASE_SECRET_KEY leaked in auth/me response!"


def test_fastapi_signup_and_login_endpoints(test_user_credentials):
    """Verify POST /api/v1/auth/signup and POST /api/v1/auth/login endpoints."""
    creds = test_user_credentials
    client = TestClient(app)

    # 1. Signup
    signup_resp = client.post(
        "/api/v1/auth/signup",
        json={
            "email": creds["email"],
            "password": creds["password"],
            "full_name": creds["full_name"],
        },
    )
    assert signup_resp.status_code == 201
    signup_data = signup_resp.json()
    assert "user_id" in signup_data

    # 2. Login
    login_resp = client.post(
        "/api/v1/auth/login",
        json={
            "email": creds["email"],
            "password": creds["password"],
        },
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert "access_token" in login_data
    token = login_data["access_token"]

    # 3. Access protected route
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200
    assert me_resp.json()["user_id"] == signup_data["user_id"]
