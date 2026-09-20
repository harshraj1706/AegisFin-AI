from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .supabase_client import get_supabase_admin_client, get_supabase_client

# Bearer token security scheme (auto_error=False to allow custom 401 handling)
bearer_scheme = HTTPBearer(auto_error=False)


class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Dict[str, Any]:
    """
    FastAPI dependency validating the Supabase Bearer token.
    1. Reads Bearer token from the Authorization header.
    2. Validates token against Supabase Auth using client.auth.get_user(token).
    3. Fetches user profile from public.profiles.
    4. Rejects missing/invalid tokens with HTTP 401.
    Never trusts client-supplied user_id.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        client = get_supabase_client()
        user_resp = client.auth.get_user(token)
        if not user_resp or not user_resp.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user = user_resp.user
        user_id = str(user.id)
        email = user.email or ""

        # Fetch profile from public.profiles
        admin = get_supabase_admin_client()
        profile_data = {}
        try:
            p_res = admin.table("profiles").select("*").eq("id", user_id).execute()
            if p_res.data:
                profile_data = p_res.data[0]
        except Exception:
            pass

        full_name = profile_data.get("full_name") or (
            user.user_metadata.get("full_name") if user.user_metadata else None
        )
        role = profile_data.get("role", "analyst")

        return {
            "user_id": user_id,
            "email": email,
            "full_name": full_name,
            "role": role,
            "authenticated": True,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def signup_user(
    email: str,
    password: str,
    full_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Registers a new user and creates their profile in public.profiles (profiles.id = auth.users.id).
    Avoids free-tier email rate-limit by auto-confirming email via admin client.
    Never stores password in profiles.
    """
    if not email or "@" not in email:
        raise ValueError("Invalid email format.")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    admin = get_supabase_admin_client()
    full_name_clean = full_name.strip() if full_name else email.split("@")[0].capitalize()

    # Create user in Supabase auth
    try:
        create_res = admin.auth.admin.create_user({
            "email": email.strip(),
            "password": password,
            "email_confirm": True,
            "user_metadata": {"full_name": full_name_clean},
        })
        if not create_res or not create_res.user:
            raise RuntimeError("User creation returned empty response from Supabase.")
        user_id = str(create_res.user.id)
    except Exception as exc:
        err_str = str(exc).lower()
        if "already registered" in err_str or "already exists" in err_str:
            raise ValueError(f"User with email '{email}' already exists.") from exc
        raise

    # Create/update row in public.profiles (profiles.id = auth.users.id)
    admin.table("profiles").upsert({
        "id": user_id,
        "full_name": full_name_clean,
        "role": "analyst",
    }).execute()

    return {
        "user_id": user_id,
        "email": email.strip(),
        "full_name": full_name_clean,
        "role": "analyst",
        "message": "User registered and profile created successfully.",
    }


def login_user(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticates a user via Supabase using email/password.
    Returns the session access token and user metadata.
    """
    client = get_supabase_client()
    try:
        res = client.auth.sign_in_with_password({
            "email": email.strip(),
            "password": password,
        })
        if not res or not res.session:
            raise ValueError("Authentication failed: No active session returned.")

        session = res.session
        user = res.user
        user_id = str(user.id)

        # Get profile if present
        admin = get_supabase_admin_client()
        full_name = None
        role = "analyst"
        try:
            p_res = admin.table("profiles").select("*").eq("id", user_id).execute()
            if p_res.data:
                full_name = p_res.data[0].get("full_name")
                role = p_res.data[0].get("role", "analyst")
        except Exception:
            pass

        if not full_name and user.user_metadata:
            full_name = user.user_metadata.get("full_name")

        return {
            "access_token": session.access_token,
            "token_type": "bearer",
            "user_id": user_id,
            "email": user.email,
            "full_name": full_name or email.split("@")[0].capitalize(),
            "role": role,
            "authenticated": True,
        }
    except Exception as exc:
        err_msg = str(exc)
        if "invalid login credentials" in err_msg.lower():
            raise ValueError("Invalid email or password.") from exc
        raise ValueError(f"Login failed: {err_msg}") from exc
