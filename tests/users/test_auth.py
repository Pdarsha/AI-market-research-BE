import pytest
import requests
import uuid

def test_signup_success(base_url):
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    payload = {
        "email": email,
        "password": "StrongPassword123!",
        "name": "Test User"
    }
    resp = requests.post(f"{base_url}/user/signup", json=payload)
    assert resp.status_code in [200, 201]
    data = resp.json()
    assert data.get("success") is True

def test_signup_duplicate_email(base_url, auth_session):
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    payload = {
        "email": email,
        "password": "StrongPassword123!",
        "name": "Test User"
    }
    requests.post(f"{base_url}/user/signup", json=payload)
    resp2 = requests.post(f"{base_url}/user/signup", json=payload)
    # The application flow for unverified users resends OTP and returns 200
    # For verified users, it returns 400. We assert either one.
    assert resp2.status_code in [200, 400]

def test_signup_invalid_email(base_url):
    payload = {
        "email": "invalid-email",
        "password": "StrongPassword123!",
        "name": "Test User"
    }
    resp = requests.post(f"{base_url}/user/signup", json=payload)
    assert resp.status_code == 422

def test_verify_otp_invalid(base_url):
    resp = requests.post(f"{base_url}/user/verify-otp", json={"email": "test@example.com", "otp": "000000"})
    assert resp.status_code in [400, 404, 422]

def test_signup_resend_otp(base_url):
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    requests.post(f"{base_url}/user/signup", json={"email": email, "password": "StrongPassword123!", "name": "User"})
    resp = requests.post(f"{base_url}/user/signup-resend-otp", json={"email": email})
    assert resp.status_code == 200

def test_login_success(base_url):
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "StrongPassword123!"
    requests.post(f"{base_url}/user/signup", json={"email": email, "password": password, "name": "User"})
    resp = requests.post(f"{base_url}/user/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    data = resp.json().get("data", {})
    assert "token" in data
    assert "access_token" in data["token"]

def test_login_invalid_password(base_url):
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "StrongPassword123!"
    requests.post(f"{base_url}/user/signup", json={"email": email, "password": password, "name": "User"})
    resp = requests.post(f"{base_url}/user/login", json={"email": email, "password": "WrongPassword1!"})
    assert resp.status_code == 401

def test_login_missing_fields(base_url):
    resp = requests.post(f"{base_url}/user/login", json={"email": "test@example.com"})
    assert resp.status_code == 422
    
def test_google_login_invalid_token(base_url):
    resp = requests.post(f"{base_url}/user/google-login", json={"id_token": "invalid_jwt_token"})
    assert resp.status_code in [400, 401, 422]

def test_sql_injection_email(base_url):
    resp = requests.post(f"{base_url}/user/login", json={"email": "admin@example.com' OR 1=1 --", "password": "pass"})
    assert resp.status_code in [400, 401, 422]
