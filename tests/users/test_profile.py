import pytest
import requests

def test_get_profile_success(base_url, auth_session):
    resp = auth_session.post(f"{base_url}/user/profile")
    assert resp.status_code == 200
    data = resp.json().get("data", {})
    assert "email" in data

def test_get_profile_unauthorized(base_url):
    resp = requests.post(f"{base_url}/user/profile")
    assert resp.status_code == 401

def test_update_profile_success(base_url, auth_session):
    payload = {"name": "Updated Name"}
    resp = auth_session.post(f"{base_url}/user/update-profile", json=payload)
    assert resp.status_code == 200
    assert resp.json().get("success") is True

def test_xss_in_profile_name(base_url, auth_session):
    payload = {"name": "<script>alert('xss')</script>"}
    resp = auth_session.post(f"{base_url}/user/update-profile", json=payload)
    assert resp.status_code in [200, 400, 422]
    
def test_change_password_missing_fields(base_url, auth_session):
    resp = auth_session.post(f"{base_url}/user/change-password", json={"new_password": "NewPassword123!"})
    assert resp.status_code == 422

def test_forgot_password_send_otp_invalid_email(base_url):
    resp = requests.post(f"{base_url}/user/forgot-password-send-otp", json={"email": "nonexistent@example.com"})
    assert resp.status_code in [400, 404, 200]

def test_reset_forgot_password_invalid(base_url):
    resp = requests.post(f"{base_url}/user/reset-forgot-password", json={
        "email": "test@example.com", 
        "otp": "000000",
        "new_password": "NewPassword123!"
    })
    assert resp.status_code in [400, 404, 422]

def test_refresh_token_missing(base_url):
    resp = requests.post(f"{base_url}/user/refresh-token", json={})
    assert resp.status_code in [400, 422]
