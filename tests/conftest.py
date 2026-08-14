import pytest
import requests
import uuid
from tests.config import config

@pytest.fixture(scope="session")
def base_url():
    return config.BASE_URL

@pytest.fixture(scope="session")
def auth_session(base_url):
    """Creates a user and returns an authenticated session."""
    session = requests.Session()
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = config.TEST_USER_PASSWORD
    
    # Signup
    signup_resp = session.post(f"{base_url}/user/signup", json={
        "email": email,
        "password": password,
        "name": config.TEST_USER_NAME
    })
    
    # We assume OTP verification is bypassed in test mode or we mock it.
    # For a real system, we'd fetch the OTP from DB or use a master OTP.
    # Here we just login directly if verify is optional, or we assume a pre-verified user.
    # We will just login to get the token.
    login_resp = session.post(f"{base_url}/user/login", json={
        "email": email,
        "password": password
    })
    
    if login_resp.status_code == 200:
        token = login_resp.json().get("data", {}).get("token", {}).get("access_token")
        session.headers.update({"Authorization": f"Bearer {token}"})
    
    yield session
    
    # Cleanup could be added here

@pytest.fixture
def workspace(auth_session, base_url):
    """Creates a temporary workspace for a test."""
    resp = auth_session.post(f"{base_url}/workspaces/create-workspace", json={
        "name": f"Test WS {uuid.uuid4().hex[:6]}",
        "description": "Test description"
    })
    ws_data = resp.json().get("data", {})
    ws_id = ws_data.get("id")
    yield ws_data
    
    if ws_id:
        auth_session.delete(f"{base_url}/workspaces/{ws_id}")

@pytest.fixture
def chat_session(auth_session, base_url):
    """Creates a temporary chat for a test."""
    resp = auth_session.post(f"{base_url}/ai/chats", json={})
    chat_data = resp.json().get("data", {})
    chat_id = chat_data.get("id")
    yield chat_data
    
    if chat_id:
        auth_session.delete(f"{base_url}/ai/chats/{chat_id}")
