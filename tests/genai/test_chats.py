import pytest
import uuid

def test_create_chat(base_url, auth_session):
    resp = auth_session.post(f"{base_url}/ai/chats", json={})
    assert resp.status_code in [200, 201]
    data = resp.json()
    assert data["success"] is True
    assert "id" in data["data"]

def test_list_chats(base_url, auth_session, chat_session):
    resp = auth_session.get(f"{base_url}/ai/chats")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert isinstance(data, list)
    assert len(data) > 0

def test_list_chats_pagination(base_url, auth_session):
    resp = auth_session.get(f"{base_url}/ai/chats?page=1000&size=10")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert isinstance(data, list)

def test_get_chat(base_url, auth_session, chat_session):
    chat_id = chat_session["id"]
    resp = auth_session.get(f"{base_url}/ai/chats/{chat_id}")
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == chat_id

def test_get_chat_unauthorized(base_url, chat_session):
    import requests
    chat_id = chat_session["id"]
    resp = requests.get(f"{base_url}/ai/chats/{chat_id}")
    assert resp.status_code == 401

def test_update_chat(base_url, auth_session, chat_session):
    chat_id = chat_session["id"]
    payload = {"title": "New Title"}
    resp = auth_session.put(f"{base_url}/ai/chats/{chat_id}", json=payload)
    assert resp.status_code == 200

def test_delete_chat(base_url, auth_session):
    resp = auth_session.post(f"{base_url}/ai/chats", json={})
    chat_id = resp.json()["data"]["id"]
    
    del_resp = auth_session.delete(f"{base_url}/ai/chats/{chat_id}")
    assert del_resp.status_code in [200, 204]
    
    get_resp = auth_session.get(f"{base_url}/ai/chats/{chat_id}")
    assert get_resp.status_code == 404

def test_get_public_chat(base_url):
    import requests
    chat_id = str(uuid.uuid4())
    # Assuming chat is NOT public or doesn't exist
    resp = requests.get(f"{base_url}/ai/chats/public/{chat_id}")
    assert resp.status_code in [403, 404]
