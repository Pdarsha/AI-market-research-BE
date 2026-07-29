import pytest
import uuid
import requests

def test_create_workspace_success(base_url, auth_session):
    payload = {
        "name": f"Workspace {uuid.uuid4().hex[:6]}",
        "description": "Test"
    }
    resp = auth_session.post(f"{base_url}/workspaces/create-workspace", json=payload)
    assert resp.status_code in [200, 201]
    assert resp.json()["success"] is True
    assert "id" in resp.json()["data"]

def test_create_workspace_missing_name(base_url, auth_session):
    resp = auth_session.post(f"{base_url}/workspaces/create-workspace", json={"description": "Test"})
    assert resp.status_code == 422

def test_list_workspaces(base_url, auth_session, workspace):
    resp = auth_session.get(f"{base_url}/workspaces/list-workspaces")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("data"), list)

def test_get_workspace(base_url, auth_session, workspace):
    ws_id = workspace["id"]
    resp = auth_session.get(f"{base_url}/workspaces/{ws_id}")
    assert resp.status_code == 200
    assert resp.json()["data"]["id"] == ws_id

def test_get_workspace_unauthorized(base_url, workspace):
    ws_id = workspace["id"]
    resp = requests.get(f"{base_url}/workspaces/{ws_id}")
    assert resp.status_code == 401

def test_update_workspace(base_url, auth_session, workspace):
    ws_id = workspace["id"]
    resp = auth_session.put(f"{base_url}/workspaces/{ws_id}", json={"name": "Updated WS"})
    assert resp.status_code == 200

def test_delete_workspace(base_url, auth_session):
    resp = auth_session.post(f"{base_url}/workspaces/create-workspace", json={"name": "To Delete"})
    ws_id = resp.json()["data"]["id"]
    
    del_resp = auth_session.delete(f"{base_url}/workspaces/{ws_id}")
    assert del_resp.status_code in [200, 204]
    
    get_resp = auth_session.get(f"{base_url}/workspaces/{ws_id}")
    assert get_resp.status_code == 404

def test_public_workspace_unauthorized(base_url):
    ws_id = str(uuid.uuid4())
    # Assuming workspace does not exist or isn't public
    resp = requests.get(f"{base_url}/workspaces/public/{ws_id}")
    assert resp.status_code in [403, 404]
