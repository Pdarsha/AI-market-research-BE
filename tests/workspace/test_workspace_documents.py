import pytest
import os

def test_upload_document_empty(base_url, auth_session, workspace):
    ws_id = workspace["id"]
    resp = auth_session.post(
        f"{base_url}/workspaces/upload-docs", 
        data={"workspace_id": ws_id}
    )
    assert resp.status_code in [400, 422]

def test_list_documents(base_url, auth_session, workspace):
    ws_id = workspace["id"]
    resp = auth_session.get(f"{base_url}/workspaces/{ws_id}/docs")
    assert resp.status_code == 200
    assert isinstance(resp.json().get("data"), list)

def test_get_document_not_found(base_url, auth_session, workspace):
    ws_id = workspace["id"]
    import uuid
    doc_id = str(uuid.uuid4())
    resp = auth_session.get(f"{base_url}/workspaces/{ws_id}/docs/{doc_id}")
    assert resp.status_code == 404

def test_delete_document_not_found(base_url, auth_session, workspace):
    ws_id = workspace["id"]
    import uuid
    doc_id = str(uuid.uuid4())
    resp = auth_session.delete(f"{base_url}/workspaces/{ws_id}/docs/{doc_id}")
    assert resp.status_code == 404
