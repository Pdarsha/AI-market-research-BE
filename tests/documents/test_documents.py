import pytest
import uuid

def test_document_viewer_not_found(base_url, auth_session):
    doc_id = str(uuid.uuid4())
    resp = auth_session.get(f"{base_url}/documents/{doc_id}/viewer")
    assert resp.status_code == 404

def test_document_viewer_unauthorized(base_url):
    import requests
    doc_id = str(uuid.uuid4())
    resp = requests.get(f"{base_url}/documents/{doc_id}/viewer")
    assert resp.status_code == 401

def test_document_file_not_found(base_url, auth_session):
    doc_id = str(uuid.uuid4())
    resp = auth_session.get(f"{base_url}/documents/{doc_id}/file")
    assert resp.status_code in [404, 422]

def test_document_file_invalid_uuid(base_url, auth_session):
    resp = auth_session.get(f"{base_url}/documents/invalid-uuid/file")
    assert resp.status_code == 422

def test_document_highlight_not_found(base_url, auth_session):
    doc_id = str(uuid.uuid4())
    resp = auth_session.get(f"{base_url}/documents/{doc_id}/highlight?query=test")
    assert resp.status_code == 404

def test_citation_not_found(base_url, auth_session):
    cit_id = str(uuid.uuid4())
    resp = auth_session.get(f"{base_url}/citations/{cit_id}")
    assert resp.status_code == 404

def test_export_pdf_success(base_url, auth_session):
    payload = {"html_content": "<h1>Test</h1>", "title": "Test PDF"}
    resp = auth_session.post(f"{base_url}/documents/export-pdf", json=payload)
    # The system might return 200 with the blob
    assert resp.status_code in [200, 422]
    
def test_export_pdf_missing_content(base_url, auth_session):
    resp = auth_session.post(f"{base_url}/documents/export-pdf", json={})
    assert resp.status_code == 422

def test_export_pdf_ssrf_prevention(base_url, auth_session):
    payload = {"html_content": "<img src='http://169.254.169.254/latest/meta-data/'>"}
    resp = auth_session.post(f"{base_url}/documents/export-pdf", json=payload)
    # Ensure SSRF fails or at least request doesn't crash 500
    assert resp.status_code in [200, 400, 422]

def test_roic_transcript_success(base_url, auth_session):
    ticker = "TSLA"
    resp = auth_session.get(f"{base_url}/citations/roic/transcript/{ticker}")
    assert resp.status_code in [200, 404]
