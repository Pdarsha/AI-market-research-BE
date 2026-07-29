import pytest

def test_get_prompt_templates(base_url, auth_session):
    resp = auth_session.get(f"{base_url}/cards/prompt-templates")
    assert resp.status_code == 200
    assert isinstance(resp.json().get("data", []), list)

def test_get_prompt_templates_without_auth(base_url):
    import requests
    resp = requests.get(f"{base_url}/cards/prompt-templates")
    assert resp.status_code == 200

def test_get_prompt_templates_filter(base_url, auth_session):
    resp = auth_session.get(f"{base_url}/cards/prompt-templates?tag=financial")
    assert resp.status_code == 200
    data = resp.json().get("data", [])
    assert isinstance(data, list)

def test_sql_injection_prompt_template_query(base_url, auth_session):
    resp = auth_session.get(f"{base_url}/cards/prompt-templates?tag=1' OR '1'='1")
    assert resp.status_code in [200, 400, 422]
