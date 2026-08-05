import pytest
import uuid
import os
from unittest import mock
import json

def test_analyze_selected_documents_returns_follow_up_questions(base_url, auth_session, workspace, chat_session):
    ws_id = workspace["id"]
    chat_id = chat_session["id"]
    
    # 1. Create a dummy txt file to upload
    dummy_text = b"This is a dummy document about Apple Inc. and its financial performance in 2024. Revenue was up."
    
    # 2. Upload the document to the workspace
    files = {
        "doc": ("dummy.txt", dummy_text, "text/plain")
    }
    upload_url = f"{base_url}/workspaces/upload-docs?workspace_id={ws_id}&chat_id={chat_id}"
    upload_resp = auth_session.post(upload_url, files=files)
    assert upload_resp.status_code == 200
    
    doc_data = upload_resp.json().get("data", {}).get("document", {})
    doc_id = doc_data.get("id")
    assert doc_id is not None
    
    # 3. Request the document analysis via the chat messaging endpoint
    payload = {
        "content": "Analyze this document",
        "mode": "auto",
        "workspace_id": ws_id,
        "document_ids": [doc_id]
    }
    
    follow_ups = []
    with auth_session.post(f"{base_url}/ai/chats/{chat_id}/messages", json=payload, stream=True) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data_str = line[len("data: "):]
                try:
                    event_data = json.loads(data_str)
                    if event_data.get("type") == "complete":
                        print("COMPLETE EVENT:", event_data)
                        msg_data = event_data.get("data", {}).get("data", {}).get("message", {})
                        if "follow_up_questions" in msg_data:
                            follow_ups = msg_data["follow_up_questions"]
                except json.JSONDecodeError:
                    print('RAW LINE:', line)
        
    assert len(follow_ups) > 0, "Follow-up questions were not found in the initial analysis."

    # 4. Now click one of those follow ups
    payload2 = {
        "content": "What is the revenue mentioned in the document?",
        "mode": "auto",
        "workspace_id": ws_id,
        "document_ids": [doc_id]
    }

    follow_ups2 = []
    with auth_session.post(f"{base_url}/ai/chats/{chat_id}/messages", json=payload2, stream=True) as resp:
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data_str = line[len("data: "):]
                try:
                    event_data = json.loads(data_str)
                    if event_data.get("type") == "complete":
                        print("COMPLETE EVENT:", event_data)
                        msg_data = event_data.get("data", {}).get("data", {}).get("message", {})
                        if "follow_up_questions" in msg_data:
                            follow_ups2 = msg_data["follow_up_questions"]
                except json.JSONDecodeError:
                    print('RAW LINE:', line)
                    
    assert len(follow_ups2) > 0, "Follow-up questions were not found in the second (follow-up) response."
