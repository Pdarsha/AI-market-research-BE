import pytest
import time
import uuid
import json

def test_analyze_selected_document_flow(auth_session, base_url, workspace):
    """
    Test that sending an 'Analyze the selected document' message properly
    invokes RAG, generates insights, and importantly, populates citations.
    """
    ws_id = workspace["id"]
    
    # 1. Upload a document with specific text so we can get a factual answer
    doc_content = "The company AlphaBeta reported an annual revenue of $100 Million in Q3 2026. The new CEO is Jane Doe."
    with open("test_analyze.txt", "w") as f:
        f.write(doc_content)
    try:
        with open("test_analyze.txt", "rb") as f:
            resp = auth_session.post(
                f"{base_url}/workspaces/upload-docs?workspace_id={ws_id}",
                files={"doc": ("test_analyze.txt", f, "text/plain")},
            )
        assert resp.status_code == 200, resp.text
        doc_id = resp.json()["data"]["document"]["id"]

        # 2. Create a chat session
        chat_resp = auth_session.post(
            f"{base_url}/ai/chats",
            json={"title": "Analysis Chat", "workspace_id": ws_id}
        )
        assert chat_resp.status_code in [200, 201], chat_resp.text
        chat_id = chat_resp.json()["data"]["id"]

        # 3. Send the analyze message
        message_resp = auth_session.post(
            f"{base_url}/ai/chats/{chat_id}/messages",
            json={
                "content": f"Analyze the selected document ({'test_analyze.txt'}).",
                "document_ids": [doc_id]
            },
            stream=True,
        )
        assert message_resp.status_code == 200, message_resp.text

        message_data = None
        for line in message_resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("type") == "complete":
                    complete_payload = event["data"]["data"]
                    message_data = complete_payload.get("message", {})
                    if not message_data and "citations" in complete_payload:
                        message_data = complete_payload
                    break

        assert message_data is not None, "Complete SSE event not received"
        
        # Verify response structure
        assert message_data.get("role") == "assistant"
        
        # Ensure citations were actually returned and linked
        citations = message_data.get("citations", [])
        assert isinstance(citations, list)
        
        # Log the citations for debugging
        print(f"Returned citations: {json.dumps(citations, indent=2)}")
        
        # We expect at least one citation since the LLM must base its analysis on the uploaded doc
        assert len(citations) > 0, "No citations generated for the document analysis!"
        
        # Verify metadata is correctly structured
        for cite in citations:
            assert (cite.get("document_name") or cite.get("title") or cite.get("file_name")) == "test_analyze.txt"
            assert "chunk_id" in cite
            assert ("excerpt" in cite or "description" in cite or "supporting_excerpt" in cite)
            assert ("page" in cite or "page_number" in cite)
            
        # Verify that source_documents is also properly hydrated for the frontend viewer
        source_docs = message_data.get("source_documents", [])
        assert len(source_docs) > 0, "Source documents should be populated"
        assert "test_analyze.txt" in source_docs

    finally:
        import os
        if os.path.exists("test_analyze.txt"):
            os.remove("test_analyze.txt")
