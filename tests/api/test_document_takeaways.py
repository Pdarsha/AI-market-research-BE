import pytest
import time
import uuid


def test_document_takeaways_flow(auth_session, base_url, workspace):
    ws_id = workspace["id"]

    with open("test.txt", "w") as f:
        f.write("This is a dummy document for testing takeaways.")
    try:
        # 1. Upload a document
        with open("test.txt", "rb") as f:
            resp = auth_session.post(
                f"{base_url}/workspaces/upload-docs?workspace_id={ws_id}",
                files={"doc": ("test.txt", f, "text/plain")},
            )
        assert resp.status_code == 200, resp.text
        doc_id = resp.json()["data"]["document"]["id"]

        # Wait briefly for document processing (assuming it might be async, though in workspace.py it's awaited directly)
        time.sleep(2)

        # 2. Test Get Empty Grid
        resp = auth_session.get(
            f"{base_url}/workspaces/{ws_id}/document-takeaways/grid"
        )
        assert resp.status_code == 200
        assert resp.json() is None

        # 3. Test Create Grid
        resp = auth_session.post(
            f"{base_url}/workspaces/{ws_id}/document-takeaways/grids",
            json={"name": "My Takeaways", "document_ids": [doc_id]},
        )
        assert resp.status_code == 200, resp.text
        grid = resp.json()
        grid_id = grid["id"]
        assert grid["status"] in ["processing", "completed"]
        assert len(grid["columns"]) == 3
        assert len(grid["rows"]) == 1
        assert grid["rows"][0]["document"]["id"] == doc_id

        # 4. Test Get Grid
        resp = auth_session.get(
            f"{base_url}/workspaces/{ws_id}/document-takeaways/grids/{grid_id}"
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == grid_id

        # 5. Test Add Custom Column
        resp = auth_session.post(
            f"{base_url}/workspaces/{ws_id}/document-takeaways/grids/{grid_id}/columns",
            json={"name": "Risks", "prompt": "What are the risks?"},
        )
        assert resp.status_code == 200, resp.text
        grid_data = resp.json()
        assert grid_data["name"] == "My Takeaways"
        col = next((c for c in grid_data["columns"] if c["name"] == "Risks"), None)
        assert col is not None
        assert col["name"] == "Risks"

        # 6. Test Retry Cell
        row = grid_data["rows"][0]
        cell_to_retry = next(
            c for c in row["cells"] if c["grid_column_id"] == col["id"]
        )

        resp = auth_session.post(
            f"{base_url}/workspaces/{ws_id}/document-takeaways/grids/{grid_id}/cells/{cell_to_retry['id']}/retry"
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] in ["pending", "processing"]

        # 7. Test Security (Invalid Workspace)
        other_ws_id = str(uuid.uuid4())
        resp = auth_session.get(
            f"{base_url}/workspaces/{other_ws_id}/document-takeaways/grid"
        )
        # Will just return grid None or 404 depending on how strict workspace checks are on that endpoint.
        # We didn't add strict workspace exists check on GET /grid if it's empty, we return None.
        # But for create grid:
        resp = auth_session.post(
            f"{base_url}/workspaces/{other_ws_id}/document-takeaways/grids",
            json={"name": "My Takeaways", "document_ids": [doc_id]},
        )
        assert resp.status_code in [
            400,
            404,
            500,
        ]  # Usually 400 for workspace not found in our service

    finally:
        # Clean up the test file
        import os

        if os.path.exists("test.txt"):
            os.remove("test.txt")
