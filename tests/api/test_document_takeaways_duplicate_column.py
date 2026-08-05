from server.models.workspace import Document


def test_document_takeaways_duplicate_column(auth_session, base_url, workspace, db_session):
    doc = Document(file_name="Test.pdf", workspace_id=workspace.id, user_id=workspace.owner_id)
    db_session.add(doc)
    db_session.commit()
    
    # create grid
    res = auth_session.post(f"{base_url}/workspaces/{workspace.id}/document-takeaways/grids", json={"name": "Test", "document_ids": [doc.id]})
    assert res.status_code == 200
    grid_id = res.json()["id"]

    # add column
    res2 = auth_session.post(f"{base_url}/document-takeaways/grids/{grid_id}/columns", json={"name": "Risks", "prompt": "Prompt"})
    assert res2.status_code == 200
    
    # add column again with different case
    res3 = auth_session.post(f"{base_url}/document-takeaways/grids/{grid_id}/columns", json={"name": " rISkS ", "prompt": "Prompt"})
    assert res3.status_code == 400
