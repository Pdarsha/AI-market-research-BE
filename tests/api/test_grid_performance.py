import time
from sqlalchemy import event
from sqlalchemy.orm import Session
from server.crud.crud_document_takeaways import crud_document_takeaways
from server.services.document_takeaways_service import document_takeaways_service
from server.models.document_takeaways import DocumentTakeawayGrid, DocumentTakeawayGridRow, DocumentTakeawayGridColumn, DocumentTakeawayGridCell
from server.models.workspace import Document


from server.db.session import SessionLocal


def test_grid_fetch_query_count_and_performance(auth_session, base_url, workspace):
    db = SessionLocal()
    ws_id = workspace["id"]
    user_id = workspace["user_id"]

    # Create 20 mock documents in DB
    doc_ids = []
    for i in range(20):
        doc = Document(
            user_id=user_id,
            workspace_id=ws_id,
            file_name=f"perf_doc_{i}.txt",
            file_format="text/plain",
        )
        db.add(doc)
        db.flush()
        doc_ids.append(str(doc.id))

    # Create a grid with 20 rows and 3 columns (60 cells total)
    grid = crud_document_takeaways.create_grid(db, ws_id, "Perf Test Grid", user_id)
    col1 = crud_document_takeaways.add_column(db, grid.id, "Document", "document", 1)
    col2 = crud_document_takeaways.add_column(db, grid.id, "Summary", "ai_generated", 2)
    col3 = crud_document_takeaways.add_column(db, grid.id, "Takeaways", "ai_generated", 3)

    for i, d_id in enumerate(doc_ids):
        row = crud_document_takeaways.add_row(db, grid.id, d_id, i + 1)
        cell1 = DocumentTakeawayGridCell(grid_row_id=row.id, grid_column_id=col1.id, value=f"Doc {i}", status="completed")
        cell2 = DocumentTakeawayGridCell(grid_row_id=row.id, grid_column_id=col2.id, value=f"Summary {i}", status="completed")
        cell3 = DocumentTakeawayGridCell(grid_row_id=row.id, grid_column_id=col3.id, value=f"Takeaways {i}", status="completed")
        db.add_all([cell1, cell2, cell3])
    db.commit()

    # Measure query count and response time for get_grid
    query_count = 0

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        nonlocal query_count
        query_count += 1

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", before_cursor_execute)

    try:
        t0 = time.perf_counter()
        fetched_grid = crud_document_takeaways.get_grid(db, grid.id)
        data = document_takeaways_service.serialize_grid(fetched_grid, db)
        elapsed = (time.perf_counter() - t0) * 1000
    finally:
        event.remove(engine, "before_cursor_execute", before_cursor_execute)

    print(f"\n[PERFORMANCE METRICS]")
    print(f"Total SQL Queries Executed for 20 Documents (60 cells): {query_count}")
    print(f"Total Serialization Execution Time: {elapsed:.2f} ms")

    # Assert that loading 20 documents hierarchy takes <= 5 SQL queries (down from 43 queries)
    assert query_count <= 5, f"Expected <= 5 SQL queries, but executed {query_count}"
    assert len(data["rows"]) == 20
    assert len(data["columns"]) == 3
    assert elapsed < 50.0  # Should execute under 50ms locally
