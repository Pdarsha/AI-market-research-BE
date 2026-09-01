"""Focused unit and API tests for Document Takeaways Matrix Grid Export.

Tests cover:
1. Export grid matrix to RFC 4180 CSV with UTF-8 BOM for Excel compatibility.
2. Export grid matrix to JSON format with complete metadata.
3. Export grid matrix to Markdown table format.
4. Unsupported format rejection with 400 Bad Request.
5. 404 Not Found error handling for non-existent grids or workspace mismatches.
6. API routes:
   - GET /v1/workspaces/{workspace_id}/document-takeaways/grids/{grid_id}/export
   - GET /v1/document-takeaways/grids/{grid_id}/export
"""

import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from server.db.base_class import Base
from server.models.user import User
from server.models.workspace import Workspace, Document
from server.models.document_takeaways import (
    DocumentTakeawayGrid,
    DocumentTakeawayGridColumn,
    DocumentTakeawayGridRow,
    DocumentTakeawayGridCell,
)
from server.services.document_takeaways_service import document_takeaways_service
from server.api import api_router
from server.utils.auth import get_current_user, get_optional_current_user
from server.api_v1.deps import get_db


@pytest.fixture
def db_session():
    """In-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_user(db_session):
    """Sample user for testing."""
    user = User(
        id="usr-export-test-1",
        email="analyst@example.com",
        display_name="Takeaway Analyst",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def populated_grid(db_session, sample_user):
    """Populate a complete document takeaways grid with documents, columns, and cells."""
    workspace = Workspace(
        id="ws-export-1",
        name="Q3 Tech Earnings",
        user_id=sample_user.id,
    )
    db_session.add(workspace)

    doc1 = Document(
        id="doc-apple-1",
        workspace_id=workspace.id,
        user_id=sample_user.id,
        file_name="Apple_Q3_10Q.pdf",
        file_format="pdf",
        storage_path="/tmp/apple.pdf",
    )
    doc2 = Document(
        id="doc-msft-2",
        workspace_id=workspace.id,
        user_id=sample_user.id,
        file_name="MSFT_Annual_Report.docx",
        file_format="docx",
        storage_path="/tmp/msft.docx",
    )
    db_session.add_all([doc1, doc2])

    grid = DocumentTakeawayGrid(
        id="grid-tech-takeaways",
        workspace_id=workspace.id,
        name="Tech Giants Q3 Analysis",
        status="completed",
    )
    db_session.add(grid)

    col1 = DocumentTakeawayGridColumn(
        id="col-1",
        grid_id=grid.id,
        name="Revenue Growth",
        column_type="default",
        display_order=1,
    )
    col2 = DocumentTakeawayGridColumn(
        id="col-2",
        grid_id=grid.id,
        name="AI Strategy",
        column_type="default",
        display_order=2,
    )
    db_session.add_all([col1, col2])

    row1 = DocumentTakeawayGridRow(
        id="row-1",
        grid_id=grid.id,
        document_id=doc1.id,
        display_order=1,
    )
    row2 = DocumentTakeawayGridRow(
        id="row-2",
        grid_id=grid.id,
        document_id=doc2.id,
        display_order=2,
    )
    db_session.add_all([row1, row2])

    cell1_1 = DocumentTakeawayGridCell(
        id="cell-1-1",
        grid_row_id=row1.id,
        grid_column_id=col1.id,
        value="+9.6% YoY driven by Services and iPhone 16",
        status="completed",
    )
    cell1_2 = DocumentTakeawayGridCell(
        id="cell-1-2",
        grid_row_id=row1.id,
        grid_column_id=col2.id,
        value="Apple Intelligence rolled out across iOS 18",
        status="completed",
    )
    cell2_1 = DocumentTakeawayGridCell(
        id="cell-2-1",
        grid_row_id=row2.id,
        grid_column_id=col1.id,
        value="+15.2% YoY driven by Azure cloud expansion",
        status="completed",
    )
    cell2_2 = DocumentTakeawayGridCell(
        id="cell-2-2",
        grid_row_id=row2.id,
        grid_column_id=col2.id,
        value="Azure OpenAI service scaled to 60k enterprise customers",
        status="completed",
    )

    db_session.add_all([cell1_1, cell1_2, cell2_1, cell2_2])

    db_session.commit()
    db_session.refresh(grid)
    return grid


@pytest.fixture
def app_client(db_session, sample_user):
    """FastAPI TestClient configured with api_router."""
    app = FastAPI()
    app.include_router(api_router, prefix="/v1")

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_current_user():
        return sample_user

    def override_get_optional_current_user():
        return sample_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_optional_current_user] = override_get_optional_current_user

    client = TestClient(app)
    return client


class TestDocumentTakeawaysExportService:
    """Test service-level export logic."""

    def test_export_to_csv(self, db_session, populated_grid):
        content_bytes, media_type, filename = document_takeaways_service.export_grid(
            grid=populated_grid,
            export_format="csv",
            db=db_session,
        )
        assert media_type == "text/csv; charset=utf-8"
        assert filename.endswith(".csv")
        assert filename == "tech-giants-q3-analysis.csv"

        # Check UTF-8 BOM
        text = content_bytes.decode("utf-8")
        assert text.startswith("\ufeff")

        # Verify headers and rows
        assert "Document Name,File Format,Revenue Growth,AI Strategy" in text
        assert "Apple_Q3_10Q.pdf,PDF,+9.6% YoY driven by Services and iPhone 16,Apple Intelligence rolled out across iOS 18" in text
        assert "MSFT_Annual_Report.docx,DOCX,+15.2% YoY driven by Azure cloud expansion,Azure OpenAI service scaled to 60k enterprise customers" in text

    def test_export_to_json(self, db_session, populated_grid):
        content_bytes, media_type, filename = document_takeaways_service.export_grid(
            grid=populated_grid,
            export_format="json",
            db=db_session,
        )
        assert media_type == "application/json"
        assert filename == "tech-giants-q3-analysis.json"

        parsed = json.loads(content_bytes.decode("utf-8"))
        assert parsed["name"] == "Tech Giants Q3 Analysis"
        assert len(parsed["columns"]) == 2
        assert len(parsed["rows"]) == 2

    def test_export_to_markdown(self, db_session, populated_grid):
        content_bytes, media_type, filename = document_takeaways_service.export_grid(
            grid=populated_grid,
            export_format="markdown",
            db=db_session,
        )
        assert "text/markdown" in media_type
        assert filename == "tech-giants-q3-analysis.md"

        text = content_bytes.decode("utf-8")
        assert "# Tech Giants Q3 Analysis" in text
        assert "| Document Name | Format | Revenue Growth | AI Strategy |" in text
        assert "| Apple_Q3_10Q.pdf | PDF |" in text

    def test_export_unsupported_format_raises_value_error(self, db_session, populated_grid):
        with pytest.raises(ValueError) as exc:
            document_takeaways_service.export_grid(
                grid=populated_grid,
                export_format="unsupported_fmt",
                db=db_session,
            )
        assert "Unsupported export format" in str(exc.value)


class TestDocumentTakeawaysExportApi:
    """Test REST API endpoints for grid export."""

    def test_export_grid_api_csv(self, app_client, populated_grid):
        resp = app_client.get(
            f"/v1/workspaces/{populated_grid.workspace_id}/document-takeaways/grids/{populated_grid.id}/export?format=csv"
        )
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert 'attachment; filename="tech-giants-q3-analysis.csv"' in resp.headers["content-disposition"]
        assert "Apple_Q3_10Q.pdf" in resp.text


    def test_export_grid_api_json(self, app_client, populated_grid):
        resp = app_client.get(
            f"/v1/document-takeaways/grids/{populated_grid.id}/export?format=json"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/json"
        body = resp.json()
        assert body["name"] == "Tech Giants Q3 Analysis"

    def test_export_grid_api_invalid_format_returns_400(self, app_client, populated_grid):
        resp = app_client.get(
            f"/v1/document-takeaways/grids/{populated_grid.id}/export?format=invalid_format"
        )
        assert resp.status_code == 400

    def test_export_grid_api_not_found_returns_404(self, app_client):
        resp = app_client.get(
            "/v1/document-takeaways/grids/non-existent-grid-id/export"
        )
        assert resp.status_code == 404
