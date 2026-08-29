from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator

# -----------------
# Models
# -----------------


class DocumentTakeawayGridBase(BaseModel):
    name: str = Field(..., description="Name of the grid")
    workspace_id: str
    status: str = "draft"


class DocumentTakeawayGridCreate(DocumentTakeawayGridBase):
    pass


class DocumentTakeawayGridUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None


class DocumentTakeawayGridColumnBase(BaseModel):
    name: str = Field(..., description="Name of the column")
    prompt: Optional[str] = Field(None, description="Custom prompt for AI generation")
    column_type: str = Field(
        ..., description="Column type: 'document' or 'ai_generated'"
    )
    display_order: int


class DocumentTakeawayGridColumn(DocumentTakeawayGridColumnBase):
    id: str
    grid_id: str
    status: Optional[str] = (
        "completed"  # Synthesized status based on cells if needed, or static
    )
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentTakeawayGridCellBase(BaseModel):
    grid_column_id: str
    value: Optional[str] = None
    status: str = Field(..., description="pending, processing, completed, failed")
    error_message: Optional[str] = None


class DocumentTakeawayGridCell(DocumentTakeawayGridCellBase):
    id: str
    grid_row_id: str
    generated_at: Optional[datetime] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentTakeawayGridDocumentMetadata(BaseModel):
    id: str
    name: str
    file_type: str


class DocumentTakeawayGridRow(BaseModel):
    id: str
    document: DocumentTakeawayGridDocumentMetadata
    cells: List[DocumentTakeawayGridCell]

    class Config:
        from_attributes = True


class DocumentTakeawayGrid(BaseModel):
    id: str
    workspace_id: str
    name: str
    status: str
    columns: List[DocumentTakeawayGridColumn]
    rows: List[DocumentTakeawayGridRow]

    class Config:
        from_attributes = True


# -----------------
# Requests
# -----------------


class CreateDocumentTakeawayGridRequest(BaseModel):
    # Defaults to the workspace name when omitted (see DocumentTakeawaysService.create_grid)
    name: Optional[str] = Field(None)
    document_ids: List[str] = Field(..., min_items=1)


class AddCustomColumnRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    prompt: str = Field(..., min_length=1, max_length=2000)

    @validator("name", "prompt")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("must not be empty or whitespace only")
        return v.strip()


class AddDocumentsToGridRequest(BaseModel):
    document_ids: List[str] = Field(..., min_items=1)


class GetCurrentWorkspaceGridResponse(BaseModel):
    grid: Optional[DocumentTakeawayGrid] = None

class DocumentTakeawayGridListItem(BaseModel):
    id: str
    name: str
    status: str
    document_count: int
    column_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class RenameGridRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
