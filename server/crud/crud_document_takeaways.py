from typing import Optional

from sqlalchemy.orm import Session, selectinload, joinedload

from server.crud.base import CRUDBase
from server.schemas.document_takeaways import (
    DocumentTakeawayGridCreate,
    DocumentTakeawayGridUpdate,
)
from server.models.document_takeaways import (
    DocumentTakeawayGrid,
    DocumentTakeawayGridColumn,
    DocumentTakeawayGridRow,
    DocumentTakeawayGridCell,
)


class CRUDDocumentTakeaways(
    CRUDBase[
        DocumentTakeawayGrid, DocumentTakeawayGridCreate, DocumentTakeawayGridUpdate
    ]
):
    def get_grid(self, db: Session, grid_id: str) -> Optional[DocumentTakeawayGrid]:
        return (
            db.query(DocumentTakeawayGrid)
            .options(
                selectinload(DocumentTakeawayGrid.columns),
                selectinload(DocumentTakeawayGrid.rows).selectinload(DocumentTakeawayGridRow.cells),
                selectinload(DocumentTakeawayGrid.rows).joinedload(DocumentTakeawayGridRow.document),
            )
            .filter(DocumentTakeawayGrid.id == grid_id)
            .first()
        )

    def get_latest_grid_by_workspace(
        self, db: Session, workspace_id: str
    ) -> Optional[DocumentTakeawayGrid]:
        return (
            db.query(DocumentTakeawayGrid)
            .options(
                selectinload(DocumentTakeawayGrid.columns),
                selectinload(DocumentTakeawayGrid.rows).selectinload(DocumentTakeawayGridRow.cells),
                selectinload(DocumentTakeawayGrid.rows).joinedload(DocumentTakeawayGridRow.document),
            )
            .filter(DocumentTakeawayGrid.workspace_id == workspace_id)
            .order_by(DocumentTakeawayGrid.created_at.desc())
            .first()
        )

    def get_grids_by_workspace(
        self, db: Session, workspace_id: str
    ) -> list[DocumentTakeawayGrid]:
        return (
            db.query(DocumentTakeawayGrid)
            .options(
                selectinload(DocumentTakeawayGrid.columns),
                selectinload(DocumentTakeawayGrid.rows).selectinload(DocumentTakeawayGridRow.cells),
                selectinload(DocumentTakeawayGrid.rows).joinedload(DocumentTakeawayGridRow.document),
            )
            .filter(DocumentTakeawayGrid.workspace_id == workspace_id)
            .order_by(DocumentTakeawayGrid.created_at.desc())
            .all()
        )

    def get_column(
        self, db: Session, column_id: str
    ) -> Optional[DocumentTakeawayGridColumn]:
        return (
            db.query(DocumentTakeawayGridColumn)
            .filter(DocumentTakeawayGridColumn.id == column_id)
            .first()
        )

    def get_cell(self, db: Session, cell_id: str) -> Optional[DocumentTakeawayGridCell]:
        return (
            db.query(DocumentTakeawayGridCell)
            .filter(DocumentTakeawayGridCell.id == cell_id)
            .first()
        )

    def create_grid(
        self, db: Session, workspace_id: str, name: str, user_id: str
    ) -> DocumentTakeawayGrid:
        grid = DocumentTakeawayGrid(
            workspace_id=workspace_id,
            name=name,
            created_by=user_id,
            status="processing",
        )
        db.add(grid)
        db.commit()
        db.refresh(grid)
        return grid

    def add_column(
        self,
        db: Session,
        grid_id: str,
        name: str,
        column_type: str,
        display_order: int,
        prompt: Optional[str] = None,
    ) -> DocumentTakeawayGridColumn:
        column = DocumentTakeawayGridColumn(
            grid_id=grid_id,
            name=name,
            prompt=prompt,
            column_type=column_type,
            display_order=display_order,
        )
        db.add(column)
        db.commit()
        db.refresh(column)
        return column

    def get_next_column_order(self, db: Session, grid_id: str) -> int:
        last_col = (
            db.query(DocumentTakeawayGridColumn)
            .filter(DocumentTakeawayGridColumn.grid_id == grid_id)
            .order_by(DocumentTakeawayGridColumn.display_order.desc())
            .first()
        )
        return (last_col.display_order + 1) if last_col else 1

    def add_row(
        self, db: Session, grid_id: str, document_id: str, display_order: int
    ) -> DocumentTakeawayGridRow:
        row = DocumentTakeawayGridRow(
            grid_id=grid_id, document_id=document_id, display_order=display_order
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def get_next_row_order(self, db: Session, grid_id: str) -> int:
        last_row = (
            db.query(DocumentTakeawayGridRow)
            .filter(DocumentTakeawayGridRow.grid_id == grid_id)
            .order_by(DocumentTakeawayGridRow.display_order.desc())
            .first()
        )
        return (last_row.display_order + 1) if last_row else 1

    def create_cell(
        self, db: Session, row_id: str, column_id: str
    ) -> DocumentTakeawayGridCell:
        cell = DocumentTakeawayGridCell(
            grid_row_id=row_id, grid_column_id=column_id, status="pending"
        )
        db.add(cell)
        db.commit()
        db.refresh(cell)
        return cell

    def update_cell(
        self, db: Session, cell: DocumentTakeawayGridCell, update_data: dict
    ) -> DocumentTakeawayGridCell:
        for k, v in update_data.items():
            setattr(cell, k, v)
        db.commit()
        db.refresh(cell)
        return cell

    def update_grid_status(self, db: Session, grid_id: str) -> DocumentTakeawayGrid:
        grid = self.get_grid(db, grid_id)
        if not grid:
            return None

        cells = (
            db.query(DocumentTakeawayGridCell)
            .join(DocumentTakeawayGridRow)
            .filter(DocumentTakeawayGridRow.grid_id == grid_id)
            .all()
        )

        has_pending = any(c.status in ["pending", "processing"] for c in cells)
        has_failed = any(c.status == "failed" for c in cells)
        all_failed = all(c.status == "failed" for c in cells) if cells else False

        new_status = "completed"
        if has_pending:
            new_status = "processing"
        elif all_failed:
            new_status = "failed"
        elif has_failed:
            new_status = "completed_with_errors"

        grid.status = new_status
        db.commit()
        db.refresh(grid)
        return grid


document_takeaways = CRUDDocumentTakeaways(DocumentTakeawayGrid)
