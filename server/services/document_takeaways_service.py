import logging
from typing import List
from datetime import datetime
import openai

from sqlalchemy.orm import Session
from fastapi import BackgroundTasks
from server import crud, schemas

from server.config import settings
from server.models.document_takeaways import DocumentTakeawayGrid
from server.prompts.document_takeaways import (
    PROMPT_VERSION,
    build_expert_description_prompt,
    build_key_takeaways_prompt,
    build_custom_column_prompt,
)
from server.services.rag_service import get_rag_service, RAGService
import json

logger = logging.getLogger(__name__)
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)


async def process_grid_cell_task(
    cell_id: str, prompt: str, document_id: str, workspace_id: str, user_id: str
):
    from server.db.session import SessionLocal

    db = SessionLocal()
    try:
        cell = crud.document_takeaways.get_cell(db, cell_id)
        if not cell or cell.status != "pending":
            return

        crud.document_takeaways.update_cell(db, cell, {"status": "processing"})
        logger.info(f"Cell processing started: {cell_id} for document: {document_id}")
        crud.document_takeaways.update_grid_status(db, cell.row.grid_id)

        rag_service = get_rag_service()

        # 1. Retrieve the document's processed text
        chunks = rag_service.retrieve(
            db=db,
            user_id=user_id,
            workspace_id=workspace_id,
            query=prompt,
            top_k=20,  # More limit for comprehensive takeaways
            document_ids=[document_id],
        )

        if not chunks:
            # Maybe the document is small enough to pass entirely, but if no chunks returned, it means no embeddings.
            response_text = (
                "The requested information is not available in this document."
            )
        else:
            context = RAGService.build_context(chunks)
            logger.info(
                f"Document context found for cell: {cell_id}, chunks: {len(chunks)}"
            )

            document_name = "Unknown Document"
            document_type = "unknown"
            doc_record = crud.document.get(db, document_id)
            if doc_record:
                document_name = doc_record.file_name or "Unknown Document"
                document_type = doc_record.file_format or "unknown"

            if cell.column.name == "Expert Description":
                messages = build_expert_description_prompt(
                    document_name=document_name,
                    document_type=document_type,
                    document_context=context,
                )
            elif cell.column.name == "Key Takeaways":
                messages = build_key_takeaways_prompt(
                    document_name=document_name,
                    document_type=document_type,
                    document_context=context,
                )
            else:
                messages = build_custom_column_prompt(
                    document_name=document_name,
                    document_type=document_type,
                    document_context=context,
                    custom_column_name=cell.column.name,
                    custom_column_instruction=prompt,
                )

            # We use json format for structured output from the LLM
            response_format = {"type": "json_object"}
            token_kwargs = {}
            if "gpt-5" in settings.LLM_MODEL.lower():
                token_kwargs["max_completion_tokens"] = 1024
                token_kwargs["temperature"] = 0
            else:
                token_kwargs["max_tokens"] = 1024
                token_kwargs["temperature"] = 0

            try:
                # Bypass generator and request JSON strictly
                completion = client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    response_format=response_format,
                    **token_kwargs,
                )
                raw_response = completion.choices[0].message.content or "{}"
                logger.info(f"AI response generated for cell: {cell_id}")

                try:
                    parsed = json.loads(raw_response)
                    response_text = parsed.get("content", "")
                    confidence = parsed.get("confidence")
                except json.JSONDecodeError:
                    response_text = raw_response
                    confidence = None
            except Exception as e:
                logger.error(f"LLM generation failed for cell {cell_id}: {e}")
                response_text = "An error occurred during AI generation."
                confidence = None
                raise e  # Trigger the except Exception below

        if not response_text or not response_text.strip():
            crud.document_takeaways.update_cell(
                db,
                cell,
                {
                    "status": "failed",
                    "error_message": "Generated response was empty.",
                },
            )
        else:
            crud.document_takeaways.update_cell(
                db,
                cell,
                {
                    "status": "completed",
                    "value": response_text.strip(),
                    "generated_at": datetime.utcnow(),
                    "model_name": settings.LLM_MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "context_chunk_count": len(chunks) if chunks else 0,
                    "confidence": confidence,
                },
            )
            logger.info(f"Cell saved successfully: {cell_id}")

    except Exception as e:
        logger.exception(f"Error processing cell {cell_id}")
        if "cell" in locals() and cell:
            crud.document_takeaways.update_cell(
                db,
                cell,
                {
                    "status": "failed",
                    "error_message": "An error occurred during AI generation.",
                },
            )
            logger.info(f"Cell failed: {cell_id}")
    finally:
        if "cell" in locals() and cell:
            crud.document_takeaways.update_grid_status(db, cell.row.grid_id)
        db.close()


class DocumentTakeawaysService:
    def create_grid(
        self,
        db: Session,
        background_tasks: BackgroundTasks,
        workspace_id: str,
        name: str,
        document_ids: List[str],
        user_id: str,
    ) -> DocumentTakeawayGrid:
        workspace = crud.workspace._get_owned_workspace(db, workspace_id, user_id)
        if not workspace:
            raise ValueError("Workspace not found")

        valid_docs = crud.document.get_multi_by_workspace(db, workspace_id=workspace_id)
        valid_doc_ids = {str(d.id) for d in valid_docs}

        unique_doc_ids = list(dict.fromkeys(document_ids))
        for doc_id in unique_doc_ids:
            if doc_id not in valid_doc_ids:
                raise ValueError(f"Document {doc_id} not found in workspace")

        # Prevent duplicate grid creation from rapid consecutive clicks
        recent_grid = crud.document_takeaways.get_latest_grid_by_workspace(db, workspace_id)
        if recent_grid and getattr(recent_grid, "created_by", None) == user_id:
            created_time = recent_grid.created_at.replace(tzinfo=None) if recent_grid.created_at else None
            if created_time and (datetime.utcnow() - created_time).total_seconds() < 5:
                grid_doc_ids = [str(r.document_id) for r in recent_grid.rows]
                if set(grid_doc_ids) == set(unique_doc_ids):
                    logger.info(f"Reusing recent grid {recent_grid.id} to prevent duplicate creation")
                    return recent_grid

        grid = crud.document_takeaways.create_grid(db, workspace_id, name, user_id)
        logger.info(f"Grid created: {grid.id} for workspace: {workspace_id}")

        col_doc = crud.document_takeaways.add_column(
            db, grid.id, "Document", "document", 1
        )
        col_expert = crud.document_takeaways.add_column(
            db,
            grid.id,
            "Expert Description",
            "ai_generated",
            2,
            "Summarize the relevant expert, author, organization, professional background, and domain expertise described in this document. If information is unavailable, return: 'Not enough information is available in the document.'",
        )
        col_takeaways = crud.document_takeaways.add_column(
            db,
            grid.id,
            "Key Takeaways",
            "ai_generated",
            3,
            "Identify the most important business insights, findings, observations, and conclusions from this document.",
        )

        for i, doc_id in enumerate(unique_doc_ids):
            row = crud.document_takeaways.add_row(db, grid.id, doc_id, i + 1)

            cell_expert = crud.document_takeaways.create_cell(db, row.id, col_expert.id)
            background_tasks.add_task(
                process_grid_cell_task,
                cell_expert.id,
                col_expert.prompt,
                doc_id,
                workspace_id,
                user_id,
            )

            cell_takeaways = crud.document_takeaways.create_cell(
                db, row.id, col_takeaways.id
            )
            background_tasks.add_task(
                process_grid_cell_task,
                cell_takeaways.id,
                col_takeaways.prompt,
                doc_id,
                workspace_id,
                user_id,
            )

        crud.document_takeaways.update_grid_status(db, grid.id)
        return grid

    def add_custom_column(
        self,
        db: Session,
        background_tasks: BackgroundTasks,
        grid_id: str,
        name: str,
        prompt: str,
        user_id: str,
    ):
        grid = crud.document_takeaways.get_grid(db, grid_id)
        if not grid:
            raise ValueError("Grid not found")

        workspace = crud.workspace._get_owned_workspace(db, grid.workspace_id, user_id)
        if not workspace:
            raise ValueError("Workspace not found")

        from sqlalchemy.exc import IntegrityError

        try:
            display_order = crud.document_takeaways.get_next_column_order(db, grid.id)
            column = crud.document_takeaways.add_column(
                db, grid.id, name, "ai_generated", display_order, prompt
            )
        except IntegrityError:
            db.rollback()
            raise ValueError("Column name already exists in this grid")

        for row in grid.rows:
            cell = crud.document_takeaways.create_cell(db, row.id, column.id)
            background_tasks.add_task(
                process_grid_cell_task,
                cell.id,
                column.prompt,
                row.document_id,
                grid.workspace_id,
                user_id,
            )

        crud.document_takeaways.update_grid_status(db, grid.id)
        db.refresh(grid)
        return grid

    def add_documents_to_grid(
        self,
        db: Session,
        background_tasks: BackgroundTasks,
        grid_id: str,
        document_ids: List[str],
        user_id: str,
    ):
        grid = crud.document_takeaways.get_grid(db, grid_id)
        if not grid:
            raise ValueError("Grid not found")

        workspace = crud.workspace._get_owned_workspace(db, grid.workspace_id, user_id)
        if not workspace:
            raise ValueError("Workspace not found")

        valid_docs = crud.document.get_multi_by_workspace(
            db, workspace_id=grid.workspace_id
        )
        valid_doc_ids = {str(d.id) for d in valid_docs}

        unique_req_doc_ids = list(dict.fromkeys(document_ids))
        for doc_id in unique_req_doc_ids:
            if doc_id not in valid_doc_ids:
                raise ValueError(f"Document {doc_id} not found in workspace")

        existing_doc_ids = {str(row.document_id) for row in grid.rows}
        new_doc_ids = [
            doc_id for doc_id in unique_req_doc_ids if doc_id not in existing_doc_ids
        ]

        if not new_doc_ids:
            return grid

        max_order = max([row.display_order for row in grid.rows] + [0])
        ai_columns = [col for col in grid.columns if col.column_type == "ai_generated"]

        for i, doc_id in enumerate(new_doc_ids):
            row = crud.document_takeaways.add_row(
                db, grid.id, doc_id, max_order + i + 1
            )
            for col in ai_columns:
                cell = crud.document_takeaways.create_cell(db, row.id, col.id)
                background_tasks.add_task(
                    process_grid_cell_task,
                    cell.id,
                    col.prompt,
                    doc_id,
                    grid.workspace_id,
                    user_id,
                )

        crud.document_takeaways.update_grid_status(db, grid.id)
        db.refresh(grid)
        return grid

    def retry_cell(
        self, db: Session, background_tasks: BackgroundTasks, cell_id: str, user_id: str
    ):
        cell = crud.document_takeaways.get_cell(db, cell_id)
        if not cell:
            raise ValueError("Cell not found")

        grid = cell.row.grid
        workspace = crud.workspace._get_owned_workspace(db, grid.workspace_id, user_id)
        if not workspace:
            raise ValueError("Workspace not found")

        if cell.status in ["pending", "processing"]:
            return cell

        crud.document_takeaways.update_cell(
            db, cell, {"status": "pending", "value": None, "error_message": None}
        )
        crud.document_takeaways.update_grid_status(db, grid.id)

        background_tasks.add_task(
            process_grid_cell_task,
            cell.id,
            cell.column.prompt,
            cell.row.document_id,
            grid.workspace_id,
            user_id,
        )
        return cell

    def duplicate_grid(
        self, db: Session, grid_id: str, user_id: str
    ) -> DocumentTakeawayGrid:
        source_grid = crud.document_takeaways.get_grid(db, grid_id)
        if not source_grid:
            raise ValueError("Grid not found")

        workspace = crud.workspace._get_owned_workspace(
            db, source_grid.workspace_id, user_id
        )
        if not workspace:
            raise ValueError("Workspace not found")

        new_grid = crud.document_takeaways.create_grid(
            db, source_grid.workspace_id, f"{source_grid.name} (Copy)", user_id
        )

        col_map = {}
        for col in source_grid.columns:
            new_col = crud.document_takeaways.add_column(
                db,
                new_grid.id,
                col.name,
                col.column_type,
                col.display_order,
                col.prompt,
            )
            col_map[col.id] = new_col.id

        for row in source_grid.rows:
            new_row = crud.document_takeaways.add_row(
                db, new_grid.id, row.document_id, row.display_order
            )
            for cell in row.cells:
                if cell.grid_column_id in col_map:
                    new_cell = crud.document_takeaways.create_cell(
                        db, new_row.id, col_map[cell.grid_column_id]
                    )
                    crud.document_takeaways.update_cell(
                        db,
                        new_cell,
                        {
                            "status": cell.status,
                            "value": cell.value,
                            "error_message": cell.error_message,
                            "generated_at": cell.generated_at,
                            "model_name": cell.model_name,
                        },
                    )

        crud.document_takeaways.update_grid_status(db, new_grid.id)
        db.refresh(new_grid)
        return new_grid

    def serialize_grid(self, grid: DocumentTakeawayGrid, db: Session = None) -> dict:
        rows_data = []
        for row in grid.rows:
            doc = getattr(row, "document", None)
            if not doc and db:
                doc = crud.document.get(db, row.document_id)
            if not doc:
                continue

            cells_data = []
            for cell in row.cells:
                cells_data.append(
                    {
                        "id": cell.id,
                        "grid_column_id": cell.grid_column_id,
                        "value": cell.value,
                        "status": cell.status,
                        "error_message": cell.error_message,
                    }
                )

            rows_data.append(
                {
                    "id": row.id,
                    "document": {
                        "id": doc.id,
                        "name": doc.file_name,
                        "file_type": doc.file_format or "unknown",
                    },
                    "cells": cells_data,
                }
            )

        return {
            "id": grid.id,
            "workspace_id": grid.workspace_id,
            "name": grid.name,
            "status": grid.status,
            "columns": [
                {
                    "id": col.id,
                    "name": col.name,
                    "type": col.column_type,
                    "display_order": col.display_order,
                }
                for col in grid.columns
            ],
            "rows": rows_data,
        }

    _grid_cache: dict = {}

    def get_cached_serialized_grid(self, grid: DocumentTakeawayGrid, db: Session = None) -> dict:
        if grid and grid.status in ["completed", "completed_with_errors"]:
            cache_key = (grid.id, str(grid.updated_at))
            if cache_key in self._grid_cache:
                return self._grid_cache[cache_key]

            payload = self.serialize_grid(grid, db)
            if len(self._grid_cache) >= 64:
                self._grid_cache.pop(next(iter(self._grid_cache)))
            self._grid_cache[cache_key] = payload
            return payload

        return self.serialize_grid(grid, db)


document_takeaways_service = DocumentTakeawaysService()
