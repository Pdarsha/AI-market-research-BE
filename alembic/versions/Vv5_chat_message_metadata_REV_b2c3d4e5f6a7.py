"""V5 chat_messages metadata (workspace_id, document_ids, source_documents)

Revision ID: b2c3d4e5f6a7
Revises: b926bcf1ac0f
Create Date: 2026-07-01 00:00:00.000000

Adds the chat metadata columns to ``chat_messages`` so it becomes the single
source of truth for conversations (replacing the now-retired GenAI table).
Written idempotently: ``Base.metadata.create_all`` at startup never alters the
pre-existing ``chat_messages`` table, so each column/index/FK is guarded.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = 'b2c3d4e5f6a7'
down_revision = 'b926bcf1ac0f'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    cm_columns = {c['name'] for c in inspector.get_columns('chat_messages')}

    if 'workspace_id' not in cm_columns:
        op.add_column(
            'chat_messages', sa.Column('workspace_id', sa.String(), nullable=True)
        )
    if 'document_ids' not in cm_columns:
        op.add_column(
            'chat_messages', sa.Column('document_ids', sa.JSON(), nullable=True)
        )
    if 'source_documents' not in cm_columns:
        op.add_column(
            'chat_messages', sa.Column('source_documents', sa.JSON(), nullable=True)
        )

    inspector = inspect(bind)
    cm_indexes = {ix['name'] for ix in inspector.get_indexes('chat_messages')}
    if 'ix_chat_messages_workspace_id' not in cm_indexes:
        op.create_index(
            op.f('ix_chat_messages_workspace_id'),
            'chat_messages',
            ['workspace_id'],
            unique=False,
        )

    cm_fks = {fk['name'] for fk in inspector.get_foreign_keys('chat_messages')}
    if 'fk_chat_messages_workspace_id_workspaces' not in cm_fks:
        op.create_foreign_key(
            'fk_chat_messages_workspace_id_workspaces',
            'chat_messages',
            'workspaces',
            ['workspace_id'],
            ['id'],
            ondelete='SET NULL',
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    cm_fks = {fk['name'] for fk in inspector.get_foreign_keys('chat_messages')}
    if 'fk_chat_messages_workspace_id_workspaces' in cm_fks:
        op.drop_constraint(
            'fk_chat_messages_workspace_id_workspaces',
            'chat_messages',
            type_='foreignkey',
        )

    cm_indexes = {ix['name'] for ix in inspector.get_indexes('chat_messages')}
    if 'ix_chat_messages_workspace_id' in cm_indexes:
        op.drop_index(
            op.f('ix_chat_messages_workspace_id'), table_name='chat_messages'
        )

    cm_columns = {c['name'] for c in inspector.get_columns('chat_messages')}
    for col in ('source_documents', 'document_ids', 'workspace_id'):
        if col in cm_columns:
            op.drop_column('chat_messages', col)
