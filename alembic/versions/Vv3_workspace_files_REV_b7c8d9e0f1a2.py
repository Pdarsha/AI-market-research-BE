"""V3 workspace files + cascade

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-07-01 00:00:00.000000

Refactors workspaces into containers for uploaded documents:
  * `embeddings` gain a `workspace_id` FK (ON DELETE CASCADE) so a file belongs
    to a workspace.
  * `chat_sessions.workspace_id` FK is switched from ON DELETE SET NULL to
    ON DELETE CASCADE, so deleting a workspace deletes the chats created inside it.

Written to be idempotent (mirrors Vv2): the startup `Base.metadata.create_all`
hook may already have created the new column/FK on a fresh DB, so every DDL step
is guarded, and the chat FK is discovered by introspection because its name may be
the model-declared `fk_chat_sessions_workspace_id_workspaces` or the Postgres
auto-generated `chat_sessions_workspace_id_fkey`.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = 'b7c8d9e0f1a2'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def _drop_workspace_fk(inspector, table: str) -> None:
    """Drop whatever FK currently backs <table>.workspace_id, by real name."""
    for fk in inspector.get_foreign_keys(table):
        if fk.get('constrained_columns') == ['workspace_id'] and fk.get('name'):
            op.drop_constraint(fk['name'], table, type_='foreignkey')


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    # --- embeddings.workspace_id -----------------------------------------
    emb_columns = {c['name'] for c in inspector.get_columns('embeddings')}
    if 'workspace_id' not in emb_columns:
        op.add_column(
            'embeddings', sa.Column('workspace_id', sa.String(), nullable=True)
        )

    inspector = inspect(bind)
    emb_indexes = {ix['name'] for ix in inspector.get_indexes('embeddings')}
    if 'ix_embeddings_workspace_id' not in emb_indexes:
        op.create_index(
            op.f('ix_embeddings_workspace_id'),
            'embeddings',
            ['workspace_id'],
            unique=False,
        )

    emb_fks = {fk['name'] for fk in inspector.get_foreign_keys('embeddings')}
    if 'fk_embeddings_workspace_id_workspaces' not in emb_fks:
        op.create_foreign_key(
            'fk_embeddings_workspace_id_workspaces',
            'embeddings',
            'workspaces',
            ['workspace_id'],
            ['id'],
            ondelete='CASCADE',
        )

    # --- chat_sessions.workspace_id FK: SET NULL -> CASCADE --------------
    inspector = inspect(bind)
    _drop_workspace_fk(inspector, 'chat_sessions')
    op.create_foreign_key(
        'fk_chat_sessions_workspace_id_workspaces',
        'chat_sessions',
        'workspaces',
        ['workspace_id'],
        ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    # chat_sessions FK: CASCADE -> SET NULL
    _drop_workspace_fk(inspector, 'chat_sessions')
    op.create_foreign_key(
        'fk_chat_sessions_workspace_id_workspaces',
        'chat_sessions',
        'workspaces',
        ['workspace_id'],
        ['id'],
        ondelete='SET NULL',
    )

    # embeddings.workspace_id
    inspector = inspect(bind)
    emb_fks = {fk['name'] for fk in inspector.get_foreign_keys('embeddings')}
    if 'fk_embeddings_workspace_id_workspaces' in emb_fks:
        op.drop_constraint(
            'fk_embeddings_workspace_id_workspaces', 'embeddings', type_='foreignkey'
        )

    emb_indexes = {ix['name'] for ix in inspector.get_indexes('embeddings')}
    if 'ix_embeddings_workspace_id' in emb_indexes:
        op.drop_index(op.f('ix_embeddings_workspace_id'), table_name='embeddings')

    emb_columns = {c['name'] for c in inspector.get_columns('embeddings')}
    if 'workspace_id' in emb_columns:
        op.drop_column('embeddings', 'workspace_id')
