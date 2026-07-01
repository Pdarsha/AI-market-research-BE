"""V2 added workspaces

Revision ID: a1b2c3d4e5f6
Revises: f3229b783f06
Create Date: 2026-06-30 00:00:00.000000

Note: this migration is written to be idempotent. The application's startup hook
(`init_db` -> `Base.metadata.create_all`) may already have created the `workspaces`
table and its index, while leaving the pre-existing `chat_sessions` table untouched
(create_all never adds columns to existing tables). Each DDL step is therefore guarded
so the migration only applies what is actually missing.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'f3229b783f06'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    # --- workspaces table -------------------------------------------------
    if 'workspaces' not in inspector.get_table_names():
        op.create_table(
            'workspaces',
            sa.Column('id', sa.String(), nullable=False),
            sa.Column('user_id', sa.String(), nullable=True),
            sa.Column('name', sa.String(), nullable=False),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id'),
        )

    inspector = inspect(bind)
    ws_indexes = {ix['name'] for ix in inspector.get_indexes('workspaces')}
    if 'ix_workspaces_user_id' not in ws_indexes:
        op.create_index(
            op.f('ix_workspaces_user_id'), 'workspaces', ['user_id'], unique=False
        )

    # --- chat_sessions.workspace_id --------------------------------------
    cs_columns = {c['name'] for c in inspector.get_columns('chat_sessions')}
    if 'workspace_id' not in cs_columns:
        op.add_column(
            'chat_sessions', sa.Column('workspace_id', sa.String(), nullable=True)
        )

    inspector = inspect(bind)
    cs_indexes = {ix['name'] for ix in inspector.get_indexes('chat_sessions')}
    if 'ix_chat_sessions_workspace_id' not in cs_indexes:
        op.create_index(
            op.f('ix_chat_sessions_workspace_id'),
            'chat_sessions',
            ['workspace_id'],
            unique=False,
        )

    cs_fks = {fk['name'] for fk in inspector.get_foreign_keys('chat_sessions')}
    if 'fk_chat_sessions_workspace_id_workspaces' not in cs_fks:
        op.create_foreign_key(
            'fk_chat_sessions_workspace_id_workspaces',
            'chat_sessions',
            'workspaces',
            ['workspace_id'],
            ['id'],
            ondelete='SET NULL',
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    cs_fks = {fk['name'] for fk in inspector.get_foreign_keys('chat_sessions')}
    if 'fk_chat_sessions_workspace_id_workspaces' in cs_fks:
        op.drop_constraint(
            'fk_chat_sessions_workspace_id_workspaces',
            'chat_sessions',
            type_='foreignkey',
        )

    cs_indexes = {ix['name'] for ix in inspector.get_indexes('chat_sessions')}
    if 'ix_chat_sessions_workspace_id' in cs_indexes:
        op.drop_index(
            op.f('ix_chat_sessions_workspace_id'), table_name='chat_sessions'
        )

    cs_columns = {c['name'] for c in inspector.get_columns('chat_sessions')}
    if 'workspace_id' in cs_columns:
        op.drop_column('chat_sessions', 'workspace_id')

    if 'workspaces' in inspector.get_table_names():
        ws_indexes = {ix['name'] for ix in inspector.get_indexes('workspaces')}
        if 'ix_workspaces_user_id' in ws_indexes:
            op.drop_index(op.f('ix_workspaces_user_id'), table_name='workspaces')
        op.drop_table('workspaces')
