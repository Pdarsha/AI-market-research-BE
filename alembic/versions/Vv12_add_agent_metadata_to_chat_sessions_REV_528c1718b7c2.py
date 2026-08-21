"""add_agent_metadata_to_chat_sessions

Revision ID: 528c1718b7c2
Revises: 428b1718b7c1
Create Date: 2026-08-18 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '528c1718b7c2'
down_revision = '428b1718b7c1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('chat_sessions', sa.Column('agent_id', sa.String(), nullable=True))
    op.add_column('chat_sessions', sa.Column('agent_name', sa.String(), nullable=True))
    op.add_column('chat_sessions', sa.Column('execution_type', sa.String(), nullable=True, server_default='chat'))
    op.add_column('chat_sessions', sa.Column('chat_metadata', sa.JSON(), nullable=True))
    op.create_index(op.f('ix_chat_sessions_agent_id'), 'chat_sessions', ['agent_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_chat_sessions_agent_id'), table_name='chat_sessions')
    op.drop_column('chat_sessions', 'chat_metadata')
    op.drop_column('chat_sessions', 'execution_type')
    op.drop_column('chat_sessions', 'agent_name')
    op.drop_column('chat_sessions', 'agent_id')
