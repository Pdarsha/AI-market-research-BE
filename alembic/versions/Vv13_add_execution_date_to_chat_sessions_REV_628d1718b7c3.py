"""add_execution_date_to_chat_sessions

Revision ID: 628d1718b7c3
Revises: 528c1718b7c2
Create Date: 2026-08-18 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '628d1718b7c3'
down_revision = '528c1718b7c2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('chat_sessions', sa.Column('execution_date', sa.Date(), nullable=True))
    op.create_index(op.f('ix_chat_sessions_execution_date'), 'chat_sessions', ['execution_date'], unique=False)
    # Unique constraint for agent daily chat reuse: (user_id, agent_id, execution_date)
    op.create_index(
        'uq_chat_sessions_agent_day',
        'chat_sessions',
        ['user_id', 'agent_id', 'execution_date'],
        unique=True,
        postgresql_where=sa.text("agent_id IS NOT NULL AND execution_date IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index('uq_chat_sessions_agent_day', table_name='chat_sessions')
    op.drop_index(op.f('ix_chat_sessions_execution_date'), table_name='chat_sessions')
    op.drop_column('chat_sessions', 'execution_date')
