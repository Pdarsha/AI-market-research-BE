"""drop_uq_chat_sessions_agent_day

Revision ID: 828f1718b7c5
Revises: 728e1718b7c4
Create Date: 2026-08-19 12:35:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '828f1718b7c5'
down_revision = '728e1718b7c4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop unique constraint that prevented creating multiple chat sessions for an agent on the same calendar day
    try:
        op.drop_index('uq_chat_sessions_agent_day', table_name='chat_sessions')
    except Exception:
        pass


def downgrade() -> None:
    op.create_index(
        'uq_chat_sessions_agent_day',
        'chat_sessions',
        ['user_id', 'agent_id', 'execution_date'],
        unique=True,
        postgresql_where=sa.text("agent_id IS NOT NULL AND execution_date IS NOT NULL"),
    )
