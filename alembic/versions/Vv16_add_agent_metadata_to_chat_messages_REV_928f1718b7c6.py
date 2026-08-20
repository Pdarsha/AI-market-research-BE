"""add_agent_metadata_to_chat_messages

Revision ID: 928f1718b7c6
Revises: 828f1718b7c5
Create Date: 2026-08-20 22:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '928f1718b7c6'
down_revision = '828f1718b7c5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('chat_messages', sa.Column('agent_id', sa.String(), nullable=True))
    op.add_column('chat_messages', sa.Column('agent_name', sa.String(), nullable=True))
    op.create_index(op.f('ix_chat_messages_agent_id'), 'chat_messages', ['agent_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_chat_messages_agent_id'), table_name='chat_messages')
    op.drop_column('chat_messages', 'agent_name')
    op.drop_column('chat_messages', 'agent_id')
