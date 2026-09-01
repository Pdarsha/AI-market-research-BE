"""add_recipient_email_to_agent_workflows

Revision ID: a28f1718b7c7
Revises: 928f1718b7c6
Create Date: 2026-08-31 10:20:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a28f1718b7c7'
down_revision = '928f1718b7c6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('agent_workflows', sa.Column('recipient_email', sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column('agent_workflows', 'recipient_email')
