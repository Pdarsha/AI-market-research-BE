"""add_schedule_interval_to_agent_workflows

Revision ID: 728e1718b7c4
Revises: 628d1718b7c3
Create Date: 2026-08-19 11:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '728e1718b7c4'
down_revision = '628d1718b7c3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('agent_workflows', sa.Column('schedule_interval', sa.Integer(), nullable=True, server_default='1'))


def downgrade() -> None:
    op.drop_column('agent_workflows', 'schedule_interval')
