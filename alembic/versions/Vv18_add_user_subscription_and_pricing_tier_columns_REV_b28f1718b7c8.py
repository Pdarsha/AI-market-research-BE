"""add_user_subscription_and_pricing_tier_columns

Revision ID: b28f1718b7c8
Revises: a28f1718b7c7
Create Date: 2026-08-31 10:50:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b28f1718b7c8'
down_revision = 'a28f1718b7c7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('subscription_tier', sa.String(length=50), nullable=True, server_default='free'))
    op.add_column('users', sa.Column('daily_credits', sa.Integer(), nullable=True, server_default='100'))
    op.add_column('users', sa.Column('credits_remaining', sa.Integer(), nullable=True, server_default='100'))
    op.add_column('users', sa.Column('subscription_status', sa.String(length=50), nullable=True, server_default='active'))
    op.add_column('users', sa.Column('subscription_period', sa.String(length=50), nullable=True, server_default='monthly'))


def downgrade() -> None:
    op.drop_column('users', 'subscription_period')
    op.drop_column('users', 'subscription_status')
    op.drop_column('users', 'credits_remaining')
    op.drop_column('users', 'daily_credits')
    op.drop_column('users', 'subscription_tier')
