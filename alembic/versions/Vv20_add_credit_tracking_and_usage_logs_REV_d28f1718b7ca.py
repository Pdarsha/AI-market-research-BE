"""add_credit_tracking_and_usage_logs

Revision ID: d28f1718b7ca
Revises: c28f1718b7c9
Create Date: 2026-08-31 11:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd28f1718b7ca'
down_revision = 'c28f1718b7c9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('last_credit_reset_at', sa.DateTime(), nullable=True, server_default=sa.func.now()))

    op.create_table(
        'credit_usage_logs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('amount', sa.Integer(), nullable=False),
        sa.Column('action_type', sa.String(length=50), nullable=False, server_default='consume'),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('balance_after', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_credit_usage_logs_user_id'), 'credit_usage_logs', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_credit_usage_logs_user_id'), table_name='credit_usage_logs')
    op.drop_table('credit_usage_logs')
    op.drop_column('users', 'last_credit_reset_at')
