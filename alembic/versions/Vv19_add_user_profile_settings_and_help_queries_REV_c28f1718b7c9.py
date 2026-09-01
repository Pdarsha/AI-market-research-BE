"""add_user_profile_settings_and_help_queries

Revision ID: c28f1718b7c9
Revises: b28f1718b7c8
Create Date: 2026-08-31 11:15:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c28f1718b7c9'
down_revision = 'b28f1718b7c8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('users', sa.Column('display_language', sa.String(length=50), nullable=True, server_default='English'))
    op.add_column('users', sa.Column('nickname', sa.String(length=255), nullable=True))
    op.add_column('users', sa.Column('occupation', sa.String(length=255), nullable=True))
    op.add_column('users', sa.Column('bio', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('ai_traits', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('custom_instructions', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('ai_data_retention', sa.Boolean(), nullable=True, server_default=sa.text('true')))
    op.add_column('users', sa.Column('settings_json', sa.Text(), nullable=True))

    op.create_table(
        'help_queries',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=True),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=True),
        sa.Column('subject', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=True, server_default='general'),
        sa.Column('status', sa.String(length=50), nullable=True, server_default='pending'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_help_queries_user_id'), 'help_queries', ['user_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_help_queries_user_id'), table_name='help_queries')
    op.drop_table('help_queries')
    op.drop_column('users', 'settings_json')
    op.drop_column('users', 'ai_data_retention')
    op.drop_column('users', 'custom_instructions')
    op.drop_column('users', 'ai_traits')
    op.drop_column('users', 'bio')
    op.drop_column('users', 'occupation')
    op.drop_column('users', 'nickname')
    op.drop_column('users', 'display_language')
