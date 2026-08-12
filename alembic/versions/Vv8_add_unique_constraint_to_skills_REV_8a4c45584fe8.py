"""V8_add_unique_constraint_to_skills

Revision ID: 8a4c45584fe8
Revises: 7a4c45584fed
Create Date: 2026-08-12 10:45:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8a4c45584fe8'
down_revision = '7a4c45584fed'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint('uix_skills_user_id_name', 'skills', ['user_id', 'name'])


def downgrade() -> None:
    op.drop_constraint('uix_skills_user_id_name', 'skills', type_='unique')
