"""Add memories table with pgvector

Revision ID: 2c45a18a3bdc
Revises: 68cce071cb87
Create Date: 2026-05-24 09:20:38.701818

"""
from typing import Sequence, Union
import os

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy


# revision identifiers, used by Alembic.
revision: str = '2c45a18a3bdc'
down_revision: Union[str, None] = '68cce071cb87'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Read embedding size from environment, matching MemoryModel.
_EMBEDDING_SIZE = int(os.getenv("EMBEDDING_MODEL_SIZE", "384"))


def upgrade() -> None:
    op.create_table('memories',
    sa.Column('memory_id', sa.UUID(), nullable=False),
    sa.Column('session_id', sa.String(), nullable=False),
    sa.Column('memory_type', sa.String(), nullable=False),
    sa.Column('content', sa.String(), nullable=False),
    sa.Column('summary', sa.String(), nullable=True),
    sa.Column('timestamp', sa.String(), nullable=False),
    sa.Column('importance_score', sa.Float(), nullable=True),
    sa.Column('embedding', pgvector.sqlalchemy.vector.VECTOR(dim=_EMBEDDING_SIZE), nullable=True),
    sa.PrimaryKeyConstraint('memory_id')
    )
    op.create_index(op.f('ix_memories_session_id'), 'memories', ['session_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_memories_session_id'), table_name='memories')
    op.drop_table('memories')