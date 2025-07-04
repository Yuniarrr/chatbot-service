"""update filename

Revision ID: 5416a73f3dd5
Revises: 9db5632e48be
Create Date: 2025-05-18 11:44:55.191744

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "5416a73f3dd5"
down_revision: Union[str, None] = "9db5632e48be"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "file",
        "file_name",
        existing_type=sa.String(length=40),
        type_=sa.Text(),
        existing_nullable=True,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "file",
        "file_name",
        existing_type=sa.Text(),
        type_=sa.String(length=40),
        existing_nullable=True,
    )
    # ### end Alembic commands ###
