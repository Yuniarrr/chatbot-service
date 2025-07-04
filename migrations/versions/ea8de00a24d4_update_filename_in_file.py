"""update filename in file

Revision ID: ea8de00a24d4
Revises: 56300a90e0e9
Create Date: 2025-03-12 22:52:13.396228

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "ea8de00a24d4"
down_revision: Union[str, None] = "56300a90e0e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


new_enum = postgresql.ENUM(
    "SUCCESS", "FAILED", "AWAITING", "DETACHED", name="filestatus"
)
old_enum = postgresql.ENUM("SUCCESS", "FAILED", "AWAITING", name="filestatus")


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Drop default constraint on column (if any)
    op.execute("ALTER TABLE file ALTER COLUMN status DROP DEFAULT")

    # Change column type to TEXT temporarily
    op.execute("ALTER TABLE file ALTER COLUMN status TYPE TEXT")

    # Drop the old ENUM type
    old_enum.drop(op.get_bind(), checkfirst=True)

    # Create the new ENUM type with the updated values
    new_enum.create(op.get_bind(), checkfirst=True)

    # Convert the column back to ENUM
    op.execute(
        "ALTER TABLE file ALTER COLUMN status TYPE filestatus USING status::filestatus"
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # Drop default constraint on column (if any)
    op.execute("ALTER TABLE file ALTER COLUMN status DROP DEFAULT")

    # Change column type to TEXT temporarily
    op.execute("ALTER TABLE file ALTER COLUMN status TYPE TEXT")

    # Drop the new ENUM type
    new_enum.drop(op.get_bind(), checkfirst=True)

    # Recreate the old ENUM type
    old_enum.create(op.get_bind(), checkfirst=True)

    # Convert the column back to old ENUM
    op.execute(
        "ALTER TABLE file ALTER COLUMN status TYPE filestatus USING status::filestatus"
    )
    # ### end Alembic commands ###
