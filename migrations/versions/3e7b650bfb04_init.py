"""init

Revision ID: 3e7b650bfb04
Revises: 
Create Date: 2025-02-23 17:05:56.939178

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3e7b650bfb04"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "user",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("full_name", sa.String(length=40), nullable=True),
        sa.Column("email", sa.String(length=40), nullable=False),
        sa.Column("phone_number", sa.String(length=20), nullable=True),
        sa.Column("password", sa.String(length=255), nullable=True),
        sa.Column(
            "role",
            sa.Enum("ADMINISTRATOR", "USER", "DEVELOPER", name="role"),
            nullable=False,
        ),
        sa.Column("profile_picture", sa.Text(), nullable=True),
        sa.Column("nrp", sa.String(), nullable=True),
        sa.Column("nip", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
        sa.UniqueConstraint("id"),
        sa.UniqueConstraint("phone_number"),
    )
    op.create_table(
        "conversation",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id"),
    )
    op.create_index(
        op.f("ix_conversation_user_id"), "conversation", ["user_id"], unique=False
    )
    op.create_table(
        "file",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("file_name", sa.String(length=40), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum("SUCCESS", "FAILED", "AWAITING", name="filestatus"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id"),
    )
    op.create_index(op.f("ix_file_file_name"), "file", ["file_name"], unique=False)
    op.create_index(op.f("ix_file_file_path"), "file", ["file_path"], unique=False)
    op.create_index(op.f("ix_file_user_id"), "file", ["user_id"], unique=False)
    op.create_table(
        "knowledge",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id"),
    )
    op.create_index(
        op.f("ix_knowledge_user_id"), "knowledge", ["user_id"], unique=False
    )
    op.create_table(
        "tool",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("specs", sa.JSON(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id"),
    )
    op.create_index(op.f("ix_tool_user_id"), "tool", ["user_id"], unique=False)
    op.create_table(
        "message",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "from_message", sa.Enum("USER", "BOT", name="frommessage"), nullable=False
        ),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("file_url", sa.String(), nullable=True),
        sa.Column("file_name", sa.String(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversation.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id"),
    )
    op.create_index(
        op.f("ix_message_conversation_id"), "message", ["conversation_id"], unique=False
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f("ix_message_conversation_id"), table_name="message")
    op.drop_table("message")
    op.drop_index(op.f("ix_tool_user_id"), table_name="tool")
    op.drop_table("tool")
    op.drop_index(op.f("ix_knowledge_user_id"), table_name="knowledge")
    op.drop_table("knowledge")
    op.drop_index(op.f("ix_file_user_id"), table_name="file")
    op.drop_index(op.f("ix_file_file_path"), table_name="file")
    op.drop_index(op.f("ix_file_file_name"), table_name="file")
    op.drop_table("file")
    op.drop_index(op.f("ix_conversation_user_id"), table_name="conversation")
    op.drop_table("conversation")
    op.drop_table("user")
    # ### end Alembic commands ###
