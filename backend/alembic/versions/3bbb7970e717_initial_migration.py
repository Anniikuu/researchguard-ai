"""Initial migration

Revision ID: 3bbb7970e717
Revises: 
Create Date: 2026-09-24 00:55:28.851102

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '3bbb7970e717'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Safe PL/pgSQL block to try enabling vector extension without aborting transaction
    op.execute("""
    DO $$
    BEGIN
        CREATE EXTENSION IF NOT EXISTS vector;
    EXCEPTION WHEN OTHERS THEN
        -- Extension not installed on system, ignore
        NULL;
    END $$;
    """)

    # Check if vector extension is active in database
    conn = op.get_bind()
    res = conn.execute(sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector';")).fetchone()
    has_vector = (res is not None)

    embedding_col_type = Vector(384) if has_vector else postgresql.ARRAY(sa.Float())

    # 1. documents table
    op.create_table(
        'documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('filename', sa.String(), nullable=False),
        sa.Column('original_filename', sa.String(), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('page_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(), nullable=False, server_default='processing'),
        sa.Column('uploaded_at', sa.DateTime(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
    )

    # 2. document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', embedding_col_type, nullable=True),
        sa.Column('token_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    # 3. questions table
    op.create_table(
        'questions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('question_text', sa.Text(), nullable=False),
        sa.Column('method', sa.String(), nullable=False, server_default='baseline'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    # 4. answers table
    op.create_table(
        'answers',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('question_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('questions.id', ondelete='CASCADE'), nullable=False, unique=True),
        sa.Column('answer_text', sa.Text(), nullable=False),
        sa.Column('context_used', sa.Text(), nullable=False),
        sa.Column('num_claims', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('num_supported', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('num_unsupported', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_confidence', sa.Float(), nullable=True),
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('method', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    # 5. ml_models table
    op.create_table(
        'ml_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_name', sa.String(), nullable=False),
        sa.Column('model_type', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=False),
        sa.Column('feature_names', sa.Text(), nullable=False),
        sa.Column('training_samples', sa.Integer(), nullable=False),
        sa.Column('test_samples', sa.Integer(), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('precision_score', sa.Float(), nullable=True),
        sa.Column('recall', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('roc_auc', sa.Float(), nullable=True),
        sa.Column('confusion_matrix', sa.Text(), nullable=True),
        sa.Column('hyperparameters', sa.Text(), nullable=True),
        sa.Column('trained_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='false'),
    )

    # 6. claims table
    op.create_table(
        'claims',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('answer_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('answers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('claim_index', sa.Integer(), nullable=False),
        sa.Column('claim_text', sa.Text(), nullable=False),
        sa.Column('verification_status', sa.String(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('verification_method', sa.String(), nullable=False),
        sa.Column('ml_model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('ml_models.id', ondelete='SET NULL'), nullable=True),
        sa.Column('cosine_similarity', sa.Float(), nullable=True),
        sa.Column('tfidf_similarity', sa.Float(), nullable=True),
        sa.Column('keyword_overlap', sa.Float(), nullable=True),
        sa.Column('claim_length', sa.Integer(), nullable=True),
        sa.Column('evidence_length', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    # 7. claim_evidence table
    op.create_table(
        'claim_evidence',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('claim_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('claims.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('document_chunks.id', ondelete='SET NULL'), nullable=True),
        sa.Column('evidence_text', sa.Text(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )

    # 8. evaluation_questions table
    op.create_table(
        'evaluation_questions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False),
        sa.Column('question_text', sa.Text(), nullable=False),
        sa.Column('expected_answer', sa.Text(), nullable=False),
        sa.Column('expected_claims', sa.Text(), nullable=False),
        sa.Column('difficulty', sa.String(), nullable=True),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('evaluation_questions')
    op.drop_table('claim_evidence')
    op.drop_table('claims')
    op.drop_table('ml_models')
    op.drop_table('answers')
    op.drop_table('questions')
    op.drop_table('document_chunks')
    op.drop_table('documents')
    try:
        op.execute("DROP EXTENSION IF EXISTS vector;")
    except Exception:
        pass
