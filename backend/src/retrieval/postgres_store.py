"""PostgreSQL-backed document storage using pgvector and native full-text search."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.documents import Document

from src.config import Config


class PostgresVectorStore:
    def __init__(self, connection_string: str, embedding_function: Any):
        from psycopg_pool import ConnectionPool

        self.embedding_function = embedding_function
        self.collection_name = Config.VECTOR_COLLECTION
        self.pool = ConnectionPool(
            conninfo=connection_string,
            min_size=1,
            max_size=Config.POSTGRES_POOL_SIZE,
            kwargs={"autocommit": True},
            open=True,
        )
        self.pool.wait(timeout=10)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self.pool.connection() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    collection TEXT NOT NULL,
                    id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({int(Config.EMBEDDING_DIMENSION)}) NOT NULL,
                    PRIMARY KEY (collection, id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rag_documents_embedding_hnsw "
                "ON rag_documents USING hnsw (embedding vector_cosine_ops)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rag_documents_metadata_gin "
                "ON rag_documents USING gin (metadata)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rag_documents_content_fts "
                "ON rag_documents USING gin (to_tsvector('english', content))"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    microsoft_tenant_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_id TEXT NOT NULL REFERENCES app_users(id),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS project_memberships (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'user', 'readonly')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (project_id, user_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_document_versions (
                    project_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'superseded', 'error')),
                    uploaded_by TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (project_id, filename, version),
                    UNIQUE (project_id, filename, file_hash)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS question_logs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
                    provider TEXT NOT NULL DEFAULT '',
                    estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
                    request_id TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS question_logs_project_created "
                "ON question_logs (project_id, created_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    results JSONB NOT NULL,
                    release_passed BOOLEAN NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    def upsert_microsoft_user(
        self, user_id: str, email: str, display_name: str, tenant_id: str
    ) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO app_users (id, email, display_name, microsoft_tenant_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    email = EXCLUDED.email,
                    display_name = EXCLUDED.display_name,
                    microsoft_tenant_id = EXCLUDED.microsoft_tenant_id
                """,
                (user_id, email, display_name, tenant_id),
            )
            has_project = connection.execute(
                "SELECT 1 FROM project_memberships WHERE user_id = %s LIMIT 1",
                (user_id,),
            ).fetchone()
            if not has_project:
                project_id = hashlib.sha256(f"{tenant_id}:{user_id}".encode()).hexdigest()[:32]
                connection.execute(
                    "INSERT INTO projects (id, name, owner_id) VALUES (%s, %s, %s) "
                    "ON CONFLICT (id) DO NOTHING",
                    (project_id, "My Project", user_id),
                )
                connection.execute(
                    "INSERT INTO project_memberships (project_id, user_id, role) "
                    "VALUES (%s, %s, 'admin') ON CONFLICT DO NOTHING",
                    (project_id, user_id),
                )

    def list_user_projects(self, user_id: str) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT p.id, p.name, m.role, p.created_at
                FROM projects p
                JOIN project_memberships m ON m.project_id = p.id
                WHERE m.user_id = %s
                ORDER BY p.created_at, p.id
                """,
                (user_id,),
            ).fetchall()
        return [
            {"id": row[0], "name": row[1], "role": row[2], "created_at": row[3].isoformat()}
            for row in rows
        ]

    def create_project(self, project_id: str, name: str, owner_id: str) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                "INSERT INTO projects (id, name, owner_id) VALUES (%s, %s, %s)",
                (project_id, name, owner_id),
            )
            connection.execute(
                "INSERT INTO project_memberships (project_id, user_id, role) "
                "VALUES (%s, %s, 'admin')",
                (project_id, owner_id),
            )

    def get_project_role(self, project_id: str, user_id: str) -> str | None:
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT role FROM project_memberships WHERE project_id = %s AND user_id = %s",
                (project_id, user_id),
            ).fetchone()
        return str(row[0]) if row else None

    def add_project_member(self, project_id: str, user_id: str, role: str) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO project_memberships (project_id, user_id, role)
                VALUES (%s, %s, %s)
                ON CONFLICT (project_id, user_id) DO UPDATE SET role = EXCLUDED.role
                """,
                (project_id, user_id, role),
            )

    def find_user_by_email_and_tenant(self, email: str, tenant_id: str) -> str | None:
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT id FROM app_users WHERE lower(email) = lower(%s) "
                "AND microsoft_tenant_id = %s",
                (email, tenant_id),
            ).fetchone()
        return str(row[0]) if row else None

    def list_project_members(self, project_id: str) -> list[dict[str, str]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT u.id, u.email, u.display_name, m.role
                FROM project_memberships m
                JOIN app_users u ON u.id = m.user_id
                WHERE m.project_id = %s
                ORDER BY u.email
                """,
                (project_id,),
            ).fetchall()
        return [
            {"user_id": row[0], "email": row[1], "display_name": row[2], "role": row[3]}
            for row in rows
        ]

    def create_document_version(
        self, project_id: str, filename: str, file_hash: str, uploaded_by: str
    ) -> int:
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"{project_id}:{filename}",),
                )
                existing = connection.execute(
                    """
                    SELECT version, status FROM rag_document_versions
                    WHERE project_id = %s AND filename = %s AND file_hash = %s
                    """,
                    (project_id, filename, file_hash),
                ).fetchone()
                if existing:
                    if existing[1] == "error":
                        connection.execute(
                            """
                            UPDATE rag_document_versions
                            SET status = 'pending', uploaded_by = %s
                            WHERE project_id = %s AND filename = %s AND file_hash = %s
                            """,
                            (uploaded_by, project_id, filename, file_hash),
                        )
                    version = int(existing[0])
                else:
                    row = connection.execute(
                        """
                        SELECT COALESCE(MAX(version), 0) + 1
                        FROM rag_document_versions
                        WHERE project_id = %s AND filename = %s
                        """,
                        (project_id, filename),
                    ).fetchone()
                    version = int(row[0])
                    connection.execute(
                        """
                        INSERT INTO rag_document_versions
                            (project_id, filename, file_hash, version, status, uploaded_by)
                        VALUES (%s, %s, %s, %s, 'pending', %s)
                        """,
                        (project_id, filename, file_hash, version, uploaded_by),
                    )
        return version

    def activate_document_version(
        self, project_id: str, filename: str, file_hash: str
    ) -> None:
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"{project_id}:{filename}",),
                )
                requested = connection.execute(
                    """
                    SELECT version FROM rag_document_versions
                    WHERE project_id = %s AND filename = %s AND file_hash = %s
                    """,
                    (project_id, filename, file_hash),
                ).fetchone()
                active = connection.execute(
                    """
                    SELECT COALESCE(MAX(version), 0)
                    FROM rag_document_versions
                    WHERE project_id = %s AND filename = %s AND status = 'active'
                    """,
                    (project_id, filename),
                ).fetchone()
                if not requested or int(requested[0]) < int(active[0]):
                    return
                connection.execute(
                    """
                    UPDATE rag_document_versions
                    SET status = CASE WHEN file_hash = %s THEN 'active' ELSE 'superseded' END
                    WHERE project_id = %s AND filename = %s
                      AND status IN ('active', 'pending', 'superseded')
                      AND version <= %s
                    """,
                    (file_hash, project_id, filename, int(requested[0])),
                )
                connection.execute(
                    """
                    UPDATE rag_documents
                    SET metadata = jsonb_set(metadata, '{active_version}', to_jsonb(file_hash = %s))
                    WHERE metadata->>'project_id' = %s AND metadata->>'filename' = %s
                    """,
                    (file_hash, project_id, filename),
                )

    def fail_document_version(
        self, project_id: str, filename: str, file_hash: str
    ) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                UPDATE rag_document_versions SET status = 'error'
                WHERE project_id = %s AND filename = %s AND file_hash = %s AND status = 'pending'
                """,
                (project_id, filename, file_hash),
            )

    def document_version_history(self, project_id: str, filename: str) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT file_hash, version, status, uploaded_by, created_at
                FROM rag_document_versions
                WHERE project_id = %s AND filename = %s
                ORDER BY version DESC
                """,
                (project_id, filename),
            ).fetchall()
        return [
            {
                "file_hash": row[0],
                "version": row[1],
                "status": row[2],
                "uploaded_by": row[3],
                "created_at": row[4].isoformat(),
            }
            for row in rows
        ]

    def list_project_document_versions(self, project_id: str) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT ON (filename)
                    filename, file_hash, version, status, uploaded_by, created_at
                FROM rag_document_versions
                WHERE project_id = %s
                ORDER BY filename, version DESC
                """,
                (project_id,),
            ).fetchall()
        return [
            {
                "filename": row[0], "file_hash": row[1], "version": row[2],
                "status": row[3], "uploaded_by": row[4],
                "created_at": row[5].isoformat(),
            }
            for row in rows
        ]

    def delete_document_versions(self, project_id: str, filename: str) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                "DELETE FROM rag_document_versions WHERE project_id = %s AND filename = %s",
                (project_id, filename),
            )

    def log_question(self, entry: dict[str, Any]) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO question_logs (
                    id, project_id, user_id, question, answer, status, error,
                    latency_ms, provider, estimated_cost_usd, request_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    entry["id"], entry["project_id"], entry["user_id"],
                    entry["question"], entry.get("answer", ""), entry["status"],
                    entry.get("error", ""), entry.get("latency_ms", 0),
                    entry.get("provider", ""), entry.get("estimated_cost_usd", 0),
                    entry.get("request_id", ""),
                ),
            )

    def list_question_logs(self, project_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, question, answer, status, error, latency_ms,
                       provider, estimated_cost_usd, request_id, created_at
                FROM question_logs
                WHERE project_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (project_id, limit),
            ).fetchall()
        return [
            {
                "id": row[0], "user_id": row[1], "question": row[2],
                "answer": row[3], "status": row[4], "error": row[5],
                "latency_ms": row[6], "provider": row[7],
                "estimated_cost_usd": row[8], "request_id": row[9],
                "created_at": row[10].isoformat(),
            }
            for row in rows
        ]

    def save_evaluation(
        self,
        run_id: str,
        project_id: str,
        created_by: str,
        results: dict[str, Any],
        release_passed: bool,
    ) -> None:
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO evaluation_runs
                    (id, project_id, created_by, results, release_passed)
                VALUES (%s, %s, %s, %s::jsonb, %s)
                """,
                (run_id, project_id, created_by, json.dumps(results, default=str), release_passed),
            )

    def list_evaluations(self, project_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, created_by, results, release_passed, created_at
                FROM evaluation_runs
                WHERE project_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (project_id, limit),
            ).fetchall()
        return [
            {
                "id": row[0], "created_by": row[1], "results": row[2],
                "release_passed": row[3], "created_at": row[4].isoformat(),
            }
            for row in rows
        ]

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        if not documents:
            return []
        if ids is None:
            generated_ids = []
            for i, doc in enumerate(documents):
                identity = (
                    f"{doc.metadata.get('source', '')}\0"
                    f"{doc.metadata.get('chunk_index', i)}\0{doc.page_content}"
                )
                generated_ids.append(hashlib.sha256(identity.encode()).hexdigest())
            ids = generated_ids
        if len(ids) != len(documents):
            raise ValueError("Each document must have exactly one ID")

        vectors = self.embedding_function.embed_documents(
            [document.page_content for document in documents]
        )
        records = [
            (
                self.collection_name,
                document_id,
                document.page_content,
                json.dumps(document.metadata, ensure_ascii=False, default=str),
                self._vector_literal(vector),
            )
            for document_id, document, vector in zip(ids, documents, vectors, strict=True)
        ]
        with self.pool.connection() as connection:
            connection.executemany(
                """
                INSERT INTO rag_documents (collection, id, content, metadata, embedding)
                VALUES (%s, %s, %s, %s::jsonb, %s::vector)
                ON CONFLICT (collection, id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                records,
            )
        return ids

    def delete(
        self,
        ids: list[str] | None = None,
        filename: str | None = None,
        project_id: str = "default",
    ) -> int:
        if ids is not None:
            if not ids:
                return 0
            with self.pool.connection() as connection:
                cursor = connection.execute(
                    "DELETE FROM rag_documents WHERE collection = %s "
                    "AND metadata->>'project_id' = %s AND id = ANY(%s)",
                    (self.collection_name, project_id, ids),
                )
                return cursor.rowcount
        if filename is not None:
            with self.pool.connection() as connection:
                cursor = connection.execute(
                    "DELETE FROM rag_documents WHERE collection = %s "
                    "AND metadata->>'project_id' = %s AND metadata->>'filename' = %s",
                    (self.collection_name, project_id, filename),
                )
                return cursor.rowcount
        raise ValueError("Pass IDs or a filename to delete")

    def count(self) -> int:
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT count(*) FROM rag_documents WHERE collection = %s",
                (self.collection_name,),
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self.pool.close()

    def list_documents(
        self,
        limit: int = 100_000,
        filename: str | None = None,
        project_id: str = "default",
        active_only: bool = False,
    ) -> list[Document]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                "SELECT id, content, metadata FROM rag_documents "
                "WHERE collection = %s AND metadata->>'project_id' = %s "
                "AND (%s::text IS NULL OR metadata->>'filename' = %s) "
                "AND (NOT %s OR metadata->>'active_version' = 'true') "
                "ORDER BY id LIMIT %s",
                (self.collection_name, project_id, filename, filename, active_only, limit),
            ).fetchall()
        documents = []
        for document_id, content, metadata in rows:
            document_metadata = dict(metadata or {})
            document_metadata["_vector_id"] = document_id
            documents.append(Document(page_content=content, metadata=document_metadata))
        return documents

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        project_id: str = "default",
    ) -> list[tuple[Document, float]]:
        query_vector = self._vector_literal(self.embedding_function.embed_query(query))
        filters = dict(metadata_filter or {})
        filters["project_id"] = project_id
        filters["active_version"] = True
        filter_json = json.dumps(filters, ensure_ascii=False)
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, content, metadata, 1 - (embedding <=> %s::vector) AS score
                FROM rag_documents
                WHERE collection = %s AND metadata @> %s::jsonb
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, self.collection_name, filter_json, query_vector, k),
            ).fetchall()
        return [
            (self._document(row[0], row[1], row[2]), float(row[3]))
            for row in rows
        ]

    def keyword_search(
        self, query: str, k: int = 5, project_id: str = "default"
    ) -> list[tuple[Document, float]]:
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, content, metadata,
                      ts_rank_cd(
                          to_tsvector('english', content),
                          plainto_tsquery('english', %s)
                      ) AS score
                FROM rag_documents
                WHERE collection = %s
                  AND metadata->>'project_id' = %s
                  AND metadata->>'active_version' = 'true'
                  AND to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, self.collection_name, project_id, query, k),
            ).fetchall()
        return [
            (self._document(row[0], row[1], row[2]), float(row[3]))
            for row in rows
        ]

    @staticmethod
    def _document(document_id: str, content: str, metadata: dict[str, Any] | None) -> Document:
        document_metadata = dict(metadata or {})
        document_metadata["_vector_id"] = document_id
        return Document(page_content=content, metadata=document_metadata)