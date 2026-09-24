"""Rebuild the Milvus `smartorder` collection so its sql_id references match the
freshly-seeded Supabase menu rows.

The old vectors point at deleted row ids (e.g. 103, 123 ...), so the RAG/hybrid
path retrieves ids that no longer exist and returns zero items. This drops the
stale collection and re-embeds every current menu row with its new id.

Run:  ./venv/Scripts/python.exe milvus_reseed.py
"""
from dotenv import load_dotenv

# Must run BEFORE `models` is imported: models.py reads HUGGINGFACE_TOKEN at
# import time, so a later load_dotenv() would be too late to help it.
load_dotenv()

from pymilvus import (
    connections, utility, Collection, FieldSchema, CollectionSchema, DataType,
)
from models import embedding_model
from supabase_client import supabase

NAME = "smartorder"

# Connect directly (can't `import milvus` — that module instantiates the
# existing collection and would raise SchemaInconsistent before we can drop it).
connections.connect(
    alias="default",
    uri="https://in03-93638f61fbeda9f.serverless.aws-eu-central-1.cloud.zilliz.com",
    token="acba16f86d47ecd9a8bd89b9244e7ddeea4b1df5503d6ec194d6762ba8885022d1485b7416664e587395371e28fee67c6753c9f3",
)


def main():
    # 1. Embed every current menu row FIRST (fall back to name when description
    #    empty). This happens before the drop on purpose: embedding is the step
    #    that talks to a third party and fails (expired HUGGINGFACE_TOKEN -> 401).
    #    Dropping first would leave the collection gone AND unrebuilt.
    rows = supabase.table("menu").select("*").execute().data
    print("menu rows to embed:", len(rows))
    embeddings, sql_ids, restaurant_ids = [], [], []
    for r in rows:
        text = (r.get("description") or "").strip() or r["name"]
        embeddings.append(embedding_model.feature_extraction(text))
        sql_ids.append(r["id"])
        restaurant_ids.append(r["restaurant_id"])
    print(f"embedded {len(embeddings)} rows — safe to replace the collection")

    # 2. Drop the stale collection (old schema / old sql_ids) if present.
    if utility.has_collection(NAME):
        utility.drop_collection(NAME)
        print("dropped stale collection")

    # 3. Recreate with the same schema as milvus.py.
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="sql_id", dtype=DataType.INT64),
        FieldSchema(name="restaurant_id", dtype=DataType.INT64),
    ]
    # NOTE: description must match milvus.py exactly, else the app's
    # `Collection(schema=..., name='smartorder')` raises SchemaInconsistent.
    schema = CollectionSchema(fields=fields, description="Menu RAG text embeddings with descriptions")
    col = Collection(schema=schema, name=NAME)

    # 4. Insert (order matches non-auto schema fields: embedding, sql_id,
    #    restaurant_id).
    col.insert([embeddings, sql_ids, restaurant_ids])
    col.flush()

    # 5. Index + load so searches work.
    col.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 8, "efConstruction": 64},
        },
    )
    col.load()
    print(f"seeded milvus with {len(sql_ids)} vectors (sql_ids {min(sql_ids)}..{max(sql_ids)})")


if __name__ == "__main__":
    main()
