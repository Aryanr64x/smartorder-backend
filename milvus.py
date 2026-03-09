from pymilvus import connections, FieldSchema, Collection, CollectionSchema, DataType

connections.connect(
    alias="default",
    uri="https://in03-93638f61fbeda9f.serverless.aws-eu-central-1.cloud.zilliz.com",
    token="acba16f86d47ecd9a8bd89b9244e7ddeea4b1df5503d6ec194d6762ba8885022d1485b7416664e587395371e28fee67c6753c9f3"
)





fields = [
    FieldSchema(
        name = 'id',
        dtype=DataType.INT64,
        is_primary=True,
        auto_id = True         
    ),

    FieldSchema(
        name='embedding', 
        dtype=DataType.FLOAT_VECTOR,
        dim = 384
        ),
    FieldSchema(
        name='text',
        dtype=DataType.VARCHAR,
        max_length = 1024
    ),
    FieldSchema(
        name="description",
        dtype=DataType.VARCHAR,
        max_length = 10000
    )
]



schema = CollectionSchema(fields=fields, description='Menu RAG text embeddings with descriptions')
collection = Collection(schema=schema, name='hydmenu3')

# collection.create_index(
#     field_name="embedding",
#     index_params={
#         "index_type": "HNSW",
#         "metric_type": "COSINE",
#         "params": {"M": 8, "efConstruction": 64}
#     }
# )

collection.load()
