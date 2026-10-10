from src.ingestion.embedder import load_vector_store

store = load_vector_store()

# Pull a sample of text chunks and table chunks
text_results  = store.similarity_search("assembly areas live load", k=5,)
table_results = store.similarity_search("assembly areas live load", k=5,
                                         filter={"type": "table"})

print("=" * 60)
print("TEXT CHUNKS")
print("=" * 60)
for i, doc in enumerate(text_results, 1):
    page = doc.metadata.get("page", "?")
    dtype = doc.metadata.get("type", "?")
    print(f"\n[{i}] page={page} type={dtype}")
    print(doc.page_content[:400])
    print("─" * 40)

print("\n" + "=" * 60)
print("TABLE CHUNKS")
print("=" * 60)
for i, doc in enumerate(table_results, 1):
    page = doc.metadata.get("page", "?")
    dtype = doc.metadata.get("type", "?")
    print(f"\n[{i}] page={page} type={dtype}")
    print(doc.page_content[:400])
    print("─" * 40)