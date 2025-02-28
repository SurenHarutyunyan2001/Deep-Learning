from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import BM25Retriever, DenseRetriever
from haystack.pipelines import Pipeline

# Создаем FAISS Document Store
document_store = FAISSDocumentStore(embedding_dim=384)  # Убедись, что embedding_dim соответствует модели

# Sparse Retriever (BM25)
bm25_retriever = BM25Retriever(document_store=document_store)

# Dense Retriever (BERT)
dense_retriever = DenseRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")

# Гибридный пайплайн
hybrid_pipeline = Pipeline()
hybrid_pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
hybrid_pipeline.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
