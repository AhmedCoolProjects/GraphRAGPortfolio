"""
ZVec Vector Store - LangChain-compatible wrapper for Alibaba's ZVec
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

import zvec


class ZVecVectorStore(VectorStore):
    """LangChain-compatible wrapper for ZVec in-process vector database."""
    
    def __init__(
        self,
        collection: Any,
        embedding: Embeddings,
        path: str,
    ):
        """Initialize ZVec vector store.
        
        Args:
            collection: ZVec collection object
            embedding: Embedding model
            path: Path to ZVec data directory
        """
        self._collection = collection
        self._embedding = embedding
        self._path = path
    
    @property
    def embeddings(self) -> Embeddings:
        return self._embedding
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        path: str = "./zvec_data",
        collection_name: str = "portfolio",
        **kwargs: Any
    ) -> "ZVecVectorStore":
        """Create a ZVec collection from texts.
        
        Args:
            texts: List of text strings
            embedding: Embedding model
            metadatas: Optional list of metadata dicts
            path: Path to store ZVec data
            collection_name: Name of the collection
            **kwargs: Additional arguments
        
        Returns:
            ZVecVectorStore instance
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = [Document(page_content=text, metadata=metadata) 
                    for text, metadata in zip(texts, metadatas)]
        
        return cls.from_documents(
            documents=documents,
            embedding=embedding,
            path=path,
            collection_name=collection_name,
            **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        path: str = "./zvec_data",
        collection_name: str = "portfolio",
        **kwargs: Any
    ) -> "ZVecVectorStore":
        """Create a new ZVec collection from documents.
        
        Args:
            documents: List of documents to add
            embedding: Embedding model
            path: Path to store ZVec data
            collection_name: Name of the collection
            **kwargs: Additional arguments
        
        Returns:
            ZVecVectorStore instance
        """
        # Ensure path exists
        os.makedirs(path, exist_ok=True)
        
        # Get embedding dimension by testing with a dummy text
        dummy_embedding = embedding.embed_query("test")
        dimension = len(dummy_embedding)
        
        # Define collection schema with fields
        content_field = zvec.FieldSchema("content", zvec.DataType.STRING)
        source_field = zvec.FieldSchema("source", zvec.DataType.STRING)
        
        schema = zvec.CollectionSchema(
            name=collection_name,
            fields=[content_field, source_field],
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, dimension),
        )
        
        # Create or open collection
        collection_path = os.path.join(path, collection_name)
        if os.path.exists(collection_path):
            # Remove existing collection to start fresh
            import shutil
            shutil.rmtree(collection_path)
        
        collection = zvec.create_and_open(path=collection_path, schema=schema)
        
        # Embed and insert documents
        texts = [doc.page_content for doc in documents]
        embeddings_list = embedding.embed_documents(texts)
        
        # Prepare documents for ZVec
        import re
        zvec_docs = []
        for i, (doc, embedding_vector) in enumerate(zip(documents, embeddings_list)):
            doc_id = doc.metadata.get("source", f"doc_{i}")
            # Sanitize ID: replace invalid characters with underscore
            # ZVec IDs must match regex: ^[a-zA-Z0-9_\-\.]+$
            sanitized_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', doc_id)
            unique_id = f"{sanitized_id}_{i}"
            
            zvec_doc = zvec.Doc(
                id=unique_id,
                vectors={"embedding": embedding_vector},
                fields={
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                }
            )
            zvec_docs.append(zvec_doc)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(zvec_docs), batch_size):
            batch = zvec_docs[i:i+batch_size]
            collection.insert(batch)
        
        return cls(collection=collection, embedding=embedding, path=collection_path)
    
    @classmethod
    def from_existing_index(
        cls,
        embedding: Embeddings,
        path: str = "./zvec_data",
        collection_name: str = "portfolio",
        **kwargs: Any
    ) -> "ZVecVectorStore":
        """Load an existing ZVec collection.
        
        Args:
            embedding: Embedding model
            path: Path to ZVec data
            collection_name: Name of the collection
            **kwargs: Additional arguments
        
        Returns:
            ZVecVectorStore instance
        """
        collection_path = os.path.join(path, collection_name)
        
        if not os.path.exists(collection_path):
            raise ValueError(f"ZVec collection not found at {collection_path}. Run ingest_data.py first.")
        
        # Open existing collection
        collection = zvec.open(path=collection_path)
        
        return cls(collection=collection, embedding=embedding, path=collection_path)
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to the collection.
        
        Args:
            documents: Documents to add
            **kwargs: Additional arguments
        
        Returns:
            List of document IDs
        """
        import re
        texts = [doc.page_content for doc in documents]
        embeddings_list = self._embedding.embed_documents(texts)
        
        zvec_docs = []
        ids = []
        for i, (doc, embedding_vector) in enumerate(zip(documents, embeddings_list)):
            doc_id = doc.metadata.get("source", f"doc_{i}")
            sanitized_id = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', doc_id)
            unique_id = f"{sanitized_id}_{i}"
            ids.append(unique_id)
            
            zvec_doc = zvec.Doc(
                id=unique_id,
                vectors={"embedding": embedding_vector},
                fields={
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                }
            )
            zvec_docs.append(zvec_doc)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(zvec_docs), batch_size):
            batch = zvec_docs[i:i+batch_size]
            self._collection.insert(batch)
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments
        
        Returns:
            List of similar documents
        """
        # Embed query
        query_embedding = self._embedding.embed_query(query)
        
        # Search
        results = self._collection.query(
            zvec.VectorQuery("embedding", vector=query_embedding),
            topk=k
        )
        
        # Convert to LangChain documents
        documents = []
        for result in results:
            content = result.fields.get("content", "")
            metadata = {k: v for k, v in result.fields.items() if k != "content"}
            metadata["score"] = result.score
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with scores.
        
        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments
        
        Returns:
            List of (document, score) tuples
        """
        # Embed query
        query_embedding = self._embedding.embed_query(query)
        
        # Search
        results = self._collection.query(
            zvec.VectorQuery("embedding", vector=query_embedding),
            topk=k
        )
        
        # Convert to LangChain documents with scores
        documents_with_scores = []
        for result in results:
            content = result.fields.get("content", "")
            metadata = {k: v for k, v in result.fields.items() if k != "content"}
            doc = Document(page_content=content, metadata=metadata)
            documents_with_scores.append((doc, result.score))
        
        return documents_with_scores
    
    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """Return a retriever for this vector store.
        
        Args:
            search_kwargs: Search configuration (e.g., {"k": 3})
            **kwargs: Additional arguments
        
        Returns:
            ZVecRetriever instance
        """
        return ZVecRetriever(vectorstore=self, search_kwargs=search_kwargs or {"k": 4})


class ZVecRetriever(BaseRetriever):
    """Simple retriever for ZVec vector store."""
    
    vectorstore: ZVecVectorStore
    search_kwargs: Dict[str, Any]
    
    def __init__(self, vectorstore: ZVecVectorStore, search_kwargs: Dict[str, Any]):
        super().__init__(vectorstore=vectorstore, search_kwargs=search_kwargs)
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query."""
        k = self.search_kwargs.get("k", 4)
        return self.vectorstore.similarity_search(query, k=k)


# Backwards compatibility alias
ZVecStore = ZVecVectorStore