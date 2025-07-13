"""
FAISS vector store integration with LangChain.
Handles vector storage and semantic search operations.
"""
import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .utils import generate_uuid
from .cleaner import preprocess_for_embedding


class FAISSStore:
    """FAISS vector store manager for semantic search."""
    
    ##TODO: Will probably need to change the embedding model to the one from Qualcomm AI Hub
    def __init__(self, index_dir: str = "faiss_index", embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize FAISS store with embedding model.
        
        Args:
            index_dir: Directory to store FAISS index files
            embedding_model_name: HuggingFace embedding model name
        """
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model_name
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vector store
        self.vector_store = None
        self.node_id_mapping = {}  # Maps FAISS document IDs to node IDs
        
        # Create index directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)
        
        # Try to load existing index
        self._load_index()
    
    def _load_index(self) -> bool:
        """
        Load existing FAISS index from disk.
        
        Returns:
            True if index loaded successfully, False otherwise
        """
        try:
            if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
                self.vector_store = FAISS.load_local(
                    self.index_dir, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                
                # Load node ID mapping
                mapping_file = os.path.join(self.index_dir, "node_mapping.txt")
                if os.path.exists(mapping_file):
                    with open(mapping_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) == 2:
                                doc_id, node_id = parts
                                self.node_id_mapping[int(doc_id)] = node_id
                
                return True
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
        
        return False
    
    def _save_index(self) -> bool:
        """
        Save FAISS index to disk.
        
        Returns:
            True if index saved successfully, False otherwise
        """
        try:
            if self.vector_store is not None:
                self.vector_store.save_local(self.index_dir)
                
                # Save node ID mapping
                mapping_file = os.path.join(self.index_dir, "node_mapping.txt")
                with open(mapping_file, 'w') as f:
                    for doc_id, node_id in self.node_id_mapping.items():
                        f.write(f"{doc_id}\t{node_id}\n")
                
                return True
        except Exception as e:
            print(f"Failed to save FAISS index: {e}")
        
        return False
    
    def add_node(self, node_id: str, summary: str, full_data: str) -> bool:
        """
        Add a node to the FAISS index.
        
        Args:
            node_id: Unique identifier for the node
            summary: Summary text to embed and search against
            full_data: Full session data (stored as metadata)
            
        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Preprocess summary for embedding
            processed_summary = preprocess_for_embedding(summary)
            
            if not processed_summary:
                return False
            
            # Create document with metadata
            doc = Document(
                page_content=processed_summary,
                metadata={
                    'node_id': node_id,
                    'summary': summary,
                    'full_data': full_data
                }
            )
            
            # Add to vector store
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents([doc], self.embedding_model)
                doc_id = 0
            else:
                # Add to existing vector store
                doc_id = len(self.vector_store.docstore._dict)
                self.vector_store.add_documents([doc])
            
            # Update node ID mapping
            self.node_id_mapping[doc_id] = node_id
            
            # Save index
            self._save_index()
            
            return True
            
        except Exception as e:
            print(f"Failed to add node to FAISS: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 5, filter: dict = None) -> List[Tuple[str, str, float]]:
        """
        Search for similar nodes using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return (default: 5)
            filter: Optional metadata filter dictionary
            
        Returns:
            List of tuples (node_id, summary, score)
            Note: Lower scores indicate higher similarity (L2 distance)
        """
        if self.vector_store is None:
            return []
        
        try:
            # Preprocess query
            processed_query = preprocess_for_embedding(query)
            
            if not processed_query:
                return []
            
            # Search vector store - this is the correct LangChain method
            results = self.vector_store.similarity_search_with_score(
                query=processed_query,
                k=k,
                filter=filter
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                node_id = doc.metadata.get('node_id', '')
                summary = doc.metadata.get('summary', '')
                formatted_results.append((node_id, summary, score))
            
            return formatted_results
            
        except Exception as e:
            print(f"Failed to search FAISS: {e}")
            return []
    
    def get_node_ids_by_similarity(self, query: str, k: int = 5, filter: dict = None) -> List[str]:
        """
        Get node IDs for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter dictionary
            
        Returns:
            List of node IDs
        """
        results = self.search_similar(query, k, filter)
        return [node_id for node_id, _, _ in results]
    
    def initialize_from_nodes(self, nodes: List[Dict[str, Any]]) -> bool:
        """
        Initialize FAISS index from existing nodes.
        
        Args:
            nodes: List of node dictionaries from database
            
        Returns:
            True if initialized successfully, False otherwise
        """
        if not nodes:
            return True
        
        try:
            documents = []
            self.node_id_mapping = {}
            
            for i, node in enumerate(nodes):
                # Preprocess summary for embedding
                processed_summary = preprocess_for_embedding(node['summary'])
                
                if processed_summary:
                    doc = Document(
                        page_content=processed_summary,
                        metadata={
                            'node_id': node['id'],
                            'summary': node['summary'],
                            'full_data': node['full_data']
                        }
                    )
                    documents.append(doc)
                    self.node_id_mapping[i] = node['id']
            
            if documents:
                # Create vector store from documents
                self.vector_store = FAISS.from_documents(documents, self.embedding_model)
                
                # Save index
                self._save_index()
                
                return True
            
        except Exception as e:
            print(f"Failed to initialize FAISS from nodes: {e}")
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.vector_store is None:
            return {
                'total_documents': 0,
                'embedding_dimension': 0,
                'index_exists': False
            }
        
        try:
            total_docs = len(self.vector_store.docstore._dict)
            embedding_dim = self.vector_store.index.d if hasattr(self.vector_store.index, 'd') else 0
            
            return {
                'total_documents': total_docs,
                'embedding_dimension': embedding_dim,
                'index_exists': True,
                'mapping_count': len(self.node_id_mapping)
            }
        except Exception as e:
            return {
                'total_documents': 0,
                'embedding_dimension': 0,
                'index_exists': False,
                'error': str(e)
            }
    
    def clear_index(self) -> bool:
        """
        Clear the FAISS index and all mappings.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            self.vector_store = None
            self.node_id_mapping = {}
            
            # Remove index files
            if os.path.exists(os.path.join(self.index_dir, "index.faiss")):
                os.remove(os.path.join(self.index_dir, "index.faiss"))
            
            if os.path.exists(os.path.join(self.index_dir, "index.pkl")):
                os.remove(os.path.join(self.index_dir, "index.pkl"))
            
            mapping_file = os.path.join(self.index_dir, "node_mapping.txt")
            if os.path.exists(mapping_file):
                os.remove(mapping_file)
            
            return True
            
        except Exception as e:
            print(f"Failed to clear FAISS index: {e}")
            return False 