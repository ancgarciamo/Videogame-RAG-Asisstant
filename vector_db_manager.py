import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from config import VECTORDB_CONFIG, EMBEDDING_MODEL
import logging


class VectorDBManager:
    """Manager for ChromaDB vector database operations"""

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.initialize_db()

    def initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=VECTORDB_CONFIG['path'])

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=VECTORDB_CONFIG['collection_name'],
                metadata={"description": "Game descriptions and metadata for semantic search"}
            )

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            logging.info("✅ Vector database initialized successfully")

        except Exception as e:
            logging.error(f"❌ Failed to initialize vector database: {e}")
            raise

    def get_embedding(self, text):
        """Generate embedding for text"""
        if not text or not text.strip():
            return None
        return self.embedding_model.encode(text).tolist()

    def add_game(self, game_id, name, description, genres, platforms, metadata=None):
        """Add a game to the vector database"""
        try:
            # Create document text for embedding
            document_text = f"{name}. {description}. Genres: {', '.join(genres)}. Platforms: {', '.join(platforms)}"

            # Generate embedding
            embedding = self.get_embedding(document_text)
            if not embedding:
                return False

            # Prepare metadata
            game_metadata = {
                "name": name,
                "genres": ", ".join(genres),
                "platforms": ", ".join(platforms),
                "game_id": str(game_id)
            }

            # Add custom metadata if provided
            if metadata:
                game_metadata.update(metadata)

            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[game_metadata],
                ids=[str(game_id)]
            )

            logging.info(f"✅ Added game to vector DB: {name} (ID: {game_id})")
            return True

        except Exception as e:
            logging.error(f"❌ Failed to add game to vector DB: {e}")
            return False

    def semantic_search(self, query, n_results=5, genre_filter=None, platform_filter=None):
        """Search games using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []

            # Prepare filters
            where_filter = {}
            if genre_filter:
                where_filter["genres"] = {"$contains": genre_filter}
            if platform_filter:
                where_filter["platforms"] = {"$contains": platform_filter}

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter if where_filter else None
            )

            # Format results
            formatted_results = []
            if results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'game_id': results['ids'][0][i],
                        'name': results['metadatas'][0][i]['name'],
                        'genres': results['metadatas'][0][i]['genres'],
                        'platforms': results['metadatas'][0][i]['platforms'],
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'document': results['documents'][0][i] if results['documents'] else None
                    })

            return formatted_results

        except Exception as e:
            logging.error(f"❌ Semantic search failed: {e}")
            return []

    def find_similar_games(self, game_id, n_results=5):
        """Find games similar to a specific game"""
        try:
            results = self.collection.get(ids=[str(game_id)])
            if not results['ids']:
                return []

            # Use the game's own embedding to find similar games
            similar_results = self.collection.query(
                query_embeddings=[results['embeddings'][0]],
                n_results=n_results + 1,  # +1 because it will include the original game
                where={"game_id": {"$ne": str(game_id)}}  # Exclude the original game
            )

            formatted_results = []
            if similar_results['ids'] and len(similar_results['ids']) > 0:
                for i in range(len(similar_results['ids'][0])):
                    formatted_results.append({
                        'game_id': similar_results['ids'][0][i],
                        'name': similar_results['metadatas'][0][i]['name'],
                        'genres': similar_results['metadatas'][0][i]['genres'],
                        'platforms': similar_results['metadatas'][0][i]['platforms'],
                        'similarity_score': 1 - similar_results['distances'][0][i]  # Convert distance to similarity
                    })

            return formatted_results[:n_results]  # Return only n_results

        except Exception as e:
            logging.error(f"❌ Similar games search failed: {e}")
            return []

    def get_collection_stats(self):
        """Get statistics about the vector database collection"""
        try:
            count = self.collection.count()
            return {
                "total_games": count,
                "collection_name": VECTORDB_CONFIG['collection_name']
            }
        except Exception as e:
            logging.error(f"❌ Failed to get collection stats: {e}")
            return {"total_games": 0, "collection_name": VECTORDB_CONFIG['collection_name']}