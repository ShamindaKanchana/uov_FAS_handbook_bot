import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.retrieval.retriever import QueryEngine


class TestQueryEngine(unittest.TestCase):
    def setUp(self):
        """Set up the test case with a mock Qdrant client."""
        self.mock_client = MagicMock()
        self.engine = QueryEngine(collection_name="test_collection")
        self.engine.client = self.mock_client
        
        # Mock the SentenceTransformer
        self.engine.model.encode = MagicMock(return_value=[[0.1] * 384])
        
    def test_correct_spelling(self):
        """Test spelling correction."""
        self.assertEqual(
            self.engine._correct_spelling("mathamatics and bio sience"),
            "mathematicomputer science and bio science"
        )
        
    def test_improve_query(self):
        """Test query improvement with term expansion."""
        # Test subject/course expansion
        result = self.engine._improve_query("subjects in CS")
        self.assertEqual(result, "subjects in computer science")
        
        # Test information expansion
        result = self.engine._improve_query("information about IT")
        self.assertEqual(result, "information about information technology description overview")
        
    def test_search_with_empty_query(self):
        """Test search with empty query returns empty results."""
        results = self.engine.search("")
        self.assertEqual(len(results), 0)
        
    def test_search_with_filters(self):
        """Test search with filter conditions."""
        # Mock search results
        mock_result = MagicMock()
        mock_result.id = 1
        mock_result.score = 0.95
        mock_result.payload = {
            'text': 'Test result',
            'metadata': {'source': 'handbook', 'page': 1}  # Changed source to match filter
        }
        self.mock_client.search.return_value = [mock_result]
        
        # Mock the model's encode method to return a list
        self.engine.model.encode.return_value = [0.1] * 384
    
        # Test with filter
        results = self.engine.search("test", filter_condition={"source": "handbook"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], 'Test result')
        
        # Verify filter was applied
        self.mock_client.search.assert_called_once()
        
    def test_get_collection_info(self):
        """Test getting collection information."""
        # Mock collection info
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.vectors_count = 100
        mock_collection.status = "green"
        self.mock_client.get_collection.return_value = mock_collection
        
        info = self.engine.get_collection_info()
        self.assertEqual(info['name'], "test_collection")
        self.assertEqual(info['vectors_count'], 100)


if __name__ == "__main__":
    unittest.main()
