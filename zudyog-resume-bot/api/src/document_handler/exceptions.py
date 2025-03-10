class PDFIndexingError(Exception):
    """Custom exception for PDF indexing errors"""
    pass


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class MetadataValidationError(Exception):
    """Custom exception for metadata validation errors."""
    pass


class EmbeddingGenerationError(Exception):
    """Custom exception for embedding generation errors."""
    pass


class PineconeUpsertError(Exception):
    """Custom exception for Pinecone upsert errors."""
    pass
