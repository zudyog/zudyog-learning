from datetime import datetime
from typing import Dict, Any
from .exceptions import MetadataValidationError


def validate_metadata(metadata: Dict[str, Any]) -> None:
    """
    Validates that the required metadata fields are present and valid

    Args:
        metadata: Dictionary containing metadata fields

    Raises:
        MetadataValidationError: If metadata is invalid
    """
    required_fields = ['document_id', 'date_uploaded']

    if not isinstance(metadata, dict):
        raise MetadataValidationError("Metadata must be a dictionary")

    for field in required_fields:
        if field not in metadata:
            raise MetadataValidationError(
                f"Missing required metadata field: {field}")

    if not isinstance(metadata['document_id'], str):
        raise MetadataValidationError("document_id must be a string")

    try:
        datetime.fromisoformat(metadata['date_uploaded'])
    except ValueError:
        raise MetadataValidationError(
            "date_uploaded must be a valid ISO format date string")
