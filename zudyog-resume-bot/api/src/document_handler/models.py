from dataclasses import dataclass
from typing import Optional


@dataclass
class IndexingResult:
    """Class to hold the result of indexing operation"""
    success: bool
    paragraphs_indexed: int
    error_message: Optional[str] = None
