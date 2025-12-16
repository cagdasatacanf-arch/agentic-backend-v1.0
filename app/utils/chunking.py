"""
Text chunking utilities for document processing.
Splits large documents into smaller chunks for better RAG performance.
"""

from typing import List, Dict
import re


class TextChunker:
    """
    Handles intelligent text chunking for document processing.
    Uses sliding window approach with overlap for context preservation.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to split on (in order of preference)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str, metadata: Dict | None = None) -> List[Dict]:
        """
        Split text into chunks with metadata.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        if not text or len(text) <= self.chunk_size:
            return [{"text": text, "metadata": metadata or {}}]

        chunks = []
        start = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If we're not at the end, try to find a good break point
            if end < len(text):
                # Try each separator in order
                best_split = end
                for separator in self.separators:
                    if not separator:
                        continue

                    # Look for separator near the end of the chunk
                    search_start = max(start, end - 100)
                    search_text = text[search_start:end]

                    if separator in search_text:
                        # Find the last occurrence of the separator
                        split_pos = search_text.rfind(separator)
                        if split_pos != -1:
                            best_split = search_start + split_pos + len(separator)
                            break

                end = best_split

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end,
                }
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

            # Move to next chunk with overlap
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    def chunk_by_sentences(self, text: str, metadata: Dict | None = None) -> List[Dict]:
        """
        Split text by sentences, grouping them into chunks.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        # Simple sentence splitting (can be improved with nltk or spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size and we have content
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": len(chunks),
                    "sentence_count": len(current_chunk),
                }
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

                # Start new chunk with overlap (keep last sentence)
                if self.chunk_overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": len(chunks),
                "sentence_count": len(current_chunk),
            }
            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks

    def chunk_by_paragraphs(
        self, text: str, metadata: Dict | None = None
    ) -> List[Dict]:
        """
        Split text by paragraphs, grouping them into chunks.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of dictionaries with 'text' and 'metadata' keys
        """
        paragraphs = text.split("\n\n")

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            # If adding this paragraph exceeds chunk size and we have content
            if current_length + para_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunk_metadata = {
                    **(metadata or {}),
                    "chunk_index": len(chunks),
                    "paragraph_count": len(current_chunk),
                }
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

                # Start new chunk
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_length + 2  # +2 for \n\n

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": len(chunks),
                "paragraph_count": len(current_chunk),
            }
            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks


# Default chunker instance
default_chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
