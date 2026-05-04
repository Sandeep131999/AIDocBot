"""
DOCUMENT LOADER
Loads documents from files and splits them into chunks.

What it does:
1. Opens a file (PDF or TXT or CSV)
2. Reads the content
3. Splits into small chunks (500 characters each)
4. Adds metadata (source file, page number, etc.)
5. Returns ready-to-embed chunks
"""
from langchain.document_loaders import PyPDFLoader, TextLoader, csv_loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Any, List


class DocumentLoader:
    """Loads and splits documents into chunks"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: How many characters in each chunk
                       (roughly 100-150 words = 500-800 characters)
            
            chunk_overlap: How many characters overlap between chunks
                          (helps preserve context at boundaries)
        
        Why these defaults?
        - 500 chars ≈ 1-2 sentences, good for embeddings
        - 50 char overlap ≈ half a sentence, prevents info loss
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create the text splitter
        # RecursiveCharacterTextSplitter is SMART - respects sentence boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Try paragraph breaks first
                "\n",    # Then line breaks
                ". ",    # Then sentences
                " ",     # Then words
                ""       # Last resort: characters
            ]
        )
    
        print(f"✓ DocumentLoader initialized")
        print(f"  Chunk size: {chunk_size} characters")
        print(f"  Overlap: {chunk_overlap} characters")

    def load_from_file(self, file_path: str) -> List[Document]:
        """
        Load a document from file (PDF or TXT).
        
        Args:
            file_path: Path to file (e.g., "data/document.pdf" or "data/guide.txt")
        
        Returns:
            List of Document objects (already split into chunks)
        
        Example:
            >>> loader = DocumentLoader()
            >>> docs = loader.load_from_file("data/guide.pdf")
            >>> print(f"Loaded {len(docs)} chunks")
            >>> print(docs[0].page_content[:100])
        """
        print(f"\n📂 Loading file: {file_path}")
        
        # Determine file type and load accordingly
        if file_path.endswith('.pdf'):
            print("   Format: PDF")
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            print("   Format: Text")
            loader = TextLoader(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {file_path}\n"
                f"Supported: .pdf, .txt"
            )
        
        # Load the raw document
        try:
            documents = loader.load()
            print(f"   ✓ Loaded {len(documents)} pages/files")
        except Exception as e:
            print(f"   ✗ Error loading file: {e}")
            raise
        
        # Split into chunks
        print(f"   Splitting into chunks...")
        chunks = self.splitter.split_documents(documents)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        return chunks

    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Load from raw text string (useful for testing).
        
        Args:
            text: Raw text content
            metadata: Extra info like {"source": "test_file", "page": 1}
        
        Returns:
            List of Document objects (chunks)
        
        Example:
            >>> text = "How to reset password: click Forgot Password button..."
            >>> docs = loader.load_from_text(text, {"source": "faq"})
            >>> print(docs[0].page_content)
        """
        
        if metadata is None:
            metadata = {}
        
        print(f"\n📝 Loading raw text ({len(text)} characters)")
        print(f"   Metadata: {metadata}")
        
        # Create a Document object
        doc = Document(page_content=text, metadata=metadata)
        
        # Split into chunks
        print(f"   Splitting into chunks...")
        chunks = self.splitter.split_documents([doc])
        print(f"   ✓ Created {len(chunks)} chunks")
        
        return chunks
     
    def print_chunks(self, chunks: List[Document], num_to_show: int = 3):
        """
        Display chunks nicely (for debugging).
        
        Args:
            chunks: List of Document objects
            num_to_show: How many to display (default 3)
        
        Example:
            >>> loader = DocumentLoader()
            >>> docs = loader.load_from_text("Your text here...")
            >>> loader.print_chunks(docs, num_to_show=2)
        """
        
        print(f"\n📋 First {num_to_show} chunks:\n")
        
        for i, chunk in enumerate[Any](chunks[:num_to_show]):
            print(f"{'='*60}")
            print(f"CHUNK {i+1}")
            print(f"{'='*60}")
            print(f"Content ({len(chunk.page_content)} chars):")
            print(f"{chunk.page_content}")
            print(f"\nMetadata:")
            for key, value in chunk.metadata.items():
                print(f"  {key}: {value}")
            print()
