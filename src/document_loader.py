"""
DOCUMENT LOADER
Loads documents from files, raw text, or JSON data and splits them into chunks.
"""

import json
import os
import re
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader,
    UnstructuredExcelLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import Any, List, Union


class DocumentLoader:
    """Loads and splits documents into chunks"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
        print(f"✓ DocumentLoader initialized")
        print(f"  Chunk size: {chunk_size} characters")
        print(f"  Overlap: {chunk_overlap} characters")

    def load_from_file(self, file_path: str) -> List[Document]:
        print(f"\n📂 Loading file: {file_path}")
        
        if file_path.endswith('.pdf'):
            print("   Format: PDF")
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            print("   Format: Text")
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            print("   Format: CSV")
            loader = CSVLoader(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            print("   Format: Excel")
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith(('.docx', '.doc')):
            print("   Format: Word")
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith('.json'):
            print("   Format: JSON")
            return self._load_json_smart(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {file_path}\n"
                f"Supported: .pdf, .txt, .csv, .xlsx, .xls, .docx, .doc, .json"
            )
        
        try:
            documents = loader.load()
            print(f"   ✓ Loaded {len(documents)} pages/files")
        except Exception as e:
            print(f"   ✗ Error loading file: {e}")
            raise
        
        print(f"   Splitting into chunks...")
        chunks = self.splitter.split_documents(documents)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        return chunks

    def _load_json_smart(self, file_path: str) -> List[Document]:
        """
        Smart JSON loading that detects structure and preserves entities.
        For employee/structured data, creates one chunk per entity.
        For long text pages, uses normal splitting.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        print(f"   JSON contains {len(data)} items")
        
        all_chunks = []
        
        for i, item in enumerate(data):
            # Detect if this is employee/structured data
            content = item.get('content', item.get('text', ''))
            
            # Check if content has employee-like patterns (Name:, Employee ID:, etc.)
            if self._is_employee_data(content):
                print(f"   Item {i}: Detected employee data → entity-aware chunking")
                entity_chunks = self._chunk_by_entity(content, item)
                all_chunks.extend(entity_chunks)
            else:
                # Regular text content — use normal splitting
                print(f"   Item {i}: Regular text → standard chunking")
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": item.get('label', 'json_data'),
                        "url": item.get('url', ''),
                        "title": item.get('title', ''),
                        "json_index": i
                    }
                )
                chunks = self.splitter.split_documents([doc])
                all_chunks.extend(chunks)
        
        print(f"   ✓ Created {len(all_chunks)} total chunks")
        return all_chunks
    
    def _is_employee_data(self, content: str) -> bool:
        """Check if content contains employee profile patterns."""
        employee_markers = [
            r'Name:\s*\w+',           # Name: Kausthub
            r'Employee ID:',           # Employee ID: 140003
            r'Designation:',           # Designation: General Manager
            r'Date Of Joining:',       # Date Of Joining: 03-Nov-2014
            r'About (Him|Her):',       # About Him:
        ]
        matches = sum(1 for pattern in employee_markers if re.search(pattern, content))
        return matches >= 2  # At least 2 markers = employee data
    
    def _chunk_by_entity(self, content: str, item: dict) -> List[Document]:
        """
        Split employee data by individual person records.
        Each person becomes one or more complete chunks.
        """
        # Pattern to match employee blocks: "Name:" followed by content until next "Name:" or end
        # This handles both formats:
        #   Name: XXX
        #   Employee ID: YYY
        #   ...
        #   Name: ZZZ  (next person)
        
        employee_blocks = re.split(r'(?=Name:\s*\w)', content)
        
        chunks = []
        base_metadata = {
            "source": item.get('label', 'json_data'),
            "url": item.get('url', ''),
            "title": item.get('title', ''),
            "page_type": "employee_profiles"
        }
        
        for block in employee_blocks:
            block = block.strip()
            if not block or len(block) < 20:
                continue
            
            # Extract name for metadata
            name_match = re.search(r'Name:\s*([^\n]+)', block)
            name = name_match.group(1).strip() if name_match else "unknown"
            
            # Extract employee ID for metadata
            id_match = re.search(r'Employee ID:\s*([^\n]+)', block)
            emp_id = id_match.group(1).strip() if id_match else ""
            
            # If block is too long, split at natural boundaries but keep name header
            if len(block) > self.chunk_size * 1.5:
                sub_chunks = self._split_large_entity(block, name, emp_id, base_metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Document(
                    page_content=block,
                    metadata={
                        **base_metadata,
                        "employee_name": name,
                        "employee_id": emp_id
                    }
                ))
        
        return chunks
    
    def _split_large_entity(self, block: str, name: str, emp_id: str, base_metadata: dict) -> List[Document]:
        """Split a large employee block while keeping name context in each chunk."""
        # Add name header to each chunk for context
        name_header = f"Name: {name}\nEmployee ID: {emp_id}\n" if emp_id else f"Name: {name}\n"
        
        # Split the block (without re-adding header to first part)
        temp_doc = Document(page_content=block, metadata={})
        split_docs = self.splitter.split_documents([temp_doc])
        
        chunks = []
        for i, doc in enumerate(split_docs):
            # Prepend name header to every chunk so it's self-contained
            content = name_header + doc.page_content if i > 0 else doc.page_content
            chunks.append(Document(
                page_content=content,
                metadata={
                    **base_metadata,
                    "employee_name": name,
                    "employee_id": emp_id,
                    "chunk_index": i
                }
            ))
        
        return chunks

    def load_from_text(self, text: str, metadata: dict = None) -> List[Document]:
        if metadata is None:
            metadata = {}
        
        print(f"\n📝 Loading raw text ({len(text)} characters)")
        doc = Document(page_content=text, metadata=metadata)
        chunks = self.splitter.split_documents([doc])
        print(f"   ✓ Created {len(chunks)} chunks")
        return chunks

    def load_from_json(self, json_data: Union[str, dict, list], 
                       text_key: str = "text", 
                       metadata: dict = None) -> List[Document]:
        if metadata is None:
            metadata = {}
        
        print(f"\n📊 Loading JSON data...")
        
        if isinstance(json_data, str):
            try:
                json_data = json.loads(json_data)
                print(f"   ✓ Parsed JSON string")
            except json.JSONDecodeError as e:
                print(f"   ✗ Invalid JSON: {e}")
                raise
        
        if isinstance(json_data, dict):
            json_data = [json_data]
            print(f"   Single document object → wrapped in list")
        elif not isinstance(json_data, list):
            raise ValueError(
                f"JSON data must be a dict, list, or JSON string. Got: {type(json_data).__name__}"
            )
        
        print(f"   Processing {len(json_data)} document(s)")
        
        documents = []
        for i, item in enumerate(json_data):
            if not isinstance(item, dict):
                print(f"   ⚠ Skipping item {i+1}: not a dict")
                continue
            
            if text_key not in item:
                raise KeyError(
                    f"Item {i+1} missing text key '{text_key}'. "
                    f"Available keys: {list(item.keys())}"
                )
            
            text_content = str(item[text_key])
            item_metadata = {k: v for k, v in item.items() if k != text_key}
            item_metadata.update(metadata)
            item_metadata["json_index"] = i
            item_metadata["source"] = item_metadata.get("source", "json_data")
            
            doc = Document(page_content=text_content, metadata=item_metadata)
            documents.append(doc)
        
        print(f"   ✓ Created {len(documents)} Document objects")
        chunks = self.splitter.split_documents(documents)
        print(f"   ✓ Created {len(chunks)} chunks")
        return chunks
     
    def print_chunks(self, chunks: List[Document], num_to_show: int = 3):
        print(f"\n📋 First {num_to_show} chunks:\n")
        for i, chunk in enumerate(chunks[:num_to_show]):
            print(f"{'='*60}")
            print(f"CHUNK {i+1}")
            print(f"{'='*60}")
            print(f"Content ({len(chunk.page_content)} chars):")
            print(f"{chunk.page_content[:300]}...")
            print(f"\nMetadata: {chunk.metadata}")
            print()