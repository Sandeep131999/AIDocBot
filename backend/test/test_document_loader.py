"""
Test DocumentLoader with real file
"""

from src.document_loader import DocumentLoader

print("\n" + "="*70)
print("TEST: Load real document from file")
print("="*70)

# Create loader
loader = DocumentLoader(chunk_size=500, chunk_overlap=50)

# Load the guide file
try:
    docs = loader.load_from_file("data/password_guide.txt")
    
    print(f"\n✓ Successfully loaded document!")
    print(f"  Total chunks: {len(docs)}")
    print(f"  Average chunk size: {sum(len(d.page_content) for d in docs) // len(docs)} characters")
    
    # Show first 3 chunks
    loader.print_chunks(docs, num_to_show=3)
    
    # Show metadata
    print("\n" + "="*70)
    print("METADATA CHECK")
    print("="*70)
    print(f"\nFirst chunk metadata:")
    for key, value in docs[0].metadata.items():
        print(f"  {key}: {value}")
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total chunks: {len(docs)}")
    print(f"Smallest chunk: {min(len(d.page_content) for d in docs)} chars")
    print(f"Largest chunk: {max(len(d.page_content) for d in docs)} chars")
    print(f"Total content: {sum(len(d.page_content) for d in docs)} chars")
    
    print("\n✓ ALL TESTS PASSED!")
    
except FileNotFoundError as e:
    print(f"\n✗ File not found: {e}")
    print(f"  Make sure you created: data/password_guide.txt")
except Exception as e:
    print(f"\n✗ Error: {e}")