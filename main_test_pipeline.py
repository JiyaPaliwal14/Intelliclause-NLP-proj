# main_test_pipeline.py
import asyncio
import hashlib
import uuid
import os

from app.database import engine, SessionLocal
from app.models_db import Base, Document
from app.pdfToText import extract_text_generator_async
from app.chunkCreator import chunk_pageText
from app.embeddings import embed_chunks_async
from app.vector_store import upsert_chunks_async, check_document_exists_async
from app.retriever import retrieve_top_chunks_async
from app.response_builder import build_final_response_async

# --- CONFIGURATION ---
# A sample insurance policy PDF available online
SAMPLE_PDF_URL = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Questions tailored to the sample PDF
QUESTIONS = [
    "What is the grace period for policy renewal?",
    "What is the initial waiting period for any illness after the policy starts?",
    "Are cataract surgeries covered, and if so, what is the financial limit?",
    "What is the definition of a 'Hospital' according to this policy document?",
    "How are claims handled if the insured person has multiple policies from different insurers?"
]

async def setup_database():
    """Initializes the database by creating all tables."""
    print("--- 🚀 Initializing Database ---")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables created successfully.")

async def process_document(doc_url: str):
    """Full pipeline: Ingest, chunk, embed, and store a document."""
    print(f"\n--- Processing Document: {doc_url} ---")
    
    # 1. Extract Text
    print("1. 📚 Extracting text from PDF...")
    full_text = ""
    try:
        async for page_text in extract_text_generator_async(doc_url):
            full_text += page_text + "\n"
    except Exception as e:
        print(f"❌ Failed to extract text: {e}")
        return

    if not full_text.strip():
        print("❌ Extracted text is empty. Aborting.")
        return
        
    print(f"✅ Text extracted successfully ({len(full_text)} characters).")

    # Generate a unique and deterministic document ID
    doc_hash = hashlib.md5(full_text.encode()).hexdigest()
    document_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_hash))
    
    # Check if this document has already been processed
    if await check_document_exists_async(document_id):
        print(f"✅ Document {document_id[:8]}... already exists in the vector store. Skipping ingestion.")
        return document_id

    # 2. Chunk Text
    print("2. 🧩 Chunking extracted text...")
    chunks = chunk_pageText(full_text)
    if not chunks:
        print("❌ No chunks were created. Aborting.")
        return
    print(f"✅ Text divided into {len(chunks)} chunks.")

    # 3. Generate Embeddings
    print("3. 🧠 Generating embeddings for chunks...")
    chunk_texts = [c['text'] for c in chunks]
    vectors = await embed_chunks_async(chunk_texts)
    
    valid_vectors = [v for v in vectors if v is not None]
    if not valid_vectors:
        print("❌ Failed to generate any valid embeddings. Aborting.")
        return
    print(f"✅ Generated {len(valid_vectors)} embeddings.")

    # 4. Prepare Metadata and Upsert
    print("4. 💾 Preparing data for storage...")
    metadata_list = [
        {
            "document_id": document_id,
            "file_name": os.path.basename(doc_url),
            "chunk_id": i,
            "page_number": 0, # Placeholder, as chunking logic doesn't retain page info
            "section_title": chunk.get("section_number", f"chunk_{i+1}")
        }
        for i, chunk in enumerate(chunks)
    ]
    
    # Filter out chunks where embedding failed
    valid_chunks = []
    valid_metadata = []
    final_vectors = []
    for i, vec in enumerate(vectors):
        if vec is not None:
            valid_chunks.append(chunk_texts[i])
            valid_metadata.append(metadata_list[i])
            final_vectors.append(vec)

    print("5. 🚀 Upserting chunks and embeddings to databases...")
    try:
        await upsert_chunks_async(document_id, valid_chunks, final_vectors, valid_metadata)
        print("✅ Document processed and stored successfully.")
        return document_id
    except Exception as e:
        print(f"❌ An error occurred during upsert: {e}")
        return None

async def answer_questions(document_id: str, questions: list):
    """Processes a list of questions against the ingested document."""
    print(f"\n--- ❓ Answering Questions for Document ID: {document_id[:8]}... ---")
    for i, question in enumerate(questions):
        print(f"\n❓ Question {i+1}: {question}")

        # 1. Retrieve relevant chunks
        print("   - Retrieving relevant context...")
        top_chunks = await retrieve_top_chunks_async(question, doc_filter=document_id, top_k=5)
        
        if not top_chunks:
            print("   - ❌ No relevant context found.")
            continue
            
        print(f"   - ✅ Retrieved {len(top_chunks)} chunks for context.")
        
        # For display, show a snippet of the most relevant chunk
        retrieved_texts = [chunk.get("chunk", "") for chunk in top_chunks]
        print(f"   - ✨ Top context snippet: \"{retrieved_texts[0][:150]}...\"")

        # 2. Build the final response
        print("   - Generating final answer...")
        answer = await build_final_response_async(question, retrieved_texts)

        print("\n" + "="*20 + " FINAL ANSWER " + "="*20)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("="*54 + "\n")

async def main():
    """Main function to run the entire RAG pipeline test."""
    await setup_database()
    document_id = await process_document(SAMPLE_PDF_URL)
    
    if document_id:
        await answer_questions(document_id, QUESTIONS)
    else:
        print("\n--- ❌ Could not proceed to question answering due to ingestion failure. ---")

if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is set
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("FATAL: GOOGLE_API_KEY environment variable is not set. Please create a .env file.")
    
    asyncio.run(main())