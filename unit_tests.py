# unit_tests.py
import asyncio
from app.chunkCreator import chunk_pageText
from app.parser import extract_question_intent_async

async def test_chunking_logic():
    """Tests the chunkCreator module with sample legal text."""
    print("\n--- 🧪 Testing Chunking Logic ---")
    
    sample_text = """
    Article 1: Definitions
    In this Policy, "We", "Us", or "Our" means the Insurance Company. "You" or "Your" means the Policyholder.

    Article 2: Coverage
    2.1. We will provide coverage for medical expenses incurred by You, subject to the terms and conditions.
    A. This includes hospitalization fees.
    B. This also includes pre-hospitalization costs up to 30 days.

    This is a paragraph without a clear header that should be chunked by size. It continues on to provide more details about the general terms and conditions that apply to all sections of the document, ensuring that even unstructured text is handled gracefully.

    3. Exclusions
    We shall not be liable for any claims arising from self-inflicted injury.
    """
    
    chunks = chunk_pageText(sample_text)
    
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} (Header: {chunk['section_number']}): \"{chunk['text'][:80]}...\"")
        
    assert len(chunks) > 0, "Chunking produced no output!"
    print("✅ Chunking test passed.")


async def test_intent_parser():
    """Tests the parser module for intent extraction."""
    print("\n--- 🧪 Testing Intent Parser Logic ---")
    
    question = "How long is the waiting period for maternity coverage?"
    
    print(f"Parsing question: \"{question}\"")
    intent = await extract_question_intent_async(question)
    
    print("Extracted Intent:")
    print(f"  - Main Topic: {intent.get('main_topic')}")
    print(f"  - Question Type: {intent.get('question_type')}")
    print(f"  - Key Entities: {intent.get('key_entities')}")
    
    assert intent['main_topic'] == 'maternity', "Failed to identify main topic."
    assert 'waiting period' in intent['key_entities'], "Failed to extract key entities."
    print("✅ Intent parser test passed.")


async def main():
    await test_chunking_logic()
    await test_intent_parser()

if __name__ == "__main__":
    asyncio.run(main())