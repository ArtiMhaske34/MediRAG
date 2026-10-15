import sys
from src.rag_pipeline import RAGPipeline

def main():
    print("Initializing Unified Offline RAG System...")
    
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"\nInitialization Error: {e}")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("System ready! Ask questions about your indexed documents.")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Ending session. Goodbye!")
                break
                
            if not user_input:
                continue

            print("\nThinking... (Retrieving context and generating answer)\n")
            
            response = pipeline.ask(user_input)
            
            print("Answer:")
            print(response['result'].strip())
            
            print("\n--- Sources used ---")
            for i, doc in enumerate(response['source_documents']):
                source_file = doc.metadata.get('source', 'Unknown File')
                print(f"[{i+1}] {source_file}")
                
        except KeyboardInterrupt:
            print("\nSession interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during generation: {e}")

if __name__ == "__main__":
    main()