import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



class RAGPipeline:
    def __init__(self, index_dir: str = "faissindex"):
        self.index_dir = index_dir
        
        print("Connecting to local Ollama embedding model (nomic-embed-text)...")
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("MEDIRAG_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            client_kwargs={"timeout": 120},
        )
        
        print("Loading local FAISS vector store...")
        if not os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_dir}/. Did you run ingest.py?"
            )
            
        self.vectorstore = FAISS.load_local(
            self.index_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        print("Connecting to local Ollama Service (Phi-3)...")
        self.llm = OllamaLLM(
            model="phi3",
            temperature=0.1,
            num_ctx=2048,
            num_predict=160,
            client_kwargs={"timeout": 120},
            stop=["<|eot_id|>"],
        )

        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a highly secure, offline Clinical AI Assistant.
STRICT RULES:
1. ONLY answer using the provided CONTEXT.
2. If missing → say: "Insufficient patient data in offline records."
3. No hallucination.
4. Clinical tone.

CONTEXT:
{context}

QUESTION:
{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        self.prompt = PromptTemplate.from_template(template)

        # ✅ Helper to format docs
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # ✅ LCEL pipeline (replacement of RetrievalQA)
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, query: str):
        docs = self.retriever.invoke(query)

        context = "\n\n".join(doc.page_content for doc in docs)

        final_prompt = self.prompt.format(
            context=context,
            question=query
        )

        answer = self.llm.invoke(final_prompt)

        return {
            "result": answer,
            "source_documents": docs
        }