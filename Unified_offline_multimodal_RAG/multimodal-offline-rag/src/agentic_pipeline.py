import os
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# IMPORT OUR DP WRAPPER
from src.dp_embeddings import DPHuggingFaceEmbeddings

# 1. DEFINE THE AGENT'S MEMORY (STATE)
class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    context: Optional[str]
    math_result: Optional[str]
    ddi_result: Optional[str]     # NEW: For drug interactions
    lab_result: Optional[str]     # NEW: For lab validations
    final_answer: Optional[str]

class AgenticRAG:
    def __init__(self, index_dir: str = "faissindex"):
        self.index_dir = index_dir
        
        print("Loading Differentially Private Embeddings...")
        self.embeddings = DPHuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            epsilon=1.2 
        )
        
        print("Loading local FAISS vector store...")
        if not os.path.exists(os.path.join(self.index_dir, "index.faiss")):
            raise FileNotFoundError("FAISS index not found! Run ingest.py first.")
            
        self.vectorstore = FAISS.load_local(
            self.index_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        print("Connecting Agent Brain to Ollama (Phi-3)...")
        self.llm = Ollama(
            model="phi3",
            temperature=0.0, # 0.0 makes the agent strictly logical for routing
            num_ctx=2048,
            stop=["<|eot_id|>"]
        )

        # Build the LangGraph
        self.app = self._build_graph()

    # --- NODE 1: THE ROUTER ---
    def _route_query(self, state: AgentState) -> AgentState:
        print("\n[AGENT] Analyzing Intent...")
        
        route_prompt = PromptTemplate.from_template("""
        You are a clinical routing assistant. Read the medical query and classify it into one of FOUR exact categories:
        1. 'search' - if the query asks for patient history, symptoms, medical records, or guidelines.
        2. 'calculate' - if the query asks to calculate BMI, dosage, or perform math.
        3. 'check_interaction' - if the query asks if two drugs are safe together or have interactions.
        4. 'validate_labs' - if the query asks if a lab result (like glucose, WBC, or HbA1c) is normal, high, or low.
        
        Query: {query}
        
        Output ONLY one of the exact words: search, calculate, check_interaction, or validate_labs. Nothing else.
        """)
        
        chain = route_prompt | self.llm | StrOutputParser()
        intent = chain.invoke({"query": state["query"]}).strip().lower()
        
        # Fallback safeguard map
        valid_intents = ["search", "calculate", "check_interaction", "validate_labs"]
        matched_intent = "search" # Default fallback
        for valid in valid_intents:
            if valid in intent:
                matched_intent = valid
                break
            
        print(f"[AGENT] Decision: Routing to [{matched_intent.upper()}] tool.")
        return {"intent": matched_intent}

    # --- NODE 2: FAISS SEARCH TOOL ---
    def _search_tool(self, state: AgentState) -> AgentState:
        print("[AGENT] Executing DP-RAG Search...")
        docs = self.retriever.invoke(state["query"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    # --- NODE 3: CLINICAL CALCULATOR TOOL ---
    def _calculator_tool(self, state: AgentState) -> AgentState:
        print("[AGENT] Executing Clinical Math Logic...")
        query = state["query"].lower()
        
        math_result = "Could not compute. Please provide valid numbers."
        try:
            if "bmi" in query:
                numbers = re.findall(r'\d+\.?\d*', query)
                if len(numbers) >= 2:
                    weight = float(numbers[0]) 
                    height = float(numbers[1]) 
                    bmi = weight / (height ** 2)
                    math_result = f"Calculated BMI: {bmi:.1f}"
        except Exception as e:
            math_result = f"Math error: {str(e)}"
            
        return {"math_result": math_result}

    # --- NODE 4: DDI CHECKER TOOL (NEW) ---
    def _ddi_checker_tool(self, state: AgentState) -> AgentState:
        print("[AGENT] Executing Drug-Drug Interaction Checker...")
        query = state["query"].lower()
        
        # Hardcoded offline dictionary mapping
        interaction_db = {
            frozenset(["gabapentin", "loratadine"]): "Safe: No known severe interactions.",
            frozenset(["metformin", "atorvastatin"]): "Safe: Commonly prescribed together.",
            frozenset(["warfarin", "aspirin"]): "Severe Contraindication: Significantly increased risk of major bleeding.",
            frozenset(["penicillin", "methotrexate"]): "Moderate Risk: Penicillin can reduce the clearance of methotrexate, increasing toxicity."
        }
        
        known_drugs = ["gabapentin", "loratadine", "metformin", "atorvastatin", "warfarin", "aspirin", "penicillin", "methotrexate"]
        found_drugs = [drug for drug in known_drugs if drug in query]
        
        if len(found_drugs) >= 2:
            query_pair = frozenset([found_drugs[0], found_drugs[1]])
            if query_pair in interaction_db:
                ddi_result = f"Interaction check for {found_drugs[0].title()} & {found_drugs[1].title()}: {interaction_db[query_pair]}"
            else:
                ddi_result = f"No specific interactions documented offline for {found_drugs[0].title()} & {found_drugs[1].title()}."
        else:
            ddi_result = "Could not identify two recognizable drugs in the query to compare."
            
        return {"ddi_result": ddi_result}

    # --- NODE 5: LAB VALIDATOR TOOL (NEW) ---
    def _lab_validator_tool(self, state: AgentState) -> AgentState:
        print("[AGENT] Executing Lab Value Validator...")
        query = state["query"].lower()
        
        # Hardcoded clinical reference ranges
        reference_ranges = {
            "glucose": {"min": 70, "max": 99, "unit": "mg/dL"},
            "wbc": {"min": 4500, "max": 11000, "unit": "/mcL"},
            "creatinine": {"min": 0.74, "max": 1.35, "unit": "mg/dL"},
            "hba1c": {"min": 4.0, "max": 5.6, "unit": "%"}
        }
        
        lab_result = "Could not identify recognizable lab values to validate."
        
        # Scan query for recognized lab tests and numbers
        for lab, ranges in reference_ranges.items():
            if lab in query:
                numbers = re.findall(r'\d+\.?\d*', query)
                if numbers:
                    val = float(numbers[0])
                    status = "NORMAL"
                    if val < ranges["min"]: status = "LOW"
                    elif val > ranges["max"]: status = "HIGH"
                    
                    lab_result = f"{lab.upper()} value of {val} {ranges['unit']} is {status}. (Reference range: {ranges['min']}-{ranges['max']} {ranges['unit']})"
                    break # Stop after finding the first match for demo simplicity
                    
        return {"lab_result": lab_result}

    # --- NODE 6: RESPONSE GENERATOR ---
    def _generate_response(self, state: AgentState) -> AgentState:
        print("[AGENT] Synthesizing Final Answer...")
        
        gen_prompt = PromptTemplate.from_template("""
        You are a highly secure Clinical AI Assistant. Answer the user's medical query based ONLY on the provided system data below.
        If the data says 'Insufficient' or 'Could not compute', state that clearly.
        
        SYSTEM DATA:
        {data}
        
        USER QUERY:
        {query}
        
        CLINICAL ANSWER:
        """)
        
        # Route the correct tool output to the LLM
        intent = state.get("intent")
        if intent == "search":
            data_to_use = state.get("context")
        elif intent == "calculate":
            data_to_use = state.get("math_result")
        elif intent == "check_interaction":
            data_to_use = state.get("ddi_result")
        elif intent == "validate_labs":
            data_to_use = state.get("lab_result")
        else:
            data_to_use = "No data retrieved."
            
        chain = gen_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"data": data_to_use, "query": state["query"]})
        
        return {"final_answer": answer}

    # --- BUILD THE GRAPH ---
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add the nodes
        workflow.add_node("router", self._route_query)
        workflow.add_node("search", self._search_tool)
        workflow.add_node("calculate", self._calculator_tool)
        workflow.add_node("check_interaction", self._ddi_checker_tool)
        workflow.add_node("validate_labs", self._lab_validator_tool)
        workflow.add_node("generator", self._generate_response)
        
        # Set the entry point
        workflow.set_entry_point("router")
        
        # Add Conditional Edges based on the Intent
        workflow.add_conditional_edges(
            "router",
            lambda state: state["intent"],
            {
                "search": "search",
                "calculate": "calculate",
                "check_interaction": "check_interaction",
                "validate_labs": "validate_labs"
            }
        )
        
        # All tools flow into the final generator
        workflow.add_edge("search", "generator")
        workflow.add_edge("calculate", "generator")
        workflow.add_edge("check_interaction", "generator")
        workflow.add_edge("validate_labs", "generator")
        workflow.add_edge("generator", END)
        
        return workflow.compile()

    # --- FASTAPI ENTRY POINT ---
    def ask(self, query: str) -> dict:
        initial_state = {"query": query}
        result = self.app.invoke(initial_state)
        
        # Dynamically map the source name based on the route taken
        intent = result.get("intent")
        if intent == "search":
            context_text = result.get("context", "")
            source_name = "FAISS Vector Database (Patient Records)"
        elif intent == "calculate":
            context_text = result.get("math_result", "")
            source_name = "Python Clinical Calculator"
        elif intent == "check_interaction":
            context_text = result.get("ddi_result", "")
            source_name = "Offline DDI Dictionary"
        elif intent == "validate_labs":
            context_text = result.get("lab_result", "")
            source_name = "Clinical Reference Range Validator"
        else:
            context_text = "No context generated."
            source_name = "Unknown Source"
            
        # Wrap the result in a proper LangChain Document object
        mock_doc = Document(
            page_content=context_text, 
            metadata={"source": source_name}
        )
        
        # Return the exact dictionary format your API expects
        return {
            "result": result["final_answer"],
            "source_documents": [mock_doc]
        }