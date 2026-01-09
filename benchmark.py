import sys
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Import your actual Assistant
from main import CourseAssistant

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Benchmark")

OUTPUT_FILE = Path("benchmark_results.csv")

# The Standard 10-Question Test Suite
TEST_QUERIES = [
    # --- Category A: Existing in Slides (Target: Answer + Citation) ---
    {"q": "How does the Boolean retrieval model work?", "type": "existing"},
    {"q": "What is the difference between stemming and lemmatization?", "type": "existing"},
    {"q": "How do k-gram indexes help with wildcard queries?", "type": "existing"},
    {"q": "Explain the formula for TF-IDF.", "type": "existing"},
    {"q": "How is cosine similarity calculated?", "type": "existing"},
    {"q": "What are the main limitations of LLMs that RAG helps to solve?", "type": "existing"},
    {"q": "Describe the architecture of the Transformer model.", "type": "existing"},

    # --- Category B: Missing from Slides (Target: Refusal) ---
    {"q": "How does Latent Dirichlet Allocation (LDA) model topics?", "type": "missing"},
    {"q": "Explain the HITS algorithm for web search.", "type": "missing"},

    # --- Category C: General Knowledge (Target: Refusal) ---
    {"q": "What is the capital of France?", "type": "general"}
]

class BenchmarkEngine:
    """
    Automated evaluation engine for the Course Assistant.
    Compares RAG vs. Pure LLM vs. Simple IR.
    """

    def __init__(self):
        try:
            logger.info("Initializing Assistant for Benchmarking...")
            self.bot = CourseAssistant()
        except Exception as e:
            logger.critical(f"Failed to initialize assistant: {e}")
            sys.exit(1)

    def evaluate_query(self, query_data: Dict[str, str]) -> Dict[str, Any]:
        """Runs a single query through all systems and scores the result."""
        query = query_data["q"]
        q_type = query_data["type"]
        
        logger.info(f"Testing: {query[:50]}...")

        # 1. RAG Response (The Main System)
        start_time = time.time()
        rag_response = self.bot.get_answer(query)
        duration = time.time() - start_time

        # 2. Simple IR (Retrieval Preview)
        # We fetch the top document directly to see what the search engine found.
        docs = self.bot.ensemble_retriever.invoke(query)
        if docs:
            top_source = docs[0].metadata.get("source", "Unknown")
            ir_preview = f"[Top Match: {top_source}] {docs[0].page_content}..."
        else:
            ir_preview = "No documents found."

        # 3. Pure LLM (Baseline - Hallucination Check)
        # We invoke the LLM directly, bypassing the RAG context.
        try:
            llm_response = self.bot.llm.invoke(query).content
        except Exception:
            llm_response = "Error invoking LLM directly."

        # 4. Auto-Scoring Logic
        score, note = self._calculate_score(q_type, rag_response)

        return {
            "Query Type": q_type,
            "Query": query,
            "RAG_Answer": rag_response,
            "Pure_LLM_Answer": llm_response,
            "Simple_IR_Preview": ir_preview,
            "Latency_Seconds": round(duration, 2),
            "Score": score,
            "Notes": note
        }

    def _calculate_score(self, q_type: str, response: str) -> tuple[float, str]:
        """
        Applies the professor's grading rubric:
        - 1.0: Perfect behavior (Cited correctly OR Refused correctly)
        - 0.2: Partial behavior (Answered without citation OR Weak refusal)
        - 0.0: Failure (Hallucination OR False Negative)
        """
        response_lower = response.lower()
        
        # Check for refusal keywords (Standardized in main.py prompt)
        is_refusal = (
            "cannot answer" in response_lower or 
            "not covered" in response_lower or
            "no information" in response_lower
        )
        
        if q_type == "existing":
            # Requirement: Must answer AND cite a slide.
            citation_markers = ["source:", ".pptx", "lecture", "slide", "file:"]
            has_citation = any(marker in response_lower for marker in citation_markers)   
                     
            if is_refusal:
                return 0.0, "False Negative (Refused valid question)"
            elif has_citation:
                return 1.0, "Success (Answered + Cited)"
            else:
                # If it's a long answer (>50 chars) but missed citation, give 0.2
                if len(response) > 50:
                    return 0.2, "Weak (Answered but missing citation)"
                return 0.0, "Wrong Answer"

        elif q_type in ["missing", "general"]:
            # Requirement: Must REFUSE to answer.
            if is_refusal:
                return 1.0, "Success (Correctly Refused)"
            else:
                return 0.0, "Failure (Hallucination / Answered out of bounds)"
        
        return 0.0, "Unknown Error"

    def run(self):
        """Main execution flow."""
        results = []
        total_score = 0.0
        
        print("\n" + "="*60)
        print("STARTING AUTOMATED BENCHMARK")
        print("="*60 + "\n")

        for i, q_data in enumerate(TEST_QUERIES):
            result = self.evaluate_query(q_data)
            results.append(result)
            total_score += result["Score"]
            
            # Print minimal status to console
            print(f"[{i+1}/10] Score: {result['Score']} | {result['Notes']}")

        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "="*60)
        print(f"FINAL SCORE: {total_score} / 10.0")
        print(f"Report saved to: {OUTPUT_FILE.absolute()}")
        print("="*60 + "\n")

if __name__ == "__main__":
    engine = BenchmarkEngine()
    engine.run()