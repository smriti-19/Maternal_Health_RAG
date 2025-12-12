#!/usr/bin/env python3
"""
Stage 1 Guardrails with LLM Label-Based Post-Processing
"""

import os, gc, json, re, unicodedata
import pandas as pd
import asyncio
import yaml
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder, SentenceTransformer

# CONFIG

#QUESTIONS_CSV = "guardrail_test_dataset_assamese.csv"
QUESTIONS_CSV = "questions.csv"
POLICY_PATH = "guardrails/policy.yaml"
#OUTPUT_FILE = "guardrails_rag_stage3_V5_b_gating_LLM_E_new_assamese.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_DIR = os.getenv("FAISS_DB_DIR")
BM25_PATH = os.getenv("BM25_PATH")
OUTPUT_FILE = "guardrail_rag_FULL_E_new_V2.json"


# Guardrails settings
ENABLE_GUARDRAILS = True
ENABLE_DEBUG = True
ENABLE_LLM_LABELS = True  # NEW: Use LLM labels for SAME_DAY detection

# Thresholds
EMERGENCY_THRESHOLD = 0.50
SAME_DAY_THRESHOLD = 0.30
SAME_DAY_THRESHOLD_LOW = 0.25

# TEMPLATES
CRISIS_TEMPLATES = {
    "NOW-MH": "Your safety is the priority. If you have thoughts about harming yourself or others, contact local emergency services now. Please contact Suicide & Crisis Helpline: India: 9152987821, US: 988, UK: 0800 689 5652.",
    "NOW-DV": "Your safety is the priority. If you are in danger, contact local emergency services now. In the United States, confidential help is available at the National Domestic Violence Hotline: 1-800-799-7233 (TTY 1-800-787-3224).",
    "NOW-MED": "IMPORTANT: This appears to be an emergency situation that requires immediate medical attention.\n\nThis chatbot cannot provide emergency medical advice. Please:\n1. Contact your healthcare provider immediately OR\n2. Go to the nearest emergency department OR\n3. Call your local emergency number for immediate assistance.\n\nDo not wait or delay seeking professional care.",
    "SAME-DAY": "This should be checked by a clinician today. Contact your OB provider or clinic. If symptoms worsen—trouble breathing, severe pain, heavy bleeding, or high fever—seek emergency care immediately."
}

# HELPERS
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"page\s*\d+", "", text)
    return text.strip()

def dedup_docs(docs):
    seen, unique = set(), []
    for d in docs:
        norm = normalize_text(d.page_content)
        if norm not in seen:
            seen.add(norm)
            d.page_content = norm
            unique.append(d)
    return unique

def reciprocal_rank_fusion(results, k=60):
    scores = {}
    for retr_name, docs in results.items():
        for rank, d in enumerate(docs):
            key = d.page_content
            score = 1 / (rank + 60)
            scores[key] = scores.get(key, 0) + score
    ranked = sorted(results["dense"], key=lambda d: scores.get(d.page_content, 0), reverse=True)
    return ranked

async def translate_query_to_english(query, llm):
    messages = [HumanMessage(content=f"Translate this query to English:\n{query}")]
    response = await llm.ainvoke(messages)
    return response.content.strip()

async def expand_query(query, llm):
    messages = [HumanMessage(content=f"Expand this medical query with synonyms and related terms:\n{query}")]
    response = await llm.ainvoke(messages)
    return response.content.strip()

def parse_llm_label(text: str) -> Tuple[str, str]:
    """
    Parse first line of LLM response for urgency label
    Returns: (label, remaining_text)
    """
    lines = text.split('\n', 1)
    first_line = lines[0].strip().upper()
    
    # Check for valid labels
    valid_labels = ['NOW-MH', 'NOW-DV', 'NOW-MED', 'SAME-DAY', 'PASS']
    for label in valid_labels:
        if label in first_line:
            remaining = lines[1] if len(lines) > 1 else ""
            return label, remaining.strip()
    
    # No valid label found
    return 'PASS', text

async def run_queries(rag, questions, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_query(i, q):
        async with semaphore:
            try:
                res = await rag.query(q, translate=True)
                return {
                    "index": i,
                    "question": q,
                    "answer": res["answer"],
                    "sources": res["sources"],
                    "guardrail_stage": res.get("guardrail_stage"),
                    "guardrail_severity": res.get("guardrail_severity"),
                    "guardrail_action": res.get("guardrail_action"),
                    "guardrail_crisis_type": res.get("guardrail_crisis_type"),
                    "guardrail_template": res.get("guardrail_template"),
                    "llm_label": res.get("llm_label")
                }
            except Exception as e:
                print(f"⚠️ Error on {i}: {e}")
                return None
    
    tasks = [process_query(i, q) for i, q in enumerate(questions)]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Running"):
        result = await coro
        if result:
            results.append(result)
        gc.collect()
        await asyncio.sleep(0.1)
    
    results.sort(key=lambda x: x["index"])
    return results

# GUARDRAILS ROUTER (Simplified - focuses on CRISIS only)
class GuardrailsRouter:
    def __init__(self, policy_path: str, llm=None):
        with open(policy_path, 'r') as f:
            self.policy = yaml.safe_load(f)
        
        self.llm = llm
        self.actions = self.policy.get('actions', {})
        self.detect_stage_config = self.policy.get('detect_stage', {})
        self.severity_gate = self.policy.get('severity_gate', {})
        self.templates = self.policy.get('templates', {})
        self.negation_config = self.policy.get('negation_guard', {})
        
        regex_flags_list = self.policy.get('regex_flags', ['IGNORECASE'])
        self._re_flags = 0
        for flag in regex_flags_list:
            if hasattr(re, flag):
                self._re_flags |= getattr(re, flag)
        
        self._regex_cache = {}
        print(f" Policy Loaded (v{self.policy.get('version')})")
        
        print(" Initializing semantic encoder...")
        self.semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Simplified: Only EMERGENCY examples (SAME_DAY will be detected by LLM)
        self.emergency_examples = [
            "I want to end my life", "I can't go on anymore", "I feel like killing myself",
            "I don't want to live anymore", "I'm thinking about suicide",
            "My husband won't let me leave the house", "I'm not safe at home",
            "He's threatening to hurt me", "I'm afraid for my safety",
            "I'm bleeding heavily and soaking through pads", "Severe bleeding that won't stop",
            "I'm passing large blood clots", "Soaking through a pad every hour",
            "I just had a seizure", "I'm having convulsions",
            "I have severe chest pain", "I can't breathe properly",
            "Severe headache with blurred vision", "Bad headache and seeing spots",
            "Severe one-sided abdominal pain with fainting",
            "I'm hearing voices telling me things",
            "Baby hasn't moved and I'm bleeding"
        ]
        
        self.emergency_embeddings = self.semantic_encoder.encode(
            self.emergency_examples, show_progress_bar=False, convert_to_numpy=True
        )
        print(f" Semantic encoder ready ({len(self.emergency_examples)} EMERGENCY examples)")
        
        self.emergency_threshold = EMERGENCY_THRESHOLD
    
    def _compile_list(self, patterns: List[str]) -> List[re.Pattern]:
        compiled = []
        for p in patterns:
            if p not in self._regex_cache:
                self._regex_cache[p] = re.compile(p, self._re_flags)
            compiled.append(self._regex_cache[p])
        return compiled
    
    def _any_match(self, text: str, patterns: List[str]) -> Tuple[bool, Optional[int]]:
        compiled = self._compile_list(patterns)
        for pattern in compiled:
            match = pattern.search(text)
            if match:
                return True, match.start()
        return False, None
    
    def _has_negation(self, text: str, match_pos: int) -> bool:
        if match_pos is None:
            return False
        window = self.negation_config.get('window_chars', 50)
        negation_terms = self.negation_config.get('terms', [])
        start = max(0, match_pos - window)
        before_text = text[start:match_pos].lower()
        for term in negation_terms:
            clean_term = term.replace('\\b', '').replace('?', '')
            if re.search(rf'\b{re.escape(clean_term)}\b', before_text):
                return True
        return False
    
    def detect_stage(self, text: str, meta: Dict = None) -> str:
        meta = meta or {}
        if 'stage' in meta and meta['stage']:
            return meta['stage']
        if 'age_group' in meta and meta['age_group']:
            return meta['age_group']
        
        # Priority order: newborn → child → postpartum → pregnant
        stage_order = ['newborn_0_2mo', 'maternal_postpartum', 'maternal_pregnant']
        for stage_name in stage_order:
            if stage_name in self.detect_stage_config:
                patterns = self.detect_stage_config[stage_name].get('any', [])
                if patterns:
                    matched, _ = self._any_match(text, patterns)
                    if matched:
                        return stage_name
        
        return 'maternal_pregnant'
    
    async def translate_if_needed(self, text: str) -> str:
        try:
            import langdetect
            lang = langdetect.detect(text)
            if lang != 'en':
                if ENABLE_DEBUG:
                    print(f"   [TRANSLATE] {lang} → en")
                messages = [HumanMessage(content=f"Translate to English:\n{text}")]
                response = await self.llm.ainvoke(messages)
                return response.content.strip()
            return text
        except:
            return text
    
    async def classify_emergency_only(self, text: str) -> Tuple[bool, str, float]:
        """
        Check ONLY for emergencies (CRISIS-level)
        Returns: (is_emergency, crisis_type, similarity)
        """
        text_en = await self.translate_if_needed(text)
        query_embedding = self.semantic_encoder.encode([text_en], show_progress_bar=False, convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.emergency_embeddings)[0]
        max_similarity = similarities.max()
        
        if ENABLE_DEBUG:
            print(f"   [EMERGENCY CHECK] sim={max_similarity:.3f}, threshold={self.emergency_threshold}")
        
        if max_similarity >= self.emergency_threshold:
            # Determine crisis type
            if any(w in text_en.lower() for w in ['suicide', 'kill myself', 'end my life', 'die', 'unalive']):
                crisis_type = "NOW-MH"
            elif any(w in text_en.lower() for w in ["won't let me leave", "not safe", "threat", "violence", "abuse"]):
                crisis_type = "NOW-DV"
            else:
                crisis_type = "NOW-MED"
            
            if ENABLE_DEBUG:
                print(f"   → EMERGENCY: {crisis_type}")
            
            return True, crisis_type, max_similarity
        
        return False, None, max_similarity
    
    async def run_guardrails(self, question: str, meta: Dict = None) -> Dict:
        """Simplified: Only check for CRISIS-level emergencies"""
        meta = meta or {}
        stage = self.detect_stage(question, meta)
        
        is_emergency, crisis_type, similarity = await self.classify_emergency_only(question)
        
        if is_emergency:
            return {
                "guardrail_stage": stage,
                "guardrail_severity": "EMERGENCY_NOW",
                "guardrail_action": "CRISIS",
                "guardrail_crisis_type": crisis_type,
                "guardrail_template": crisis_type,
                "pass_through": False,
                "emergency_sim": similarity
            }
        else:
            return {
                "guardrail_stage": stage,
                "guardrail_severity": "NONE",
                "guardrail_action": "PASS_THROUGH",
                "guardrail_crisis_type": None,
                "guardrail_template": None,
                "pass_through": True,
                "emergency_sim": similarity
            }

# RAG CLASS WITH LLM LABELS
class EnhancedMedicalRAG:
    def __init__(self, vector_store, bm25, use_reranking=True, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", enable_guardrails=True, policy_path=POLICY_PATH):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.enable_guardrails = enable_guardrails
        
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1, openai_api_key=OPENAI_API_KEY, max_retries=3, request_timeout=60)
        
        if enable_guardrails:
            self.guardrails = GuardrailsRouter(policy_path, self.llm)
        
        self.use_reranking = use_reranking
        if use_reranking:
            try:
                self.reranker = CrossEncoder(reranker_model)
                print(f"✅ Reranker initialized")
            except:
                self.use_reranking = False
        
        # UPDATED PROMPT WITH LABEL INSTRUCTION
        self.rag_prompt = PromptTemplate(
            template="""You are a supportive maternal health assistant providing information to pregnant women and new mothers.

Response Guidelines:
1. CRITICAL INSTRUCTION: Your FIRST LINE must be ONLY ONE urgency label from this list (but do not change any NOW-MED labels to SAME-DAY or PASS):
- NOW-MH (suicidal thoughts, self-harm, poor mental health)
- NOW-DV (domestic violence, unsafe, threatened or control issues with/without current danger)
- SAME-DAY (needs clinical assessment or medical intervention (eg: decreased fetal movement))
- PASS (routine question, educational content, normal pregnancy changes)


After the label, on a new line, provide your detailed response.

2. Answer ONLY using the provided context. Do not speculate or generate information not found in the retrieved data. If the provided context doesn't fully answer the question, state the limitations. Example: "I don't have enough information to answer that accurately."

3. Be reassuring but honest. Do not minimize legitimate concerns. If the topic is serious, clearly explain when to seek medical help.

4. Use clear, non-technical language. Avoid jargon and explain complex terms simply.

5. Warning signs: When discussing symptoms or warning signs, clearly distinguish between normal variations and symptoms that require medical attention.

6. Do not provide: Any specific drug brand or prescription recommendations; Any information related to fetal gender detection or selection.

7. Do not answer off-topic questions (questions not related to pregnancy, postpartum, or maternal/newborn wellness). If off-topic, say: "I'm sorry, I can only help with pregnancy and maternal-health questions."

8. Respond in the same language as the original question. If the language is Hindi written in English, do the same.


Context:
{context}

Question:
{question}

Answer (label first, then explanation):""",
            input_variables=["context", "question"]
        )
        
        self.stage_pattern = re.compile(r"\b(trimester|week|month|postpartum)\b", re.I)
        self.concern_pattern = re.compile(r"\b(pain|bleeding|movement|nutrition|sleep|emotional|discharge|swelling|fever|gas|acidity)\b", re.I)
    
    async def hybrid_retrieve(self, query: str, top_k: int = 15):
        expanded_text = await expand_query(query, self.llm)
        expanded_queries = [query] + [x.strip("-• \n") for x in expanded_text.split("\n") if x.strip()]
        dense_results, bm25_results = [], []
        
        for q in expanded_queries:
            dense_results.extend(self.vector_store.similarity_search(q, k=top_k))
            bm25_results.extend(self.bm25.get_relevant_documents(q))
        
        combined = reciprocal_rank_fusion({"dense": dense_results, "bm25": bm25_results})
        stage_hits, concern_hits = [], []
        if self.stage_pattern.search(query):
            stage_hits = self.vector_store.similarity_search("pregnancy stage info", k=3)
        if self.concern_pattern.search(query):
            concern_hits = self.vector_store.similarity_search("maternal health concerns", k=3)
        
        all_docs = combined + stage_hits + concern_hits
        all_docs = dedup_docs(all_docs)
        return all_docs[:top_k]
    
    def rerank_documents(self, query: str, docs: List, top_k: int = 7):
        if not self.use_reranking or not docs:
            return docs[:top_k]
        try:
            pairs = [(query, d.page_content) for d in docs]
            scores = self.reranker.predict(pairs)
            return [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)][:top_k]
        except:
            return docs[:top_k]
    
    async def query(self, question: str, translate: bool = True, user_metadata: Dict = None):
        original = question
        user_metadata = user_metadata or {}
        
        # STEP 1: Pre-LLM Emergency Check
        if self.enable_guardrails:
            route = await self.guardrails.run_guardrails(question, user_metadata)
            
            if route['guardrail_action'] == 'CRISIS':
                if ENABLE_DEBUG:
                    print(f"[PRE-LLM CRISIS] {route['guardrail_crisis_type']}: {question[:50]}...")
                
                # For SAME-DAY, we need to get LLM answer too
                # But pre-LLM guardrails typically catch NOW-* only
                # This is here for completeness in case SAME-DAY is detected pre-LLM
                if route['guardrail_crisis_type'] == 'SAME-DAY':
                    # Need to generate LLM answer for SAME-DAY
                    translated = await translate_query_to_english(question, self.llm) if translate else question
                    docs = await self.hybrid_retrieve(translated)
                    docs = self.rerank_documents(translated, docs)
                    context = "\n\n".join(d.page_content for d in docs)
                    
                    messages = [HumanMessage(content=self.rag_prompt.format(context=context, question=original))]
                    response = await self.llm.ainvoke(messages)
                    llm_answer = response.content.strip()
                    
                    template = CRISIS_TEMPLATES[route['guardrail_crisis_type']]
                    answer = f"{template}\n\n---\n\nAdditional Information:\n{llm_answer}"
                    sources = [{"content": d.page_content[:200], "metadata": d.metadata} for d in docs]
                else:
                    # For NOW-* cases, template only
                    answer = CRISIS_TEMPLATES[route['guardrail_crisis_type']]
                    sources = []
                
                return {
                    "question_original": original,
                    "question_translated": None,
                    "answer": answer,
                    "sources": sources,
                    "guardrail_stage": route['guardrail_stage'],
                    "guardrail_severity": route['guardrail_severity'],
                    "guardrail_action": route['guardrail_action'],
                    "guardrail_crisis_type": route['guardrail_crisis_type'],
                    "guardrail_template": route['guardrail_crisis_type'],
                    "llm_label": None
                }
        else:
            route = {"guardrail_stage": None, "guardrail_action": "PASS_THROUGH"}
        
        # STEP 2: Pass-Through → RAG with LLM Label
        translated = await translate_query_to_english(question, self.llm) if translate else question
        docs = await self.hybrid_retrieve(translated)
        docs = self.rerank_documents(translated, docs)
        context = "\n\n".join(d.page_content for d in docs)
        
        messages = [HumanMessage(content=self.rag_prompt.format(context=context, question=original))]
        response = await self.llm.ainvoke(messages)
        full_response = response.content.strip()
        
        # STEP 3: Parse LLM Label
        if ENABLE_LLM_LABELS:
            llm_label, answer_text = parse_llm_label(full_response)
            
            if ENABLE_DEBUG:
                print(f"[LLM LABEL] {llm_label}: {question[:50]}...")
            
            # If LLM detected urgency, handle based on type
            if llm_label in ['NOW-MH', 'NOW-DV', 'NOW-MED', 'SAME-DAY']:
                # For NOW-* cases: template only
                # For SAME-DAY: template + LLM answer
                if llm_label == 'SAME-DAY':
                    template = CRISIS_TEMPLATES.get(llm_label, "")
                    # Combine template with LLM answer
                    combined_answer = f"{template}\n\n---\n\nAdditional Information:\n{answer_text}"
                    answer = combined_answer
                else:
                    # For crisis cases (NOW-*), use template only
                    answer = CRISIS_TEMPLATES.get(llm_label, answer_text)
                
                action = "CRISIS" if llm_label.startswith('NOW-') else "SAME_DAY"
                severity = "EMERGENCY_NOW" if llm_label.startswith('NOW-') else "SAME_DAY"
                
                return {
                    "question_original": original,
                    "question_translated": translated,
                    "answer": answer,
                    "sources": [] if llm_label.startswith('NOW-') else [{"content": d.page_content[:200], "metadata": d.metadata} for d in docs],
                    "guardrail_stage": route['guardrail_stage'],
                    "guardrail_severity": severity,
                    "guardrail_action": action,
                    "guardrail_crisis_type": llm_label,
                    "guardrail_template": llm_label,
                    "llm_label": llm_label
                }
            else:
                # PASS - use LLM answer
                return {
                    "question_original": original,
                    "question_translated": translated,
                    "answer": answer_text or full_response,
                    "sources": [{"content": d.page_content[:200], "metadata": d.metadata} for d in docs],
                    "guardrail_stage": route['guardrail_stage'],
                    "guardrail_severity": "NONE",
                    "guardrail_action": "PASS_THROUGH",
                    "guardrail_crisis_type": None,
                    "guardrail_template": None,
                    "llm_label": llm_label
                }
        else:
            # Labels disabled, return full response
            return {
                "question_original": original,
                "question_translated": translated,
                "answer": full_response,
                "sources": [{"content": d.page_content[:200], "metadata": d.metadata} for d in docs],
                "guardrail_stage": route['guardrail_stage'],
                "guardrail_severity": "NONE",
                "guardrail_action": "PASS_THROUGH",
                "guardrail_crisis_type": None,
                "guardrail_template": None,
                "llm_label": None
            }

# MAIN
if __name__ == "__main__":
    print("="*70)
    print("STAGE 1: PRE-LLM CRISIS + LLM LABEL ROUTING")
    print("="*70)
    
    print(f"Loading FAISS from {DB_DIR}...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct", model_kwargs={"device": "cuda"}, encode_kwargs={"batch_size": 32})
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    print(" FAISS loaded")
    
    import pickle
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print(" BM25 loaded")
    
    df = pd.read_csv(QUESTIONS_CSV)
    questions = df.iloc[:, 0].dropna().astype(str).tolist()
    print(f" Loaded {len(questions)} questions\n")
    
    rag = EnhancedMedicalRAG(db, bm25, enable_guardrails=ENABLE_GUARDRAILS, policy_path=POLICY_PATH)
    
    if ENABLE_DEBUG:
        print("\nQuick test (3 questions):")
        async def test():
            for i in range(min(3, len(questions))):
                print(f"\n{i+1}. {questions[i][:80]}")
                result = await rag.query(questions[i])
                print(f"   → {result['guardrail_action']} (LLM: {result.get('llm_label')})")
        asyncio.run(test())
        print()
    
    print(f"Running on {len(questions)} questions...")
    results = asyncio.run(run_queries(rag, questions))
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    total = len(results)
    crisis = sum(1 for r in results if r.get('guardrail_action') == 'CRISIS')
    same_day = sum(1 for r in results if r.get('guardrail_action') == 'SAME_DAY')
    pass_through = sum(1 for r in results if r.get('guardrail_action') == 'PASS_THROUGH')
    
    # LLM label breakdown
    llm_labels = {}
    for r in results:
        label = r.get('llm_label', 'None')
        llm_labels[label] = llm_labels.get(label, 0) + 1
    
    
    print(f"\n Saved to {OUTPUT_FILE}\n")