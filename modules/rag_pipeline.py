import re
from typing import List, Dict, Any, Tuple
from modules.retriever import Retriever
from modules.llm.factory import LLMFactory

# Automotive brand makes and common vehicle models
KNOWN_MAKES = [
    "subaru", "toyota", "honda", "ford", "chevrolet", "chevy", "nissan",
    "hyundai", "kia", "mazda", "volkswagen", "vw", "bmw", "mercedes",
    "mercedes-benz", "audi", "lexus", "volvo", "tesla", "jeep", "dodge",
    "ram", "chrysler", "porsche", "mitsubishi", "land rover", "cadillac",
    "buick", "gmc", "acura", "infiniti", "lincoln", "genesis"
]

KNOWN_MODELS = [
    "outback", "forester", "crosstrek", "impreza", "wrx", "ascent", "legacy", "brz", "solterra",
    "rav4", "camry", "corolla", "highlander", "tacoma", "tundra", "prius", "4runner", "sienna", "venza", "crown",
    "civic", "accord", "cr-v", "crv", "pilot", "hr-v", "hrv", "odyssey", "ridgeline",
    "f-150", "f150", "mustang", "explorer", "escape", "bronco", "ranger", "edge", "expedition",
    "silverado", "equinox", "malibu", "tahoe", "suburban", "colorado", "traverse",
    "rogue", "altima", "sentra", "pathfinder", "frontier", "murano",
    "tucson", "santa fe", "elantra", "sonata", "palisade", "kona", "ioniq",
    "sportage", "sorento", "telluride", "forte", "k5", "ev6", "seltos",
    "cx-5", "cx5", "cx-30", "cx30", "cx-50", "cx50", "cx-90", "cx90", "mazda3", "mazda6",
    "golf", "jetta", "tiguan", "passat", "atlas", "taos", "id.4"
]

AUTOMOTIVE_KEYWORDS = {
    "car", "cars", "vehicle", "vehicles", "automobile", "automobiles",
    "engine", "motor", "oil", "tire", "tires", "tyre", "tyres", "wheel", "wheels",
    "brake", "brakes", "braking", "battery", "fluid", "fluids", "transmission",
    "coolant", "radiator", "pressure", "psi", "maintenance", "service", "warranty",
    "mileage", "fuel", "gas", "diesel", "hybrid", "electric", "ev", "tank",
    "filter", "spark", "plug", "fuse", "headlight", "wiper", "wipers", "cargo",
    "towing", "capacity", "torque", "horsepower", "specs", "specification",
    "specifications", "manual", "steering", "suspension", "exhaust", "airbag",
    "abs", "awd", "4wd", "fwd", "rwd", "cylinder", "cylinders", "liters", "litres"
}

ORDINAL_FIRST_PATTERNS = [
    r"\bthe\s+first\s+(?:one|car|vehicle)\b",
    r"\bthe\s+former\b"
]

ORDINAL_SECOND_PATTERNS = [
    r"\bthe\s+second\s+(?:one|car|vehicle)\b",
    r"\bthe\s+latter\b"
]

ORDINAL_OTHER_PATTERNS = [
    r"\bthe\s+other\s+(?:one|car|vehicle)\b"
]

def extract_vehicle_mentions(text: str) -> List[str]:
    """
    Extract vehicle make/model mentions from text in appearance order.
    Combines Year (optional) + Make + Model when present.
    """
    if not text:
        return []
    t_lower = text.lower()
    found = []
    matched_spans = []

    sorted_makes = sorted(KNOWN_MAKES, key=len, reverse=True)
    sorted_models = sorted(KNOWN_MODELS, key=len, reverse=True)

    # 1. Check compound Make + Model (with optional year)
    for make in sorted_makes:
        for model in sorted_models:
            pattern = rf"\b(?:((?:19|20)\d\d)\s+)?({re.escape(make)})\s+({re.escape(model)})\b"
            for m in re.finditer(pattern, t_lower):
                span = m.span()
                if not any(s[0] <= span[0] and span[1] <= s[1] for s in matched_spans):
                    matched_spans.append(span)
                    found.append((span[0], text[span[0]:span[1]].strip()))

    # 2. Check standalone Makes
    for make in sorted_makes:
        pattern = rf"\b(?:((?:19|20)\d\d)\s+)?({re.escape(make)})\b"
        for m in re.finditer(pattern, t_lower):
            span = m.span()
            if not any(s[0] <= span[0] and span[1] <= s[1] for s in matched_spans):
                matched_spans.append(span)
                found.append((span[0], text[span[0]:span[1]].strip()))

    # 3. Check standalone Models
    for model in sorted_models:
        pattern = rf"\b(?:((?:19|20)\d\d)\s+)?({re.escape(model)})\b"
        for m in re.finditer(pattern, t_lower):
            span = m.span()
            if not any(s[0] <= span[0] and span[1] <= s[1] for s in matched_spans):
                matched_spans.append(span)
                found.append((span[0], text[span[0]:span[1]].strip()))

    found.sort(key=lambda x: x[0])
    return [item[1] for item in found]

def is_relevant_automotive_query(text: str) -> bool:
    """
    Check if a query is relevant to automobiles, vehicle systems, or manuals.
    Rejects general conversational chatter (jokes, greetings, riddles).
    """
    if not text or not text.strip():
        return False
    t_lower = text.lower().strip()

    # Reject social chatter or joke requests
    conversational_patterns = [
        r"\b(tell\s+me\s+a\s+joke|joke|poem|riddle|story|song)\b",
        r"\b(hello|hi|hey|how\s+are\s+you|who\s+are\s+you|what\s+is\s+your\s+name)\b",
        r"\b(thank\s+you|thanks|good\s+morning|good\s+night)\b"
    ]
    if any(re.search(pat, t_lower) for pat in conversational_patterns):
        return False

    # Positive signal if vehicle mentions exist
    if extract_vehicle_mentions(text):
        return True

    # Positive signal if automotive keywords exist
    words = set(re.findall(r"\b[a-z0-9\-]+\b", t_lower))
    if words & AUTOMOTIVE_KEYWORDS:
        return True

    return False

class RAGPipeline:
    def __init__(self, retriever: Retriever, llm_provider: str = "gemini", 
                 llm_model: str = None, llm_api_key: str = None, temperature: float = 0.2,
                 similarity_threshold: float = None):
        self.retriever = retriever
        self.similarity_threshold = similarity_threshold
        self.llm = LLMFactory.get_provider(
            provider_name=llm_provider, 
            api_key=llm_api_key, 
            model=llm_model, 
            temperature=temperature
        )

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a single context string with citations.
        Handles missing or incomplete metadata gracefully.
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk.get("metadata") or {}
            doc_name = meta.get("document_name") or "Unknown Document"
            page = meta.get("page_number")
            if page is None:
                page = 1
            text = (chunk.get("text") or "").strip()
            if text:
                context_parts.append(f"--- Document: {doc_name} | Page: {page} ---\n{text}\n")
        return "\n".join(context_parts)

    def extract_sources(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract sources from retrieved chunks for the UI.
        Preserves all distinct relevant chunks (even on the same page)
        while avoiding duplicate entries for identical chunks.
        Handles missing or incomplete metadata gracefully.
        """
        sources = []
        seen_chunk_keys = set()
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk.get("metadata") or {}
            doc_name = meta.get("document_name") or "Unknown Document"
            page = meta.get("page_number")
            if page is None:
                page = 1
            chunk_id = meta.get("chunk_id") or chunk.get("id") or f"{doc_name}_p{page}_c{i}"
            relevance = chunk.get("relevance", 0.0)
            text = (chunk.get("text") or "").strip()

            # Deduplicate by unique chunk identifier
            if chunk_id not in seen_chunk_keys:
                seen_chunk_keys.add(chunk_id)
                sources.append({
                    "document": doc_name,
                    "page": page,
                    "relevance": relevance,
                    "text": text,
                    "chunk_id": chunk_id
                })
        return sources

    def build_retrieval_query(self, query: str, conversation_history: List[Dict] = None) -> str:
        """
        Build an effective retrieval query by incorporating prior user context
        when the current query is anaphoric, referential, or a brief follow-up.
        
        Guarantees:
        - Does not append old vehicle context if the new query names a different vehicle.
        - Resolves ordinal references ("the second one", "the former") only when clearly established.
        - Does not append unrelated conversational history (e.g. jokes, greetings).
        - Keeps original query intact for LLM generation and user display.
        """
        if not query or not query.strip():
            return ""

        raw_query = query.strip()
        if not conversation_history or not isinstance(conversation_history, list):
            return raw_query

        # Find the most recent prior user query
        last_user_query = None
        all_prior_user_queries = []
        for msg in reversed(conversation_history):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = (msg.get("content") or "").strip()
                if content and content.lower() != raw_query.lower():
                    if last_user_query is None:
                        last_user_query = content
                    all_prior_user_queries.append(content)

        if not last_user_query:
            return raw_query

        q_lower = raw_query.lower()
        current_vehicles = extract_vehicle_mentions(raw_query)
        prior_vehicles = extract_vehicle_mentions(last_user_query)

        # 1. Vehicle switching or explicit new vehicle named in follow-up
        if current_vehicles:
            current_tokens = set(" ".join(current_vehicles).lower().split())
            prior_tokens = set(" ".join(prior_vehicles).lower().split()) if prior_vehicles else set()
            # If current query mentions a vehicle that doesn't overlap with prior vehicles,
            # a new vehicle was introduced -> do not append old vehicle context
            if not (current_tokens & prior_tokens):
                return raw_query
            # If same vehicle is already explicitly named in the new query, return raw query
            return raw_query

        # 2. Ordinal references resolution ("the first one", "the second one", "the former", "the latter", "the other vehicle")
        has_first = any(re.search(pat, q_lower) for pat in ORDINAL_FIRST_PATTERNS)
        has_second = any(re.search(pat, q_lower) for pat in ORDINAL_SECOND_PATTERNS)
        has_other = any(re.search(pat, q_lower) for pat in ORDINAL_OTHER_PATTERNS)

        if has_first or has_second or has_other:
            # Look for established vehicles in prior user queries
            established_vehicles = []
            for uq in reversed(all_prior_user_queries):
                for v in extract_vehicle_mentions(uq):
                    if v not in established_vehicles:
                        established_vehicles.append(v)
            last_turn_vehicles = extract_vehicle_mentions(last_user_query)
            if len(last_turn_vehicles) >= 2:
                established_vehicles = last_turn_vehicles

            if len(established_vehicles) >= 2:
                if has_first:
                    target = established_vehicles[0]
                    return f"{raw_query} ({target})"
                elif has_second:
                    target = established_vehicles[1]
                    return f"{raw_query} ({target})"
                elif has_other:
                    target = established_vehicles[1]
                    return f"{raw_query} ({target})"
            else:
                # Do not invent a vehicle when the conversation does not establish it clearly
                return raw_query

        # 3. Check if prior query contains relevant automotive context
        if not is_relevant_automotive_query(last_user_query):
            return raw_query

        # 4. Normal contextual follow-ups (referential pronouns or brief follow-up)
        referential_indicators = [
            "it", "its", "this", "that", "these", "those",
            "they", "them", "the car", "the vehicle", "the engine",
            "same", "how much", "how many", "what about", "how about",
            "where is", "when should", "why", "which one", "recommend"
        ]
        words = q_lower.split()
        is_short = len(words) <= 6
        has_referential = any(re.search(rf"\b{re.escape(ind)}\b", q_lower) for ind in referential_indicators)

        if has_referential or is_short:
            clean_prior = re.sub(r"[?!.]+$", "", last_user_query).strip()
            return f"{raw_query} ({clean_prior})"

        return raw_query

    def answer_question(self, query: str, top_k: int = 5, filter_document_name: str = None, 
                       conversation_history: List[Dict] = None,
                       threshold: float = None) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run the full RAG pipeline to answer a question.
        Returns (answer_string, unique_sources, retrieved_chunks)
        """
        if not query or not query.strip():
            return "Please enter a question to search the automobile knowledge base.", [], []

        # Contextual retrieval query clarification for follow-ups
        retrieval_query = self.build_retrieval_query(query, conversation_history)

        effective_threshold = threshold if threshold is not None else self.similarity_threshold
        retrieved_chunks = self.retriever.retrieve(
            query=retrieval_query, 
            top_k=top_k, 
            filter_document_name=filter_document_name,
            threshold=effective_threshold
        )

        if not retrieved_chunks:
            return "I could not find sufficient information in the uploaded automobile documents to answer this question.", [], []

        context = self.format_context(retrieved_chunks)
        sources = self.extract_sources(retrieved_chunks)
        
        # Generation uses the original user query and full conversation history
        answer = self.llm.generate_answer(
            query=query, 
            context=context, 
            conversation_history=conversation_history
        )
        
        return answer, sources, retrieved_chunks

    def compare_vehicles(self, query: str, docs_to_compare: List[str], top_k_per_doc: int = 5,
                         threshold: float = None) -> Tuple[str, List[Dict]]:
        """
        Compare information across multiple selected documents.
        """
        if not query or not query.strip():
            return "Please enter a comparison question.", []

        if not docs_to_compare or len(docs_to_compare) < 2:
            return "Please select at least two documents to compare.", []

        effective_threshold = threshold if threshold is not None else self.similarity_threshold
        all_retrieved_chunks = []
        
        for doc in docs_to_compare:
            chunks = self.retriever.retrieve(
                query=query, 
                top_k=top_k_per_doc, 
                filter_document_name=doc,
                threshold=effective_threshold
            )
            all_retrieved_chunks.extend(chunks)

        if not all_retrieved_chunks:
            return "I could not find sufficient information to compare these vehicles.", []

        context = self.format_context(all_retrieved_chunks)
        sources = self.extract_sources(all_retrieved_chunks)
        
        # Strict negative constraint to avoid hallucinating missing vehicle specs
        comparison_query = (
            f"Compare the following vehicles based strictly on the retrieved context for the user question: {query}\n"
            "Output a structured comparison (e.g., a Markdown table if applicable) highlighting differences and specifications.\n"
            "Important: If a specification is not found or not specified in the retrieved context for any vehicle, "
            "explicitly state 'Not specified in provided manual'. Do not invent, assume, or extrapolate specifications."
        )
        
        answer = self.llm.generate_answer(
            query=comparison_query, 
            context=context, 
            conversation_history=[]
        )
        
        return answer, sources
