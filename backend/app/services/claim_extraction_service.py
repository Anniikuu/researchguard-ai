import json
import re
import logging
from typing import List, Dict, Any, Optional

from .ollama_service import ollama_service, OllamaServiceError

logger = logging.getLogger(__name__)

CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise academic claim extraction engine.\n"
    "Your task is to decompose a given text into a list of atomic, standalone factual claims.\n\n"
    "CRITICAL RULES:\n"
    "1. Extract ONLY factual assertions made in the text.\n"
    "2. Split compound sentences into individual atomic claims.\n"
    "3. Ensure each claim is self-contained and retains its original meaning.\n"
    "4. Do NOT verify, evaluate, or judge whether claims are true or false.\n"
    "5. Do NOT label claims as SUPPORTED, UNSUPPORTED, TRUE, or FALSE.\n"
    "6. Output MUST be valid JSON with a single key \"claims\" containing an array of claim strings.\n\n"
    "EXAMPLE INPUT:\n"
    "Logistic regression is a supervised learning algorithm. It is used for binary classification.\n\n"
    "EXAMPLE OUTPUT:\n"
    "{\n"
    "  \"claims\": [\n"
    "    \"Logistic regression is a supervised learning algorithm.\",\n"
    "    \"Logistic regression is used for binary classification.\"\n"
    "  ]\n"
    "}"
)

def _clean_and_parse_json(text: str) -> Optional[List[str]]:
    """
    Parses JSON output from LLM, attempting to extract 'claims' array cleanly.
    """
    if not text:
        return None

    # 1. Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "claims" in data and isinstance(data["claims"], list):
            claims = [str(c).strip() for c in data["claims"] if str(c).strip()]
            return claims
    except Exception:
        pass

    # 2. Extract JSON block from markdown ```json ... ``` or regex pattern
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\"claims\"\s*:\s*\[.*?\].*\})", text, re.DOTALL)
        
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "claims" in data and isinstance(data["claims"], list):
                claims = [str(c).strip() for c in data["claims"] if str(c).strip()]
                return claims
        except Exception:
            pass

    return None

def _fallback_sentence_splitter(text: str) -> List[str]:
    """
    Deterministic fallback: splits text into sentence-based claims if LLM fails JSON format.
    """
    if not text.strip():
        return []
    # Remove citations like [Page X] or [Doc Y] for clean claim extraction
    cleaned = re.sub(r"\[(?:Page|Doc|Source)[^\]]*\]", "", text, flags=re.IGNORECASE)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    claims = []
    for s in sentences:
        s_clean = s.strip(" \t\n\r-\u2022")
        if len(s_clean) >= 10:  # Ignore trivial non-factual fragments
            claims.append(s_clean)
    return claims

async def extract_claims_from_answer(answer_text: str) -> List[str]:
    """
    Extracts individual atomic claims from a generated RAG answer using local Ollama (qwen2.5:7b-instruct).
    Does NOT make any verification decision (no SUPPORTED/UNSUPPORTED labels).
    """
    if not answer_text or not answer_text.strip():
        return []

    # If the answer indicates no information was found, return no claims
    if "documents do not contain enough information" in answer_text.lower():
        return []

    user_prompt = f"Extract atomic factual claims from the following text:\n\n{answer_text}"

    try:
        res = await ollama_service.generate_response(
            prompt=user_prompt,
            system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            temperature=0.0
        )
        raw_llm_output = res.get("response", "").strip()
        parsed_claims = _clean_and_parse_json(raw_llm_output)

        if parsed_claims is not None:
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for c in parsed_claims:
                if c.lower() not in seen:
                    seen.add(c.lower())
                    deduped.append(c)
            return deduped
        else:
            logger.warning("LLM claim extraction did not return valid JSON; using fallback sentence splitter.")
            return _fallback_sentence_splitter(answer_text)

    except OllamaServiceError as err:
        logger.warning(f"Ollama claim extraction service failed ({err.message}); falling back to sentence splitting.")
        return _fallback_sentence_splitter(answer_text)
    except Exception as err:
        logger.error(f"Unexpected error during claim extraction: {err}")
        return _fallback_sentence_splitter(answer_text)
