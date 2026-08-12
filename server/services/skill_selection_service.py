"""Skill Selection Service.

Provides automatic selection of relevant skills from a user's published skill library
based on intent-aware query analysis and lightweight skill metadata (name, description, trigger_description).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from sqlalchemy.orm import Session

from server import crud
from server.config import settings
from server.models.skill import Skill, SkillStatus

logger = logging.getLogger(__name__)


class SkillSelectionService:
    """Service to evaluate and automatically select relevant skills for a user query."""

    def __init__(self, client: Optional[OpenAI] = None, model: Optional[str] = None):
        self.client = client or OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model or settings.LLM_MODEL

    def get_candidate_skills(self, db: Session, user_id: str) -> List[Skill]:
        """Fetch candidate published skills belonging to user_id from database."""
        return crud.skill.get_candidate_skills(db, user_id=user_id)

    def prepare_metadata_payload(self, skills: List[Skill]) -> List[Dict[str, Any]]:
        """Extract lightweight metadata for candidate matching (excluding full instructions)."""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description or "",
                "trigger_description": s.trigger_description or "",
            }
            for s in skills
        ]

    def select_skills(
        self,
        db: Session,
        user_id: str,
        query: str,
        max_skills: int = 5,
    ) -> List[Skill]:
        """Automatically select relevant published skills for the user query (up to max_skills).

        Uses only lightweight skill metadata (name, description, trigger_description).
        Returns empty list if no candidate skills exist or no skill is relevant.
        """
        if not query or not query.strip():
            return []

        candidate_skills = self.get_candidate_skills(db, user_id=user_id)
        if not candidate_skills:
            return []

        candidate_map = {s.id: s for s in candidate_skills}
        metadata_payload = self.prepare_metadata_payload(candidate_skills)

        selected_ids = self._evaluate_relevance(
            query=query.strip(),
            candidates=metadata_payload,
            max_skills=max_skills,
        )

        selected_skills: List[Skill] = []
        for sid in selected_ids:
            if sid in candidate_map and len(selected_skills) < max_skills:
                selected_skills.append(candidate_map[sid])

        return selected_skills

    def _evaluate_relevance(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        max_skills: int = 5,
    ) -> List[str]:
        """Evaluate candidate skill metadata against user intent using structured LLM output.

        Follows Intent-Aware Relevance Evaluation Rules:
        - Understand user intent before selecting skills.
        - Match skills based on name, description, and trigger_description.
        - Prefer precision over recall (Level 1 Direct Match & Level 2 Strong Supporting Match only).
        - Reject Level 3 Weak/Indirect matches and keyword-only overlap.
        - Order descending by relevance, max 5 skills.
        - Return valid candidate IDs only.
        """
        if not candidates:
            return []

        valid_candidate_ids = {str(c["id"]) for c in candidates}

        system_prompt = (
            "You are an expert AI skill routing and relevance evaluator.\n"
            "Your objective is to evaluate user queries against candidate skill metadata and determine which skills "
            "are genuinely relevant to fulfilling the user's explicit intent.\n\n"
            "CRITICAL RELEVANCE RULES:\n"
            "1. INTENT-BASED MATCHING: Understand the user's actual requested task and intent before evaluating candidates.\n"
            "2. PRECISION OVER RECALL: Prefer selecting zero skills over selecting loosely related, generic, or non-essential skills.\n"
            "3. RELEVANCE HIERARCHY:\n"
            "   - Level 1 (Direct Match): The skill directly handles or automates the requested task -> SELECT.\n"
            "   - Level 2 (Strong Supporting Match): The skill provides domain-specific instructions or guidelines that materially help complete the task -> SELECT.\n"
            "   - Level 3 (Weak/Indirect Match): The skill only shares general keywords or topic domain -> DO NOT SELECT.\n"
            "   - Level 4 (No Match): Unrelated skill -> DO NOT SELECT.\n"
            "4. NO KEYWORD-ONLY MATCHING: Do NOT select a skill solely because it shares keywords with the query if the actual task is different.\n"
            "5. ABSOLUTE CONSTRAINTS:\n"
            f"   - Select AT MOST {max_skills} skills.\n"
            "   - Order selected skills in DESCENDING order of relevance (most relevant first).\n"
            "   - Never invent, fabricate, or return a skill ID that does not exist in the candidate list.\n"
            "   - Do NOT select extra skills just to reach the maximum limit.\n"
            "   - If no candidate skill is genuinely a Direct or Strong Supporting match, return an empty array [].\n\n"
            "REQUIRED RESPONSE FORMAT:\n"
            'Return ONLY a valid JSON object with key "selected_skill_ids": ["id-1", "id-2"]'
        )

        user_prompt = (
            f'User Query: "{query}"\n\n'
            "Available Candidate Skills (Metadata Only):\n"
            f"{json.dumps(candidates, indent=2)}\n\n"
            "Evaluate candidate skills based on user intent. "
            f'Return JSON: {{"selected_skill_ids": [...]}} containing valid candidate IDs only, ordered by descending relevance.'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            selected_ids = parsed.get("selected_skill_ids", [])
            
            if isinstance(selected_ids, list):
                # Filter to valid candidate IDs only and enforce max_skills limit
                valid_selected = []
                for sid in selected_ids:
                    sid_str = str(sid)
                    if sid_str in valid_candidate_ids and sid_str not in valid_selected:
                        valid_selected.append(sid_str)
                        if len(valid_selected) >= max_skills:
                            break
                return valid_selected
            return []
        except Exception as e:
            logger.error("Error during automatic skill selection relevance evaluation: %s", e)
            return self._heuristic_fallback(query, candidates, max_skills)

    def _heuristic_fallback(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        max_skills: int = 5,
    ) -> List[str]:
        """Fallback keyword matching when LLM API call fails."""
        query_lower = query.lower()
        matched_ids = []
        for cand in candidates:
            name = cand["name"].lower()
            desc = cand["description"].lower()
            trig = cand["trigger_description"].lower()
            
            # Match exact trigger or specific skill name phrase
            name_words = [w for w in name.split() if len(w) > 3]
            trig_words = [w for w in trig.split() if len(w) > 3]
            
            if (trig and trig in query_lower) or (name and name in query_lower) or \
               (name_words and any(w in query_lower for w in name_words)):
                if cand["id"] not in matched_ids:
                    matched_ids.append(cand["id"])
                    if len(matched_ids) >= max_skills:
                        break
        return matched_ids


def get_skill_selection_service() -> SkillSelectionService:
    """Return singleton instance of SkillSelectionService."""
    return SkillSelectionService()
