"""
Integration of Online Evidence Retrieval with Local Evidence

Manages retrieval, merging, and conflict detection between local and online evidence.
Enforces tier-based verification policies and provides conflict reporting.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from src.retrieval.evidence_store import Evidence
from src.retrieval.online_retriever import OnlineRetriever, OnlineSpan, create_retriever
from src.retrieval.authority_sources import (
    AuthorityTier,
    get_allowlist,
    is_allowed_source,
    get_source_tier,
    get_source_weight
)
import config

logger = logging.getLogger(__name__)


@dataclass
class ConflictReport:
    """Report of conflict between local and online evidence."""
    claim_text: str
    conflict_detected: bool
    conflict_type: str  # "none", "text_contradiction", "authority_mismatch"
    severity: str  # "none", "low", "medium", "high"
    local_evidence: Optional[Dict[str, Any]] = None
    online_evidence: Optional[Dict[str, Any]] = None
    resolution: str = ""  # How the conflict was resolved
    detected_at: str = ""


@dataclass
class MergedEvidence:
    """Evidence merged from local and online sources."""
    evidence_id: str
    source_id: str
    text: str
    authority_tier: str
    authority_weight: float
    origin: str  # "local", "online", "merged"
    local_source: Optional[str] = None
    online_source: Optional[str] = None
    is_corroborated: bool = False
    corroboration_count: int = 0


class OnlineEvidenceIntegrator:
    """Integrate online authority evidence with local evidence."""

    def __init__(
        self,
        enable_online: bool = False,
        enforce_policies: bool = True,
        conflict_threshold: float = 0.3  # Similarity threshold for conflicts
    ):
        """
        Initialize integrator.

        Args:
            enable_online: Enable online evidence retrieval
            enforce_policies: Enforce tier-based verification policies
            conflict_threshold: Similarity threshold (0-1) for detecting conflicts
        """
        self.enable_online = enable_online
        self.enforce_policies = enforce_policies
        self.conflict_threshold = conflict_threshold
        self.online_retriever = create_retriever() if enable_online else None
        self.allowlist = get_allowlist()
        self.conflicts: List[ConflictReport] = []

    def retrieve_online_evidence(
        self,
        claim_text: str,
        num_results: int = 5
    ) -> List[OnlineSpan]:
        """
        Retrieve online evidence for a claim.

        Args:
            claim_text: Claim to search for
            num_results: Maximum online sources to retrieve

        Returns:
            List of OnlineSpan objects with online evidence
        """
        if not self.enable_online or not self.online_retriever:
            return []

        max_urls = getattr(config, "ONLINE_MAX_SOURCES_PER_CLAIM", 5)

        try:
            # For now, return empty list (full search implementation would use
            # information retrieval or search engines)
            logger.debug(f"Online retrieval for: {claim_text}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving online evidence: {e}")
            return []

    def merge_evidence(
        self,
        local_evidence: List[Evidence],
        online_evidence: List[OnlineSpan],
        prefer_local: bool = True
    ) -> List[MergedEvidence]:
        """
        Merge local and online evidence with deduplication.

        Args:
            local_evidence: Local evidence from session
            online_evidence: Online evidence from authority sources
            prefer_local: If True, prioritize local evidence in merges

        Returns:
            List of merged evidence items
        """
        merged = []

        # Add local evidence
        for local in local_evidence:
            merged_item = MergedEvidence(
                evidence_id=local.evidence_id,
                source_id=local.source_id,
                text=local.text,
                authority_tier="LOCAL",
                authority_weight=0.5,  # Default for local evidence
                origin="local",
                local_source=local.source_id,
            )
            merged.append(merged_item)

        # Add online evidence if not duplicating local
        if online_evidence:
            for online in online_evidence:
                # Check for corroboration
                is_dup = self._is_duplicate(
                    online.text,
                    [m.text for m in merged if m.origin == "local"]
                )

                if is_dup:
                    # Corroborate local evidence
                    for m in merged:
                        if m.origin == "local" and m.text == online.text:
                            m.is_corroborated = True
                            m.corroboration_count += 1
                else:
                    # Add as new evidence
                    merged_item = MergedEvidence(
                        evidence_id=f"online_{online.span_id}",
                        source_id=online.source_id,
                        text=online.text,
                        authority_tier=online.authority_tier.name,
                        authority_weight=float(online.authority_weight),
                        origin="online",
                        online_source=online.source_id,
                    )
                    merged.append(merged_item)

        return merged

    def _is_duplicate(self, text: str, existing_texts: List[str]) -> bool:
        """
        Check if text is a duplicate of existing evidence.

        Simple implementation: check for substring overlap.
        """
        if not existing_texts:
            return False

        # Normalize texts
        text_norm = text.lower().strip()

        for existing in existing_texts:
            existing_norm = existing.lower().strip()

            # Check for overlap
            if len(text_norm) > 50 and len(existing_norm) > 50:
                # For longer texts, check substring match
                if text_norm in existing_norm or existing_norm in text_norm:
                    return True

        return False

    def detect_conflicts(
        self,
        claim_text: str,
        local_evidence: List[Evidence],
        online_evidence: List[OnlineSpan]
    ) -> Tuple[bool, Optional[ConflictReport]]:
        """
        Detect conflicts between local and online evidence.

        Args:
            claim_text: Claim being verified
            local_evidence: Local evidence items
            online_evidence: Online evidence items

        Returns:
            (has_conflict, conflict_report)
        """
        if not local_evidence or not online_evidence:
            return False, None

        try:
            # Check for contradictory assertions
            local_texts = [e.text for e in local_evidence]
            online_texts = [o.text for o in online_evidence]

            for local_text in local_texts:
                for online_text in online_texts:
                    # Naive conflict detection: very different texts about same claim
                    if self._is_contradictory(local_text, online_text):
                        report = ConflictReport(
                            claim_text=claim_text,
                            conflict_detected=True,
                            conflict_type="text_contradiction",
                            severity="high",
                            local_evidence={"text": local_text},
                            online_evidence={"text": online_text},
                            detected_at=datetime.utcnow().isoformat(),
                        )
                        self.conflicts.append(report)
                        logger.warning(f"Conflict detected for claim: {claim_text}")
                        return True, report

            return False, None

        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            return False, None

    def _is_contradictory(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are contradictory.

        Simple heuristic: texts are contradictory if they negatively reference each other.
        """
        keywords = ["not", "no", "never", "contradiction", "incorrect", "false"]

        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # If one text contains negation and references key terms from the other
        has_negation = any(kw in text1_lower or kw in text2_lower for kw in keywords)

        if has_negation:
            # Very crude: if substantially different despite discussing same topic
            common_len = len(set(text1_lower.split()) & set(text2_lower.split()))
            if common_len < 3:  # Very few words in common
                return True

        return False

    def enforce_verification_policy(
        self,
        claim_text: str,
        local_evidence: List[Evidence],
        online_evidence: List[OnlineSpan],
        min_tier_for_solo_online: AuthorityTier = AuthorityTier.TIER_2
    ) -> Tuple[bool, str]:
        """
        Enforce tier-based verification policies.

        Policy:
        - Tier 1/2 online evidence can verify claims independently
        - Tier 3 online evidence requires 2+ independent sources OR local corroboration
        - Any local evidence always acceptable (from session)

        Args:
            claim_text: Claim text
            local_evidence: Local evidence items
            online_evidence: Online evidence items
            min_tier_for_solo_online: Minimum tier for solo online verification

        Returns:
            (is_valid, reason)
        """
        if not self.enforce_policies:
            return True, "Policy enforcement disabled"

        if local_evidence:
            # Local evidence present - always valid
            return True, "Local evidence present"

        if not online_evidence:
            return False, "No evidence available"

        # Check online evidence tiers
        tier_1_count = sum(1 for e in online_evidence if e.authority_tier == AuthorityTier.TIER_1)
        tier_2_count = sum(1 for e in online_evidence if e.authority_tier == AuthorityTier.TIER_2)
        tier_3_count = sum(1 for e in online_evidence if e.authority_tier == AuthorityTier.TIER_3)

        # Tier 1 or 2 sources can verify independently
        if tier_1_count > 0 or (tier_2_count > 0 and min_tier_for_solo_online.value >= 2):
            return True, "Sufficient Tier 1/2 authority evidence"

        # Tier 3 only sources require corroboration
        if tier_3_count >= 2:
            return True, "Multiple Tier 3 sources provide corroboration"

        return False, "Insufficient authority tier for online-only verification"

    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get summary of detected conflicts."""
        if not self.conflicts:
            return {
                "conflict_count": 0,
                "conflicts": [],
            }

        return {
            "conflict_count": len(self.conflicts),
            "high_severity": sum(1 for c in self.conflicts if c.severity == "high"),
            "medium_severity": sum(1 for c in self.conflicts if c.severity == "medium"),
            "conflicts": [asdict(c) for c in self.conflicts],
        }


def create_integrator(
    enable_online: Optional[bool] = None,
    enforce_policies: Optional[bool] = None
) -> OnlineEvidenceIntegrator:
    """
    Create configured online evidence integrator.

    Uses config defaults if not specified.
    """
    if enable_online is None:
        enable_online = getattr(config, "ENABLE_ONLINE_VERIFICATION", False)

    if enforce_policies is None:
        enforce_policies = getattr(config, "ONLINE_ENFORCE_POLICIES", True)

    return OnlineEvidenceIntegrator(
        enable_online=enable_online,
        enforce_policies=enforce_policies
    )


# Convenience functions for quick access
def should_retrieve_online() -> bool:
    """Check if online retrieval is enabled."""
    return getattr(config, "ENABLE_ONLINE_VERIFICATION", False)


def get_min_tier_for_solo_verification() -> AuthorityTier:
    """Get minimum tier required for solo online verification."""
    tier_val = getattr(config, "ONLINE_MIN_TIER_FOR_SOLO_VERIFICATION", 2)
    if tier_val == 1:
        return AuthorityTier.TIER_1
    elif tier_val == 2:
        return AuthorityTier.TIER_2
    else:
        return AuthorityTier.TIER_3
