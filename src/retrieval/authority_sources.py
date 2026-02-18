"""
Authoritative Evidence Sources Allowlist

Maintains a curated list of trusted sources for evidence verification,
organized by tier and authority weight.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional
import re
from urllib.parse import urlparse


class AuthorityTier(Enum):
    """Hierarchical authority tier for evidence sources."""
    TIER_1 = 1  # Official technical standards and reference implementations
    TIER_2 = 2  # Reputable academic and institutional sources
    TIER_3 = 3  # Community and supplementary sources


@dataclass
class AuthoritySource:
    """A trusted source for evidence."""
    domain: str
    tier: AuthorityTier
    authority_weight: float  # 0.0-1.0, higher = more authoritative
    category: str  # e.g., "rfc", "python_official", "university", "community"
    description: str = ""
    https_required: bool = True
    url_pattern: Optional[str] = None  # Regex for path validation


class AuthorityAllowlist:
    """Manages curated list of authoritative sources."""
    
    def __init__(self):
        self.sources: Dict[str, AuthoritySource] = {}
        self._initialize_allowlist()
    
    def _initialize_allowlist(self):
        """Populate the authoritative sources allowlist."""
        
        # =====================================================================
        # TIER 1: Authoritative technical standards and reference implementations
        # =====================================================================
        tier_1_sources = [
            AuthoritySource(
                domain="rfc-editor.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=1.0,
                category="rfc",
                description="IETF RFC Editor - Internet standards and RFCs"
            ),
            AuthoritySource(
                domain="docs.python.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=1.0,
                category="python_official",
                description="Python official documentation"
            ),
            AuthoritySource(
                domain="python.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.95,
                category="python_official",
                description="Python.org homepage and PEPs"
            ),
            AuthoritySource(
                domain="openjdk.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=1.0,
                category="java_official",
                description="OpenJDK - official Java implementation"
            ),
            AuthoritySource(
                domain="kubernetes.io",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.99,
                category="kubernetes_official",
                description="Kubernetes official documentation"
            ),
            AuthoritySource(
                domain="docs.microsoft.com",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.95,
                category="microsoft_official",
                description="Microsoft official documentation"
            ),
            AuthoritySource(
                domain="docs.aws.amazon.com",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.95,
                category="aws_official",
                description="AWS official documentation"
            ),
            AuthoritySource(
                domain="developer.mozilla.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.98,
                category="web_standards",
                description="MDN Web Docs - Web standards and APIs"
            ),
            AuthoritySource(
                domain="nodejs.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.96,
                category="nodejs_official",
                description="Node.js official documentation"
            ),
            AuthoritySource(
                domain="www.rust-lang.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.98,
                category="rust_official",
                description="Rust official documentation and book"
            ),
            AuthoritySource(
                domain="golang.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.98,
                category="go_official",
                description="Go official documentation"
            ),
            AuthoritySource(
                domain="cplusplus.com",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.92,
                category="cpp_reference",
                description="C++ reference documentation"
            ),
            AuthoritySource(
                domain="isocpp.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.99,
                category="cpp_standards",
                description="ISO C++ standards committee"
            ),
            AuthoritySource(
                domain="getdocs.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.9,
                category="documentation",
                description="Aggregated official documentation"
            ),
            AuthoritySource(
                domain="gnu.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.95,
                category="gnu_official",
                description="GNU project (GCC, Emacs, etc.)"
            ),
            AuthoritySource(
                domain="linux.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.94,
                category="linux_official",
                description="Linux official resources"
            ),
            AuthoritySource(
                domain="kernel.org",
                tier=AuthorityTier.TIER_1,
                authority_weight=0.99,
                category="linux_kernel",
                description="Linux Kernel Archive"
            ),
        ]
        
        for source in tier_1_sources:
            self.sources[source.domain] = source
        
        # =====================================================================
        # TIER 2: Reputable academic and institutional sources
        # =====================================================================
        tier_2_sources = [
            AuthoritySource(
                domain="ocw.mit.edu",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.95,
                category="university_ocw",
                description="MIT OpenCourseWare"
            ),
            AuthoritySource(
                domain="cs109.github.io",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.85,
                category="university_course",
                description="Harvard CS109 Data Science course"
            ),
            AuthoritySource(
                domain="stanford.edu",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.90,
                category="university",
                description="Stanford University materials"
            ),
            AuthoritySource(
                domain="berkeley.edu",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.90,
                category="university",
                description="UC Berkeley materials"
            ),
            AuthoritySource(
                domain="cam.ac.uk",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.88,
                category="university",
                description="Cambridge University materials"
            ),
            AuthoritySource(
                domain="ox.ac.uk",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.88,
                category="university",
                description="Oxford University materials"
            ),
            AuthoritySource(
                domain="polymath.blog",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.80,
                category="research_blog",
                description="Polymath collaborative research"
            ),
            AuthoritySource(
                domain="arxiv.org",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.85,
                category="preprint",
                description="ArXiv pre-print repository (peer community)"
            ),
            AuthoritySource(
                domain="scholar.google.com",
                tier=AuthorityTier.TIER_2,
                authority_weight=0.80,
                category="research_index",
                description="Google Scholar (aggregator)"
            ),
        ]
        
        for source in tier_2_sources:
            self.sources[source.domain] = source
        
        # =====================================================================
        # TIER 3: Community and supplementary sources (lower confidence)
        # =====================================================================
        tier_3_sources = [
            AuthoritySource(
                domain="wikipedia.org",
                tier=AuthorityTier.TIER_3,
                authority_weight=0.60,
                category="wiki",
                description="Wikipedia (supplementary only, community-edited)"
            ),
            AuthoritySource(
                domain="en.wikipedia.org",
                tier=AuthorityTier.TIER_3,
                authority_weight=0.60,
                category="wiki",
                description="English Wikipedia"
            ),
            AuthoritySource(
                domain="github.com",
                tier=AuthorityTier.TIER_3,
                authority_weight=0.65,
                category="open_source",
                description="GitHub repositories (variable quality)"
            ),
            AuthoritySource(
                domain="stackoverflow.com",
                tier=AuthorityTier.TIER_3,
                authority_weight=0.65,
                category="community",
                description="Stack Overflow (community Q&A)"
            ),
            AuthoritySource(
                domain="medium.com",
                tier=AuthorityTier.TIER_3,
                authority_weight=0.50,
                category="blog",
                description="Medium (user-generated content)"
            ),
            AuthoritySource(
                domain="hashnode.com",
                tier=AuthorityTier.TIER_3,
                authority_weight=0.55,
                category="blog",
                description="Hashnode (developer community blog)"
            ),
            AuthoritySource(
                domain="dev.to",
                tier=AuthorityTier.TIER_3,
                authority_weight=0.60,
                category="community",
                description="DEV Community"
            ),
        ]
        
        for source in tier_3_sources:
            self.sources[source.domain] = source
    
    def get_source(self, url: str) -> Optional[AuthoritySource]:
        """
        Retrieve authority source info from URL.
        
        Args:
            url: Full URL or domain
        
        Returns:
            AuthoritySource if domain is in allowlist, None otherwise
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or url
            
            # Remove www. prefix for matching
            if domain.startswith("www."):
                domain = domain[4:]
            
            return self.sources.get(domain)
        except Exception:
            return None
    
    def is_allowed(self, url: str) -> bool:
        """Check if URL is from an allowlisted domain."""
        return self.get_source(url) is not None
    
    def get_tier(self, url: str) -> Optional[AuthorityTier]:
        """Get tier for URL, None if not allowlisted."""
        source = self.get_source(url)
        return source.tier if source else None
    
    def get_authority_weight(self, url: str) -> float:
        """
        Get authority weight for URL.
        
        Returns:
            Weight between 0.0-1.0, or 0.0 if not allowlisted
        """
        source = self.get_source(url)
        return source.authority_weight if source else 0.0
    
    def get_sources_by_tier(self, tier: AuthorityTier) -> List[AuthoritySource]:
        """Get all sources in a specific tier."""
        return [s for s in self.sources.values() if s.tier == tier]
    
    def add_custom_source(
        self,
        domain: str,
        tier: AuthorityTier,
        authority_weight: float,
        category: str,
        description: str = ""
    ) -> AuthoritySource:
        """
        Add custom source to allowlist (for testing or expansion).
        
        Args:
            domain: Domain to allow
            tier: AuthorityTier
            authority_weight: 0.0-1.0
            category: Category string
            description: Optional description
        
        Returns:
            Created AuthoritySource
        """
        source = AuthoritySource(
            domain=domain,
            tier=tier,
            authority_weight=authority_weight,
            category=category,
            description=description
        )
        self.sources[domain] = source
        return source
    
    def validate_source(
        self,
        url: str,
        require_tier: Optional[AuthorityTier] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if URL is from an allowed source.
        
        Args:
            url: URL to validate
            require_tier: If specified, source must be at least this tier
        
        Returns:
            (is_valid, reason) tuple
        """
        if not url:
            return False, "Empty URL"
        
        source = self.get_source(url)
        
        if not source:
            domain = urlparse(url).netloc
            return False, f"Domain '{domain}' not in allowlist"
        
        if require_tier and source.tier.value > require_tier.value:
            return False, f"Source tier ({source.tier.name}) below required ({require_tier.name})"
        
        return True, None
    
    def get_statistics(self) -> Dict[str, int]:
        """Get allowlist statistics."""
        by_tier = {}
        for tier in AuthorityTier:
            count = len(self.get_sources_by_tier(tier))
            by_tier[tier.name] = count
        
        return {
            "total_sources": len(self.sources),
            "by_tier": by_tier,
            "avg_authority_weight": sum(s.authority_weight for s in self.sources.values()) / max(len(self.sources), 1)
        }


# Global singleton allowlist
_allowlist: Optional[AuthorityAllowlist] = None


def get_allowlist() -> AuthorityAllowlist:
    """Get or create the global allowlist."""
    global _allowlist
    if _allowlist is None:
        _allowlist = AuthorityAllowlist()
    return _allowlist


# Export for convenience
is_allowed_source = lambda url: get_allowlist().is_allowed(url)
get_source_tier = lambda url: get_allowlist().get_tier(url)
get_source_weight = lambda url: get_allowlist().get_authority_weight(url)
