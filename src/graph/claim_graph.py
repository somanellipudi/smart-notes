"""
Claim-Evidence Graph for dependency analysis and metric computation.

Builds a NetworkX directed graph of claims and evidence relationships,
computes metrics like redundancy, diversity, support depth, and conflicts.

Research-grade export formats:
- GraphML (Gephi/Cytoscape compatible)
- JSON adjacency list
- PNG visualization
"""

import logging
import json
from typing import List, Dict, Set, Tuple, Optional, Any
import networkx as nx
from pathlib import Path

from src.claims.schema import LearningClaim, GraphMetrics
from src.graph.graph_sanitize import (
    export_graphml_bytes,
    export_graphml_string,
    sanitize_graph_for_graphml,
)

logger = logging.getLogger(__name__)


def export_adjacency_json(graph: nx.Graph) -> str:
    """
    Export a graph as a JSON adjacency structure with sanitized attributes.
    """
    from src.graph.graph_sanitize import _sanitize_value
    
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        safe_attrs = {k: _sanitize_value(v) for k, v in attrs.items()}
        nodes.append({"id": node_id, **safe_attrs})

    edges = []
    for source, target, attrs in graph.edges(data=True):
        safe_attrs = {k: _sanitize_value(v) for k, v in attrs.items()}
        edges.append({"source": source, "target": target, **safe_attrs})

    return json.dumps({"nodes": nodes, "edges": edges}, indent=2, default=str)


class ClaimGraph:
    """
    NetworkX-based graph for claim-evidence relationships and analysis.
    
    Structure:
    - Claim nodes: represent learning claims (with claim_id, status, confidence)
    - Evidence nodes: represent evidence items
    - Edges: directed from claims to evidence (support relationship)
    
    Metrics computed:
    - Redundancy: average evidence per claim
    - Diversity: proportion of different source types
    - Support Depth: average path length from evidence
    - Conflict Count: number of contradictory evidence pairs
    """
    
    def __init__(self, claims: List[LearningClaim]):
        """
        Build graph from claims and their evidence.
        
        Args:
            claims: List of LearningClaim objects with evidence attached
        """
        self.graph = nx.DiGraph()
        self.claims = claims
        self.claim_map = {c.claim_id: c for c in claims}
        self.evidence_map = {}  # evidence_id → EvidenceItem
        
        self._build_graph()
        logger.info(f"ClaimGraph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _build_graph(self):
        """Build graph nodes and edges."""
        # Add claim nodes
        for claim in self.claims:
            self.graph.add_node(
                claim.claim_id,
                node_type="claim",
                claim_type=claim.claim_type.value if claim.claim_type else None,
                status=claim.status.value if claim.status else None,
                confidence=claim.confidence or 0.0
            )
        
        # Add evidence nodes and edges
        for claim in self.claims:
            if not hasattr(claim, 'evidence_objects') or not claim.evidence_objects:
                continue
            
            for i, evidence in enumerate(claim.evidence_objects):
                # Create unique evidence node ID
                evidence_id = f"{claim.claim_id}_ev_{i}"
                self.evidence_map[evidence_id] = evidence
                
                # Add evidence node
                self.graph.add_node(
                    evidence_id,
                    node_type="evidence",
                    source_id=evidence.source_id,
                    source_type=evidence.source_type,
                    similarity=evidence.similarity,
                    snippet=evidence.snippet[:50]  # Truncate for display
                )
                
                # Add edge from claim to evidence
                self.graph.add_edge(claim.claim_id, evidence_id, weight=evidence.similarity)
    
    def compute_metrics(self) -> GraphMetrics:
        """
        Compute all graph metrics.
        
        Returns:
            GraphMetrics object with all computed values
        """
        total_evidence = sum(
            len(c.evidence_objects) if hasattr(c, 'evidence_objects') else 0
            for c in self.claims
        )
        
        metrics = GraphMetrics(
            avg_redundancy=self._compute_redundancy(),
            avg_diversity=self._compute_diversity(),
            avg_support_depth=self._compute_support_depth(),
            conflict_count=self._compute_conflict_count(),
            total_claims=len(self.claims),
            total_evidence=total_evidence
        )
        
        logger.info(
            f"Graph metrics: redundancy={metrics.avg_redundancy:.2f}, "
            f"diversity={metrics.avg_diversity:.2f}, "
            f"depth={metrics.avg_support_depth:.2f}, "
            f"conflicts={metrics.conflict_count}"
        )
        
        return metrics
    
    def _compute_redundancy(self) -> float:
        """
        Compute redundancy: average evidence per claim (0-infinity).
        
        High redundancy means claims are well-supported by multiple sources.
        
        Returns:
            Average number of evidence items per claim
        """
        if not self.claims:
            return 0.0
        
        total_evidence = 0
        for claim in self.claims:
            if hasattr(claim, 'evidence_objects') and claim.evidence_objects:
                total_evidence += len(claim.evidence_objects)
        
        redundancy = total_evidence / len(self.claims)
        return round(redundancy, 2)
    
    def _compute_diversity(self) -> float:
        """
        Compute diversity: proportion of different source types (0-1).
        
        High diversity means evidence comes from varied sources.
        
        Returns:
            Ratio of unique source types to total source types
        """
        if not self.claims:
            return 0.0
        
        source_types = set()
        total_sources = 0
        
        for claim in self.claims:
            if hasattr(claim, 'evidence_objects') and claim.evidence_objects:
                for evidence in claim.evidence_objects:
                    source_types.add(evidence.source_type)
                    total_sources += 1
        
        if total_sources == 0:
            return 0.0
        
        diversity = len(source_types) / max(1, len(source_types))  # Normalized
        return round(diversity, 2)
    
    def _compute_support_depth(self) -> float:
        """
        Compute support depth: average evidence path length (0-1 typically).
        
        Higher depth means stronger hierarchical support structure.
        
        Returns:
            Average normalized path length
        """
        if not self.claims:
            return 0.0
        
        total_depth = 0
        depth_count = 0
        
        for claim in self.claims:
            if hasattr(claim, 'evidence_objects') and claim.evidence_objects:
                # For each claim, depth is 1 (direct evidence)
                # Could be extended for transitive dependencies
                total_depth += 1.0
                depth_count += 1
        
        if depth_count == 0:
            return 0.0
        
        # Normalize to 0-1 (current implementation: always ~1 for direct evidence)
        depth = (total_depth / depth_count) / 2.0  # Normalize
        return round(depth, 2)
    
    def _compute_conflict_count(self) -> int:
        """
        Compute conflict count: number of contradictory evidence pairs.
        
        Detects explicit contradictions in evidence snippets.
        
        Returns:
            Count of conflicting evidence pairs
        """
        conflict_count = 0
        contradiction_keywords = [
            ('always', 'never'),
            ('true', 'false'),
            ('yes', 'no'),
            ('increase', 'decrease'),
            ('positive', 'negative')
        ]
        
        for claim in self.claims:
            if not hasattr(claim, 'evidence_objects') or len(claim.evidence_objects) < 2:
                continue
            
            evidence_list = claim.evidence_objects
            for i in range(len(evidence_list)):
                for j in range(i + 1, len(evidence_list)):
                    snippet1 = evidence_list[i].snippet.lower()
                    snippet2 = evidence_list[j].snippet.lower()
                    
                    # Check for contradictions
                    for pos, neg in contradiction_keywords:
                        if pos in snippet1 and neg in snippet2:
                            conflict_count += 1
                        elif pos in snippet2 and neg in snippet1:
                            conflict_count += 1
        
        return conflict_count
    
    def compute_centrality(self, claim_id: str) -> float:
        """
        Compute betweenness centrality for a claim node.
        
        High centrality means the claim is important for connecting
        other claims in the graph.
        
        Args:
            claim_id: Claim to analyze
        
        Returns:
            Centrality score in [0, 1]
        """
        if claim_id not in self.graph:
            logger.warning(f"Claim {claim_id} not in graph for centrality")
            return 0.0
        
        # Check if graph has enough nodes for meaningful centrality
        if self.graph.number_of_nodes() < 3:
            return 0.0
        
        try:
            centrality_dict = nx.betweenness_centrality(self.graph)
            return round(centrality_dict.get(claim_id, 0.0), 4)
        except Exception as e:
            logger.error(f"Failed to compute centrality for {claim_id}: {e}")
            return 0.0
    
    def compute_support_depth(self, claim_id: str) -> int:
        """
        Compute maximum evidence support depth for a claim.
        
        Depth = longest path from claim to evidence node.
        
        Args:
            claim_id: Claim to analyze
        
        Returns:
            Maximum depth (number of hops)
        """
        if claim_id not in self.graph:
            logger.warning(f"Claim {claim_id} not in graph for support depth")
            return 0
        
        try:
            # Get all successors (evidence nodes)
            successors = list(self.graph.successors(claim_id))
            
            if not successors:
                return 0
            
            # For simple claim-evidence graph, depth is 1
            # For hierarchical graphs, compute longest path
            max_depth = 0
            for successor in successors:
                try:
                    # BFS from claim to evidence
                    path_lengths = nx.single_source_shortest_path_length(
                        self.graph, claim_id
                    )
                    depth = path_lengths.get(successor, 0)
                    max_depth = max(max_depth, depth)
                except Exception:
                    continue
            
            return max_depth
        except Exception as e:
            logger.error(f"Failed to compute support depth for {claim_id}: {e}")
            return 0
    
    def compute_redundancy_score(self, claim_id: str) -> float:
        """
        Compute evidence redundancy for a single claim.
        
        Redundancy = (evidence count - 1) / max_possible
        Higher redundancy means more supporting evidence.
        
        Args:
            claim_id: Claim to analyze
        
        Returns:
            Redundancy score in [0, 1]
        """
        if claim_id not in self.graph:
            logger.warning(f"Claim {claim_id} not in graph for redundancy")
            return 0.0
        
        try:
            evidence_count = self.graph.out_degree(claim_id)
            
            # Normalize by assuming max of 10 evidence pieces is "full" redundancy
            max_evidence = 10
            redundancy = min(evidence_count, max_evidence) / max_evidence
            
            return round(redundancy, 4)
        except Exception as e:
            logger.error(f"Failed to compute redundancy for {claim_id}: {e}")
            return 0.0
    
    def get_connected_components(self) -> List[Set[str]]:
        """
        Get independent claim clusters.
        
        Returns:
            List of sets, each set is a connected component
        """
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        return components
    
    def get_dependency_paths(self, claim_id: str) -> List[List[str]]:
        """
        Get all evidence paths supporting a claim.
        
        Args:
            claim_id: Claim identifier
        
        Returns:
            List of paths (each path is a list of node IDs)
        """
        if claim_id not in self.graph:
            return []
        
        # For direct evidence model, return immediate successors
        successors = list(self.graph.successors(claim_id))
        return [[claim_id, s] for s in successors]
    
    def get_claim_stats(self, claim_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific claim.
        
        Args:
            claim_id: Claim identifier
        
        Returns:
            Dict with claim statistics
        """
        if claim_id not in self.claim_map:
            return {}
        
        claim = self.claim_map[claim_id]
        evidence_count = len(claim.evidence_objects) if hasattr(claim, 'evidence_objects') else 0
        
        if evidence_count > 0:
            avg_similarity = sum(e.similarity for e in claim.evidence_objects) / evidence_count
        else:
            avg_similarity = 0.0
        
        return {
            "claim_id": claim_id,
            "claim_type": claim.claim_type,
            "status": claim.status.value if claim.status else None,
            "confidence": claim.confidence or 0.0,
            "evidence_count": evidence_count,
            "avg_similarity": round(avg_similarity, 2)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to dict representation for JSON serialization.
        
        Returns:
            Dict with nodes, edges, and metrics
        """
        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            nodes.append({"id": node_id, **attrs})
        
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({"source": source, "target": target, **attrs})
        
        metrics = self.compute_metrics()
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "avg_redundancy": metrics.avg_redundancy,
                "avg_diversity": metrics.avg_diversity,
                "avg_support_depth": metrics.avg_support_depth,
                "conflict_count": metrics.conflict_count
            },
            "summary": {
                "total_claims": len(self.claims),
                "total_evidence": sum(
                    len(c.evidence_objects) if hasattr(c, 'evidence_objects') else 0
                    for c in self.claims
                ),
                "verified_claims": len([c for c in self.claims if c.status and c.status.value == "verified"]),
                "rejected_claims": len([c for c in self.claims if c.status and c.status.value == "rejected"])
            }
        }
    
    def export_graphml(self, filepath: str) -> bool:
        """
        Export graph to GraphML format (Gephi/Cytoscape compatible).
        Uses sanitization to handle complex attributes.
        
        Args:
            filepath: Path to save GraphML file
        
        Returns:
            True if successful
        """
        try:
            graphml_bytes = export_graphml_bytes(self.graph)
            with open(filepath, 'wb') as f:
                f.write(graphml_bytes)
            logger.info(f"Graph exported to GraphML: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export GraphML: {e}")
            return False
    
    def get_graphml_bytes(self) -> bytes:
        """
        Get GraphML export as bytes for downloads.
        
        Returns:
            GraphML XML as UTF-8 encoded bytes
        """
        try:
            return export_graphml_bytes(self.graph)
        except Exception as e:
            logger.error(f"Failed to generate GraphML bytes: {e}")
            return b""
    
    def export_adjacency_json(self) -> str:
        """
        Export graph as JSON adjacency list.
        
        Returns:
            JSON string representation of graph
        """
        try:
            adjacency = nx.node_link_data(self.graph)
            import json
            return json.dumps(adjacency, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to export adjacency JSON: {e}")
            return "{}"
    
    def render_graph_png(self, filepath: str, figsize: Tuple[int, int] = (14, 10)) -> bool:
        """
        Render graph as PNG image with matplotlib.
        
        Shows:
        - Claim nodes (circles) colored by status (green/yellow/red)
        - Evidence nodes (squares) in blue
        - Edges weighted by similarity
        - Legend and title
        
        Args:
            filepath: Path to save PNG
            figsize: Figure size (width, height)
        
        Returns:
            True if successful
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')
            
            # Separate nodes by type
            claim_nodes = [node for node, attr in self.graph.nodes(data=True) 
                          if attr.get("node_type") == "claim"]
            evidence_nodes = [node for node, attr in self.graph.nodes(data=True) 
                             if attr.get("node_type") == "evidence"]
            
            # Layout
            if evidence_nodes:
                # Hierarchical: claims top, evidence bottom
                pos = {}
                for i, node in enumerate(claim_nodes):
                    pos[node] = (i * 2, 1)
                for i, node in enumerate(evidence_nodes):
                    pos[node] = (i * 1.5, 0)
            else:
                # Circular layout for claims only
                pos = nx.spring_layout(self.graph, k=3, iterations=50, seed=42)
            
            # Color claims by status
            status_colors = {
                "verified": "#28a745",
                "low_confidence": "#ffc107",
                "rejected": "#dc3545"
            }
            
            claim_colors = []
            claim_labels = {}
            for node in claim_nodes:
                node_data = self.graph.nodes[node].get("data", {})
                status = node_data.get("status", "rejected")
                status_lower = str(status).lower() if status else "rejected"
                
                claim_colors.append(status_colors.get(status_lower, "#6c757d"))
                
                claim_type = node_data.get("claim_type", "claim")
                if isinstance(claim_type, str):
                    claim_type = claim_type.split(".")[-1].replace("_", " ").title()
                confidence = node_data.get("confidence", 0)
                claim_labels[node] = f"{claim_type}\n{confidence:.0%}"
            
            # Draw claim nodes
            if claim_nodes:
                nx.draw_networkx_nodes(
                    self.graph, pos, nodelist=claim_nodes,
                    node_color=claim_colors,
                    node_size=2000,
                    node_shape='o',
                    edgecolors='#333333',
                    linewidths=2,
                    ax=ax
                )
            
            # Draw evidence nodes
            if evidence_nodes:
                nx.draw_networkx_nodes(
                    self.graph, pos, nodelist=evidence_nodes,
                    node_color="#17a2b8",
                    node_size=1200,
                    node_shape='s',
                    edgecolors='#333333',
                    linewidths=2,
                    ax=ax,
                    alpha=0.8
                )
                
                evidence_labels = {node: f"Ev\n{node[:4]}" for node in evidence_nodes}
                nx.draw_networkx_labels(self.graph, pos, evidence_labels, font_size=7, font_weight='bold', ax=ax)
            
            # Draw edges
            if self.graph.number_of_edges() > 0:
                edge_weights = [self.graph[u][v].get('weight', 1.0) for u, v in self.graph.edges()]
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edge_color='#666666',
                    arrows=True,
                    arrowsize=15,
                    arrowstyle='-|>',
                    width=[w * 2 for w in edge_weights],
                    ax=ax,
                    alpha=0.6,
                    connectionstyle='arc3,rad=0.1'
                )
            
            # Draw claim labels
            nx.draw_networkx_labels(self.graph, pos, claim_labels, font_size=8, font_weight='bold', ax=ax)
            
            # Title
            if evidence_nodes:
                title = f"Claim-Evidence Network: {len(claim_nodes)} Claims, {len(evidence_nodes)} Sources"
            else:
                title = f"Extracted Claims: {len(claim_nodes)} Claims (No Evidence)"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Legend
            legend_elements = [
                mpatches.Patch(color='#28a745', label='✓ Verified'),
                mpatches.Patch(color='#ffc107', label='⚠ Low Confidence'),
                mpatches.Patch(color='#dc3545', label='✗ Rejected')
            ]
            if evidence_nodes:
                legend_elements.append(mpatches.Patch(color='#17a2b8', label='📄 Evidence'))
            
            ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
            ax.axis('off')
            plt.tight_layout()
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f"Graph rendered to PNG: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to render PNG: {e}")
            return False
    
    def render_subgraph_png(self, claim_id: str, filepath: str, figsize: Tuple[int, int] = (10, 8)) -> bool:
        """
        Render ego graph for a specific claim (claim + evidence neighbors).
        
        Args:
            claim_id: ID of claim to center on
            filepath: Path to save PNG
            figsize: Figure size
        
        Returns:
            True if successful
        """
        try:
            if claim_id not in self.graph:
                logger.warning(f"Claim {claim_id} not found in graph")
                return False
            
            # Get ego graph (claim + 1-hop neighbors)
            ego = nx.ego_graph(self.graph, claim_id, radius=1)
            
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(figsize=figsize)
            fig.patch.set_facecolor('#f8f9fa')
            ax.set_facecolor('#ffffff')
            
            # Layout
            pos = nx.spring_layout(ego, k=2, iterations=50, seed=42)
            
            # Color nodes
            node_colors = []
            node_list = list(ego.nodes())
            for node in node_list:
                if node == claim_id:
                    node_colors.append('#FF6B6B')  # Red for central claim
                else:
                    node_data = ego.nodes[node].get('data', {})
                    node_type = ego.nodes[node].get('node_type', 'evidence')
                    if node_type == 'evidence':
                        node_colors.append('#4ECDC4')  # Teal for evidence
                    else:
                        node_colors.append('#95E1D3')  # Light teal for related claims
            
            nx.draw_networkx_nodes(
                ego, pos,
                node_color=node_colors,
                node_size=1500,
                edgecolors='#333333',
                linewidths=2,
                ax=ax
            )
            
            # Draw edges
            if ego.number_of_edges() > 0:
                nx.draw_networkx_edges(
                    ego, pos,
                    edge_color='#999999',
                    arrows=True,
                    arrowsize=15,
                    width=2,
                    ax=ax,
                    alpha=0.7
                )
            
            # Labels
            labels = {}
            for node in node_list:
                if node == claim_id:
                    labels[node] = "Central\nClaim"
                else:
                    labels[node] = node[:8]
            
            nx.draw_networkx_labels(ego, pos, labels, font_size=8, font_weight='bold', ax=ax)
            
            ax.set_title(f"Ego Graph: {claim_id[:8]} + Evidence Sources", fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            logger.info(f"Subgraph rendered to PNG: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to render subgraph PNG: {e}")
            return False
