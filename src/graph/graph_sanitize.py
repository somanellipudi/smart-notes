"""
Graph sanitization utilities for GraphML export compatibility.

GraphML format requires node/edge attributes to be simple types (string, int, float, bool).
This module handles conversion of complex Python objects to GraphML-compatible formats.
"""

import json
import base64
import logging
from typing import Any, Dict
import networkx as nx

logger = logging.getLogger(__name__)


def _sanitize_value(value: Any, max_str_length: int = 500) -> Any:
    """
    Sanitize a single attribute value for GraphML compatibility.
    
    Args:
        value: Value to sanitize
        max_str_length: Maximum string length for text attributes
    
    Returns:
        Sanitized value (str, int, float, or bool)
    """
    if value is None:
        return ""
    
    # Bytes: try UTF-8 decode, fallback to base64
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8', errors='replace')[:max_str_length]
        except Exception:
            return base64.b64encode(value).decode('ascii')[:max_str_length]
    
    # Simple types: keep as is
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > max_str_length:
            return value[:max_str_length] + "..."
        return value
    
    # Enums: convert to value
    if hasattr(value, '__class__') and hasattr(value.__class__, '__mro__'):
        import enum
        if enum.Enum in value.__class__.__mro__:
            return str(value.value) if hasattr(value, 'value') else str(value)
    
    # Collections: JSON serialize
    if isinstance(value, (dict, list, tuple, set)):
        try:
            json_str = json.dumps(value, ensure_ascii=False, default=str)
            if len(json_str) > max_str_length:
                return json_str[:max_str_length] + "..."
            return json_str
        except Exception as e:
            logger.warning(f"Failed to serialize {type(value)}: {e}")
            return str(value)[:max_str_length]
    
    # Pydantic models: convert via model_dump()
    if hasattr(value, 'model_dump'):
        try:
            data = value.model_dump()
            json_str = json.dumps(data, ensure_ascii=False, default=str)
            if len(json_str) > max_str_length:
                return json_str[:max_str_length] + "..."
            return json_str
        except Exception as e:
            logger.warning(f"Failed to dump Pydantic model {type(value)}: {e}")
            return str(value)[:max_str_length]
    
    # Dataclasses or objects with .dict()
    if hasattr(value, 'dict'):
        try:
            data = value.dict()
            json_str = json.dumps(data, ensure_ascii=False, default=str)
            if len(json_str) > max_str_length:
                return json_str[:max_str_length] + "..."
            return json_str
        except Exception as e:
            logger.warning(f"Failed to dump object with .dict(): {e}")
            return str(value)[:max_str_length]
    
    # Fallback: string representation
    try:
        str_val = str(value)
        if len(str_val) > max_str_length:
            return str_val[:max_str_length] + "..."
        return str_val
    except Exception:
        return "<non-serializable>"


def sanitize_graph_for_graphml(
    graph: nx.Graph,
    max_str_length: int = 500
) -> nx.Graph:
    """
    Create a GraphML-safe copy of a graph by sanitizing all node/edge attributes.
    
    Args:
        graph: Original NetworkX graph
        max_str_length: Maximum string length for attributes
    
    Returns:
        New graph with sanitized attributes
    """
    sanitized = graph.copy()
    
    # Sanitize node attributes
    for node, attrs in list(sanitized.nodes(data=True)):
        for key, value in list(attrs.items()):
            sanitized.nodes[node][key] = _sanitize_value(value, max_str_length)
    
    # Sanitize edge attributes
    for source, target, attrs in list(sanitized.edges(data=True)):
        for key, value in list(attrs.items()):
            sanitized.edges[source, target][key] = _sanitize_value(value, max_str_length)
    
    logger.info(
        f"Sanitized graph: {sanitized.number_of_nodes()} nodes, "
        f"{sanitized.number_of_edges()} edges"
    )
    
    return sanitized


def export_graphml_string(graph: nx.Graph, max_str_length: int = 500) -> str:
    """
    Export graph to GraphML string with automatic sanitization.
    
    Args:
        graph: NetworkX graph
        max_str_length: Maximum string length for attributes
    
    Returns:
        GraphML XML as string
    """
    from io import StringIO, BytesIO
    import tempfile
    import os
    
    sanitized = sanitize_graph_for_graphml(graph, max_str_length)
    
    # Use temporary file to avoid StringIO/BytesIO issues with nx.write_graphml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
    
    try:
        nx.write_graphml(sanitized, tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def export_graphml_bytes(graph: nx.Graph, max_str_length: int = 500) -> bytes:
    """
    Export graph to GraphML bytes (UTF-8 encoded) for file downloads.
    
    Args:
        graph: NetworkX graph
        max_str_length: Maximum string length for attributes
    
    Returns:
        GraphML XML as UTF-8 encoded bytes
    """
    graphml_str = export_graphml_string(graph, max_str_length)
    return graphml_str.encode('utf-8')
