"""Interactive knowledge graph visualization using pyvis.

Generates a standalone HTML file with a force-directed graph layout.
Entities colored by type, edges colored by relation type, with
interactive color pickers, type toggles, community filters, search,
and detail sidebar.
"""

import json
import logging
import math
import random
import webbrowser
from pathlib import Path

import networkx as nx

from sift_kg.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Node color palette — auto-assigned to entity types, max hue separation
NODE_PALETTE = [
    "#42A5F5",  # blue
    "#66BB6A",  # green
    "#FFA726",  # orange
    "#AB47BC",  # purple
    "#26C6DA",  # cyan
    "#EF5350",  # red
    "#FFEE58",  # yellow
    "#EC407A",  # pink
    "#9CCC65",  # lime
    "#FF7043",  # deep orange
    "#7E57C2",  # deep purple
    "#29B6F6",  # light blue
    "#8D6E63",  # brown
    "#78909C",  # gray
    "#FFD54F",  # amber
    "#26A69A",  # teal
    "#CE93D8",  # light purple
    "#5C6BC0",  # indigo
    "#D4E157",  # yellow-green
    "#FF8A65",  # light deep orange
]

# Semantic entity colors — common types get stable meaningful colors
SEMANTIC_ENTITY_COLORS = {
    "PERSON":      "#42A5F5",  # blue — people
    "ORGANIZATION": "#66BB6A",  # green — groups
    "LOCATION":    "#AB47BC",  # purple — places
    "EVENT":       "#FFA726",  # orange — happenings
    "DOCUMENT":    "#78909C",  # gray — records
}

# Distinct edge color palette — auto-assigned to relation types
EDGE_PALETTE = [
    "#4CAF50",
    "#FF7043",
    "#42A5F5",
    "#AB47BC",
    "#26A69A",
    "#EC407A",
    "#FFA726",
    "#66BB6A",
    "#7E57C2",
    "#29B6F6",
    "#EF5350",
    "#8D6E63",
    "#78909C",
    "#FFCA28",
    "#5C6BC0",
    "#D4E157",
    "#26C6DA",
    "#FF8A65",
    "#9CCC65",
    "#CE93D8",
]

def _generate_community_colors(n: int) -> list[str]:
    """Generate n maximally-separated colors using golden angle hue spacing."""
    import colorsys
    colors = []
    golden_angle = 137.508  # degrees
    for i in range(n):
        hue = ((i * golden_angle) % 360) / 360.0
        r, g, b = colorsys.hls_to_rgb(hue, 0.65, 0.75)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


# Semantic edge colors — named relation types get meaningful colors
SEMANTIC_EDGE_COLORS = {
    "EXTENDS":         "#AB47BC",  # purple — lineage/inheritance
    "PROPOSED_BY":     "#26C6DA",  # cyan — attribution
    "SUPPORTS":        "#66BB6A",  # green — evidence for
    "CONTRADICTS":     "#EF5350",  # red — evidence against
    "USES_METHOD":     "#42A5F5",  # blue — methodology
    "IMPLEMENTS":      "#FFA726",  # orange — concrete realization
    "ASSOCIATED_WITH": "#78909C",  # gray — generic fallback
    "MENTIONED_IN":    "#546E7A",  # dark gray — provenance (low emphasis)
}


def _color_for_entity(entity_type: str, entity_color_map: dict[str, str]) -> str:
    """Get or assign a color for an entity type."""
    if entity_type in entity_color_map:
        return entity_color_map[entity_type]
    if entity_type in SEMANTIC_ENTITY_COLORS:
        entity_color_map[entity_type] = SEMANTIC_ENTITY_COLORS[entity_type]
    else:
        idx = len(entity_color_map) % len(NODE_PALETTE)
        entity_color_map[entity_type] = NODE_PALETTE[idx]
    return entity_color_map[entity_type]


def _color_for_relation(rel_type: str, rel_color_map: dict[str, str]) -> str:
    """Get or assign a color for a relation type."""
    if rel_type in rel_color_map:
        return rel_color_map[rel_type]
    if rel_type in SEMANTIC_EDGE_COLORS:
        rel_color_map[rel_type] = SEMANTIC_EDGE_COLORS[rel_type]
    else:
        idx = len(rel_color_map) % len(EDGE_PALETTE)
        rel_color_map[rel_type] = EDGE_PALETTE[idx]
    return rel_color_map[rel_type]


def generate_view(
    kg: KnowledgeGraph,
    output_path: Path,
    open_browser: bool = True,
    height: str = "100%",
    width: str = "100%",
    descriptions_path: Path | None = None,
) -> Path:
    """Generate an interactive HTML visualization of the knowledge graph."""
    # Load entity descriptions if available
    entity_descriptions: dict[str, str] = {}
    if descriptions_path and descriptions_path.exists():
        entity_descriptions = json.loads(descriptions_path.read_text())
        logger.info(f"Loaded {len(entity_descriptions)} entity descriptions for viewer")

    # Load or compute community assignments
    community_map: dict[str, str] = {}
    communities_path = output_path.parent / "communities.json"
    if communities_path.exists():
        community_map = json.loads(communities_path.read_text())
        logger.info(f"Loaded {len(set(community_map.values()))} communities for viewer")
    if not community_map:
        try:
            undirected = kg.graph.to_undirected()
            raw = nx.community.louvain_communities(undirected)
            if len(raw) > 1:
                for i, comm in enumerate(sorted(raw, key=len, reverse=True)):
                    for nid in comm:
                        community_map[nid] = f"Community {i + 1}"
        except Exception:
            pass

    unique_communities = sorted(set(community_map.values()))
    comm_colors = _generate_community_colors(len(unique_communities))
    community_color_map = {
        label: comm_colors[i]
        for i, label in enumerate(unique_communities)
    }

    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError(
            "pyvis is required for graph visualization.\nInstall it with: pip install pyvis"
        ) from exc

    from sift_kg.graph.postprocessor import strip_metadata

    kg = strip_metadata(kg)

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        select_menu=False,
        filter_menu=False,
    )

    # Physics: strong repulsion, freeze after stabilization
    net.set_options("""{
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -350,
                "centralGravity": 0.002,
                "springLength": 400,
                "springConstant": 0.03,
                "damping": 0.4,
                "avoidOverlap": 0.9
            },
            "solver": "forceAtlas2Based",
            "stabilization": { "enabled": true, "iterations": 300 },
            "enabled": true
        },
        "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 1.2 } },
            "smooth": { "type": "curvedCW", "roundness": 0.1 },
            "color": { "opacity": 0.2 },
            "font": { "size": 0 }
        },
        "nodes": {
            "font": { "size": 0, "face": "Inter, sans-serif" },
            "borderWidth": 1.5,
            "borderWidthSelected": 3
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 999999,
            "zoomView": true,
            "dragView": true,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true
        }
    }""")

    degrees = dict(kg.graph.degree())

    # Seed initial positions by community so clusters start separated
    community_centers: dict[str, tuple[float, float]] = {}
    if unique_communities:
        radius = 1200
        for i, comm in enumerate(unique_communities):
            angle = 2 * math.pi * i / len(unique_communities)
            community_centers[comm] = (radius * math.cos(angle), radius * math.sin(angle))

    # Collect entity types — auto-assign colors as they appear
    entity_types_present: set[str] = set()
    entity_color_map: dict[str, str] = {}

    # Add nodes
    for node_id, data in kg.graph.nodes(data=True):
        entity_type = data.get("entity_type", "UNKNOWN")
        entity_types_present.add(entity_type)
        name = data.get("name", node_id)
        confidence = data.get("confidence", 0)
        entity_color = _color_for_entity(entity_type, entity_color_map)
        degree = degrees.get(node_id, 0)

        # Community border color
        comm_label = community_map.get(node_id, "")
        comm_color = community_color_map.get(comm_label, "#333")
        node_color = {
            "background": entity_color,
            "border": comm_color,
            "highlight": {"background": entity_color, "border": comm_color},
        }
        border_w = 2.0 if comm_label else 1.5

        tooltip_parts = [
            name,
            f"Type: {entity_type}",
            f"Connections: {degree}",
        ]
        if comm_label:
            tooltip_parts.append(f"Community: {comm_label}")
        if isinstance(confidence, (int, float)):
            tooltip_parts.append(f"Confidence: {confidence:.0%}")
        source_docs = data.get("source_documents", [])
        if source_docs:
            tooltip_parts.append(f"Sources: {', '.join(source_docs[:3])}")
        attrs = data.get("attributes", {})
        aliases_raw = []
        if isinstance(attrs, dict):
            for k, v in list(attrs.items())[:4]:
                tooltip_parts.append(f"{k}: {v}")
            aliases_raw = attrs.get("aliases", []) or attrs.get("also_known_as", [])
            if isinstance(aliases_raw, str):
                aliases_raw = [aliases_raw]
        aliases_str = ", ".join(str(a) for a in aliases_raw) if aliases_raw else ""
        tooltip = "\n".join(p for p in tooltip_parts if p)

        size = max(8, min(50, 6 + degree * 2.5))

        desc = entity_descriptions.get(node_id, "")

        # Seed position by community center + jitter
        center = community_centers.get(comm_label, (0, 0))
        jitter = 250
        init_x = center[0] + random.uniform(-jitter, jitter)
        init_y = center[1] + random.uniform(-jitter, jitter)

        net.add_node(
            node_id,
            label=name,
            title=tooltip,
            color=node_color,
            size=size,
            shape="dot",
            borderWidth=border_w,
            x=init_x,
            y=init_y,
            entity_type=entity_type,
            community=comm_label,
            node_degree=degree,
            full_name=name,
            aliases=aliases_str,
            description=desc,
        )

    # Add edges — colored by relation type
    rel_color_map: dict[str, str] = {}
    seen_edges: set[tuple[str, str, str]] = set()

    for source, target, _key, data in kg.graph.edges(data=True, keys=True):
        relation_type = data.get("relation_type", "UNKNOWN")
        edge_key = (source, target, relation_type)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        edge_color = _color_for_relation(relation_type, rel_color_map)
        confidence = data.get("confidence", 0)
        evidence = data.get("evidence", "")
        support_count_raw = data.get("support_count", 1)
        try:
            support_count = max(1, int(support_count_raw))
        except (TypeError, ValueError):
            support_count = 1
        support_docs = data.get("support_documents", [])
        if not isinstance(support_docs, list):
            support_docs = []

        # Look up source/target names
        source_name = kg.graph.nodes[source].get("name", source)
        target_name = kg.graph.nodes[target].get("name", target)

        tooltip_parts = [
            f"{source_name} {relation_type} {target_name}",
        ]
        if isinstance(confidence, (int, float)):
            tooltip_parts.append(f"Confidence: {confidence:.0%}")
        tooltip_parts.append(f"Support mentions: {support_count}")
        if support_docs:
            tooltip_parts.append(f"Support docs: {len(support_docs)}")
        if evidence:
            ev_display = evidence[:150] + "..." if len(evidence) > 150 else evidence
            tooltip_parts.append(f"Evidence: {ev_display}")
        tooltip = "\n".join(p for p in tooltip_parts if p)

        # Width driven by support_count — clear steps at low end without being too thick
        width = min(10.0, 1.5 + support_count * 2.0)

        net.add_edge(
            source,
            target,
            title=tooltip,
            color=edge_color,
            width=width,
            relation_type=relation_type,
            source_name=source_name,
            target_name=target_name,
            full_evidence=evidence or "",
            edge_confidence=float(confidence) if isinstance(confidence, (int, float)) else 0,
            support_count=support_count,
            support_doc_count=len(support_docs),
        )

    # Write HTML then inject UI + fix Firefox height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path))
    _fix_firefox_height(output_path)
    _inject_ui(output_path, kg, entity_types_present, entity_color_map, rel_color_map, community_color_map)

    logger.info(
        f"View generated: {kg.entity_count} entities, "
        f"{kg.relation_count} relations \u2192 {output_path}"
    )

    if open_browser:
        webbrowser.open(f"file://{output_path.resolve()}")

    return output_path


def _fix_firefox_height(html_path: Path) -> None:
    """Fix graph container height for Firefox.

    pyvis sets #mynetwork to height:100% but doesn't set explicit heights
    on html/body/parent elements. Chrome infers the height, Firefox doesn't.
    """
    html = html_path.read_text()
    fix_css = "<style>html, body { height: 100%; margin: 0; padding: 0; overflow: hidden; } .card { height: 100%; }</style>"
    html = html.replace("</head>", f"{fix_css}\n</head>")
    html_path.write_text(html)


def _inject_ui(
    html_path: Path,
    kg: KnowledgeGraph,
    entity_types: set[str],
    entity_color_map: dict[str, str],
    rel_color_map: dict[str, str],
    community_color_map: dict[str, str] | None = None,
) -> None:
    """Inject sidebar with search, entity/relation/community toggles + color pickers + detail panel."""
    from string import Template

    viewer_dir = Path(__file__).parent / "viewer"

    # Load template files
    css = (viewer_dir / "styles.css").read_text()
    controls_template = Template((viewer_dir / "controls.html").read_text())
    app_js = (viewer_dir / "app.js").read_text()

    # Entity type controls
    entity_items = ""
    for et in sorted(entity_types):
        color = entity_color_map.get(et, NODE_PALETTE[0])
        entity_items += (
            f'<div class="type-row">'
            f'<input type="checkbox" checked data-etype="{et}" onchange="toggleEntityType(this)">'
            f'<input type="color" value="{color}" data-etype-color="{et}" onchange="changeEntityColor(this)">'
            f'<span class="type-label">{et}</span>'
            f"</div>"
        )

    # Community controls
    community_section = ""
    if community_color_map:
        community_items = ""
        for label in sorted(community_color_map.keys()):
            color = community_color_map[label]
            community_items += (
                f'<div class="type-row">'
                f'<input type="checkbox" checked data-community="{label}" onchange="toggleCommunity(this)">'
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                f'background:{color};border:1px solid #555;flex-shrink:0"></span>'
                f'<span class="type-label">{label}</span>'
                f"</div>"
            )
        community_section = f'<div class="section-header" style="border-top:none">Communities</div>{community_items}'

    # Relation type controls — count edges per relation type
    rel_counts: dict[str, int] = {}
    for _, _, _, edata in kg.graph.edges(data=True, keys=True):
        rt = edata.get("relation_type", "UNKNOWN")
        rel_counts[rt] = rel_counts.get(rt, 0) + 1

    relation_items = ""
    for rt in sorted(rel_color_map.keys(), key=lambda r: rel_counts.get(r, 0), reverse=True):
        color = rel_color_map[rt]
        checked = "" if rt == "MENTIONED_IN" else " checked"
        count = rel_counts.get(rt, 0)
        relation_items += (
            f'<div class="type-row">'
            f'<input type="checkbox"{checked} data-rtype="{rt}" onchange="toggleRelationType(this)">'
            f'<span class="type-label">{rt}</span>'
            f'<span style="margin-left:auto;font-size:10px;color:#666;flex-shrink:0">{count}</span>'
            f"</div>"
        )

    # Adaptive degree filter: small graphs default to 0 so all nodes are visible
    substantive = kg.relation_count - rel_counts.get("MENTIONED_IN", 0)
    default_degree = 0 if substantive < 100 else 2

    # Substitute HTML template
    controls_html = controls_template.substitute(
        entity_items=entity_items,
        relation_items=relation_items,
        community_section=community_section,
        default_degree=default_degree,
        entity_count=kg.entity_count,
        relation_count=kg.relation_count,
    )

    # Build config JSON for JS runtime values
    config_json = json.dumps({
        "communityColors": community_color_map or {},
        "defaultDegree": default_degree,
    })

    # Assemble: CSS + controls + config script + app script
    injected = (
        f"<style>{css}</style>\n"
        f"{controls_html}\n"
        f"<script>var SIFT_CONFIG={config_json};</script>\n"
        f"<script>{app_js}</script>"
    )

    html = html_path.read_text()
    html = html.replace("</body>", f"{injected}</body>")
    html_path.write_text(html)
