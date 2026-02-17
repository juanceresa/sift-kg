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
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.4 } },
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

        # Width driven by support_count — linear so differences are obvious
        width = min(12.0, max(1.0, support_count * 2.5))

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

    controls_html = f"""
    <style>
        #sift-controls {{
            position:fixed; top:12px; left:12px; z-index:999;
            background:rgba(26,26,46,0.95); border:1px solid #333;
            border-radius:10px; padding:14px 16px;
            font-family:Inter,system-ui,sans-serif; font-size:13px;
            color:#e0e0e0; width:320px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            max-height: calc(100vh - 24px);
            overflow-y: auto;
        }}
        #sift-controls::-webkit-scrollbar {{ width: 4px; }}
        #sift-controls::-webkit-scrollbar-thumb {{ background: #444; border-radius: 2px; }}
        .section-header {{
            font-weight:600; margin:12px 0 6px; font-size:11px;
            color:#888; text-transform:uppercase; letter-spacing:0.5px;
            border-top: 1px solid #333; padding-top: 10px;
        }}
        .section-header:first-of-type {{ border-top: none; margin-top: 6px; }}
        .type-row {{
            display:flex; align-items:center; gap:6px; margin:3px 0; cursor:pointer;
        }}
        .type-row input[type="checkbox"] {{ margin:0; cursor:pointer; }}
        .type-row input[type="color"] {{
            width:18px; height:18px; border:none; padding:0;
            background:none; cursor:pointer; border-radius:3px;
        }}
        .type-label {{ font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        #search-input {{
            width:100%; padding:6px 10px; border-radius:6px;
            border:1px solid #444; background:#16213e;
            color:#e0e0e0; font-size:12px; margin-bottom:4px;
            box-sizing:border-box; outline:none;
        }}
        #search-input:focus {{ border-color:#4FC3F7; }}
        /* Detail sidebar */
        #detail-panel {{
            position:fixed; top:0; right:0; z-index:998;
            width:320px; height:100vh;
            background:rgba(26,26,46,0.97); border-left:1px solid #333;
            font-family:Inter,system-ui,sans-serif; font-size:13px;
            color:#e0e0e0; display:none; flex-direction:column;
            overflow:hidden;
        }}
        #detail-panel.open {{ display:flex; }}
        #detail-header {{
            padding:12px 14px; border-bottom:1px solid #333;
            display:flex; align-items:center; justify-content:space-between;
        }}
        #detail-header h3 {{
            margin:0; font-size:14px; font-weight:600; color:#fff;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
            max-width:240px;
        }}
        #detail-close {{
            background:none; border:none; color:#888; cursor:pointer;
            font-size:18px; padding:0 4px; line-height:1;
        }}
        #detail-close:hover {{ color:#fff; }}
        #detail-body {{
            flex:1; overflow-y:auto; padding:12px 14px;
        }}
        #detail-body::-webkit-scrollbar {{ width: 4px; }}
        #detail-body::-webkit-scrollbar-thumb {{ background: #444; border-radius: 2px; }}
        .d-field {{ margin-bottom:8px; }}
        .d-label {{
            font-size:10px; text-transform:uppercase; letter-spacing:0.5px;
            color:#666; margin-bottom:2px;
        }}
        .d-val {{ font-size:13px; color:#e0e0e0; word-wrap:break-word; line-height:1.4; }}
        .d-badge {{
            display:inline-block; padding:2px 8px; border-radius:4px;
            font-size:11px; font-weight:600;
        }}
        .d-conn {{
            padding:5px 0; border-bottom:1px solid #2a2a3e;
            font-size:12px; cursor:pointer;
        }}
        .d-conn:hover {{ color:#4FC3F7; }}
        .d-conn.active {{ background:rgba(79,195,247,0.15); color:#4FC3F7; border-left:2px solid #4FC3F7; padding-left:6px; }}
        .d-evidence {{
            font-size:12px; color:#ccc; line-height:1.6;
            background:rgba(255,255,255,0.03); border-radius:6px;
            padding:10px 12px; border-left:3px solid #4FC3F7;
            white-space:pre-wrap;
        }}
        /* Focus mode banner */
        #focus-banner {{
            position:fixed; top:12px; left:50%; transform:translateX(-50%);
            z-index:1000; background:rgba(66,165,245,0.95);
            color:#fff; padding:8px 18px; border-radius:8px;
            font-family:Inter,system-ui,sans-serif; font-size:13px;
            font-weight:600; display:none; align-items:center; gap:10px;
            box-shadow:0 2px 12px rgba(0,0,0,0.3);
        }}
        #focus-banner.visible {{ display:flex; }}
        #focus-exit {{
            background:none; border:none; color:#fff; cursor:pointer;
            font-size:18px; padding:0 2px; line-height:1; opacity:0.8;
        }}
        #focus-exit:hover {{ opacity:1; }}
    </style>
    <div id="sift-controls">
        <div style="font-weight:700;margin-bottom:10px;font-size:15px;color:#fff">sift-kg</div>
        <input id="search-input" type="text" placeholder="Search entities..." oninput="searchEntity(this.value)">

        <div style="margin:8px 0 4px">
            <label style="font-size:11px;color:#888">Min connections: <span id="deg-val">{default_degree}</span></label>
            <input id="deg-slider" type="range" min="0" max="20" value="{default_degree}" style="width:100%;margin:2px 0;accent-color:#4FC3F7" oninput="filterByDegree(this.value)">
        </div>

        {community_section}

        <div class="section-header">Entity Types</div>
        {entity_items}

        <div class="section-header">Relation Types</div>
        {relation_items}

        <div style="margin-top:12px;font-size:11px;color:#555;border-top:1px solid #333;padding-top:8px">
            {kg.entity_count} entities &middot; {kg.relation_count} relations
        </div>
    </div>
    <div id="focus-banner">
        <span id="focus-label">Focused on: —</span>
        <button id="focus-exit" onclick="exitFocusMode()">&times;</button>
    </div>
    <div id="detail-panel">
        <div id="detail-header">
            <h3 id="detail-title">Detail</h3>
            <button id="detail-close" onclick="closeDetail()">&times;</button>
        </div>
        <div id="detail-body"></div>
    </div>
    <div id="edge-tooltip" style="display:none;position:fixed;pointer-events:none;
        background:#23233a;border:1px solid #555;border-radius:6px;padding:6px 10px;
        font-size:12px;color:#e0e0e0;z-index:9999;max-width:260px;
        box-shadow:0 2px 8px rgba(0,0,0,0.4);white-space:nowrap"></div>
    """

    # NOTE: Use a regular string (not raw) so \\n produces literal \n in the JS output.
    # In the JS below, we use \\n to produce the JS string literal \n for split().
    script = """
    <script>
    var allNodes = nodes.get();
    var allEdges = edges.get();

    // --- Community region data ---
    var communityHulls = {};   // comm -> [{x,y}, ...] padded convex hull points
    var communityCentroids = {}; // comm -> {x, y}
    var communityColors = COMMUNITY_COLORS_JS;

    // Convex hull (Graham scan)
    function convexHull(points) {
        if (points.length < 3) return points.slice();
        points.sort(function(a, b) { return a.x === b.x ? a.y - b.y : a.x - b.x; });
        var lower = [];
        for (var i = 0; i < points.length; i++) {
            while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], points[i]) <= 0) lower.pop();
            lower.push(points[i]);
        }
        var upper = [];
        for (var i = points.length - 1; i >= 0; i--) {
            while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], points[i]) <= 0) upper.pop();
            upper.push(points[i]);
        }
        upper.pop(); lower.pop();
        return lower.concat(upper);
    }
    function cross(o, a, b) { return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x); }

    // Pad hull outward from centroid
    function padHull(hull, cx, cy, padding) {
        return hull.map(function(p) {
            var dx = p.x - cx, dy = p.y - cy;
            var dist = Math.sqrt(dx * dx + dy * dy) || 1;
            return { x: p.x + (dx / dist) * padding, y: p.y + (dy / dist) * padding };
        });
    }

    function computeCommunityRegions() {
        var positions = network.getPositions();
        var commPoints = {};
        allNodes.forEach(function(n) {
            if (!n.community) return;
            var p = positions[n.id];
            if (!p) return;
            if (!commPoints[n.community]) commPoints[n.community] = [];
            commPoints[n.community].push({ x: p.x, y: p.y });
        });
        communityHulls = {};
        communityCentroids = {};
        for (var comm in commPoints) {
            var pts = commPoints[comm];
            // Centroid
            var cx = 0, cy = 0;
            for (var i = 0; i < pts.length; i++) { cx += pts[i].x; cy += pts[i].y; }
            cx /= pts.length; cy /= pts.length;
            communityCentroids[comm] = { x: cx, y: cy };
            // Hull with padding
            if (pts.length < 3) {
                communityHulls[comm] = padHull(pts, cx, cy, 80);
            } else {
                var hull = convexHull(pts);
                communityHulls[comm] = padHull(hull, cx, cy, 80);
            }
        }
    }

    network.once('stabilizationIterationsDone', function() {
        network.setOptions({ physics: { enabled: false } });
        computeCommunityRegions();
    });

    // Draw filled community regions BEHIND nodes
    network.on('beforeDrawing', function(ctx) {
        if (focusedNodeId !== null) return;
        for (var comm in communityHulls) {
            var hull = communityHulls[comm];
            if (hull.length < 2) continue;
            var color = communityColors[comm] || '#ffffff';
            var r = parseInt(color.slice(1,3), 16);
            var g = parseInt(color.slice(3,5), 16);
            var b = parseInt(color.slice(5,7), 16);

            ctx.save();
            ctx.beginPath();
            // Draw smooth rounded hull using quadratic curves
            if (hull.length === 2) {
                ctx.moveTo(hull[0].x, hull[0].y);
                ctx.lineTo(hull[1].x, hull[1].y);
            } else {
                // Start at midpoint of first edge
                var mx = (hull[0].x + hull[hull.length - 1].x) / 2;
                var my = (hull[0].y + hull[hull.length - 1].y) / 2;
                ctx.moveTo(mx, my);
                for (var i = 0; i < hull.length; i++) {
                    var next = (i + 1) % hull.length;
                    var mx2 = (hull[i].x + hull[next].x) / 2;
                    var my2 = (hull[i].y + hull[next].y) / 2;
                    ctx.quadraticCurveTo(hull[i].x, hull[i].y, mx2, my2);
                }
            }
            ctx.closePath();
            ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.08)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(' + r + ',' + g + ',' + b + ',0.25)';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
        }
    });

    // Draw community labels ON TOP of nodes — scale-compensated
    network.on('afterDrawing', function(ctx) {
        if (focusedNodeId !== null) return;
        var scale = network.getScale();
        var fontSize = Math.round(16 / scale);
        var strokeW = Math.max(2, Math.round(3 / scale));
        for (var comm in communityCentroids) {
            var pos = communityCentroids[comm];
            var color = communityColors[comm] || '#ffffff';
            ctx.save();
            ctx.font = '700 ' + fontSize + 'px Inter, system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.strokeStyle = '#1a1a2e';
            ctx.lineWidth = strokeW;
            ctx.lineJoin = 'round';
            ctx.strokeText(comm, pos.x, pos.y);
            ctx.fillStyle = color;
            ctx.fillText(comm, pos.x, pos.y);
            ctx.restore();
        }
    });

    var hiddenEntityTypes = new Set();
    var hiddenRelationTypes = new Set();
    var hiddenCommunities = new Set();
    var focusedNodeId = null;
    var focusConnIndex = -1;
    var focusHistory = [];

    // --- Detail sidebar ---
    var dp = document.getElementById('detail-panel');
    var dt = document.getElementById('detail-title');
    var db = document.getElementById('detail-body');

    function esc(s) {
        var d = document.createElement('div');
        d.textContent = String(s);
        return d.innerHTML;
    }

    function closeDetail() { dp.classList.remove('open'); }

    function focusNode(nid) {
        network.focus(nid, { scale: 1.5, animation: { duration: 400 } });
        network.selectNodes([nid]);
        showNodeDetail(nid);
    }

    // event delegation for clickable connections
    db.addEventListener('click', function(ev) {
        var el = ev.target.closest('[data-nid]');
        if (el) focusNode(el.getAttribute('data-nid'));
    });

    network.on('click', function(params) {
        if (focusedNodeId !== null && params.nodes && params.nodes.length > 0) {
            // In focus mode: click neighbor to shift focus
            focusHistory.push(focusedNodeId);
            enterFocusMode(params.nodes[0]);
            return;
        }
        if (focusedNodeId !== null && (!params.nodes || params.nodes.length === 0) && (!params.edges || params.edges.length === 0)) {
            // Click empty canvas → exit focus
            exitFocusMode();
            return;
        }
        if (params.nodes && params.nodes.length > 0) {
            showNodeDetail(params.nodes[0]);
        } else if (params.edges && params.edges.length > 0) {
            showEdgeDetail(params.edges[0]);
        }
    });

    function showNodeDetail(nodeId) {
        var node = nodes.get(nodeId);
        if (!node) return;
        dt.textContent = node.full_name || node.label || nodeId;
        dp.classList.add('open');

        // gather connections — group by neighbor, merge relation types
        var connMap = {};
        allEdges.forEach(function(e) {
            var dir, nid, name;
            if (e.from === nodeId) { dir = '\\u2192'; nid = e.to; name = e.target_name || e.to; }
            else if (e.to === nodeId) { dir = '\\u2190'; nid = e.from; name = e.source_name || e.from; }
            else return;
            var key = dir + '|' + nid;
            if (!connMap[key]) connMap[key] = { rels: [], name: name, dir: dir, nid: nid, maxSupport: 0 };
            if (connMap[key].rels.indexOf(e.relation_type) < 0) connMap[key].rels.push(e.relation_type);
            connMap[key].maxSupport = Math.max(connMap[key].maxSupport, e.support_count || 1);
        });
        var conns = [];
        for (var k in connMap) {
            var g = connMap[k];
            g.rels.sort();
            g.rel = g.rels.join(', ');
            conns.push(g);
        }
        conns.sort(function(a, b) { return a.rel.localeCompare(b.rel) || a.name.localeCompare(b.name); });
        window._conns = conns;

        var h = '';
        // type
        var tc = node.color || '#B0BEC5';
        if (typeof tc === 'object') tc = tc.background || '#B0BEC5';
        h += '<div class="d-field"><div class="d-label">Type</div>';
        h += '<span class="d-badge" style="background:' + esc(tc) + ';color:#1a1a2e">' + esc(node.entity_type || 'UNKNOWN') + '</span></div>';

        // community
        if (node.community) {
            h += '<div class="d-field"><div class="d-label">Community</div>';
            h += '<div class="d-val">' + esc(node.community) + '</div></div>';
        }

        // narrative description
        if (node.description) {
            h += '<div class="d-field"><div class="d-label">Description</div>';
            h += '<div class="d-evidence">' + esc(node.description) + '</div></div>';
        }

        // parse tooltip fields
        var lines = (node.title || '').split('\\n');
        for (var i = 1; i < lines.length; i++) {
            var ln = lines[i].trim();
            if (!ln) continue;
            if (ln.startsWith('Community:')) continue;
            var ci = ln.indexOf(': ');
            if (ci > 0) {
                h += '<div class="d-field"><div class="d-label">' + esc(ln.substring(0, ci)) + '</div>';
                h += '<div class="d-val">' + esc(ln.substring(ci + 2)) + '</div></div>';
            }
        }

        // unique relation types for filter (flatten from grouped rels)
        var relTypes = [];
        var seenRel = {};
        for (var r = 0; r < conns.length; r++) {
            for (var ri = 0; ri < conns[r].rels.length; ri++) {
                var rt = conns[r].rels[ri];
                if (!seenRel[rt]) { relTypes.push(rt); seenRel[rt] = true; }
            }
        }
        relTypes.sort();

        // filter select + connections header
        h += '<div class="d-field" style="margin-top:14px">';
        h += '<div class="d-label">Connections (' + conns.length + ')</div>';
        if (relTypes.length > 1) {
            h += '<select id="conn-filter" onchange="filterConns(this.value)" style="width:100%;padding:4px 6px;margin:4px 0 6px;border-radius:4px;border:1px solid #444;background:#16213e;color:#e0e0e0;font-size:11px;outline:none">';
            h += '<option value="">All types</option>';
            for (var t = 0; t < relTypes.length; t++) {
                var cnt = conns.filter(function(c){ return c.rels.indexOf(relTypes[t]) >= 0; }).length;
                h += '<option value="' + esc(relTypes[t]) + '">' + esc(relTypes[t]) + ' (' + cnt + ')</option>';
            }
            h += '</select>';
        }
        h += '<div id="conn-list"></div></div>';

        db.innerHTML = h;
        filterConns('');
    }

    function filterConns(relFilter) {
        var list = document.getElementById('conn-list');
        if (!list) return;
        var conns = window._conns || [];
        var h = '';
        var shown = 0;
        for (var j = 0; j < conns.length; j++) {
            var c = conns[j];
            if (relFilter && c.rels.indexOf(relFilter) < 0) continue;
            shown++;
            h += '<div class="d-conn" data-nid="' + esc(c.nid) + '">';
            var relLabel = c.rels.map(function(r){ return formatRelLabel(r); }).join(', ');
            h += '<span style="color:#888;font-size:10px">' + esc(c.dir) + ' ' + esc(relLabel).toUpperCase() + '</span><br>';
            h += '<span>' + esc(c.name) + '</span></div>';
        }
        list.innerHTML = h;
    }

    function showEdgeDetail(edgeId) {
        var edge = edges.get(edgeId);
        if (!edge) return;
        var rt = edge.relation_type || 'UNKNOWN';
        dt.textContent = rt;
        dp.classList.add('open');

        var ec = edge.color || '#555';
        if (typeof ec === 'object') ec = ec.color || '#555';

        var h = '';
        h += '<div class="d-field"><span class="d-badge" style="background:' + esc(ec) + ';color:#1a1a2e">' + esc(rt) + '</span></div>';
        h += '<div class="d-field"><div class="d-label">From</div>';
        h += '<div class="d-val" style="cursor:pointer;color:#4FC3F7" data-nid="' + esc(edge.from) + '">' + esc(edge.source_name || edge.from) + '</div></div>';
        h += '<div class="d-field"><div class="d-label">To</div>';
        h += '<div class="d-val" style="cursor:pointer;color:#4FC3F7" data-nid="' + esc(edge.to) + '">' + esc(edge.target_name || edge.to) + '</div></div>';

        if (edge.edge_confidence) {
            h += '<div class="d-field"><div class="d-label">Confidence</div>';
            h += '<div class="d-val">' + Math.round(edge.edge_confidence * 100) + '%</div></div>';
        }
        var sc = edge.support_count || 1;
        var sdc = edge.support_doc_count || 0;
        h += '<div class="d-field"><div class="d-label">Support</div>';
        h += '<div class="d-val">' + sc + (sc === 1 ? ' mention' : ' mentions');
        if (sdc > 1) h += ' across ' + sdc + ' docs';
        h += '</div></div>';
        if (edge.full_evidence) {
            h += '<div class="d-field"><div class="d-label">Evidence</div>';
            h += '<div class="d-evidence">' + esc(edge.full_evidence) + '</div></div>';
        }
        db.innerHTML = h;
    }

    // --- Entity type toggle ---
    function toggleEntityType(cb) {
        var type = cb.dataset.etype;
        if (cb.checked) hiddenEntityTypes.delete(type);
        else hiddenEntityTypes.add(type);
        applyFilters();
    }

    // --- Community toggle ---
    function toggleCommunity(cb) {
        var comm = cb.dataset.community;
        if (cb.checked) hiddenCommunities.delete(comm);
        else hiddenCommunities.add(comm);
        applyFilters();
    }

    // --- Relation type toggle ---
    function toggleRelationType(cb) {
        var type = cb.dataset.rtype;
        if (cb.checked) hiddenRelationTypes.delete(type);
        else hiddenRelationTypes.add(type);
        applyEdgeFilters();
    }

    // --- Entity color picker ---
    function changeEntityColor(input) {
        var type = input.dataset.etypeColor;
        var color = input.value;
        var updates = [];
        allNodes.forEach(function(n) {
            if (n.entity_type === type) {
                var c = n.color;
                var border = (typeof c === 'object' && c.border) ? c.border : '#333';
                updates.push({ id: n.id, color: { background: color, border: border, highlight: { background: color, border: border } } });
            }
        });
        nodes.update(updates);
        allNodes = nodes.get();
    }

    function applyFilters() {
        var updates = [];
        allNodes.forEach(function(node) {
            var hidden = hiddenEntityTypes.has(node.entity_type) ||
                         (node.community && hiddenCommunities.has(node.community)) ||
                         (node.node_degree || 0) < minDegree;
            updates.push({ id: node.id, hidden: hidden });
        });
        nodes.update(updates);
    }

    function applyEdgeFilters() {
        var updates = [];
        allEdges.forEach(function(edge) {
            updates.push({ id: edge.id, hidden: hiddenRelationTypes.has(edge.relation_type) });
        });
        edges.update(updates);
    }

    // --- Degree filter ---
    var minDegree = __DEFAULT_DEGREE__;
    function filterByDegree(val) {
        minDegree = parseInt(val, 10);
        document.getElementById('deg-val').textContent = minDegree;
        applyFilters();
    }

    // --- Startup: apply smart defaults ---
    (function() {
        // MENTIONED_IN starts unchecked — add to hidden set
        var mentionedCb = document.querySelector('[data-rtype="MENTIONED_IN"]');
        if (mentionedCb && !mentionedCb.checked) {
            hiddenRelationTypes.add('MENTIONED_IN');
        }
        applyEdgeFilters();
        applyFilters();
    })();

    // --- Search ---
    function searchEntity(query) {
        if (!query || query.length < 2) {
            var reset = allNodes.map(function(n) {
                return { id: n.id, opacity: 1.0, font: { size: 0 }, borderWidth: n.community ? 2 : 1.5 };
            });
            nodes.update(reset);
            return;
        }
        query = query.toLowerCase();
        var matchIds = new Set();
        allNodes.forEach(function(n) {
            var name = (n.full_name || n.label || '').toLowerCase();
            var als = (n.aliases || '').toLowerCase();
            if (name.includes(query) || als.includes(query)) matchIds.add(n.id);
        });
        var neighborIds = new Set();
        allEdges.forEach(function(e) {
            if (matchIds.has(e.from)) neighborIds.add(e.to);
            if (matchIds.has(e.to)) neighborIds.add(e.from);
        });
        var updates = allNodes.map(function(n) {
            if (matchIds.has(n.id))
                return { id: n.id, opacity: 1.0, font: { size: 18, color: '#ffffff' }, borderWidth: 3 };
            else if (neighborIds.has(n.id))
                return { id: n.id, opacity: 0.8, font: { size: 12 }, borderWidth: n.community ? 2 : 1.5 };
            else
                return { id: n.id, opacity: 0.1, font: { size: 0 }, borderWidth: 1 };
        });
        nodes.update(updates);
        if (matchIds.size > 0) {
            network.focus(matchIds.values().next().value, { scale: 1.5, animation: { duration: 500 } });
        }
    }

    // --- Focus mode (ego-graph explorer) ---
    function formatRelLabel(rt) {
        return rt.toLowerCase().replace(/_/g, ' ');
    }

    network.on('doubleClick', function(params) {
        if (params.nodes && params.nodes.length > 0) {
            enterFocusMode(params.nodes[0]);
        }
    });

    document.addEventListener('keydown', function(ev) {
        if (ev.key === 'Escape') {
            exitFocusMode();
            return;
        }
        if (focusedNodeId === null) return;
        var rows = document.querySelectorAll('#conn-list .d-conn');
        if (!rows.length) return;

        if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
            ev.preventDefault();
            if (ev.key === 'ArrowDown') focusConnIndex = Math.min(focusConnIndex + 1, rows.length - 1);
            else focusConnIndex = focusConnIndex - 1;
            if (focusConnIndex < 0) {
                // Back to full neighborhood view
                focusConnIndex = -1;
                rows.forEach(function(r) { r.classList.remove('active'); });
                var restoreId = focusedNodeId;
                focusedNodeId = null;  // clear so enterFocusMode guard allows re-entry
                enterFocusMode(restoreId);
                return;
            }
            highlightConn(focusConnIndex);
        } else if (ev.key === 'ArrowRight' || ev.key === 'Enter') {
            ev.preventDefault();
            if (focusConnIndex >= 0 && focusConnIndex < rows.length) {
                focusHistory.push(focusedNodeId);
                enterFocusMode(rows[focusConnIndex].getAttribute('data-nid'));
            }
        } else if (ev.key === 'Backspace' || ev.key === 'Delete' || ev.key === 'ArrowLeft') {
            ev.preventDefault();
            if (focusHistory.length > 0) {
                var prevId = focusHistory.pop();
                focusedNodeId = null;
                enterFocusMode(prevId);
            }
        }
    });

    function highlightConn(idx) {
        var rows = document.querySelectorAll('#conn-list .d-conn');
        rows.forEach(function(r, i) { r.classList.toggle('active', i === idx); });
        if (rows[idx]) rows[idx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        if (idx < 0 || idx >= rows.length) return;
        var c = { nid: rows[idx].getAttribute('data-nid') };

        network.selectNodes([c.nid]);

        // Find neighbor's adjacents, ranked by degree — cap at 10
        var adjList = [];
        var adjDeg = {};
        allEdges.forEach(function(e) {
            var other = null;
            if (e.from === c.nid && e.to !== focusedNodeId) other = e.to;
            if (e.to === c.nid && e.from !== focusedNodeId) other = e.from;
            if (other && !adjDeg[other]) {
                adjDeg[other] = 0;
                adjList.push(other);
            }
            if (other) adjDeg[other]++;
        });
        adjList.sort(function(a, b) {
            var da = 0, db = 0;
            allNodes.forEach(function(n) {
                if (n.id === a) da = n.node_degree || 0;
                if (n.id === b) db = n.node_degree || 0;
            });
            return db - da;
        });
        var MAX_ADJ = 10;
        var adjIds = new Set(adjList.slice(0, MAX_ADJ));

        // Nodes: primary pair full, adjacents ghosted
        var nodeUpdates = [];
        allNodes.forEach(function(n) {
            var isPrimary = n.id === focusedNodeId || n.id === c.nid;
            var isAdj = adjIds.has(n.id);
            nodeUpdates.push({ id: n.id, hidden: !(isPrimary || isAdj), opacity: isAdj ? 0.35 : 1.0 });
        });
        nodes.update(nodeUpdates);

        // Edges: primary thick + labeled, adjacent thin + unlabeled
        var edgeUpdates = [];
        allEdges.forEach(function(e) {
            var isPrimary = (e.from === focusedNodeId && e.to === c.nid) || (e.to === focusedNodeId && e.from === c.nid);
            var isAdj = (e.from === c.nid && adjIds.has(e.to)) || (e.to === c.nid && adjIds.has(e.from));
            var show = (isPrimary || isAdj) && !hiddenRelationTypes.has(e.relation_type);
            var origWidth = Math.min(12, Math.max(1, (e.support_count || 1) * 2.5));
            var origColor = (typeof e.color === 'string') ? e.color : (e.color && e.color.color) || '#888';
            edgeUpdates.push({
                id: e.id,
                hidden: !show,
                color: { color: origColor, opacity: isPrimary ? 1.0 : 0.25 },
                label: isPrimary ? formatRelLabel(e.relation_type) : '',
                width: isPrimary ? Math.max(3, origWidth) : 1
            });
        });
        edges.update(edgeUpdates);

        // Camera: center on PRIMARY PAIR only, adjacents are peripheral
        var positions = network.getPositions([focusedNodeId, c.nid]);
        var p1 = positions[focusedNodeId];
        var p2 = positions[c.nid];
        var midX = (p1.x + p2.x) / 2;
        var midY = (p1.y + p2.y) / 2;

        var pad = 250;
        var rangeX = Math.abs(p1.x - p2.x) + pad * 2;
        var rangeY = Math.abs(p1.y - p2.y) + pad * 2;

        var canvasEl = network.canvas.frame.canvas;
        var totalW = canvasEl.clientWidth;
        var totalH = canvasEl.clientHeight;
        var leftW = 352;
        var rightW = dp.classList.contains('open') ? 320 : 0;
        var visW = totalW - leftW - rightW;
        var visH = totalH - 80;

        var scale = Math.min(visW / rangeX, visH / rangeY, 1.5);

        var visCenterPx = leftW + visW / 2;
        var shiftPx = visCenterPx - totalW / 2;

        network.moveTo({
            position: { x: midX - shiftPx / scale, y: midY },
            scale: scale,
            animation: { duration: 300, easingFunction: 'easeInOutQuad' }
        });

        // Fonts: primary pair large + white, adjacents small + dim
        (function() {
            var nodeFontSize = Math.round(20 / scale);
            var adjFontSize = Math.round(9 / scale);
            var edgeFontSize = Math.round(16 / scale);
            var strokeW = Math.max(3, Math.round(3 / scale));

            var fontUpdatesN = [];
            allNodes.forEach(function(n) {
                if (n.id === focusedNodeId || n.id === c.nid) {
                    fontUpdatesN.push({ id: n.id, font: { size: nodeFontSize, color: '#ffffff', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
                } else if (adjIds.has(n.id)) {
                    fontUpdatesN.push({ id: n.id, font: { size: adjFontSize, color: '#555555', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
                }
            });
            nodes.update(fontUpdatesN);

            var fontUpdatesE = [];
            allEdges.forEach(function(e) {
                var isPrimary = (e.from === focusedNodeId && e.to === c.nid) || (e.to === focusedNodeId && e.from === c.nid);
                if (isPrimary && !hiddenRelationTypes.has(e.relation_type)) {
                    fontUpdatesE.push({ id: e.id, font: { size: edgeFontSize, color: '#ffffff', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
                }
            });
            edges.update(fontUpdatesE);
        })();
    }

    // --- Overview hover preview: show name + highlight connections ---
    var overviewHoveredId = null;
    network.on('hoverNode', function(params) {
        if (focusedNodeId !== null) return;  // focus mode handles its own display
        var nid = params.node;
        overviewHoveredId = nid;
        var scale = network.getScale();
        var labelSize = Math.round(12 / scale);
        var neighborSize = Math.round(9 / scale);
        var strokeW = Math.max(2, Math.round(2 / scale));

        // Find connected neighbors
        var neighborIds = new Set();
        var connEdgeIds = new Set();
        allEdges.forEach(function(e) {
            if (e.from === nid) { neighborIds.add(e.to); connEdgeIds.add(e.id); }
            if (e.to === nid) { neighborIds.add(e.from); connEdgeIds.add(e.id); }
        });

        // Show hovered node label + dim neighbor labels
        var nodeUpdates = [];
        nodeUpdates.push({ id: nid, font: { size: labelSize, color: '#ffffff', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
        neighborIds.forEach(function(id) {
            nodeUpdates.push({ id: id, font: { size: neighborSize, color: '#999999', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
        });
        nodes.update(nodeUpdates);

        // Brighten connected edges
        var edgeUpdates = [];
        connEdgeIds.forEach(function(eid) {
            edgeUpdates.push({ id: eid, color: { opacity: 0.7 } });
        });
        edges.update(edgeUpdates);
    });

    network.on('blurNode', function(params) {
        if (focusedNodeId !== null) return;
        if (overviewHoveredId === null) return;
        overviewHoveredId = null;

        // Reset all node fonts to 0 (overview default)
        var nodeResets = [];
        allNodes.forEach(function(n) {
            nodeResets.push({ id: n.id, font: { size: 0 } });
        });
        nodes.update(nodeResets);

        // Reset all edge opacities
        var edgeResets = [];
        allEdges.forEach(function(e) {
            edgeResets.push({ id: e.id, color: { opacity: 0.2 } });
        });
        edges.update(edgeResets);
    });

    // Hover-to-reveal edge labels + support tooltip in focus mode
    var edgeTooltip = document.getElementById('edge-tooltip');
    network.on('hoverEdge', function(params) {
        if (focusedNodeId === null) return;
        var e = edges.get(params.edge);
        if (!e) return;
        // Show edge label only in neighborhood view (pair view already has labels)
        if (focusConnIndex < 0) {
            edges.update({ id: e.id, font: { size: 12, color: '#fff', strokeWidth: 3, strokeColor: '#1a1a2e' }, label: formatRelLabel(e.relation_type) });
        }
        // Show support tooltip in all focus sub-modes
        var parts = [formatRelLabel(e.relation_type)];
        if (e.edge_confidence) parts.push(Math.round(e.edge_confidence * 100) + '%');
        var sc = e.support_count || 1;
        parts.push(sc + (sc === 1 ? ' mention' : ' mentions'));
        if (e.support_doc_count > 1) parts.push(e.support_doc_count + ' docs');
        edgeTooltip.innerHTML = parts.join(' &middot; ');
        edgeTooltip.style.display = 'block';
    });
    network.on('blurEdge', function(params) {
        if (focusedNodeId === null) return;
        var e = edges.get(params.edge);
        if (!e) return;
        if (focusConnIndex < 0) {
            edges.update({ id: e.id, font: { size: 0 }, label: '' });
        }
        edgeTooltip.style.display = 'none';
    });
    // Track mouse to position tooltip
    document.getElementById('mynetwork').addEventListener('mousemove', function(ev) {
        if (edgeTooltip.style.display === 'block') {
            edgeTooltip.style.left = (ev.clientX + 14) + 'px';
            edgeTooltip.style.top = (ev.clientY - 28) + 'px';
        }
    });

    function enterFocusMode(nodeId) {
        if (focusedNodeId === nodeId && focusConnIndex < 0) return;  // already in neighborhood view
        overviewHoveredId = null;  // clear any hover preview state
        focusedNodeId = nodeId;
        var node = nodes.get(nodeId);
        if (!node) return;

        // Show banner
        var banner = document.getElementById('focus-banner');
        document.getElementById('focus-label').textContent = 'Focused on: ' + (node.full_name || node.label || nodeId);
        banner.classList.add('visible');

        // Find all 1-hop neighbors, ranked by degree
        var allNeighbors = [];
        var neighborDeg = {};
        var connectedEdgeIds = new Set();
        allEdges.forEach(function(e) {
            if (e.from === nodeId) { connectedEdgeIds.add(e.id); if (!neighborDeg[e.to]) { neighborDeg[e.to] = 0; allNeighbors.push(e.to); } neighborDeg[e.to]++; }
            if (e.to === nodeId) { connectedEdgeIds.add(e.id); if (!neighborDeg[e.from]) { neighborDeg[e.from] = 0; allNeighbors.push(e.from); } neighborDeg[e.from]++; }
        });

        // Sort by global degree, cap visible neighbors for dense hubs
        allNeighbors.sort(function(a, b) {
            var da = 0, db = 0;
            allNodes.forEach(function(n) { if (n.id === a) da = n.node_degree || 0; if (n.id === b) db = n.node_degree || 0; });
            return db - da;
        });
        var MAX_NEIGHBORS = 25;
        var visibleNeighbors = new Set(allNeighbors.slice(0, MAX_NEIGHBORS));
        visibleNeighbors.add(nodeId);

        // Show top neighbors, hide rest
        var nodeUpdates = [];
        allNodes.forEach(function(n) {
            var isVisible = visibleNeighbors.has(n.id);
            var isFocused = n.id === nodeId;
            var belowDegree = (n.node_degree || 0) < minDegree;
            nodeUpdates.push({ id: n.id, hidden: !isVisible || (!isFocused && belowDegree), opacity: 1.0, font: { size: 14, color: '#e0e0e0' } });
        });
        nodes.update(nodeUpdates);

        // Count parallel edges per node pair to offset curves
        var pairCount = {};
        var pairIndex = {};
        allEdges.forEach(function(e) {
            if (!connectedEdgeIds.has(e.id)) return;
            var key = [e.from, e.to].sort().join('||');
            if (!pairCount[key]) pairCount[key] = 0;
            pairIndex[e.id] = pairCount[key];
            pairCount[key]++;
        });

        // Show static edge labels only for small neighborhoods
        var showStaticLabels = visibleNeighbors.size <= 20;

        // Show edges only to visible neighbors
        var edgeUpdates = [];
        allEdges.forEach(function(e) {
            if (connectedEdgeIds.has(e.id)) {
                var other = e.from === nodeId ? e.to : e.from;
                if (!visibleNeighbors.has(other)) {
                    edgeUpdates.push({ id: e.id, hidden: true });
                    return;
                }
                var key = [e.from, e.to].sort().join('||');
                var total = pairCount[key] || 1;
                var idx = pairIndex[e.id] || 0;
                var roundness = total > 1 ? 0.15 + idx * 0.2 : 0.1;
                edgeUpdates.push({
                    id: e.id,
                    hidden: hiddenRelationTypes.has(e.relation_type),
                    color: { opacity: 0.6 },
                    font: { size: showStaticLabels ? 11 : 0, color: '#ccc', strokeWidth: 3, strokeColor: '#1a1a2e' },
                    label: showStaticLabels ? formatRelLabel(e.relation_type) : '',
                    smooth: { type: 'curvedCW', roundness: roundness }
                });
            } else {
                edgeUpdates.push({ id: e.id, hidden: true });
            }
        });
        edges.update(edgeUpdates);

        focusConnIndex = -1;

        // Sidebar-aware camera fit
        var fitIds = [];
        visibleNeighbors.forEach(function(nid) { fitIds.push(nid); });
        var positions = network.getPositions(fitIds);
        var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        fitIds.forEach(function(nid) {
            var p = positions[nid];
            if (!p) return;
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.y > maxY) maxY = p.y;
        });
        var midX = (minX + maxX) / 2;
        var midY = (minY + maxY) / 2;

        var pad = 150;
        var rangeX = (maxX - minX) + pad * 2;
        var rangeY = (maxY - minY) + pad * 2;

        var canvasEl = network.canvas.frame.canvas;
        var totalW = canvasEl.clientWidth;
        var totalH = canvasEl.clientHeight;
        var leftW = 352;
        var rightW = dp.classList.contains('open') ? 320 : 0;
        var visW = totalW - leftW - rightW;
        var visH = totalH - 80;

        var scale = Math.min(visW / rangeX, visH / rangeY, 1.2);

        var visCenterPx = leftW + visW / 2;
        var shiftPx = visCenterPx - totalW / 2;

        network.moveTo({
            position: { x: midX - shiftPx / scale, y: midY },
            scale: scale,
            animation: { duration: 400, easingFunction: 'easeInOutQuad' }
        });

        // After camera settles, apply scale-aware labels on top neighbors
        setTimeout(function() {
            var s = network.getScale();
            var fontSize = Math.round(14 / s);
            var strokeW = Math.max(2, Math.round(2 / s));

            // Label top 15 by degree + focused node
            var maxLabeled = 15;
            var labeledSet = new Set();
            labeledSet.add(nodeId);
            for (var i = 0; i < Math.min(maxLabeled, allNeighbors.length); i++) {
                if (visibleNeighbors.has(allNeighbors[i])) labeledSet.add(allNeighbors[i]);
            }

            var updates = [];
            allNodes.forEach(function(n) {
                if (visibleNeighbors.has(n.id)) {
                    var show = labeledSet.has(n.id);
                    updates.push({ id: n.id, font: { size: show ? fontSize : 0, color: '#e0e0e0', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
                }
            });
            nodes.update(updates);
        }, 450);

        showNodeDetail(nodeId);
    }

    function exitFocusMode() {
        if (focusedNodeId === null) {
            closeDetail();
            return;
        }
        focusedNodeId = null;
        focusConnIndex = -1;
        focusHistory = [];

        // Hide banner
        document.getElementById('focus-banner').classList.remove('visible');

        // Restore all nodes per current filters + reset fonts to overview (no labels)
        applyFilters();
        var fontResets = [];
        allNodes.forEach(function(n) {
            fontResets.push({ id: n.id, font: { size: 0, color: '#e0e0e0' }, opacity: 1.0 });
        });
        nodes.update(fontResets);

        // Restore edges — re-apply edge filters, reset labels and curves
        var edgeUpdates = [];
        allEdges.forEach(function(e) {
            edgeUpdates.push({
                id: e.id,
                hidden: hiddenRelationTypes.has(e.relation_type),
                color: { opacity: 0.2 },
                font: { size: 0 },
                label: '',
                smooth: { type: 'curvedCW', roundness: 0.1 }
            });
        });
        edges.update(edgeUpdates);

        // Fit camera back to full graph
        network.fit({ animation: { duration: 400, easingFunction: 'easeInOutQuad' } });
    }
    </script>
    """

    html = html_path.read_text()
    # Inject community color map into JS
    comm_colors_json = json.dumps(community_color_map or {})
    script_with_colors = script.replace("COMMUNITY_COLORS_JS", comm_colors_json).replace(
        "__DEFAULT_DEGREE__", str(default_degree)
    )
    html = html.replace("</body>", f"{controls_html}{script_with_colors}</body>")
    html_path.write_text(html)
