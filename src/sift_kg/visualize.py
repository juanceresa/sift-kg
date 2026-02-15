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

# Color palette for entity types — visually distinct, dark-mode friendly
ENTITY_COLORS = {
    # Academic entity types
    "CONCEPT":     "#42A5F5",  # blue — abstract ideas
    "THEORY":      "#AB47BC",  # purple — frameworks
    "METHOD":      "#66BB6A",  # green — techniques
    "SYSTEM":      "#FFA726",  # orange — concrete implementations
    "FINDING":     "#FFEE58",  # yellow — results
    "RESEARCHER":  "#26C6DA",  # cyan — people
    "PHENOMENON":  "#EF5350",  # red — observable things
    "PUBLICATION": "#78909C",  # gray — papers
    "FIELD":       "#8D6E63",  # brown — disciplines
    "DATASET":     "#9CCC65",  # lime — data
    # OSINT entity types (backward compat)
    "PERSON":      "#4FC3F7",
    "ORGANIZATION": "#81C784",
    "SHELL_COMPANY": "#FFB74D",
    "FINANCIAL_INSTRUMENT": "#F06292",
    "FINANCIAL_ACCOUNT": "#F06292",
    "GOVERNMENT_AGENCY": "#CE93D8",
    "LOCATION":    "#BA68C8",
    "DOCUMENT":    "#90A4AE",
    "EVENT":       "#FFD54F",
}
DEFAULT_ENTITY_COLOR = "#B0BEC5"

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

# Community border colors — distinct from entity fill colors
COMMUNITY_COLORS = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DDA0DD",
    "#98D8C8",
    "#F7DC6F",
    "#BB8FCE",
    "#85C1E9",
    "#F0B27A",
    "#82E0AA",
]


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


def _color_for_relation(rel_type: str, rel_color_map: dict[str, str]) -> str:
    """Get or assign a color for a relation type."""
    if rel_type in rel_color_map:
        return rel_color_map[rel_type]
    # Use semantic color if available, otherwise auto-assign from palette
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
    community_color_map = {
        label: COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
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
            "font": { "size": 0 }
        },
        "nodes": {
            "font": { "size": 14, "face": "Inter, sans-serif" },
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

    # Collect entity types
    entity_types_present: set[str] = set()

    # Add nodes
    for node_id, data in kg.graph.nodes(data=True):
        entity_type = data.get("entity_type", "UNKNOWN")
        entity_types_present.add(entity_type)
        name = data.get("name", node_id)
        confidence = data.get("confidence", 0)
        entity_color = ENTITY_COLORS.get(entity_type, DEFAULT_ENTITY_COLOR)
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

        # Look up source/target names
        source_name = kg.graph.nodes[source].get("name", source)
        target_name = kg.graph.nodes[target].get("name", target)

        tooltip_parts = [
            f"{source_name} {relation_type} {target_name}",
        ]
        if isinstance(confidence, (int, float)):
            tooltip_parts.append(f"Confidence: {confidence:.0%}")
        if evidence:
            ev_display = evidence[:150] + "..." if len(evidence) > 150 else evidence
            tooltip_parts.append(f"Evidence: {ev_display}")
        tooltip = "\n".join(p for p in tooltip_parts if p)

        width = 1.0 if not isinstance(confidence, (int, float)) else max(0.5, confidence * 2.5)

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
        )

    # Write HTML then inject UI + fix Firefox height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path))
    _fix_firefox_height(output_path)
    _inject_ui(output_path, kg, entity_types_present, rel_color_map, community_color_map)

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
    rel_color_map: dict[str, str],
    community_color_map: dict[str, str] | None = None,
) -> None:
    """Inject sidebar with search, entity/relation/community toggles + color pickers + detail panel."""

    # Entity type controls
    entity_items = ""
    for et in sorted(entity_types):
        color = ENTITY_COLORS.get(et, DEFAULT_ENTITY_COLOR)
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
        community_section = f'<div class="section-header">Communities</div>{community_items}'

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
            f'<input type="color" value="{color}" data-rtype-color="{rt}" onchange="changeRelationColor(this)">'
            f'<span class="type-label">{rt}</span>'
            f'<span style="margin-left:auto;font-size:10px;color:#666;flex-shrink:0">{count}</span>'
            f"</div>"
        )

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
            <label style="font-size:11px;color:#888">Min connections: <span id="deg-val">2</span></label>
            <input id="deg-slider" type="range" min="0" max="20" value="2" style="width:100%;margin:2px 0;accent-color:#4FC3F7" oninput="filterByDegree(this.value)">
        </div>

        <div class="section-header" style="border-top:none">Entity Types</div>
        {entity_items}

        {community_section}

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
    """

    # NOTE: Use a regular string (not raw) so \\n produces literal \n in the JS output.
    # In the JS below, we use \\n to produce the JS string literal \n for split().
    script = """
    <script>
    // Freeze after stabilization
    network.once('stabilizationIterationsDone', function() {
        network.setOptions({ physics: { enabled: false } });
    });

    var allNodes = nodes.get();
    var allEdges = edges.get();
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

        // gather connections
        var conns = [];
        allEdges.forEach(function(e) {
            if (e.from === nodeId) conns.push({ rel: e.relation_type, name: e.target_name || e.to, dir: '\\u2192', nid: e.to });
            if (e.to === nodeId) conns.push({ rel: e.relation_type, name: e.source_name || e.from, dir: '\\u2190', nid: e.from });
        });
        conns.sort(function(a, b) { return a.rel.localeCompare(b.rel); });
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

        // unique relation types for filter
        var relTypes = [];
        var seen = {};
        for (var r = 0; r < conns.length; r++) {
            if (!seen[conns[r].rel]) { relTypes.push(conns[r].rel); seen[conns[r].rel] = true; }
        }
        relTypes.sort();

        // filter select + connections header
        h += '<div class="d-field" style="margin-top:14px">';
        h += '<div class="d-label">Connections (' + conns.length + ')</div>';
        if (relTypes.length > 1) {
            h += '<select id="conn-filter" onchange="filterConns(this.value)" style="width:100%;padding:4px 6px;margin:4px 0 6px;border-radius:4px;border:1px solid #444;background:#16213e;color:#e0e0e0;font-size:11px;outline:none">';
            h += '<option value="">All types</option>';
            for (var t = 0; t < relTypes.length; t++) {
                var cnt = conns.filter(function(c){ return c.rel === relTypes[t]; }).length;
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
            if (relFilter && c.rel !== relFilter) continue;
            shown++;
            h += '<div class="d-conn" data-nid="' + esc(c.nid) + '">';
            h += '<span style="color:#888;font-size:10px;text-transform:uppercase">' + esc(c.dir + ' ' + c.rel) + '</span><br>';
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

    // --- Relation color picker ---
    function changeRelationColor(input) {
        var type = input.dataset.rtypeColor;
        var color = input.value;
        var updates = [];
        allEdges.forEach(function(e) {
            if (e.relation_type === type) {
                updates.push({ id: e.id, color: color });
            }
        });
        edges.update(updates);
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
    var minDegree = 2;
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
                return { id: n.id, opacity: 1.0, font: { size: 14 }, borderWidth: n.community ? 2 : 1.5 };
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
        if (ev.key === 'Escape' && focusedNodeId !== null) {
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
        // Update sidebar highlight
        rows.forEach(function(r, i) { r.classList.toggle('active', i === idx); });
        // Scroll into view
        if (rows[idx]) rows[idx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        if (idx < 0 || idx >= rows.length) return;
        var c = { nid: rows[idx].getAttribute('data-nid') };

        // Select neighbor node in graph
        network.selectNodes([c.nid]);

        // Hide all nodes/edges except focused + highlighted neighbor
        var nodeUpdates = [];
        allNodes.forEach(function(n) {
            var visible = n.id === focusedNodeId || n.id === c.nid;
            nodeUpdates.push({ id: n.id, hidden: !visible });
        });
        nodes.update(nodeUpdates);

        var edgeUpdates = [];
        allEdges.forEach(function(e) {
            var isThisConn = (e.from === focusedNodeId && e.to === c.nid) || (e.to === focusedNodeId && e.from === c.nid);
            edgeUpdates.push({
                id: e.id,
                hidden: !isThisConn || hiddenRelationTypes.has(e.relation_type),
                label: isThisConn ? formatRelLabel(e.relation_type) : '',
                width: isThisConn ? 4 : undefined
            });
        });
        edges.update(edgeUpdates);

        // Center the pair in the visible area between sidebars
        var positions = network.getPositions([focusedNodeId, c.nid]);
        var p1 = positions[focusedNodeId];
        var p2 = positions[c.nid];
        var midX = (p1.x + p2.x) / 2;
        var midY = (p1.y + p2.y) / 2;

        var pad = 200;
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

        // Scale font sizes so they appear constant on screen (~20px nodes, ~16px edges)
        var nodeFontSize = Math.round(20 / scale);
        var edgeFontSize = Math.round(16 / scale);

        // Re-apply with distance-aware font sizes
        var fontUpdatesN = [];
        allNodes.forEach(function(n) {
            var visible = n.id === focusedNodeId || n.id === c.nid;
            if (visible) fontUpdatesN.push({ id: n.id, font: { size: nodeFontSize, color: '#ffffff', strokeWidth: Math.max(3, Math.round(3 / scale)), strokeColor: '#1a1a2e' } });
        });
        nodes.update(fontUpdatesN);

        var fontUpdatesE = [];
        allEdges.forEach(function(e) {
            var isThisConn = (e.from === focusedNodeId && e.to === c.nid) || (e.to === focusedNodeId && e.from === c.nid);
            if (isThisConn && !hiddenRelationTypes.has(e.relation_type)) {
                fontUpdatesE.push({ id: e.id, font: { size: edgeFontSize, color: '#ffffff', strokeWidth: Math.max(3, Math.round(3 / scale)), strokeColor: '#1a1a2e' } });
            }
        });
        edges.update(fontUpdatesE);

        var visCenterPx = leftW + visW / 2;
        var shiftPx = visCenterPx - totalW / 2;

        network.moveTo({
            position: { x: midX - shiftPx / scale, y: midY },
            scale: scale,
            animation: { duration: 300, easingFunction: 'easeInOutQuad' }
        });
    }

    // Hover-to-reveal edge labels in focus mode (for dense neighborhoods)
    network.on('hoverEdge', function(params) {
        if (focusedNodeId === null) return;
        var e = edges.get(params.edge);
        if (e) {
            edges.update({ id: e.id, font: { size: 12, color: '#fff', strokeWidth: 3, strokeColor: '#1a1a2e' }, label: formatRelLabel(e.relation_type) });
        }
    });
    network.on('blurEdge', function(params) {
        if (focusedNodeId === null) return;
        if (focusConnIndex >= 0) return;  // in pair view, don't undo labels
        var e = edges.get(params.edge);
        if (!e) return;
        edges.update({ id: e.id, font: { size: 0 }, label: '' });
    });

    function enterFocusMode(nodeId) {
        if (focusedNodeId === nodeId && focusConnIndex < 0) return;  // already in neighborhood view
        focusedNodeId = nodeId;
        var node = nodes.get(nodeId);
        if (!node) return;

        // Show banner
        var banner = document.getElementById('focus-banner');
        document.getElementById('focus-label').textContent = 'Focused on: ' + (node.full_name || node.label || nodeId);
        banner.classList.add('visible');

        // Find 1-hop neighbors
        var neighborIds = new Set();
        neighborIds.add(nodeId);
        var connectedEdgeIds = new Set();
        allEdges.forEach(function(e) {
            if (e.from === nodeId) { neighborIds.add(e.to); connectedEdgeIds.add(e.id); }
            if (e.to === nodeId) { neighborIds.add(e.from); connectedEdgeIds.add(e.id); }
        });

        // Hide non-neighbor nodes, show neighbors (respecting min degree filter)
        // Also reset font from pair-view back to default
        var nodeUpdates = [];
        allNodes.forEach(function(n) {
            var isNeighbor = neighborIds.has(n.id);
            var isFocused = n.id === nodeId;
            var belowDegree = (n.node_degree || 0) < minDegree;
            nodeUpdates.push({ id: n.id, hidden: !isNeighbor || (!isFocused && belowDegree), font: { size: 14, color: '#e0e0e0' } });
        });
        nodes.update(nodeUpdates);

        // Count parallel edges per node pair to offset their curves
        var pairCount = {};
        var pairIndex = {};
        allEdges.forEach(function(e) {
            if (!connectedEdgeIds.has(e.id)) return;
            var key = [e.from, e.to].sort().join('||');
            if (!pairCount[key]) pairCount[key] = 0;
            pairIndex[e.id] = pairCount[key];
            pairCount[key]++;
        });

        // Show static labels only when neighborhood is small enough to read
        var showStaticLabels = neighborIds.size <= 20;

        // Show connected edges, offset parallel edges so labels don't overlap
        var edgeUpdates = [];
        allEdges.forEach(function(e) {
            if (connectedEdgeIds.has(e.id)) {
                var key = [e.from, e.to].sort().join('||');
                var total = pairCount[key] || 1;
                var idx = pairIndex[e.id] || 0;
                var roundness = total > 1 ? 0.15 + idx * 0.2 : 0.1;
                edgeUpdates.push({
                    id: e.id,
                    hidden: hiddenRelationTypes.has(e.relation_type),
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

        // Fit camera to visible neighborhood
        var visibleIds = [];
        neighborIds.forEach(function(nid) { visibleIds.push(nid); });
        network.fit({ nodes: visibleIds, animation: { duration: 400, easingFunction: 'easeInOutQuad' } });

        // After fit, show labels only on top neighbors by degree to avoid overlap
        setTimeout(function() {
            var scale = network.getScale();
            var fontSize = Math.round(14 / scale);
            var strokeW = Math.max(2, Math.round(2 / scale));

            // Rank neighbors by degree, label top 15 + the focused node
            var ranked = [];
            allNodes.forEach(function(n) {
                if (neighborIds.has(n.id) && n.id !== nodeId) {
                    ranked.push({ id: n.id, deg: n.node_degree || 0 });
                }
            });
            ranked.sort(function(a, b) { return b.deg - a.deg; });
            var maxLabeled = 15;
            var labeledSet = new Set();
            labeledSet.add(nodeId);
            for (var i = 0; i < Math.min(maxLabeled, ranked.length); i++) {
                labeledSet.add(ranked[i].id);
            }

            var updates = [];
            allNodes.forEach(function(n) {
                if (neighborIds.has(n.id)) {
                    var show = labeledSet.has(n.id);
                    updates.push({ id: n.id, font: { size: show ? fontSize : 0, color: '#e0e0e0', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
                }
            });
            nodes.update(updates);
        }, 450);

        showNodeDetail(nodeId);
    }

    function exitFocusMode() {
        if (focusedNodeId === null) return;
        focusedNodeId = null;
        focusConnIndex = -1;
        focusHistory = [];

        // Hide banner
        document.getElementById('focus-banner').classList.remove('visible');

        // Restore all nodes per current filters
        applyFilters();

        // Restore edges — remove labels, re-apply edge filters
        var edgeUpdates = [];
        allEdges.forEach(function(e) {
            edgeUpdates.push({
                id: e.id,
                hidden: hiddenRelationTypes.has(e.relation_type),
                font: { size: 0 },
                label: ''
            });
        });
        edges.update(edgeUpdates);
    }
    </script>
    """

    html = html_path.read_text()
    html = html.replace("</body>", f"{controls_html}{script}</body>")
    html_path.write_text(html)
