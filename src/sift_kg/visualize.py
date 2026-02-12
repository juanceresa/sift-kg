"""Interactive knowledge graph visualization using pyvis.

Generates a standalone HTML file with a force-directed graph layout.
Entities colored by type, edges colored by relation type, with
interactive color pickers, type toggles, search, and detail sidebar.
"""

import logging
import webbrowser
from pathlib import Path

from sift_kg.graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Color palette for entity types — visually distinct, dark-mode friendly
ENTITY_COLORS = {
    "PERSON": "#4FC3F7",
    "ORGANIZATION": "#81C784",
    "SHELL_COMPANY": "#FFB74D",
    "FINANCIAL_INSTRUMENT": "#F06292",
    "FINANCIAL_ACCOUNT": "#F06292",
    "GOVERNMENT_AGENCY": "#CE93D8",
    "LOCATION": "#BA68C8",
    "DOCUMENT": "#90A4AE",
    "EVENT": "#FFD54F",
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


def _color_for_relation(rel_type: str, rel_color_map: dict[str, str]) -> str:
    """Get or assign a color for a relation type."""
    if rel_type in rel_color_map:
        return rel_color_map[rel_type]
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
        import json

        entity_descriptions = json.loads(descriptions_path.read_text())
        logger.info(f"Loaded {len(entity_descriptions)} entity descriptions for viewer")
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
                "gravitationalConstant": -150,
                "centralGravity": 0.005,
                "springLength": 250,
                "springConstant": 0.04,
                "damping": 0.4,
                "avoidOverlap": 0.8
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
            "tooltipDelay": 50,
            "zoomView": true,
            "dragView": true,
            "hideEdgesOnDrag": true,
            "hideEdgesOnZoom": true
        }
    }""")

    degrees = dict(kg.graph.degree())

    # Collect entity types
    entity_types_present: set[str] = set()

    # Add nodes
    for node_id, data in kg.graph.nodes(data=True):
        entity_type = data.get("entity_type", "UNKNOWN")
        entity_types_present.add(entity_type)
        name = data.get("name", node_id)
        confidence = data.get("confidence", 0)
        color = ENTITY_COLORS.get(entity_type, DEFAULT_ENTITY_COLOR)
        degree = degrees.get(node_id, 0)

        tooltip_parts = [
            name,
            f"Type: {entity_type}",
            f"Connections: {degree}",
        ]
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

        net.add_node(
            node_id,
            label=name,
            title=tooltip,
            color=color,
            size=size,
            shape="dot",
            entity_type=entity_type,
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
    _inject_ui(output_path, kg, entity_types_present, rel_color_map)

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
) -> None:
    """Inject sidebar with search, entity/relation type toggles + color pickers + detail panel."""

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

    # Relation type controls
    relation_items = ""
    for rt in sorted(rel_color_map.keys()):
        color = rel_color_map[rt]
        relation_items += (
            f'<div class="type-row">'
            f'<input type="checkbox" checked data-rtype="{rt}" onchange="toggleRelationType(this)">'
            f'<input type="color" value="{color}" data-rtype-color="{rt}" onchange="changeRelationColor(this)">'
            f'<span class="type-label">{rt}</span>'
            f"</div>"
        )

    controls_html = f"""
    <style>
        #sift-controls {{
            position:fixed; top:12px; left:12px; z-index:999;
            background:rgba(26,26,46,0.95); border:1px solid #333;
            border-radius:10px; padding:14px 16px;
            font-family:Inter,system-ui,sans-serif; font-size:13px;
            color:#e0e0e0; width:210px;
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
        .d-evidence {{
            font-size:12px; color:#ccc; line-height:1.6;
            background:rgba(255,255,255,0.03); border-radius:6px;
            padding:10px 12px; border-left:3px solid #4FC3F7;
            white-space:pre-wrap;
        }}
    </style>
    <div id="sift-controls">
        <div style="font-weight:700;margin-bottom:10px;font-size:15px;color:#fff">sift-kg</div>
        <input id="search-input" type="text" placeholder="Search entities..." oninput="searchEntity(this.value)">

        <div class="section-header" style="border-top:none">Entity Types</div>
        {entity_items}

        <div class="section-header">Relation Types</div>
        {relation_items}

        <div style="margin-top:12px;font-size:11px;color:#555;border-top:1px solid #333;padding-top:8px">
            {kg.entity_count} entities &middot; {kg.relation_count} relations
        </div>
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
                updates.push({ id: n.id, color: color });
            }
        });
        nodes.update(updates);
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
            updates.push({ id: node.id, hidden: hiddenEntityTypes.has(node.entity_type) });
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

    // --- Search ---
    function searchEntity(query) {
        if (!query || query.length < 2) {
            var reset = allNodes.map(function(n) {
                return { id: n.id, opacity: 1.0, font: { size: 14 }, borderWidth: 1.5 };
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
                return { id: n.id, opacity: 0.8, font: { size: 12 }, borderWidth: 1.5 };
            else
                return { id: n.id, opacity: 0.1, font: { size: 0 }, borderWidth: 1 };
        });
        nodes.update(updates);
        if (matchIds.size > 0) {
            network.focus(matchIds.values().next().value, { scale: 1.5, animation: { duration: 500 } });
        }
    }
    </script>
    """

    html = html_path.read_text()
    html = html.replace("</body>", f"{controls_html}{script}</body>")
    html_path.write_text(html)
