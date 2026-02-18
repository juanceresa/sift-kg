var allNodes = nodes.get();
var allEdges = edges.get();

// --- Community region data ---
var communityHulls = {};   // comm -> [{x,y}, ...] padded convex hull points
var communityCentroids = {}; // comm -> {x, y}
var communityColors = SIFT_CONFIG.communityColors;

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
var detailBackBtn = document.getElementById('detail-back');
var detailLastNodeId = null;

function esc(s) {
    var d = document.createElement('div');
    d.textContent = String(s);
    return d.innerHTML;
}

// --- Trail breadcrumb helpers ---
function findRelationBetween(fromId, toId) {
    var outgoing = [], incoming = [];
    allEdges.forEach(function(e) {
        if (e.from === fromId && e.to === toId) {
            if (outgoing.indexOf(e.relation_type) < 0) outgoing.push(e.relation_type);
        } else if (e.to === fromId && e.from === toId) {
            if (incoming.indexOf(e.relation_type) < 0) incoming.push(e.relation_type);
        }
    });
    // Prefer outgoing; if only incoming, mark as incoming
    if (outgoing.length > 0) return { rel: outgoing.join(', '), dir: '\u2192' };
    if (incoming.length > 0) return { rel: incoming.join(', '), dir: '\u2190' };
    return { rel: '', dir: '' };
}

function getTrailPath() {
    var path = [];
    for (var i = 0; i < focusHistory.length; i++) {
        var entry = focusHistory[i];
        path.push({ nodeId: entry.nodeId, edgeRel: entry.relationType || '', edgeDir: entry.relationDir || '' });
    }
    path.push({ nodeId: focusedNodeId, edgeRel: null, edgeDir: null });
    return path;
}

function getTrailNodeIds() {
    var ids = new Set();
    for (var i = 0; i < focusHistory.length; i++) {
        ids.add(focusHistory[i].nodeId);
    }
    if (focusedNodeId) ids.add(focusedNodeId);
    return ids;
}

function getTrailEdgeIds() {
    var edgeIds = new Set();
    var trail = getTrailPath();
    for (var i = 0; i < trail.length - 1; i++) {
        var fromId = trail[i].nodeId;
        var toId = trail[i + 1].nodeId;
        allEdges.forEach(function(e) {
            if ((e.from === fromId && e.to === toId) || (e.to === fromId && e.from === toId)) {
                edgeIds.add(e.id);
            }
        });
    }
    return edgeIds;
}

// Map trail edge IDs to the traversal "from" node for correct label perspective
function getTrailEdgePerspectives() {
    var map = {};
    var trail = getTrailPath();
    for (var i = 0; i < trail.length - 1; i++) {
        var fromId = trail[i].nodeId;
        var toId = trail[i + 1].nodeId;
        allEdges.forEach(function(e) {
            if ((e.from === fromId && e.to === toId) || (e.to === fromId && e.from === toId)) {
                map[e.id] = fromId;
            }
        });
    }
    return map;
}

function buildNodeDetailHtml(nodeId) {
    var node = nodes.get(nodeId);
    if (!node) return '';

    var connMap = {};
    allEdges.forEach(function(e) {
        var dir, nid, name;
        if (e.from === nodeId) { dir = '\u2192'; nid = e.to; name = e.target_name || e.to; }
        else if (e.to === nodeId) { dir = '\u2190'; nid = e.from; name = e.source_name || e.from; }
        else return;
        var key = dir + '|' + nid;
        if (!connMap[key]) connMap[key] = { rels: [], edges: [], name: name, dir: dir, nid: nid, maxSupport: 0 };
        if (connMap[key].rels.indexOf(e.relation_type) < 0) connMap[key].rels.push(e.relation_type);
        connMap[key].maxSupport = Math.max(connMap[key].maxSupport, e.support_count || 1);
        connMap[key].edges.push({
            relation_type: e.relation_type,
            confidence: e.edge_confidence || 0,
            support_count: e.support_count || 1,
            support_doc_count: e.support_doc_count || 0,
            evidence: e.full_evidence || ''
        });
    });
    var conns = [];
    for (var k in connMap) {
        var g = connMap[k];
        g.rels.sort();
        g.rel = g.rels.join(', ');
        conns.push(g);
    }
    // Sort: outgoing first, then incoming; within each group, by rel then name
    conns.sort(function(a, b) {
        var aOut = a.dir === '\u2192' ? 0 : 1;
        var bOut = b.dir === '\u2192' ? 0 : 1;
        if (aOut !== bOut) return aOut - bOut;
        return a.rel.localeCompare(b.rel) || a.name.localeCompare(b.name);
    });

    var h = '';
    var tc = node.color || '#B0BEC5';
    if (typeof tc === 'object') tc = tc.background || '#B0BEC5';
    h += '<div class="d-field"><div class="d-label">Type</div>';
    h += '<span class="d-badge" style="background:' + esc(tc) + ';color:#1a1a2e">' + esc(node.entity_type || 'UNKNOWN') + '</span></div>';

    if (node.community) {
        h += '<div class="d-field"><div class="d-label">Community</div>';
        h += '<div class="d-val">' + esc(node.community) + '</div></div>';
    }

    if (node.description) {
        h += '<div class="d-field"><div class="d-label">Description</div>';
        h += '<div class="d-evidence">' + esc(node.description) + '</div></div>';
    }

    var lines = (node.title || '').split('\n');
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

    var relTypes = [];
    var seenRel = {};
    for (var r = 0; r < conns.length; r++) {
        for (var ri = 0; ri < conns[r].rels.length; ri++) {
            var rt = conns[r].rels[ri];
            if (!seenRel[rt]) { relTypes.push(rt); seenRel[rt] = true; }
        }
    }
    relTypes.sort();

    h += '<div class="d-field" style="margin-top:14px">';
    h += '<div class="d-label">Connections (' + conns.length + ')</div>';
    if (relTypes.length > 1) {
        h += '<select class="trail-conn-filter" onchange="filterTrailConns(this, \'' + esc(nodeId) + '\')" style="width:100%;padding:4px 6px;margin:4px 0 6px;border-radius:4px;border:1px solid #444;background:#16213e;color:#e0e0e0;font-size:11px;outline:none">';
        h += '<option value="">All types</option>';
        for (var t = 0; t < relTypes.length; t++) {
            var cnt = conns.filter(function(c){ return c.rels.indexOf(relTypes[t]) >= 0; }).length;
            h += '<option value="' + esc(relTypes[t]) + '">' + esc(relTypes[t]) + ' (' + cnt + ')</option>';
        }
        h += '</select>';
    }
    h += '<div class="trail-conn-list" data-node-id="' + esc(nodeId) + '">';
    h += buildConnListHtml(conns, '');
    h += '</div></div>';

    if (!window._trailConns) window._trailConns = {};
    window._trailConns[nodeId] = conns;

    return h;
}

function buildConnRowHtml(c) {
    var isIncoming = c.dir === '\u2190';
    var relLabel = c.rels.map(function(r){ return isIncoming ? inverseRelLabel(r) : formatRelLabel(r); }).join(', ');
    var h = '';
    h += '<div class="d-conn" data-nid="' + esc(c.nid) + '">';
    h += '<div class="d-conn-header">';
    h += '<div class="d-conn-summary">';
    h += '<span style="color:#888;font-size:10px">' + esc(relLabel).toUpperCase() + '</span><br>';
    h += '<span>' + esc(c.name) + '</span>';
    h += '</div>';
    h += '<button class="d-conn-nav" data-nid="' + esc(c.nid) + '" title="Go to ' + esc(c.name) + '">\u2192</button>';
    h += '</div>';
    h += '<div class="d-conn-detail">';
    var edges = c.edges || [];
    for (var ei = 0; ei < edges.length; ei++) {
        var ed = edges[ei];
        h += '<div class="d-edge-card">';
        h += '<div class="d-label">' + esc(formatRelLabel(ed.relation_type)).toUpperCase() + '</div>';
        var parts = [];
        if (ed.confidence) parts.push(Math.round(ed.confidence * 100) + '% confidence');
        parts.push(ed.support_count + (ed.support_count === 1 ? ' mention' : ' mentions'));
        if (ed.support_doc_count > 1) parts.push(ed.support_doc_count + ' docs');
        h += '<div class="d-val" style="color:#999">' + esc(parts.join(' \u00b7 ')) + '</div>';
        if (ed.evidence) {
            h += '<div class="d-edge-evidence">' + esc(ed.evidence) + '</div>';
        }
        h += '</div>';
    }
    h += '</div>';
    h += '</div>';
    return h;
}

function buildConnListHtml(conns, relFilter) {
    var outgoing = [];
    var incoming = [];
    for (var j = 0; j < conns.length; j++) {
        var c = conns[j];
        if (relFilter && c.rels.indexOf(relFilter) < 0) continue;
        if (c.dir === '\u2190') { incoming.push(c); } else { outgoing.push(c); }
    }
    var h = '';
    if (outgoing.length > 0) {
        h += '<div class="d-conn-group-header">Outgoing</div>';
        for (var o = 0; o < outgoing.length; o++) h += buildConnRowHtml(outgoing[o]);
    }
    if (incoming.length > 0) {
        h += '<div class="d-conn-group-header">Incoming</div>';
        for (var i = 0; i < incoming.length; i++) h += buildConnRowHtml(incoming[i]);
    }
    return h;
}

function filterTrailConns(select, nodeId) {
    var list = document.querySelector('.trail-conn-list[data-node-id="' + nodeId + '"]');
    if (!list) return;
    var conns = (window._trailConns || {})[nodeId] || [];
    list.innerHTML = buildConnListHtml(conns, select.value);
}

function renderTrail() {
    var trail = getTrailPath();
    if (trail.length <= 1) {
        showNodeDetail(focusedNodeId);
        return;
    }

    window._trailConns = {};
    dt.textContent = 'Trail';
    detailBackBtn.style.display = 'none';
    dp.classList.add('open');

    var h = '<div class="trail-header">Trail (' + trail.length + ' nodes)</div>';

    for (var i = 0; i < trail.length; i++) {
        var step = trail[i];
        var node = nodes.get(step.nodeId);
        if (!node) continue;
        var isCurrent = (i === trail.length - 1);
        var isExpanded = isCurrent;

        var tc = node.color || '#B0BEC5';
        if (typeof tc === 'object') tc = tc.background || '#B0BEC5';

        h += '<div class="trail-card' + (isCurrent ? ' current' : '') + (isExpanded ? ' expanded' : '') + '" data-trail-idx="' + i + '" data-trail-nid="' + esc(step.nodeId) + '">';
        h += '<div class="trail-card-header" onclick="toggleTrailCard(' + i + ')">';
        h += '<span class="d-badge" style="background:' + esc(tc) + ';color:#1a1a2e">' + esc(node.entity_type || '?') + '</span>';
        h += '<span>' + esc(node.full_name || node.label || step.nodeId) + '</span>';
        h += '<span class="trail-expand-icon">\u25B8</span>';
        h += '</div>';
        h += '<div class="trail-card-body">';
        h += buildNodeDetailHtml(step.nodeId);
        h += '</div>';
        h += '</div>';

        if (step.edgeRel) {
            var isIncomingTrail = step.edgeDir === '\u2190';
            var trailRelLabel = step.edgeRel.split(', ').map(function(r) {
                return isIncomingTrail ? inverseRelLabel(r.trim()) : formatRelLabel(r.trim());
            }).join(', ');
            h += '<div class="trail-rel-label">\u2193 ' + esc(trailRelLabel).toUpperCase() + '</div>';
        }
    }

    db.innerHTML = h;

    var currentNodeId = trail[trail.length - 1].nodeId;
    window._conns = (window._trailConns || {})[currentNodeId] || [];
}

function toggleTrailCard(idx) {
    var cards = document.querySelectorAll('.trail-card');
    cards.forEach(function(card, i) {
        if (i === idx) {
            card.classList.toggle('expanded');
        } else {
            card.classList.remove('expanded');
        }
    });
}

function closeDetail() {
    dp.classList.remove('open');
    detailLastNodeId = null;
    if (focusedNodeId !== null) exitFocusMode();
}

function detailGoBack() {
    if (detailLastNodeId) showNodeDetail(detailLastNodeId);
}

function focusNode(nid) {
    if (focusedNodeId !== null) {
        var info = findRelationBetween(focusedNodeId, nid);
        focusHistory.push({ nodeId: focusedNodeId, connIndex: focusConnIndex, relationType: info.rel, relationDir: info.dir, targetId: nid });
        enterFocusMode(nid);
        return;
    }
    network.focus(nid, { scale: 1.5, animation: { duration: 400 } });
    network.selectNodes([nid]);
    showNodeDetail(nid);
}

// event delegation for connection rows
db.addEventListener('click', function(ev) {
    // Navigate button → go to neighbor
    var navBtn = ev.target.closest('.d-conn-nav');
    if (navBtn) {
        var nid = navBtn.getAttribute('data-nid');
        if (nid) focusNode(nid);
        return;
    }
    // Clickable node names in edge detail (From/To)
    var nodeLink = ev.target.closest('[data-nid]:not(.d-conn-nav):not(.d-conn)');
    if (nodeLink) {
        focusNode(nodeLink.getAttribute('data-nid'));
        return;
    }
    // Connection header → toggle expand
    var header = ev.target.closest('.d-conn-header');
    if (header) {
        var conn = header.closest('.d-conn');
        if (conn) conn.classList.toggle('expanded');
        return;
    }
});

network.on('click', function(params) {
    if (focusedNodeId !== null && params.nodes && params.nodes.length > 0) {
        // In focus mode: click neighbor to shift focus
        var clickedNid = params.nodes[0];
        var clickInfo = findRelationBetween(focusedNodeId, clickedNid);
        focusHistory.push({ nodeId: focusedNodeId, connIndex: focusConnIndex, relationType: clickInfo.rel, relationDir: clickInfo.dir, targetId: clickedNid });
        enterFocusMode(clickedNid);
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
    detailLastNodeId = nodeId;
    detailBackBtn.style.display = 'none';
    dt.textContent = node.full_name || node.label || nodeId;
    dp.classList.add('open');

    // gather connections — group by neighbor, merge relation types, store edge data
    var connMap = {};
    allEdges.forEach(function(e) {
        var dir, nid, name;
        if (e.from === nodeId) { dir = '\u2192'; nid = e.to; name = e.target_name || e.to; }
        else if (e.to === nodeId) { dir = '\u2190'; nid = e.from; name = e.source_name || e.from; }
        else return;
        var key = dir + '|' + nid;
        if (!connMap[key]) connMap[key] = { rels: [], edges: [], name: name, dir: dir, nid: nid, maxSupport: 0 };
        if (connMap[key].rels.indexOf(e.relation_type) < 0) connMap[key].rels.push(e.relation_type);
        connMap[key].maxSupport = Math.max(connMap[key].maxSupport, e.support_count || 1);
        connMap[key].edges.push({
            relation_type: e.relation_type,
            confidence: e.edge_confidence || 0,
            support_count: e.support_count || 1,
            support_doc_count: e.support_doc_count || 0,
            evidence: e.full_evidence || ''
        });
    });
    var conns = [];
    for (var k in connMap) {
        var g = connMap[k];
        g.rels.sort();
        g.rel = g.rels.join(', ');
        conns.push(g);
    }
    // Sort: outgoing first, then incoming; within each group, by rel then name
    conns.sort(function(a, b) {
        var aOut = a.dir === '\u2192' ? 0 : 1;
        var bOut = b.dir === '\u2192' ? 0 : 1;
        if (aOut !== bOut) return aOut - bOut;
        return a.rel.localeCompare(b.rel) || a.name.localeCompare(b.name);
    });
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
    var lines = (node.title || '').split('\n');
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
    list.innerHTML = buildConnListHtml(conns, relFilter);
}

function showEdgeDetail(edgeId) {
    var edge = edges.get(edgeId);
    if (!edge) return;
    var rt = edge.relation_type || 'UNKNOWN';
    detailBackBtn.style.display = detailLastNodeId ? '' : 'none';
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
var minDegree = SIFT_CONFIG.defaultDegree;
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

// Direction-aware edge label relative to a perspective node
function edgeLabelFrom(e, perspectiveNodeId) {
    if (e.to === perspectiveNodeId && e.from !== perspectiveNodeId) {
        return inverseRelLabel(e.relation_type);
    }
    return formatRelLabel(e.relation_type);
}

// Convert relation type to passive/inverse form for incoming edges
// e.g. "CONTRADICTS" → "CONTRADICTED BY", "GOVERNS" → "GOVERNED BY"
function inverseRelLabel(rt) {
    var inv = {
        'CONTRADICTS': 'CONTRADICTED BY',
        'GOVERNS': 'GOVERNED BY',
        'INVOLVES': 'INVOLVED IN',
        'RESPONSIBLE_FOR': 'UNDER RESPONSIBILITY OF',
        'ENABLED_BY': 'ENABLES',
        'ASSOCIATED_WITH': 'ASSOCIATED WITH'
    };
    var upper = rt.toUpperCase().replace(/\s+/g, '_');
    if (inv[upper]) return formatRelLabel(inv[upper]);
    // Fallback: try common English verb patterns
    var lower = rt.toLowerCase().replace(/_/g, ' ');
    if (lower.endsWith('s') && !lower.endsWith('ss')) {
        // "contradicts" → "contradicted by" (rough heuristic)
        return lower.replace(/s$/, 'd by');
    }
    return lower + ' (inverse)';
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
    var connList = document.querySelector('.trail-card.current .trail-conn-list') || document.getElementById('conn-list');
    var rows = connList ? connList.querySelectorAll('.d-conn') : [];
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
            var targetNid = rows[focusConnIndex].getAttribute('data-nid');
            var currentConns = focusHistory.length > 0
                ? ((window._trailConns || {})[focusedNodeId] || [])
                : (window._conns || []);
            var relType = '';
            var relDir = '';
            if (currentConns[focusConnIndex]) {
                relType = currentConns[focusConnIndex].rels.join(', ');
                relDir = currentConns[focusConnIndex].dir || '';
            }
            focusHistory.push({ nodeId: focusedNodeId, connIndex: focusConnIndex, relationType: relType, relationDir: relDir, targetId: targetNid });
            enterFocusMode(targetNid);
        }
    } else if (ev.key === ' ') {
        ev.preventDefault();
        if (focusConnIndex >= 0 && focusConnIndex < rows.length) {
            rows[focusConnIndex].classList.toggle('expanded');
            var detail = rows[focusConnIndex].querySelector('.d-conn-detail');
            if (detail && rows[focusConnIndex].classList.contains('expanded')) {
                setTimeout(function() { detail.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }, 10);
            }
        }
    } else if (ev.key === 'Backspace' || ev.key === 'Delete' || ev.key === 'ArrowLeft') {
        ev.preventDefault();
        if (focusHistory.length > 0) {
            var prev = focusHistory.pop();
            focusedNodeId = null;
            enterFocusMode(prev.nodeId, prev.connIndex >= 0);
            if (prev.connIndex >= 0) {
                focusConnIndex = prev.connIndex;
                highlightConn(prev.connIndex);
            }
        }
    }
});

function highlightConn(idx) {
    var connList = document.querySelector('.trail-card.current .trail-conn-list') || document.getElementById('conn-list');
    var rows = connList ? connList.querySelectorAll('.d-conn') : [];
    rows.forEach(function(r, i) {
        r.classList.toggle('active', i === idx);
        if (i !== idx) r.classList.remove('expanded');
    });
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

    // Collect focused node's 1-hop neighbors so they stay faintly visible
    var focusNeighborIds = new Set();
    allEdges.forEach(function(e) {
        if (e.from === focusedNodeId && e.to !== c.nid) focusNeighborIds.add(e.to);
        if (e.to === focusedNodeId && e.from !== c.nid) focusNeighborIds.add(e.from);
    });

    // Nodes: primary pair full, trail visible, adjacents ghosted, focus neighbors faint
    var trailNids = getTrailNodeIds();
    var nodeUpdates = [];
    allNodes.forEach(function(n) {
        var isPrimary = n.id === focusedNodeId || n.id === c.nid;
        var isAdj = adjIds.has(n.id);
        var isTrail = trailNids.has(n.id);
        var isFocusNeighbor = focusNeighborIds.has(n.id);
        var visible = isPrimary || isAdj || isTrail || isFocusNeighbor;
        var opacity = isPrimary || isTrail ? 1.0 : isAdj ? 0.35 : isFocusNeighbor ? 0.15 : 1.0;
        nodeUpdates.push({ id: n.id, hidden: !visible, opacity: opacity });
    });
    nodes.update(nodeUpdates);

    // Count parallel edges between primary pair for curve offsets
    var primaryCount = 0;
    var primaryIndex = {};
    allEdges.forEach(function(e) {
        var isPrimary = (e.from === focusedNodeId && e.to === c.nid) || (e.to === focusedNodeId && e.from === c.nid);
        if (isPrimary) {
            primaryIndex[e.id] = primaryCount;
            primaryCount++;
        }
    });

    // Edges: primary thick + labeled, trail visible + labeled, adjacent thin, focus-neighbor faint
    var trailEids = getTrailEdgeIds();
    var trailPersp = getTrailEdgePerspectives();
    var edgeUpdates = [];
    allEdges.forEach(function(e) {
        var isPrimary = (e.from === focusedNodeId && e.to === c.nid) || (e.to === focusedNodeId && e.from === c.nid);
        var isAdj = (e.from === c.nid && adjIds.has(e.to)) || (e.to === c.nid && adjIds.has(e.from));
        var isTrail = trailEids.has(e.id);
        var isFocusEdge = (e.from === focusedNodeId && focusNeighborIds.has(e.to)) || (e.to === focusedNodeId && focusNeighborIds.has(e.from));
        var show = (isPrimary || isAdj || isTrail || isFocusEdge) && !hiddenRelationTypes.has(e.relation_type);
        var origWidth = Math.min(10, 1.5 + (e.support_count || 1) * 2);
        var origColor = (typeof e.color === 'string') ? e.color : (e.color && e.color.color) || '#888';
        var edgeOpacity = isPrimary || isTrail ? 1.0 : isAdj ? 0.25 : isFocusEdge ? 0.1 : 0.25;
        var edgePerspective = isTrail ? (trailPersp[e.id] || focusedNodeId) : focusedNodeId;
        var update = {
            id: e.id,
            hidden: !show,
            color: { color: origColor, opacity: edgeOpacity },
            label: (isPrimary || isTrail) ? edgeLabelFrom(e, edgePerspective) : '',
            width: isPrimary ? Math.max(3, origWidth) : (isTrail ? origWidth : 1)
        };
        if (isPrimary && primaryCount > 1) {
            var roundness = 0.15 + (primaryIndex[e.id] || 0) * 0.2;
            update.smooth = { type: 'curvedCW', roundness: roundness };
        }
        edgeUpdates.push(update);
    });
    edges.update(edgeUpdates);

    // Camera: fit trail + primary pair, adjacents are peripheral
    var fitNodeIds = [focusedNodeId, c.nid];
    trailNids.forEach(function(nid) { if (fitNodeIds.indexOf(nid) < 0) fitNodeIds.push(nid); });
    var positions = network.getPositions(fitNodeIds);
    var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    fitNodeIds.forEach(function(nid) {
        var p = positions[nid];
        if (!p) return;
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    });
    var midX = (minX + maxX) / 2;
    var midY = (minY + maxY) / 2;

    var pad = 250;
    var rangeX = (maxX - minX) + pad * 2;
    var rangeY = (maxY - minY) + pad * 2;

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

    // Fonts: primary pair large + white, trail nodes medium + gray, adjacents small + dim
    (function() {
        var nodeFontSize = Math.round(20 / scale);
        var adjFontSize = Math.round(9 / scale);
        var ghostFontSize = Math.round(6 / scale);
        var edgeFontSize = Math.round(16 / scale);
        var strokeW = Math.max(3, Math.round(3 / scale));

        var fontUpdatesN = [];
        allNodes.forEach(function(n) {
            if (n.id === focusedNodeId || n.id === c.nid) {
                fontUpdatesN.push({ id: n.id, font: { size: nodeFontSize, color: '#ffffff', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
            } else if (trailNids.has(n.id)) {
                fontUpdatesN.push({ id: n.id, font: { size: adjFontSize, color: '#aaaaaa', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
            } else if (adjIds.has(n.id)) {
                fontUpdatesN.push({ id: n.id, font: { size: adjFontSize, color: '#555555', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
            } else if (focusNeighborIds.has(n.id)) {
                fontUpdatesN.push({ id: n.id, font: { size: ghostFontSize, color: '#333333', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
            }
        });
        nodes.update(fontUpdatesN);

        var fontUpdatesE = [];
        var smallEdgeFont = Math.round(4 / scale);
        allEdges.forEach(function(e) {
            var isPrimary = (e.from === focusedNodeId && e.to === c.nid) || (e.to === focusedNodeId && e.from === c.nid);
            var isTrail = trailEids.has(e.id);
            if ((isPrimary || isTrail) && !hiddenRelationTypes.has(e.relation_type)) {
                fontUpdatesE.push({ id: e.id, font: { size: edgeFontSize, color: '#ffffff', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
            } else {
                fontUpdatesE.push({ id: e.id, font: { size: smallEdgeFont, color: '#333333', strokeWidth: 0, strokeColor: '#1a1a2e' } });
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
        edges.update({ id: e.id, font: { size: 12, color: '#fff', strokeWidth: 3, strokeColor: '#1a1a2e' }, label: edgeLabelFrom(e, focusedNodeId) });
    }
    // Show support tooltip in all focus sub-modes
    var parts = [edgeLabelFrom(e, focusedNodeId)];
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

function enterFocusMode(nodeId, skipCamera) {
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
    var trailNodeIds = getTrailNodeIds();
    trailNodeIds.forEach(function(nid) { visibleNeighbors.add(nid); });

    // Show top neighbors, hide rest
    var nodeUpdates = [];
    allNodes.forEach(function(n) {
        var isVisible = visibleNeighbors.has(n.id);
        var isFocused = n.id === nodeId;
        var isTrail = trailNodeIds.has(n.id);
        var belowDegree = (n.node_degree || 0) < minDegree;
        nodeUpdates.push({ id: n.id, hidden: !isVisible || (!isFocused && !isTrail && belowDegree), opacity: 1.0, font: { size: 14, color: '#e0e0e0' } });
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

    // Show edges only to visible neighbors + trail edges
    var trailEdgeIds = getTrailEdgeIds();
    var trailEdgePersp = getTrailEdgePerspectives();
    var edgeUpdates = [];
    allEdges.forEach(function(e) {
        var isTrailEdge = trailEdgeIds.has(e.id);
        if (isTrailEdge) {
            edgeUpdates.push({
                id: e.id,
                hidden: false,
                color: { opacity: 0.6 },
                font: { size: 11, color: '#ccc', strokeWidth: 3, strokeColor: '#1a1a2e' },
                label: edgeLabelFrom(e, trailEdgePersp[e.id] || nodeId),
                smooth: { type: 'curvedCW', roundness: 0.1 }
            });
        } else if (connectedEdgeIds.has(e.id)) {
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
                label: showStaticLabels ? edgeLabelFrom(e, nodeId) : '',
                smooth: { type: 'curvedCW', roundness: roundness }
            });
        } else {
            edgeUpdates.push({ id: e.id, hidden: true });
        }
    });
    edges.update(edgeUpdates);

    if (!skipCamera) focusConnIndex = -1;

    if (!skipCamera) {
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
            var trailIds = getTrailNodeIds();
            trailIds.forEach(function(nid) { labeledSet.add(nid); });
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

            // Upscale trail edge labels to match current zoom
            var trailEids = getTrailEdgeIds();
            if (trailEids.size > 0) {
                var trailEdgeFontSize = Math.round(12 / s);
                var edgeUpdatesTrail = [];
                trailEids.forEach(function(eid) {
                    edgeUpdatesTrail.push({ id: eid, font: { size: trailEdgeFontSize, color: '#cccccc', strokeWidth: strokeW, strokeColor: '#1a1a2e' } });
                });
                edges.update(edgeUpdatesTrail);
            }
        }, 450);
    }

    if (focusHistory.length > 0) {
        renderTrail();
    } else {
        showNodeDetail(nodeId);
    }
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
