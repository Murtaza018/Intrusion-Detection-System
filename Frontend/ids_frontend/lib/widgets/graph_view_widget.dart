// lib/widgets/graph_view_widget.dart
import 'dart:collection' show LinkedHashMap;
import 'dart:math' as math;
import 'package:flutter/foundation.dart' show setEquals;
import 'package:flutter/material.dart';
import '../models/graph_data.dart';
import '../models/node.dart';
import '../models/edge.dart';
import '../services/graph_layout_service.dart';

class GraphViewWidget extends StatefulWidget {
  final GraphData graphData;
  final Size size;

  /// Rolling-window cap. When a new node arrives and the window is full,
  /// the oldest node (by first-seen order) is evicted.
  static const int kMaxNodes = 60;

  const GraphViewWidget({
    Key? key,
    required this.graphData,
    required this.size,
  }) : super(key: key);

  @override
  State<GraphViewWidget> createState() => _GraphViewWidgetState();
}

class _GraphViewWidgetState extends State<GraphViewWidget>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  Offset _panOffset = Offset.zero;
  double _scale = 1.0;
  bool _showControls = true;
  GraphNode? _selectedNode;
  Set<int> _highlightedNodes = {};
  late GraphData _layoutGraphData;

  // Insertion-ordered map: node.id → fully positioned GraphNode.
  // LinkedHashMap preserves insertion order so we can evict the oldest key.
  final LinkedHashMap<int, GraphNode> _nodeWindow = LinkedHashMap();
  Set<int> _laidOutNodeIds = {};

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat(reverse: true);

    _layoutGraphData =
        GraphData(nodes: const [], edges: const [], timestamp: 0);
    _applyLayout();
    WidgetsBinding.instance.addPostFrameCallback((_) => _autoFit());
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  void didUpdateWidget(GraphViewWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.graphData != widget.graphData) {
      _applyLayout();
    }
  }

  // ── Normalisation ────────────────────────────────────────────────────────
  double _norm(double raw) => raw.clamp(0.0, 1.0);

  // ── Rolling window ───────────────────────────────────────────────────────
  void _mergeIntoWindow(List<GraphNode> incoming) {
    for (final node in incoming) {
      if (_nodeWindow.containsKey(node.id)) {
        // Update anomaly in-place, keep existing position.
        final existing = _nodeWindow[node.id]!;
        _nodeWindow[node.id] = existing.copyWith(anomaly: _norm(node.anomaly));
      } else {
        // New node: add with raw position (layout will assign x/y later).
        _nodeWindow[node.id] = node.copyWith(anomaly: _norm(node.anomaly));
      }
    }
    // Evict oldest while over cap.
    while (_nodeWindow.length > GraphViewWidget.kMaxNodes) {
      _nodeWindow.remove(_nodeWindow.keys.first);
    }
  }

  // ── Layout ───────────────────────────────────────────────────────────────
  void _applyLayout() {
    _mergeIntoWindow(widget.graphData.nodes);

    final currentIds = _nodeWindow.keys.toSet();
    final topologyChanged = !setEquals(currentIds, _laidOutNodeIds);

    // Keep only edges whose both endpoints are in the window.
    final filteredEdges = widget.graphData.edges
        .where((e) =>
            currentIds.contains(e.source) && currentIds.contains(e.target))
        .toList();

    if (topologyChanged) {
      // Nodes that already have a position keep it; only brand-new nodes
      // start at (0,0) and will be repositioned by the layout pass.
      final nodeList = _nodeWindow.values.toList();
      final allAtOrigin =
          nodeList.every((n) => n.x.abs() < 0.1 && n.y.abs() < 0.1);

      GraphData positioned;
      if (allAtOrigin && nodeList.isNotEmpty) {
        positioned = GraphLayoutService.applyForceDirectedLayout(
          GraphData(
              nodes: nodeList,
              edges: filteredEdges,
              timestamp: widget.graphData.timestamp),
          // Higher repulsion + lower attraction = nodes spread far apart
          // in world-space so zooming in always reveals clear separation.
          iterations: 150,
          repulsionStrength: 8000.0,
          attractionStrength: 0.06,
          initialRadius: 600.0,
          minDistance: 80.0,
        );
      } else {
        positioned = GraphData(
            nodes: nodeList,
            edges: filteredEdges,
            timestamp: widget.graphData.timestamp);
      }

      // Write positions back into the window.
      for (final n in positioned.nodes) {
        _nodeWindow[n.id] = n.copyWith(anomaly: _norm(n.anomaly));
      }

      _laidOutNodeIds = currentIds;
      WidgetsBinding.instance.addPostFrameCallback((_) => _autoFit());
    }

    setState(() {
      _layoutGraphData = GraphData(
        nodes: _nodeWindow.values.toList(),
        edges: filteredEdges,
        timestamp: widget.graphData.timestamp,
      );
    });
  }

  // ── Viewport ─────────────────────────────────────────────────────────────
  void _autoFit() {
    final bounds = _calculateBounds();
    if (bounds.width == 0 || bounds.height == 0) return;
    setState(() {
      final sx = (widget.size.width - 120) / bounds.width;
      final sy = (widget.size.height - 120) / bounds.height;
      _scale = math.min(sx, sy).clamp(0.2, 3.0);
      _panOffset = Offset(
        widget.size.width / 2 - bounds.center.dx * _scale,
        widget.size.height / 2 - bounds.center.dy * _scale,
      );
    });
  }

  Rect _calculateBounds() {
    if (_layoutGraphData.nodes.isEmpty) return Rect.zero;
    double minX = double.infinity, minY = double.infinity;
    double maxX = double.negativeInfinity, maxY = double.negativeInfinity;
    for (final n in _layoutGraphData.nodes) {
      if (!n.x.isFinite || !n.y.isFinite) continue;
      minX = math.min(minX, n.x);
      minY = math.min(minY, n.y);
      maxX = math.max(maxX, n.x);
      maxY = math.max(maxY, n.y);
    }
    if (!minX.isFinite) return Rect.zero;
    const p = 60.0;
    return Rect.fromLTRB(minX - p, minY - p, maxX + p, maxY + p);
  }

  void _zoomBy(double factor) =>
      setState(() => _scale = (_scale * factor).clamp(0.1, 8.0));

  // ── Selection ─────────────────────────────────────────────────────────────
  void _onNodeTap(GraphNode node) {
    setState(() {
      _selectedNode = node;
      _highlightedNodes = {node.id};
      for (final e in _layoutGraphData.edges) {
        if (e.source == node.id) _highlightedNodes.add(e.target);
        if (e.target == node.id) _highlightedNodes.add(e.source);
      }
    });
  }

  void _clearSelection() => setState(() {
        _selectedNode = null;
        _highlightedNodes = {};
      });

  // ── Stats helpers ─────────────────────────────────────────────────────────
  int get _criticalCount =>
      _layoutGraphData.nodes.where((n) => n.anomaly > 0.6).length;
  int get _warningCount => _layoutGraphData.nodes
      .where((n) => n.anomaly > 0.3 && n.anomaly <= 0.6)
      .length;
  double get _avgAnomaly => _layoutGraphData.nodes.isEmpty
      ? 0
      : _layoutGraphData.nodes.fold(0.0, (s, n) => s + n.anomaly) /
          _layoutGraphData.nodes.length;

  // ── Degree map (number of connections per node) ───────────────────────────
  Map<int, int> get _degreeMap {
    final m = <int, int>{};
    for (final e in _layoutGraphData.edges) {
      m[e.source] = (m[e.source] ?? 0) + 1;
      m[e.target] = (m[e.target] ?? 0) + 1;
    }
    return m;
  }

  // ── Build ─────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final degreeMap = _degreeMap;

    return Scaffold(
      backgroundColor: const Color(0xFF080B10),
      body: Stack(children: [
        // ── Dot-grid background
        Positioned.fill(child: CustomPaint(painter: _DotGridPainter())),

        // ── Graph canvas
        Positioned.fill(
          child: GestureDetector(
            onScaleStart: (_) {
              setState(() => _showControls = false);
              _clearSelection();
            },
            onScaleUpdate: (d) => setState(() {
              _scale = (_scale * d.scale).clamp(0.1, 8.0);
              _panOffset += d.focalPointDelta;
            }),
            onScaleEnd: (_) => setState(() => _showControls = true),
            onDoubleTap: _autoFit,
            onTapUp: (d) {
              final tap = d.localPosition;
              GraphNode? hit;
              // Check in reverse so topmost-drawn node is hit first.
              for (final node in _layoutGraphData.nodes.reversed) {
                final c = Offset(
                  node.x * _scale + _panOffset.dx,
                  node.y * _scale + _panOffset.dy,
                );
                final deg = degreeMap[node.id] ?? 0;
                final r = _nodeRadius(node.anomaly, deg) * _scale;
                if ((tap - c).distance <= r + 6) {
                  hit = node;
                  break;
                }
              }
              hit != null ? _onNodeTap(hit) : _clearSelection();
            },
            child: AnimatedBuilder(
              animation: _pulseController,
              builder: (_, __) => CustomPaint(
                size: widget.size,
                painter: _GraphPainter(
                  nodes: _layoutGraphData.nodes,
                  edges: _layoutGraphData.edges,
                  panOffset: _panOffset,
                  scale: _scale,
                  pulseValue: _pulseController.value,
                  selectedNode: _selectedNode,
                  highlightedNodes: _highlightedNodes,
                  degreeMap: degreeMap,
                ),
              ),
            ),
          ),
        ),

        // ── Stats panel (top-right)
        Positioned(top: 16, right: 16, child: _buildStatsPanel()),

        // ── Node detail panel (top-left, when selected)
        if (_selectedNode != null)
          Positioned(
              top: 16,
              left: 16,
              child: _buildDetailPanel(_selectedNode!, degreeMap)),

        // ── Toolbar (bottom-left)
        if (_showControls)
          Positioned(bottom: 20, left: 20, child: _buildToolbar()),

        // ── Legend (bottom-right)
        Positioned(bottom: 20, right: 20, child: _buildLegend()),
      ]),
    );
  }

  // ── Panel builders ────────────────────────────────────────────────────────

  Widget _buildStatsPanel() {
    return _Panel(
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          _PulseDot(color: Colors.cyanAccent),
          const SizedBox(width: 8),
          const Text('NETWORK STATUS',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.6)),
        ]),
        const SizedBox(height: 12),
        const Divider(color: Colors.white12, height: 1),
        const SizedBox(height: 12),
        _StatRow2(
            'Visible Nodes',
            '${_layoutGraphData.nodes.length} / ${GraphViewWidget.kMaxNodes}',
            Colors.cyanAccent),
        _StatRow2('Connections', '${_layoutGraphData.edges.length}',
            Colors.blueAccent),
        const SizedBox(height: 6),
        _StatRow2('🔥 Critical', '$_criticalCount', Colors.redAccent),
        _StatRow2('⚠️ Warnings', '$_warningCount', Colors.orange),
        const SizedBox(height: 6),
        _StatRow2('Avg Anomaly', '${(_avgAnomaly * 100).toStringAsFixed(1)}%',
            _anomalyColor(_avgAnomaly)),
      ]),
    );
  }

  // Widget _buildDetailPanel(GraphNode node, Map<int, int> degreeMap) {
  //   final conns = degreeMap[node.id] ?? 0;
  //   final severity = _severityLabel(node.anomaly);
  //   final sColor = _anomalyColor(node.anomaly);

  //   return _Panel(
  //     minWidth: 260,
  //     child: Column(
  //         crossAxisAlignment: CrossAxisAlignment.start,
  //         mainAxisSize: MainAxisSize.min,
  //         children: [
  //           // Header
  //           Row(children: [
  //             Container(
  //                 width: 10,
  //                 height: 10,
  //                 decoration:
  //                     BoxDecoration(color: sColor, shape: BoxShape.circle)),
  //             const SizedBox(width: 8),
  //             const Text('NODE DETAILS',
  //                 style: TextStyle(
  //                     color: Colors.white70,
  //                     fontSize: 11,
  //                     fontWeight: FontWeight.bold,
  //                     letterSpacing: 1.3)),
  //             const Spacer(),
  //             GestureDetector(
  //                 onTap: _clearSelection,
  //                 child:
  //                     const Icon(Icons.close, color: Colors.white38, size: 16)),
  //           ]),
  //           const SizedBox(height: 10),
  //           const Divider(color: Colors.white12, height: 1),
  //           const SizedBox(height: 10),
  //           // Fields
  //           _DetailField('IP Address', node.ip,
  //               valueStyle: const TextStyle(
  //                   color: Colors.white,
  //                   fontSize: 13,
  //                   fontWeight: FontWeight.w600,
  //                   fontFamily: 'monospace')),
  //           _DetailField('Node ID', '#${node.id}',
  //               valueStyle:
  //                   const TextStyle(color: Colors.cyanAccent, fontSize: 12)),
  //           _DetailField(
  //             'Anomaly Score',
  //             '${(node.anomaly * 100).toStringAsFixed(2)}%',
  //             valueStyle: TextStyle(
  //                 color: sColor, fontSize: 13, fontWeight: FontWeight.bold),
  //           ),
  //           _DetailField('Connections', '$conns peers',
  //               valueStyle:
  //                   const TextStyle(color: Colors.white70, fontSize: 12)),
  //           const SizedBox(height: 8),
  //           // Severity badge
  //           Container(
  //             padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
  //             decoration: BoxDecoration(
  //               color: sColor.withOpacity(0.15),
  //               borderRadius: BorderRadius.circular(20),
  //               border: Border.all(color: sColor.withOpacity(0.5)),
  //             ),
  //             child: Text(severity,
  //                 style: TextStyle(
  //                     color: sColor,
  //                     fontSize: 11,
  //                     fontWeight: FontWeight.bold,
  //                     letterSpacing: 1.2)),
  //           ),
  //         ]),
  //   );
  // }

  Widget _buildDetailPanel(GraphNode node, Map<int, int> degreeMap) {
    final conns = degreeMap[node.id] ?? 0;
    final severity = _severityLabel(node.anomaly);
    final sColor = _anomalyColor(node.anomaly);

    return _Panel(
      minWidth: 260,
      child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // Header
            Row(children: [
              Container(
                  width: 10,
                  height: 10,
                  decoration:
                      BoxDecoration(color: sColor, shape: BoxShape.circle)),
              const SizedBox(width: 8),
              const Text('NODE DETAILS',
                  style: TextStyle(
                      color: Colors.white70,
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.3)),
              const Spacer(),
              GestureDetector(
                  onTap: _clearSelection,
                  child:
                      const Icon(Icons.close, color: Colors.white38, size: 16)),
            ]),
            const SizedBox(height: 10),
            const Divider(color: Colors.white12, height: 1),
            const SizedBox(height: 10),

            // --- UPDATED FIELDS START HERE ---
            _DetailField('IP Address', node.ip,
                valueStyle: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    fontFamily: 'monospace')),
            _DetailField('Node ID', '#${node.id}',
                valueStyle:
                    const TextStyle(color: Colors.cyanAccent, fontSize: 12)),

            // NEW: Subnet, DMZ, and Gateway Info
            _DetailField('Subnet', node.subnet,
                valueStyle:
                    const TextStyle(color: Colors.white70, fontSize: 12)),

            // 1. Updated Zone to handle External, DMZ, and Internal
            _DetailField(
                'Zone',
                node.isDmz
                    ? 'DMZ 🛡️'
                    : (node.subnet.contains('External')
                        ? 'External 🌍'
                        : 'Internal 🔒'),
                valueStyle: TextStyle(
                    color: node.isDmz ? Colors.orangeAccent : Colors.white70,
                    fontSize: 12,
                    fontWeight:
                        node.isDmz ? FontWeight.bold : FontWeight.normal)),

            // 2. Updated Gateway to always show True or False
            _DetailField('Gateway', node.isGateway ? 'True 🌐' : 'False ❌',
                valueStyle: TextStyle(
                    color: node.isGateway ? Colors.greenAccent : Colors.white54,
                    fontSize: 12,
                    fontWeight:
                        node.isGateway ? FontWeight.bold : FontWeight.normal)),

            _DetailField('Connections', '$conns peers',
                valueStyle:
                    const TextStyle(color: Colors.white70, fontSize: 12)),
            _DetailField(
              'Anomaly Score',
              '${(node.anomaly * 100).toStringAsFixed(2)}%',
              valueStyle: TextStyle(
                  color: sColor, fontSize: 13, fontWeight: FontWeight.bold),
            ),
            // --- UPDATED FIELDS END HERE ---

            const SizedBox(height: 8),
            // Severity badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: sColor.withOpacity(0.15),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: sColor.withOpacity(0.5)),
              ),
              child: Text(severity,
                  style: TextStyle(
                      color: sColor,
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2)),
            ),
          ]),
    );
  }

  Widget _buildToolbar() {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      _ToolBtn(Icons.add, 'Zoom In', () => _zoomBy(1.3)),
      const SizedBox(width: 6),
      _ToolBtn(Icons.remove, 'Zoom Out', () => _zoomBy(1 / 1.3)),
      const SizedBox(width: 6),
      _ToolBtn(Icons.fit_screen, 'Fit', _autoFit),
      const SizedBox(width: 6),
      _ToolBtn(Icons.refresh, 'Reset', () {
        setState(() {
          _scale = 1.0;
          _panOffset = Offset.zero;
        });
        Future.delayed(const Duration(milliseconds: 80), _autoFit);
      }),
    ]);
  }

  Widget _buildLegend() {
    return _Panel(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: const [
            _LegendRow(Colors.green, 'Normal  < 10%'),
            _LegendRow(Colors.yellow, 'Low      10–30%'),
            _LegendRow(Colors.orange, 'Medium  30–60%'),
            _LegendRow(Colors.redAccent, 'Critical  > 60%'),
          ]),
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  static double _nodeRadius(double anomaly, int degree) {
    // Base size grows slightly with connection count (hub nodes are bigger).
    final degBonus = math.log(degree + 1) * 1.5;
    return 5.0 + anomaly * 10.0 + degBonus;
  }

  static Color _anomalyColor(double a) {
    if (a > 0.6) return Colors.redAccent;
    if (a > 0.3) return Colors.orange;
    if (a > 0.1) return Colors.yellow;
    return Colors.greenAccent;
  }

  static String _severityLabel(double a) {
    if (a > 0.6) return 'CRITICAL';
    if (a > 0.3) return 'MEDIUM';
    if (a > 0.1) return 'LOW';
    return 'NORMAL';
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Graph painter
// ═══════════════════════════════════════════════════════════════════════════

class _GraphPainter extends CustomPainter {
  final List<GraphNode> nodes;
  final List<GraphEdge> edges;
  final Offset panOffset;
  final double scale;
  final double pulseValue;
  final GraphNode? selectedNode;
  final Set<int> highlightedNodes;
  final Map<int, int> degreeMap;

  _GraphPainter({
    required this.nodes,
    required this.edges,
    required this.panOffset,
    required this.scale,
    required this.pulseValue,
    required this.highlightedNodes,
    required this.degreeMap,
    this.selectedNode,
  });

  Offset _toScreen(double x, double y) =>
      Offset(x * scale + panOffset.dx, y * scale + panOffset.dy);

  @override
  void paint(Canvas canvas, Size size) {
    if (nodes.isEmpty) return;

    final nodeMap = <int, GraphNode>{for (final n in nodes) n.id: n};
    final hasSelection = highlightedNodes.isNotEmpty;

    // ── Edges ──────────────────────────────────────────────────────────────
    const maxEdges = 150;
    int drawnEdges = 0;

    // Sort: highlighted first, then by weight desc.
    final sorted = List<GraphEdge>.from(edges)
      ..sort((a, b) {
        final aH = highlightedNodes.contains(a.source) &&
            highlightedNodes.contains(a.target);
        final bH = highlightedNodes.contains(b.source) &&
            highlightedNodes.contains(b.target);
        if (aH != bH) return aH ? -1 : 1;
        return b.weight.compareTo(a.weight);
      });

    for (final edge in sorted) {
      final n1 = nodeMap[edge.source];
      final n2 = nodeMap[edge.target];
      if (n1 == null || n2 == null) continue;

      final isHL = highlightedNodes.contains(edge.source) &&
          highlightedNodes.contains(edge.target);

      if (hasSelection && !isHL) continue;
      if (!hasSelection && drawnEdges >= maxEdges) continue;
      if (!hasSelection) drawnEdges++;

      final p1 = _toScreen(n1.x, n1.y);
      final p2 = _toScreen(n2.x, n2.y);
      if (!p1.dx.isFinite || !p2.dx.isFinite) continue;

      // Curved edges: control point offset perpendicular to mid-point.
      final mid = (p1 + p2) / 2;
      final dx = p2.dx - p1.dx;
      final dy = p2.dy - p1.dy;
      final len = math.sqrt(dx * dx + dy * dy);
      final curve = len * 0.18;
      final ctrl = Offset(mid.dx - dy / len * curve, mid.dy + dx / len * curve);

      final path = Path()
        ..moveTo(p1.dx, p1.dy)
        ..quadraticBezierTo(ctrl.dx, ctrl.dy, p2.dx, p2.dy);

      final t = (edge.weight / 5.0).clamp(0.0, 1.0);
      final color = isHL
          ? Colors.cyanAccent.withOpacity(0.9)
          : Color.lerp(Colors.white.withOpacity(0.08),
              Colors.cyanAccent.withOpacity(0.4), t)!;

      canvas.drawPath(
          path,
          Paint()
            ..color = color
            ..strokeWidth = isHL ? 1.8 * scale.clamp(0.5, 2) : 1.0
            ..style = PaintingStyle.stroke
            ..strokeCap = StrokeCap.round);

      // Arrowhead on highlighted edges to show directionality.
      if (isHL) {
        _drawArrow(canvas, ctrl, p2, color);
      }
    }

    // ── Nodes ──────────────────────────────────────────────────────────────
    for (final node in nodes) {
      final c = _toScreen(node.x, node.y);
      if (!c.dx.isFinite || !c.dy.isFinite) continue;

      final isHL = highlightedNodes.contains(node.id);
      final isSel = selectedNode?.id == node.id;
      final dimmed = hasSelection && !isHL;
      final opacity = dimmed ? 0.2 : 1.0;

      final deg = degreeMap[node.id] ?? 0;
      // FIX: radius is in world-space, multiply by scale for screen size.
      // Previously divided by scale which made nodes shrink on zoom-in.
      final baseR = _GraphViewWidgetState._nodeRadius(node.anomaly, deg);
      final r = baseR * scale.clamp(0.3, 4.0);

      final color = _GraphViewWidgetState._anomalyColor(node.anomaly);
      final pulse =
          (node.anomaly > 0.6 && !dimmed) ? 1.0 + pulseValue * 0.18 : 1.0;

      // Glow ring (critical only)
      if (node.anomaly > 0.6 && !dimmed) {
        canvas.drawCircle(
            c,
            r * 2.2 * pulse,
            Paint()
              ..color = Colors.red.withOpacity(0.12 * pulseValue)
              ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 12));
      }

      // Outer anomaly ring
      if (node.anomaly > 0.1 && !dimmed) {
        canvas.drawCircle(
            c,
            r * 1.5 * pulse,
            Paint()
              ..color = color.withOpacity(node.anomaly > 0.6 ? 0.45 : 0.25)
              ..style = PaintingStyle.stroke
              ..strokeWidth = 1.2);
      }

      // Main fill with radial gradient
      final grad = RadialGradient(colors: [
        color.withOpacity(opacity),
        color.withOpacity(opacity * 0.6),
      ]);
      canvas.drawCircle(
          c,
          r * pulse,
          Paint()
            ..shader = grad
                .createShader(Rect.fromCircle(center: c, radius: r * pulse)));

      // Inner specular highlight
      canvas.drawCircle(Offset(c.dx - r * 0.25, c.dy - r * 0.25), r * 0.3,
          Paint()..color = Colors.white.withOpacity(0.25 * opacity));

      // Selection ring
      if (isSel) {
        canvas.drawCircle(
            c,
            r * 1.8,
            Paint()
              ..color = Colors.white.withOpacity(0.9)
              ..style = PaintingStyle.stroke
              ..strokeWidth = 1.5);
        // Second outer ring for emphasis
        canvas.drawCircle(
            c,
            r * 2.1,
            Paint()
              ..color = Colors.cyanAccent.withOpacity(0.4)
              ..style = PaintingStyle.stroke
              ..strokeWidth = 0.8);
      }

      // Connection-count badge (only when not dimmed and has edges)
      if (deg > 0 && !dimmed && scale > 0.5) {
        final badgePos = Offset(c.dx + r * 0.75, c.dy - r * 0.75);
        final badgeR = (6.0 * scale).clamp(4.0, 10.0);
        canvas.drawCircle(
            badgePos, badgeR, Paint()..color = const Color(0xFF0A0E18));
        canvas.drawCircle(
            badgePos,
            badgeR,
            Paint()
              ..color = Colors.cyanAccent.withOpacity(0.7)
              ..style = PaintingStyle.stroke
              ..strokeWidth = 0.8);
        final badge = TextPainter(
          text: TextSpan(
            text: deg > 99 ? '99+' : '$deg',
            style: TextStyle(
                color: Colors.cyanAccent,
                fontSize: (badgeR * 1.1).clamp(6.0, 10.0),
                fontWeight: FontWeight.bold),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        badge.paint(
            canvas, badgePos - Offset(badge.width / 2, badge.height / 2));
      }

      // IP label with background pill
      if (scale > 0.45 || isSel) {
        final text =
            node.ip.length > 15 ? '${node.ip.substring(0, 13)}…' : node.ip;
        final fs = ((isSel ? 10.5 : 9.0) * scale.clamp(0.5, 1.8));
        final tp = TextPainter(
          text: TextSpan(
              text: text,
              style: TextStyle(
                  color: Colors.white.withOpacity(dimmed ? 0.15 : 0.92),
                  fontSize: fs,
                  fontWeight: isSel ? FontWeight.bold : FontWeight.w500,
                  letterSpacing: 0.3,
                  shadows: const [Shadow(blurRadius: 4, color: Colors.black)])),
          textAlign: TextAlign.center,
          textDirection: TextDirection.ltr,
        )..layout(minWidth: 0, maxWidth: 160);

        final labelY = c.dy + r * pulse + 5;
        final lx = c.dx - tp.width / 2;

        // Semi-transparent pill behind label
        if (!dimmed) {
          final pill = RRect.fromRectAndRadius(
              Rect.fromLTWH(lx - 4, labelY - 1, tp.width + 8, tp.height + 2),
              const Radius.circular(4));
          canvas.drawRRect(pill, Paint()..color = const Color(0xCC080B10));
        }
        tp.paint(canvas, Offset(lx, labelY));
      }
    }
  }

  void _drawArrow(Canvas canvas, Offset from, Offset to, Color color) {
    final dx = to.dx - from.dx;
    final dy = to.dy - from.dy;
    final len = math.sqrt(dx * dx + dy * dy);
    if (len < 1) return;
    final ux = dx / len, uy = dy / len;
    final arrowLen = (8 * scale).clamp(4.0, 14.0);
    const spread = 0.4;
    final p1 = Offset(to.dx - ux * arrowLen - uy * arrowLen * spread,
        to.dy - uy * arrowLen + ux * arrowLen * spread);
    final p2 = Offset(to.dx - ux * arrowLen + uy * arrowLen * spread,
        to.dy - uy * arrowLen - ux * arrowLen * spread);
    canvas.drawPath(
        Path()
          ..moveTo(to.dx, to.dy)
          ..lineTo(p1.dx, p1.dy)
          ..lineTo(p2.dx, p2.dy)
          ..close(),
        Paint()..color = color);
  }

  @override
  bool shouldRepaint(covariant _GraphPainter old) =>
      old.pulseValue != pulseValue ||
      old.selectedNode != selectedNode ||
      old.highlightedNodes != highlightedNodes ||
      old.panOffset != panOffset ||
      old.scale != scale ||
      old.nodes.length != nodes.length;
}

// ═══════════════════════════════════════════════════════════════════════════
// Background dot grid
// ═══════════════════════════════════════════════════════════════════════════

class _DotGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()..color = Colors.white.withOpacity(0.045);
    const step = 40.0;
    for (double x = 0; x < size.width; x += step) {
      for (double y = 0; y < size.height; y += step) {
        canvas.drawCircle(Offset(x, y), 1.0, p);
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter _) => false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Reusable UI components
// ═══════════════════════════════════════════════════════════════════════════

class _Panel extends StatelessWidget {
  final Widget child;
  final double? minWidth;
  final EdgeInsetsGeometry padding;

  const _Panel({
    required this.child,
    this.minWidth,
    this.padding = const EdgeInsets.all(18),
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: BoxConstraints(maxWidth: 260, minWidth: minWidth ?? 0),
      padding: padding,
      decoration: BoxDecoration(
        color: const Color(0xEE0D1117),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
        boxShadow: [
          BoxShadow(
              color: Colors.black.withOpacity(0.5),
              blurRadius: 20,
              offset: const Offset(0, 4))
        ],
      ),
      child: child,
    );
  }
}

class _PulseDot extends StatelessWidget {
  final Color color;
  const _PulseDot({required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 8,
      height: 8,
      decoration: BoxDecoration(
        color: color,
        shape: BoxShape.circle,
        boxShadow: [
          BoxShadow(
              color: color.withOpacity(0.6), blurRadius: 6, spreadRadius: 1)
        ],
      ),
    );
  }
}

class _StatRow2 extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  const _StatRow2(this.label, this.value, this.color);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3.5),
      child: Row(children: [
        Expanded(
            child: Text(label,
                style: TextStyle(color: Colors.grey[500], fontSize: 12))),
        Text(value,
            style: TextStyle(
                color: color, fontSize: 13, fontWeight: FontWeight.w700)),
      ]),
    );
  }
}

class _DetailField extends StatelessWidget {
  final String label;
  final String value;
  final TextStyle? valueStyle;
  const _DetailField(this.label, this.value, {this.valueStyle});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        SizedBox(
            width: 108,
            child: Text(label,
                style: TextStyle(color: Colors.grey[500], fontSize: 11))),
        Expanded(
            child: Text(value,
                style: valueStyle ??
                    const TextStyle(color: Colors.white70, fontSize: 12))),
      ]),
    );
  }
}

class _LegendRow extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendRow(this.color, this.label) : super();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Container(
            width: 9,
            height: 9,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 8),
        Text(label, style: TextStyle(color: Colors.grey[400], fontSize: 11)),
      ]),
    );
  }
}

class _ToolBtn extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _ToolBtn(this.icon, this.label, this.onTap);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: const Color(0xCC0D1117),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.white.withOpacity(0.12)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, color: Colors.white70, size: 15),
          const SizedBox(width: 5),
          Text(label,
              style: const TextStyle(
                  color: Colors.white54,
                  fontSize: 11,
                  fontWeight: FontWeight.w500)),
        ]),
      ),
    );
  }
}
