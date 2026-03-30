// lib/widgets/graph_view_widget.dart - OPTIMIZED FOR NETWORK ANOMALY DETECTION
import 'package:flutter/foundation.dart' show setEquals;
import 'package:flutter/material.dart';
import '../models/graph_data.dart';
import '../models/node.dart';
import '../models/edge.dart';
import '../services/graph_layout_service.dart';
import 'dart:math' as math;

class GraphViewWidget extends StatefulWidget {
  final GraphData graphData;
  final Size size;

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

  // FIX (infinite layout loop): Track which node-ID set has already been laid
  // out. graphStream() emits fresh GraphData every 2 s with nodes at (0,0),
  // so without this guard didUpdateWidget → _applyLayout fires on every tick
  // and the expensive force-directed pass runs continuously.
  Set<int> _laidOutNodeIds = {};

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat(reverse: true);

    _applyLayout();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      _autoFit();
    });
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

  /// Normalise a raw anomaly value to [0.0, 1.0].
  /// Backend sends scores in 0–3 range. Values above 1.0 are genuine
  /// high-anomaly scores (not percentage encoding), so we simply clamp.
  double _normaliseAnomaly(double raw) => raw.clamp(0.0, 1.0);

  void _applyLayout() {
    final incomingIds = widget.graphData.nodes.map((n) => n.id).toSet();

    // FIX: Use setEquals() for content-based Set comparison.
    // Dart's != on two Set objects compares by *identity*, so two Sets with
    // identical elements are always != — causing layout to re-run every tick.
    final topologyChanged = !setEquals(incomingIds, _laidOutNodeIds);

    if (topologyChanged) {
      final needsPositioning = widget.graphData.nodes.every(
        (node) => node.x.abs() < 0.1 && node.y.abs() < 0.1,
      );

      GraphData positioned;
      if (needsPositioning && widget.graphData.nodes.isNotEmpty) {
        print('Applying force-directed layout for ${incomingIds.length} nodes');
        positioned = GraphLayoutService.applyForceDirectedLayout(
          widget.graphData,
          iterations: 150,
          repulsionStrength: 2000.0,
          attractionStrength: 0.15,
        );
      } else {
        positioned = widget.graphData;
      }

      // Normalise anomaly on the freshly laid-out nodes too.
      _layoutGraphData = GraphData(
        nodes: positioned.nodes
            .map((n) => n.copyWith(anomaly: _normaliseAnomaly(n.anomaly)))
            .toList(),
        edges: positioned.edges,
        timestamp: positioned.timestamp,
      );

      _laidOutNodeIds = incomingIds;
      WidgetsBinding.instance.addPostFrameCallback((_) => _autoFit());
    } else {
      // Same topology — keep positions, refresh anomaly scores only.
      final updatedAnomalyMap = {
        for (final n in widget.graphData.nodes) n.id: n.anomaly
      };

      _layoutGraphData = GraphData(
        nodes: _layoutGraphData.nodes.map((n) {
          final raw = updatedAnomalyMap[n.id] ?? n.anomaly;
          return n.copyWith(anomaly: _normaliseAnomaly(raw));
        }).toList(),
        edges: _layoutGraphData.edges,
        timestamp: widget.graphData.timestamp,
      );
    }
  }

  void _autoFit() {
    if (_layoutGraphData.nodes.isEmpty) return;

    final bounds = _calculateBounds();
    if (bounds.width == 0 || bounds.height == 0) return;

    setState(() {
      final scaleX = (widget.size.width - 100) / bounds.width;
      final scaleY = (widget.size.height - 100) / bounds.height;
      _scale = math.min(scaleX, scaleY).clamp(0.3, 3.0);

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

    for (final node in _layoutGraphData.nodes) {
      if (!node.x.isFinite || !node.y.isFinite) continue;
      minX = math.min(minX, node.x);
      minY = math.min(minY, node.y);
      maxX = math.max(maxX, node.x);
      maxY = math.max(maxY, node.y);
    }

    if (!minX.isFinite || !minY.isFinite || !maxX.isFinite || !maxY.isFinite) {
      return Rect.zero;
    }

    const padding = 50.0;
    return Rect.fromLTRB(
      minX - padding,
      minY - padding,
      maxX + padding,
      maxY + padding,
    );
  }

  int _attackCount() =>
      _layoutGraphData.nodes.where((node) => node.anomaly > 0.3).length;

  int _criticalCount() =>
      _layoutGraphData.nodes.where((node) => node.anomaly > 0.6).length;

  void _onNodeTap(GraphNode node) {
    setState(() {
      _selectedNode = node;
      _highlightedNodes.clear();

      for (final edge in _layoutGraphData.edges) {
        if (edge.source == node.id) {
          _highlightedNodes.add(edge.target);
        } else if (edge.target == node.id) {
          _highlightedNodes.add(edge.source);
        }
      }
      _highlightedNodes.add(node.id);
    });
  }

  void _clearSelection() {
    setState(() {
      _selectedNode = null;
      _highlightedNodes.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    final attacks = _attackCount();
    final critical = _criticalCount();

    return Scaffold(
      backgroundColor: const Color(0xFF0A0E12),
      body: Stack(
        children: [
          Positioned.fill(child: CustomPaint(painter: GridPainter())),

          // Graph canvas
          Positioned.fill(
            child: GestureDetector(
              onScaleStart: (_) {
                setState(() => _showControls = false);
                _clearSelection();
              },
              onScaleUpdate: (details) {
                setState(() {
                  _scale = (_scale * details.scale).clamp(0.2, 5.0);
                  _panOffset += details.focalPointDelta;
                });
              },
              onScaleEnd: (_) => setState(() => _showControls = true),
              onDoubleTap: _autoFit,
              onTapUp: (details) {
                final tapPos = details.localPosition;
                GraphNode? tappedNode;

                for (final node in _layoutGraphData.nodes) {
                  final nodePos = Offset(
                    node.x * _scale + _panOffset.dx,
                    node.y * _scale + _panOffset.dy,
                  );
                  final radius =
                      (10 + node.anomaly * 15) / _scale.clamp(0.5, 2.0);

                  if ((tapPos - nodePos).distance <= radius * 2) {
                    tappedNode = node;
                    break;
                  }
                }

                if (tappedNode != null) {
                  _onNodeTap(tappedNode);
                } else {
                  _clearSelection();
                }
              },
              child: AnimatedBuilder(
                animation: _pulseController,
                builder: (context, child) {
                  return CustomPaint(
                    size: widget.size,
                    painter: GraphPainter(
                      nodes: _layoutGraphData.nodes,
                      edges: _layoutGraphData.edges,
                      panOffset: _panOffset,
                      scale: _scale,
                      pulseValue: _pulseController.value,
                      selectedNode: _selectedNode,
                      highlightedNodes: _highlightedNodes,
                    ),
                  );
                },
              ),
            ),
          ),

          // Stats Panel
          Positioned(
            top: 20,
            right: 20,
            child: Material(
              color: Colors.black.withOpacity(0.8),
              borderRadius: BorderRadius.circular(16),
              elevation: 12,
              child: Container(
                padding: const EdgeInsets.all(20),
                constraints: const BoxConstraints(maxWidth: 200),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: BoxDecoration(
                            color: Colors.cyan,
                            shape: BoxShape.circle,
                            boxShadow: [
                              BoxShadow(
                                color: Colors.cyan.withOpacity(0.5),
                                blurRadius: 8,
                                spreadRadius: 2,
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 10),
                        const Text(
                          'NETWORK STATUS',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 13,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1.5,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    const Divider(color: Colors.white24, height: 1),
                    const SizedBox(height: 16),
                    _StatRow('Total Nodes', '${_layoutGraphData.nodes.length}',
                        Colors.blue),
                    _StatRow('Connections', '${_layoutGraphData.edges.length}',
                        Colors.blue),
                    const SizedBox(height: 8),
                    _StatRow('🔥 Critical', '$critical', Colors.red),
                    _StatRow(
                        '⚠️ Warnings', '${attacks - critical}', Colors.orange),
                    const SizedBox(height: 8),
                    _StatRow(
                      'Avg Anomaly',
                      _layoutGraphData.nodes.isEmpty
                          ? '0%'
                          : '${(_layoutGraphData.nodes.fold<double>(0.0, (sum, n) => sum + n.anomaly) / _layoutGraphData.nodes.length * 100).toStringAsFixed(1)}%',
                      _getAnomalyColor(_layoutGraphData.nodes.isEmpty
                          ? 0.0
                          : _layoutGraphData.nodes.fold<double>(
                                  0.0, (sum, n) => sum + n.anomaly) /
                              _layoutGraphData.nodes.length),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Selected Node Info
          if (_selectedNode != null)
            Positioned(
              top: 20,
              left: 20,
              child: Material(
                color: Colors.black.withOpacity(0.9),
                borderRadius: BorderRadius.circular(12),
                elevation: 12,
                child: Container(
                  padding: const EdgeInsets.all(16),
                  constraints: const BoxConstraints(maxWidth: 280),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Row(
                        children: [
                          Container(
                            width: 12,
                            height: 12,
                            decoration: BoxDecoration(
                              color: _getNodeColor(_selectedNode!.anomaly),
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 8),
                          const Text(
                            'NODE DETAILS',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 11,
                              fontWeight: FontWeight.bold,
                              letterSpacing: 1.2,
                            ),
                          ),
                          const Spacer(),
                          GestureDetector(
                            onTap: _clearSelection,
                            child: const Icon(Icons.close,
                                color: Colors.white54, size: 18),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'IP: ${_selectedNode!.ip}',
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text('Node ID: ${_selectedNode!.id}',
                          style:
                              TextStyle(color: Colors.grey[400], fontSize: 12)),
                      const SizedBox(height: 8),
                      Row(
                        children: [
                          Text('Anomaly Score: ',
                              style: TextStyle(
                                  color: Colors.grey[400], fontSize: 12)),
                          Text(
                            '${(_selectedNode!.anomaly * 100).toStringAsFixed(1)}%',
                            style: TextStyle(
                              color: _getNodeColor(_selectedNode!.anomaly),
                              fontSize: 13,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text('Connections: ${_highlightedNodes.length - 1}',
                          style:
                              TextStyle(color: Colors.grey[400], fontSize: 12)),
                    ],
                  ),
                ),
              ),
            ),

          // Controls
          if (_showControls)
            Positioned(
              bottom: 20,
              left: 20,
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _ControlButton(
                      icon: Icons.zoom_out_map, label: 'Fit', onTap: _autoFit),
                  const SizedBox(width: 8),
                  _ControlButton(
                    icon: Icons.refresh,
                    label: 'Reset',
                    onTap: () {
                      setState(() {
                        _scale = 1.0;
                        _panOffset = Offset.zero;
                      });
                      Future.delayed(
                          const Duration(milliseconds: 100), _autoFit);
                    },
                  ),
                ],
              ),
            ),

          // Legend
          Positioned(
            bottom: 20,
            right: 20,
            child: Material(
              color: Colors.black.withOpacity(0.7),
              borderRadius: BorderRadius.circular(12),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _LegendItem(Colors.green, 'Normal (< 10%)'),
                    _LegendItem(Colors.yellow, 'Low (10-30%)'),
                    _LegendItem(Colors.orange, 'Medium (30-60%)'),
                    _LegendItem(Colors.red, 'Critical (> 60%)'),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Color _getNodeColor(double anomaly) {
    if (anomaly > 0.6) return Colors.red;
    if (anomaly > 0.3) return Colors.orange;
    if (anomaly > 0.1) return Colors.yellow;
    return Colors.green;
  }

  Color _getAnomalyColor(double avgAnomaly) {
    if (avgAnomaly > 0.6) return Colors.red;
    if (avgAnomaly > 0.3) return Colors.orange;
    if (avgAnomaly > 0.1) return Colors.yellow;
    return Colors.green;
  }
}

// ── Supporting widgets ────────────────────────────────────────────────────────

class _StatRow extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _StatRow(this.label, this.value, this.color);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
              flex: 3,
              child: Text(label,
                  style: TextStyle(color: Colors.grey[400], fontSize: 12))),
          Expanded(
            flex: 2,
            child: Text(value,
                style: TextStyle(
                    color: color, fontSize: 14, fontWeight: FontWeight.bold),
                textAlign: TextAlign.right),
          ),
        ],
      ),
    );
  }
}

class _LegendItem extends StatelessWidget {
  final Color color;
  final String label;

  const _LegendItem(this.color, this.label);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
              width: 10,
              height: 10,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
          const SizedBox(width: 8),
          Text(label,
              style: const TextStyle(color: Colors.white70, fontSize: 11)),
        ],
      ),
    );
  }
}

class _ControlButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ControlButton(
      {required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.1),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: Colors.white.withOpacity(0.25)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: Colors.white, size: 16),
            const SizedBox(width: 6),
            Text(label,
                style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 11,
                    fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }
}

// ── Painters ──────────────────────────────────────────────────────────────────

class GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.03)
      ..strokeWidth = 1;

    const spacing = 50.0;
    for (double x = 0; x < size.width; x += spacing) {
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), paint);
    }
    for (double y = 0; y < size.height; y += spacing) {
      canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class GraphPainter extends CustomPainter {
  final List<GraphNode> nodes;
  final List<GraphEdge> edges;
  final Offset panOffset;
  final double scale;
  final double pulseValue;
  final GraphNode? selectedNode;
  final Set<int> highlightedNodes;

  GraphPainter({
    required this.nodes,
    required this.edges,
    required this.panOffset,
    required this.scale,
    required this.pulseValue,
    this.selectedNode,
    required this.highlightedNodes,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (nodes.isEmpty) return;

    final nodeMap = <int, GraphNode>{for (final n in nodes) n.id: n};
    final hasSelection = highlightedNodes.isNotEmpty;

    // ── Edges ────────────────────────────────────────────────────────────────
    // With 1000+ edges drawing every line creates a solid blob.
    // Strategy: always draw highlighted edges; for the rest, only draw the
    // top-weight edges up to a cap so the graph stays readable.
    const maxUnselectedEdges = 120;

    // Sort edges: highlighted first, then by weight descending
    final sortedEdges = List<GraphEdge>.from(edges)
      ..sort((a, b) {
        final aHl = highlightedNodes.contains(a.source) &&
            highlightedNodes.contains(a.target);
        final bHl = highlightedNodes.contains(b.source) &&
            highlightedNodes.contains(b.target);
        if (aHl != bHl) return aHl ? -1 : 1;
        return b.weight.compareTo(a.weight);
      });

    int unselectedCount = 0;
    final strokeBase = 1.2 / scale.clamp(0.4, 2.0);

    for (final edge in sortedEdges) {
      final node1 = nodeMap[edge.source];
      final node2 = nodeMap[edge.target];
      if (node1 == null || node2 == null) continue;

      final isHighlighted = highlightedNodes.contains(edge.source) &&
          highlightedNodes.contains(edge.target);

      // Skip low-priority edges once we hit the cap (keeps graph readable)
      if (!isHighlighted && hasSelection) continue;
      if (!isHighlighted && unselectedCount >= maxUnselectedEdges) continue;
      if (!isHighlighted) unselectedCount++;

      final p1 = Offset(
          node1.x * scale + panOffset.dx, node1.y * scale + panOffset.dy);
      final p2 = Offset(
          node2.x * scale + panOffset.dx, node2.y * scale + panOffset.dy);

      if (!p1.dx.isFinite ||
          !p1.dy.isFinite ||
          !p2.dx.isFinite ||
          !p2.dy.isFinite) continue;

      Color edgeColor;
      double strokeWidth;
      if (isHighlighted) {
        edgeColor = Colors.cyan.withOpacity(0.85);
        strokeWidth = strokeBase * 1.8;
      } else {
        // Colour by weight: heavier = more cyan, lighter = dimmer grey
        final t = (edge.weight / 5.0).clamp(0.0, 1.0);
        edgeColor = Color.lerp(
          Colors.white.withOpacity(0.12),
          Colors.cyan.withOpacity(0.45),
          t,
        )!;
        strokeWidth = strokeBase * (0.8 + t * 0.6);
      }

      canvas.drawLine(
          p1,
          p2,
          Paint()
            ..color = edgeColor
            ..strokeWidth = strokeWidth
            ..style = PaintingStyle.stroke);
    }

    // ── Nodes ────────────────────────────────────────────────────────────────
    for (final node in nodes) {
      final center =
          Offset(node.x * scale + panOffset.dx, node.y * scale + panOffset.dy);
      if (!center.dx.isFinite || !center.dy.isFinite) continue;

      final isHighlighted = highlightedNodes.contains(node.id);
      final isSelected = selectedNode?.id == node.id;
      final dimmed = hasSelection && !isHighlighted;
      final opacity = dimmed ? 0.25 : 1.0;

      // Keep node size modest — anomaly drives colour, not huge radius
      final radius = (6.0 + node.anomaly * 8.0) / scale.clamp(0.4, 2.0);
      final nodeColor = _getNodeColor(node.anomaly);

      // Subtle pulse only for critical nodes — no giant glow
      final pulseMultiplier =
          (node.anomaly > 0.6 && !dimmed) ? 1.0 + (pulseValue * 0.15) : 1.0;

      // Thin outer ring on anomalous nodes (replaces the heavy glow)
      if (node.anomaly > 0.3 && !dimmed) {
        canvas.drawCircle(
          center,
          radius * 1.45 * pulseMultiplier,
          Paint()
            ..color = nodeColor.withOpacity(node.anomaly > 0.6 ? 0.55 : 0.35)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.5 / scale.clamp(0.4, 2.0),
        );
      }

      // Main filled circle
      canvas.drawCircle(
        center,
        radius * pulseMultiplier,
        Paint()
          ..color = nodeColor.withOpacity(opacity)
          ..style = PaintingStyle.fill,
      );

      // Dark inner dot so nodes look solid, not flat
      canvas.drawCircle(
        center,
        radius * 0.35 * pulseMultiplier,
        Paint()
          ..color = Colors.black.withOpacity(0.35 * opacity)
          ..style = PaintingStyle.fill,
      );

      // Selection ring
      if (isSelected) {
        canvas.drawCircle(
          center,
          radius * 1.7,
          Paint()
            ..color = Colors.white
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2.0 / scale.clamp(0.4, 2.0),
        );
      }

      // IP label — show whenever zoomed in enough or node is selected
      if (scale > 0.5 || isSelected) {
        final displayText =
            node.ip.length > 15 ? '${node.ip.substring(0, 13)}...' : node.ip;
        final tp = TextPainter(
          text: TextSpan(
            text: displayText,
            style: TextStyle(
              color: Colors.white.withOpacity(dimmed ? 0.2 : 0.88),
              fontSize: (isSelected ? 11.0 : 9.0) / scale.clamp(0.4, 2.0),
              fontWeight: isSelected ? FontWeight.bold : FontWeight.w500,
              shadows: [
                Shadow(blurRadius: 4, color: Colors.black.withOpacity(0.9))
              ],
            ),
          ),
          textAlign: TextAlign.center,
          textDirection: TextDirection.ltr,
        )..layout(minWidth: 0, maxWidth: 140);

        tp.paint(
            canvas,
            Offset(center.dx - tp.width / 2,
                center.dy + radius * pulseMultiplier + 4));
      }
    }
  }

  Color _getNodeColor(double anomaly) {
    if (anomaly > 0.6) return Colors.red;
    if (anomaly > 0.3) return Colors.orange;
    if (anomaly > 0.1) return Colors.yellow;
    return Colors.green;
  }

  @override
  bool shouldRepaint(covariant GraphPainter oldDelegate) {
    return oldDelegate.pulseValue != pulseValue ||
        oldDelegate.selectedNode != selectedNode ||
        oldDelegate.highlightedNodes != highlightedNodes ||
        oldDelegate.panOffset != panOffset ||
        oldDelegate.scale != scale;
  }
}
