import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import '../models/graph_data.dart';
import '../models/node.dart';
import '../models/edge.dart';

// Add normalize to Offset
extension OffsetExtension on Offset {
  Offset normalize() {
    final ds = distanceSquared;
    if (ds < 0.001) return Offset.zero;
    final l = sqrt(ds);
    return Offset(dx / l, dy / l);
  }
}

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

class _GraphViewWidgetState extends State<GraphViewWidget> {
  final Map<int, Offset> _nodePositions = {};
  final Map<int, double> _nodeSizes = {};
  final Map<int, Color> _nodeColors = {};

  late final Size _size;
  late double _scale;
  late Offset _panOffset;

  @override
  void initState() {
    _size = widget.size;
    _scale = 1.0;
    _panOffset = Offset.zero;

    _initPositions();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _tick();
    });
    super.initState();
  }

  void _initPositions() {
    for (final node in widget.graphData.nodes) {
      _nodePositions[node.id] = Offset(
        Random().nextDouble() * _size.width * 0.8 + _size.width * 0.1,
        Random().nextDouble() * _size.height * 0.8 + _size.height * 0.1,
      );
      _nodeSizes[node.id] =
          lerpDouble(8, 24, node.anomaly.clamp(0.0, 1.0)) ?? 8;
      _nodeColors[node.id] = _anomalyColor(node.anomaly);
    }
  }

  Color _anomalyColor(double anomaly) {
    final a = anomaly.clamp(0.0, 1.0);
    final h = 0.6 * (1.0 - a);
    return HSVColor.fromAHSV(1.0, h * 360, 1.0, 0.9).toColor();
  }

  void _tick() {
    final dt = 0.05;
    final kSpring = 0.008;
    final kRepel = 50000.0;
    final damping = 0.96;

    final velocities = <int, Offset>{};

    // 1. Spring force along edges
    for (final edge in widget.graphData.edges) {
      final src = widget.graphData.nodes.firstWhere(
        (n) => n.id == edge.source,
        orElse: () => GraphNode(id: 0, ip: "", anomaly: 0),
      );
      final dst = widget.graphData.nodes.firstWhere(
        (n) => n.id == edge.target,
        orElse: () => GraphNode(id: 0, ip: "", anomaly: 0),
      );

      final srcPos = (_nodePositions[src.id] ?? Offset.zero) - _panOffset;
      final dstPos = (_nodePositions[dst.id] ?? Offset.zero) - _panOffset;

      final direction = dstPos - srcPos;
      final length = max(1.0, direction.distance);
      final desired = 120.0;
      final springDist = length - desired;

      final springForce = direction.normalize() * springDist * kSpring;

      velocities[edge.source] =
          (velocities[edge.source] ?? Offset.zero) + springForce;
      velocities[edge.target] =
          (velocities[edge.target] ?? Offset.zero) - springForce;
    }

    // 2. Repel all nodes from each other
    for (final src in widget.graphData.nodes) {
      for (final dst in widget.graphData.nodes) {
        if (src.id == dst.id) continue;
        final a = (_nodePositions[src.id] ?? Offset.zero) - _panOffset;
        final b = (_nodePositions[dst.id] ?? Offset.zero) - _panOffset;

        final repelVec = a - b;
        final dist = max(1.0, repelVec.distance);
        final repel = repelVec.normalize() * (kRepel / (dist * dist));

        velocities[src.id] = (velocities[src.id] ?? Offset.zero) + repel;
      }
    }

    // 3. Update positions
    for (final node in widget.graphData.nodes) {
      final v = (velocities[node.id] ?? Offset.zero) * damping;
      final newPos = (_nodePositions[node.id] ?? Offset.zero) + v * dt * 100.0;
      _nodePositions[node.id] = newPos;
    }

    // 4. Keep in bounds
    final w = _size.width;
    final h = _size.height;

    for (final node in widget.graphData.nodes) {
      var pos = _nodePositions[node.id] ?? Offset.zero;
      pos = Offset(
        pos.dx.clamp(w * 0.05, w * 0.95),
        pos.dy.clamp(h * 0.05, h * 0.95),
      );
      _nodePositions[node.id] = pos;
    }

    if (mounted) {
      setState(() {});
    }

    WidgetsBinding.instance.addPostFrameCallback((_) => _tick());
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onScaleUpdate: (details) {
        setState(() {
          _scale = (_scale * details.scale).clamp(0.1, 5.0);
          _panOffset += details.focalPointDelta;
        });
      },
      onScaleEnd: (details) {
        // Optional: apply momentum / smoothing
      },
      onDoubleTap: () {
        setState(() {
          _panOffset = Offset.zero;
          _scale = 1.0;
        });
      },
      child: CustomPaint(
        size: _size,
        painter: _GraphPainter(
          graphData: widget.graphData,
          nodePositions: _nodePositions,
          nodeSizes: _nodeSizes,
          nodeColors: _nodeColors,
          panOffset: _panOffset,
          scale: _scale,
        ),
      ),
    );
  }
}

class _GraphPainter extends CustomPainter {
  final GraphData graphData;
  final Map<int, Offset> nodePositions;
  final Map<int, double> nodeSizes;
  final Map<int, Color> nodeColors;
  final Offset panOffset;
  final double scale;

  _GraphPainter({
    required this.graphData,
    required this.nodePositions,
    required this.nodeColors,
    required this.nodeSizes,
    required this.panOffset,
    required this.scale,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final edgePaint = Paint()
      ..style = PaintingStyle.stroke
      ..color = Colors.white12
      ..strokeWidth = 0.5;

    // Draw edges
    for (final edge in graphData.edges) {
      final src = graphData.nodes.firstWhere(
        (n) => n.id == edge.source,
        orElse: () => GraphNode(id: 0, ip: "", anomaly: 0),
      );
      final dst = graphData.nodes.firstWhere(
        (n) => n.id == edge.target,
        orElse: () => GraphNode(id: 0, ip: "", anomaly: 0),
      );

      if (src.id == 0 && dst.id == 0) continue;

      final a = (nodePositions[src.id] ?? Offset.zero) - panOffset;
      final b = (nodePositions[dst.id] ?? Offset.zero) - panOffset;

      canvas.drawLine(
        Offset(a.dx, a.dy) * scale + size.center(Offset.zero),
        Offset(b.dx, b.dy) * scale + size.center(Offset.zero),
        edgePaint,
      );
    }

    final textPaint = TextPainter(
      textDirection: TextDirection.ltr,
      maxLines: 1,
      textAlign: TextAlign.center,
    );

    // Draw nodes
    for (final node in graphData.nodes) {
      final pos = (nodePositions[node.id] ?? Offset.zero) - panOffset;
      final radius = (nodeSizes[node.id] ?? 10) * 0.5 * scale.clamp(0.5, 2.0);
      final center = Offset(pos.dx, pos.dy) * scale + size.center(Offset.zero);

      // Glow / halo
      final glowPaint = Paint()
        ..color = nodeColors[node.id]?.withOpacity(0.15) ?? Colors.white12
        ..style = PaintingStyle.fill;
      canvas.drawCircle(center, radius * 2, glowPaint);

      // Node circle
      final nodePaint = Paint()
        ..color = nodeColors[node.id] ?? Colors.teal
        ..style = PaintingStyle.fill;
      canvas.drawCircle(center, radius, nodePaint);

      nodePaint
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.0
        ..color = Colors.white54;
      canvas.drawCircle(center, radius, nodePaint);

      // Label (ip)
      textPaint.text = TextSpan(
        text: node.ip.split('.').last,
        style: TextStyle(
          fontSize: 9 * scale.clamp(0.5, 1.5),
          fontWeight: FontWeight.bold,
          color: Colors.white,
        ),
      );
      textPaint.layout();

      textPaint.paint(
        canvas,
        Offset(
          center.dx - textPaint.width / 2,
          center.dy - textPaint.height / 2,
        ),
      );
    }
  }

  @override
  bool shouldRepaint(covariant _GraphPainter oldDelegate) {
    return oldDelegate.graphData != graphData ||
        oldDelegate.nodePositions != nodePositions ||
        oldDelegate.scale != scale ||
        oldDelegate.panOffset != panOffset;
  }
}
