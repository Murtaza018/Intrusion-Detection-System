// lib/services/graph_layout_service.dart
import 'dart:math' as math;
import '../models/graph_data.dart';
import '../models/node.dart';
import '../models/edge.dart';

class GraphLayoutService {
  /// Force-directed (Fruchterman-Reingold style) layout.
  ///
  /// Key tuning levers:
  ///  • [initialRadius]   — how wide the starting circle is. Larger = nodes
  ///                        start farther apart and end up farther apart.
  ///  • [repulsionStrength] — how hard nodes push each other away.
  ///  • [attractionStrength] — spring constant pulling connected nodes together.
  ///  • [minDistance]     — hard floor: no two nodes end up closer than this
  ///                        in world-space, so zooming in always shows space.
  static GraphData applyForceDirectedLayout(
    GraphData graphData, {
    int iterations = 150,
    double repulsionStrength = 8000.0,
    double attractionStrength = 0.06,
    double damping = 0.82,
    double initialRadius = 600.0,
    double minDistance = 80.0, // world-space minimum gap between node centres
  }) {
    if (graphData.nodes.isEmpty) return graphData;

    final n = graphData.nodes.length;
    final random = math.Random(42);
    final positions = <int, _Vec2>{};
    final velocities = <int, _Vec2>{};

    // Place nodes evenly on a circle with a small random jitter so no two
    // nodes start exactly on top of each other.
    for (int i = 0; i < n; i++) {
      final node = graphData.nodes[i];
      final angle = (i / n) * 2 * math.pi;
      // Stagger inner/outer rings for denser graphs so layout converges faster.
      final ring = initialRadius * (0.6 + (i % 3) * 0.25) +
          random.nextDouble() * initialRadius * 0.15;
      positions[node.id] = _Vec2(
        math.cos(angle) * ring,
        math.sin(angle) * ring,
      );
      velocities[node.id] = _Vec2(0, 0);
    }

    // ── Main simulation loop ──────────────────────────────────────────────
    for (int iter = 0; iter < iterations; iter++) {
      final forces = <int, _Vec2>{
        for (final node in graphData.nodes) node.id: _Vec2(0, 0)
      };

      // Cooling temperature: starts at 1.0, ends at 0.0.
      final temp = 1.0 - (iter / iterations);

      // Repulsion (O(n²) — acceptable for n ≤ 100)
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          final id1 = graphData.nodes[i].id;
          final id2 = graphData.nodes[j].id;
          final delta = positions[id1]! - positions[id2]!;
          final dist = delta.length();

          _Vec2 force;
          if (dist < 0.1) {
            // Exactly overlapping — push in a random direction.
            final dir =
                _Vec2(random.nextDouble() - 0.5, random.nextDouble() - 0.5)
                    .normalized();
            force = dir * repulsionStrength;
          } else {
            // Standard inverse-square repulsion, clamped to avoid blow-up.
            final mag = (repulsionStrength / (dist * dist))
                .clamp(0.0, repulsionStrength * 2.0);
            force = delta.normalized() * mag;
          }

          forces[id1] = forces[id1]! + force;
          forces[id2] = forces[id2]! - force;
        }
      }

      // Attraction along edges (spring model)
      for (final edge in graphData.edges) {
        final p1 = positions[edge.source];
        final p2 = positions[edge.target];
        if (p1 == null || p2 == null) continue;

        final delta = p2 - p1;
        final dist = delta.length();
        if (dist < 0.1) continue;

        final mag = dist * attractionStrength * edge.weight;
        final force = delta.normalized() * mag;
        forces[edge.source] = forces[edge.source]! + force;
        forces[edge.target] = forces[edge.target]! - force;
      }

      // Integrate with velocity damping + temperature cooling
      for (final node in graphData.nodes) {
        var vel = (velocities[node.id]! + forces[node.id]!) * damping * temp;

        // Hard velocity cap to prevent runaway
        const maxVel = 80.0;
        vel = _Vec2(
          vel.x.clamp(-maxVel, maxVel),
          vel.y.clamp(-maxVel, maxVel),
        );

        velocities[node.id] = vel;
        positions[node.id] = positions[node.id]! + vel;
      }

      // ── Minimum distance enforcement ──────────────────────────────────
      // After each integration step, push any pair of nodes that ended up
      // closer than minDistance apart back to that minimum separation.
      // This guarantees there is always visible space between nodes at any
      // zoom level, preventing the "clumped blob" appearance.
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          final id1 = graphData.nodes[i].id;
          final id2 = graphData.nodes[j].id;
          final delta = positions[id1]! - positions[id2]!;
          final dist = delta.length();
          if (dist < minDistance && dist > 0.001) {
            final push = delta.normalized() * ((minDistance - dist) / 2.0);
            positions[id1] = positions[id1]! + push;
            positions[id2] = positions[id2]! - push;
          } else if (dist < 0.001) {
            // Exact overlap fallback
            positions[id1] = positions[id1]! +
                _Vec2(random.nextDouble() * minDistance,
                    random.nextDouble() * minDistance);
          }
        }
      }
    }

    // Sanitise and return
    final updated = graphData.nodes.map((node) {
      final pos = positions[node.id]!;
      return node.copyWith(
        x: pos.x.isFinite ? pos.x : 0.0,
        y: pos.y.isFinite ? pos.y : 0.0,
      );
    }).toList();

    return GraphData(
      nodes: updated,
      edges: graphData.edges,
      timestamp: graphData.timestamp,
    );
  }

  // ── Other layouts (unchanged) ─────────────────────────────────────────────

  static GraphData applyCircularLayout(GraphData graphData,
      {double radius = 300.0}) {
    if (graphData.nodes.isEmpty) return graphData;
    final updated = <GraphNode>[];
    for (int i = 0; i < graphData.nodes.length; i++) {
      final angle = (i / graphData.nodes.length) * 2 * math.pi;
      updated.add(graphData.nodes[i].copyWith(
        x: math.cos(angle) * radius,
        y: math.sin(angle) * radius,
      ));
    }
    return GraphData(
        nodes: updated, edges: graphData.edges, timestamp: graphData.timestamp);
  }

  static GraphData applyGridLayout(GraphData graphData,
      {double spacing = 150.0}) {
    if (graphData.nodes.isEmpty) return graphData;
    final updated = <GraphNode>[];
    final cols = math.sqrt(graphData.nodes.length).ceil();
    for (int i = 0; i < graphData.nodes.length; i++) {
      final row = i ~/ cols;
      final col = i % cols;
      updated.add(graphData.nodes[i].copyWith(
        x: col * spacing - (cols * spacing / 2),
        y: row * spacing - (cols * spacing / 2),
      ));
    }
    return GraphData(
        nodes: updated, edges: graphData.edges, timestamp: graphData.timestamp);
  }

  static GraphData applyHierarchicalLayout(
    GraphData graphData, {
    double levelSpacing = 200.0,
    double nodeSpacing = 120.0,
  }) {
    if (graphData.nodes.isEmpty) return graphData;

    final adjacency = <int, List<int>>{};
    final inDegree = <int, int>{};
    for (final node in graphData.nodes) {
      adjacency[node.id] = [];
      inDegree[node.id] = 0;
    }
    for (final edge in graphData.edges) {
      adjacency[edge.source]?.add(edge.target);
      inDegree[edge.target] = (inDegree[edge.target] ?? 0) + 1;
    }

    final levels = <int, List<int>>{};
    final nodeLevel = <int, int>{};
    final queue = <int>[];

    for (final node in graphData.nodes) {
      if (inDegree[node.id] == 0) {
        queue.add(node.id);
        nodeLevel[node.id] = 0;
        levels.putIfAbsent(0, () => []).add(node.id);
      }
    }
    if (queue.isEmpty && graphData.nodes.isNotEmpty) {
      final firstId = graphData.nodes.first.id;
      queue.add(firstId);
      nodeLevel[firstId] = 0;
      levels.putIfAbsent(0, () => []).add(firstId);
    }

    while (queue.isNotEmpty) {
      final nodeId = queue.removeAt(0);
      final level = nodeLevel[nodeId]!;
      for (final neighborId in adjacency[nodeId] ?? []) {
        if (!nodeLevel.containsKey(neighborId)) {
          nodeLevel[neighborId] = level + 1;
          levels.putIfAbsent(level + 1, () => []).add(neighborId);
          queue.add(neighborId);
        }
      }
    }

    final nodeMap = <int, GraphNode>{for (final n in graphData.nodes) n.id: n};
    final updated = <GraphNode>[];

    for (final entry in levels.entries) {
      final nodesInLevel = entry.value;
      final y = entry.key * levelSpacing;
      for (int i = 0; i < nodesInLevel.length; i++) {
        final node = nodeMap[nodesInLevel[i]];
        if (node != null) {
          updated.add(node.copyWith(
            x: (i - nodesInLevel.length / 2) * nodeSpacing,
            y: y,
          ));
        }
      }
    }

    final random = math.Random(0);
    for (final node in graphData.nodes) {
      if (!nodeLevel.containsKey(node.id)) {
        updated.add(node.copyWith(
          x: random.nextDouble() * 400 - 200,
          y: random.nextDouble() * 400 - 200,
        ));
      }
    }

    return GraphData(
        nodes: updated, edges: graphData.edges, timestamp: graphData.timestamp);
  }
}

// ── 2D vector helper ──────────────────────────────────────────────────────────

class _Vec2 {
  final double x;
  final double y;
  const _Vec2(this.x, this.y);

  double length() => math.sqrt(x * x + y * y);

  _Vec2 normalized() {
    final l = length();
    return l > 0 ? _Vec2(x / l, y / l) : const _Vec2(0, 0);
  }

  _Vec2 operator +(_Vec2 o) => _Vec2(x + o.x, y + o.y);
  _Vec2 operator -(_Vec2 o) => _Vec2(x - o.x, y - o.y);
  _Vec2 operator *(double s) => _Vec2(x * s, y * s);
}
