// lib/services/graph_layout_service.dart
import 'dart:math' as math;
import '../models/graph_data.dart';
import '../models/node.dart';
import '../models/edge.dart';

class GraphLayoutService {
  /// Applies a force-directed layout algorithm to position nodes
  static GraphData applyForceDirectedLayout(
    GraphData graphData, {
    int iterations = 100,
    double repulsionStrength = 1000.0,
    double attractionStrength = 0.1,
    double damping = 0.85,
  }) {
    if (graphData.nodes.isEmpty) return graphData;

    final random = math.Random(42); // Fixed seed for consistency
    final positions = <int, _Vector2D>{};
    final velocities = <int, _Vector2D>{};

    // Initialize positions randomly in a circle
    for (int i = 0; i < graphData.nodes.length; i++) {
      final node = graphData.nodes[i];
      final angle = (i / graphData.nodes.length) * 2 * math.pi;
      final radius = 200.0 + random.nextDouble() * 100;

      positions[node.id] = _Vector2D(
        math.cos(angle) * radius,
        math.sin(angle) * radius,
      );
      velocities[node.id] = _Vector2D(0, 0);
    }

    // Create adjacency map for faster lookup
    final adjacency = <int, Set<int>>{};
    for (final edge in graphData.edges) {
      adjacency.putIfAbsent(edge.source, () => {}).add(edge.target);
      adjacency.putIfAbsent(edge.target, () => {}).add(edge.source);
    }

    // Force-directed iterations
    for (int iter = 0; iter < iterations; iter++) {
      final forces = <int, _Vector2D>{};

      // Initialize forces
      for (final node in graphData.nodes) {
        forces[node.id] = _Vector2D(0, 0);
      }

      // Repulsion between all nodes (like charged particles)
      for (int i = 0; i < graphData.nodes.length; i++) {
        for (int j = i + 1; j < graphData.nodes.length; j++) {
          final pos1 = positions[graphData.nodes[i].id]!;
          final pos2 = positions[graphData.nodes[j].id]!;

          final delta = pos1 - pos2;
          final distance = delta.length();

          if (distance < 0.1) {
            // If they are on top of each other, push them apart in a random direction
            final randomDir =
                _Vector2D(random.nextDouble() - 0.5, random.nextDouble() - 0.5)
                    .normalized();
            final force = randomDir * repulsionStrength;
            forces[graphData.nodes[i].id] =
                forces[graphData.nodes[i].id]! + force;
            forces[graphData.nodes[j].id] =
                forces[graphData.nodes[j].id]! - force;
          } else {
            // FIX: Clamp the repulsion force magnitude to prevent blow-up.
            // Without this, small distances (e.g. 0.11) produce enormous forces
            // (2000 / 0.0121 ≈ 165,000) that compound over iterations into Infinity/NaN.
            final forceMag = (repulsionStrength / (distance * distance))
                .clamp(0.0, repulsionStrength * 2);
            final force = delta.normalized() * forceMag;
            forces[graphData.nodes[i].id] =
                forces[graphData.nodes[i].id]! + force;
            forces[graphData.nodes[j].id] =
                forces[graphData.nodes[j].id]! - force;
          }
        }
      }

      // Attraction between connected nodes (like springs)
      for (final edge in graphData.edges) {
        final pos1 = positions[edge.source];
        final pos2 = positions[edge.target];

        if (pos1 == null || pos2 == null) continue;

        final delta = pos2 - pos1;
        final distance = delta.length();

        if (distance > 0.1) {
          final force = delta.normalized() *
              (distance * attractionStrength * edge.weight);
          forces[edge.source] = forces[edge.source]! + force;
          forces[edge.target] = forces[edge.target]! - force;
        }
      }

      // Apply forces with damping
      final temp = 1.0 - (iter / iterations); // Temperature decreases over time
      for (final node in graphData.nodes) {
        var vel = (velocities[node.id]! + forces[node.id]!) * damping * temp;

        // FIX: Clamp velocity each iteration to prevent exponential blow-up.
        // Accumulated repulsion forces can push velocities toward Infinity over
        // many iterations; a hard cap keeps positions finite and canvas-safe.
        const maxVel = 50.0;
        vel = _Vector2D(
          vel.x.clamp(-maxVel, maxVel),
          vel.y.clamp(-maxVel, maxVel),
        );

        velocities[node.id] = vel;
        positions[node.id] = positions[node.id]! + vel;
      }
    }

    // Create new nodes with updated positions.
    // FIX: Sanitize positions before handing them to the painter.
    // If Infinity or NaN slipped through (e.g. via a degenerate graph),
    // Flutter's CanvasKit validator will throw on the very first drawCircle/drawLine call.
    final updatedNodes = graphData.nodes.map((node) {
      final pos = positions[node.id]!;
      return node.copyWith(
        x: pos.x.isFinite ? pos.x : 0.0,
        y: pos.y.isFinite ? pos.y : 0.0,
      );
    }).toList();

    return GraphData(
      nodes: updatedNodes,
      edges: graphData.edges,
      timestamp: graphData.timestamp,
    );
  }

  /// Applies a circular layout (simple but effective)
  static GraphData applyCircularLayout(GraphData graphData,
      {double radius = 300.0}) {
    if (graphData.nodes.isEmpty) return graphData;

    final updatedNodes = <GraphNode>[];

    for (int i = 0; i < graphData.nodes.length; i++) {
      final node = graphData.nodes[i];
      final angle = (i / graphData.nodes.length) * 2 * math.pi;

      updatedNodes.add(node.copyWith(
        x: math.cos(angle) * radius,
        y: math.sin(angle) * radius,
      ));
    }

    return GraphData(
      nodes: updatedNodes,
      edges: graphData.edges,
      timestamp: graphData.timestamp,
    );
  }

  /// Applies a grid layout
  static GraphData applyGridLayout(GraphData graphData,
      {double spacing = 150.0}) {
    if (graphData.nodes.isEmpty) return graphData;

    final updatedNodes = <GraphNode>[];
    final cols = math.sqrt(graphData.nodes.length).ceil();

    for (int i = 0; i < graphData.nodes.length; i++) {
      final node = graphData.nodes[i];
      final row = i ~/ cols;
      final col = i % cols;

      updatedNodes.add(node.copyWith(
        x: col * spacing - (cols * spacing / 2),
        y: row * spacing - (cols * spacing / 2),
      ));
    }

    return GraphData(
      nodes: updatedNodes,
      edges: graphData.edges,
      timestamp: graphData.timestamp,
    );
  }

  /// Applies a hierarchical layout based on connectivity
  static GraphData applyHierarchicalLayout(
    GraphData graphData, {
    double levelSpacing = 200.0,
    double nodeSpacing = 120.0,
  }) {
    if (graphData.nodes.isEmpty) return graphData;

    // Build adjacency list
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

    // Topological sort to determine levels
    final levels = <int, List<int>>{};
    final nodeLevel = <int, int>{};
    final queue = <int>[];

    // Start with nodes that have no incoming edges
    for (final node in graphData.nodes) {
      if (inDegree[node.id] == 0) {
        queue.add(node.id);
        nodeLevel[node.id] = 0;
        levels.putIfAbsent(0, () => []).add(node.id);
      }
    }

    // If no starting nodes, just use the first node
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

    // Assign positions based on levels
    final updatedNodes = <GraphNode>[];
    final nodeMap = <int, GraphNode>{};
    for (final node in graphData.nodes) {
      nodeMap[node.id] = node;
    }

    for (final entry in levels.entries) {
      final level = entry.key;
      final nodesInLevel = entry.value;
      final y = level * levelSpacing;

      for (int i = 0; i < nodesInLevel.length; i++) {
        final nodeId = nodesInLevel[i];
        final node = nodeMap[nodeId];
        if (node != null) {
          final x = (i - nodesInLevel.length / 2) * nodeSpacing;
          updatedNodes.add(node.copyWith(x: x, y: y));
        }
      }
    }

    // Add any nodes that weren't processed (disconnected nodes)
    for (final node in graphData.nodes) {
      if (!nodeLevel.containsKey(node.id)) {
        updatedNodes.add(node.copyWith(
          x: math.Random().nextDouble() * 400 - 200,
          y: math.Random().nextDouble() * 400 - 200,
        ));
      }
    }

    return GraphData(
      nodes: updatedNodes,
      edges: graphData.edges,
      timestamp: graphData.timestamp,
    );
  }
}

/// Helper class for 2D vector operations
class _Vector2D {
  final double x;
  final double y;

  _Vector2D(this.x, this.y);

  double length() => math.sqrt(x * x + y * y);

  _Vector2D normalized() {
    final len = length();
    return len > 0 ? _Vector2D(x / len, y / len) : _Vector2D(0, 0);
  }

  _Vector2D operator +(_Vector2D other) => _Vector2D(x + other.x, y + other.y);
  _Vector2D operator -(_Vector2D other) => _Vector2D(x - other.x, y - other.y);
  _Vector2D operator *(double scalar) => _Vector2D(x * scalar, y * scalar);
}
