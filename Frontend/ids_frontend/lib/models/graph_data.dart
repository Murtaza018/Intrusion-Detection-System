import 'node.dart';
import 'edge.dart';

class GraphData {
  final List<GraphNode> nodes;
  final List<GraphEdge> edges;
  final double timestamp;

  GraphData({
    required this.nodes,
    required this.edges,
    this.timestamp = 0.0,
  });

  factory GraphData.fromJson(Map<String, dynamic> json) {
    final nodesList = (json['nodes'] as List? ?? [])
        .map((e) => GraphNode.fromJson(e as Map<String, dynamic>))
        .toList();
    final edgesList = (json['edges'] as List? ?? [])
        .map((e) => GraphEdge.fromJson(e as Map<String, dynamic>))
        .toList();

    return GraphData(
      nodes: nodesList,
      edges: edgesList,
      timestamp: (json['timestamp'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
