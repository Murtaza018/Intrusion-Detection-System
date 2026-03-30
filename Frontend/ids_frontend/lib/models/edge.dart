class GraphEdge {
  final int source;
  final int target;
  final double weight;

  GraphEdge({
    required this.source,
    required this.target,
    this.weight = 1.0,
  });

  factory GraphEdge.fromJson(Map<String, dynamic> json) {
    return GraphEdge(
      source: json['source'] as int,
      target: json['target'] as int,
      weight: (json['weight'] as num?)?.toDouble() ?? 1.0,
    );
  }
}
