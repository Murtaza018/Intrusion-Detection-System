class GraphNode {
  final int id;
  final String ip;
  final double anomaly;
  final double x;
  final double y;

  // Normal constructor
  GraphNode({
    required this.id,
    required this.ip,
    this.anomaly = 0.0,
    this.x = 0.0,
    this.y = 0.0,
  });

  // Optional: const constructor (if you ever want to use it in const contexts)
  // const GraphNode.const({
  //   required this.id,
  //   required this.ip,
  //   this.anomaly = 0.0,
  //   this.x = 0.0,
  //   this.y = 0.0,
  // });

  factory GraphNode.fromJson(Map<String, dynamic> json) {
    return GraphNode(
      id: json['id'] as int,
      ip: json['ip'] as String,
      anomaly: (json['anomaly'] as num?)?.toDouble() ?? 0.0,
    );
  }

  GraphNode copyWith({double? x, double? y}) {
    return GraphNode(
      id: id,
      ip: ip,
      anomaly: anomaly,
      x: x ?? this.x,
      y: y ?? this.y,
    );
  }
}
