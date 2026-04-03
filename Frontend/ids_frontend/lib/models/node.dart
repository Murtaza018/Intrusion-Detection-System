// class GraphNode {
//   final int id;
//   final String ip;
//   final double anomaly;
//   final double x;
//   final double y;

//   GraphNode({
//     required this.id,
//     required this.ip,
//     this.anomaly = 0.0,
//     this.x = 0.0,
//     this.y = 0.0,
//   });

//   factory GraphNode.fromJson(Map<String, dynamic> json) {
//     return GraphNode(
//       id: json['id'] as int,
//       ip: json['ip'] as String,
//       anomaly: (json['anomaly'] as num?)?.toDouble() ?? 0.0,
//     );
//   }

//   GraphNode copyWith({double? x, double? y, double? anomaly}) {
//     return GraphNode(
//       id: id,
//       ip: ip,
//       anomaly: anomaly ?? this.anomaly,
//       x: x ?? this.x,
//       y: y ?? this.y,
//     );
//   }
// }

// lib/models/node.dart
class GraphNode {
  final int id;
  final String ip;
  final double anomaly;
  final double x;
  final double y;

  // 1. Add the new fields
  final String subnet;
  final bool isDmz;
  final bool isGateway;

  GraphNode({
    required this.id,
    required this.ip,
    this.anomaly = 0.0,
    this.x = 0.0,
    this.y = 0.0,
    this.subnet = 'Unknown',
    this.isDmz = false,
    this.isGateway = false,
  });

  factory GraphNode.fromJson(Map<String, dynamic> json) {
    return GraphNode(
      id: json['id'] as int,
      ip: json['ip'] as String,
      anomaly: (json['anomaly'] as num?)?.toDouble() ?? 0.0,
      // 2. Parse them from the incoming API JSON
      subnet: json['subnet'] as String? ?? 'Unknown',
      isDmz: json['isDmz'] as bool? ?? false,
      isGateway: json['isGateway'] as bool? ?? false,
    );
  }

  GraphNode copyWith({
    double? x,
    double? y,
    double? anomaly,
    String? subnet,
    bool? isDmz,
    bool? isGateway,
  }) {
    return GraphNode(
      id: id,
      ip: ip,
      anomaly: anomaly ?? this.anomaly,
      x: x ?? this.x,
      y: y ?? this.y,
      // 3. Include them in the copy function used by the layout engine
      subnet: subnet ?? this.subnet,
      isDmz: isDmz ?? this.isDmz,
      isGateway: isGateway ?? this.isGateway,
    );
  }
}
