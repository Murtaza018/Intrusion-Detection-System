// models/packet.dart

class Packet {
  final int id;
  final String summary;
  final String srcIp;
  final String dstIp;
  final String protocol;
  final int srcPort;
  final int dstPort;
  final int length;
  final DateTime timestamp;
  final String status;
  final double confidence;
  final double maeAnomaly;
  final double gnnAnomaly;

  final Map<String, dynamic>? explanation;
  String? userLabel;

  Packet({
    required this.id,
    required this.summary,
    required this.srcIp,
    required this.dstIp,
    required this.protocol,
    required this.srcPort,
    required this.dstPort,
    required this.length,
    required this.timestamp,
    required this.status,
    required this.maeAnomaly,
    required this.gnnAnomaly,
    this.confidence = 0.0,
    this.explanation,
    this.userLabel,
  });
}
