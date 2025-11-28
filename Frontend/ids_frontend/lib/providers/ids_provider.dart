import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

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
  final String status; // 'normal', 'known_attack', 'zero_day'
  final double confidence;
  final Map<String, dynamic>? explanation;

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
    this.confidence = 0.0,
    this.explanation,
  });
}

class IdsProvider with ChangeNotifier {
  bool _isRunning = false;
  List<Packet> _packets = [];
  int _totalPackets = 0;
  int _normalCount = 0;
  int _attackCount = 0;
  int _zeroDayCount = 0;
  Timer? _packetTimer;

  bool get isRunning => _isRunning;
  List<Packet> get packets => _packets;
  int get totalPackets => _totalPackets;
  int get normalCount => _normalCount;
  int get attackCount => _attackCount;
  int get zeroDayCount => _zeroDayCount;

  static const String BASE_URL = "http://127.0.0.1:5000";

  Future<void> startPipeline() async {
    _isRunning = true;
    notifyListeners();

    try {
      final response = await http.post(
        Uri.parse('$BASE_URL/api/pipeline/start'),
        headers: {'X-API-Key': 'MySuperSecretKey12345!'},
      );

      if (response.statusCode == 200) {
        // Start listening for packets - FIXED METHOD NAME
        _startRealPacketStream();
      } else {
        _isRunning = false;
        notifyListeners();
      }
    } catch (e) {
      print('Error starting pipeline: $e');
      _isRunning = false;
      notifyListeners();
    }
  }

  Future<void> stopPipeline() async {
    _isRunning = false;
    _packetTimer?.cancel(); // Stop the timer
    notifyListeners();

    try {
      await http.post(
        Uri.parse('$BASE_URL/api/pipeline/stop'),
        headers: {'X-API-Key': 'MySuperSecretKey12345!'},
      );
    } catch (e) {
      print('Error stopping pipeline: $e');
    }
  }

  // FIXED: This is the correct method name now
  void _startRealPacketStream() {
    // Cancel any existing timer
    _packetTimer?.cancel();

    // Start polling for packets every 2 seconds
    _packetTimer = Timer.periodic(Duration(seconds: 2), (timer) async {
      if (!_isRunning) {
        timer.cancel();
        return;
      }

      try {
        final response = await http.get(
          Uri.parse('$BASE_URL/api/packets/recent?limit=5'),
          headers: {'X-API-Key': 'MySuperSecretKey12345!'},
        );

        if (response.statusCode == 200) {
          final data = jsonDecode(response.body);
          final List packets = data['packets'];

          for (var packetData in packets) {
            _addPacketFromApi(packetData);
          }
        }
      } catch (e) {
        print('Error fetching packets: $e');
      }
    });
  }

  void _addPacketFromApi(Map<String, dynamic> data) {
    // Check if we already have this packet to avoid duplicates
    if (_packets.any((packet) => packet.id == data['id'])) {
      return;
    }

    final packet = Packet(
      id: data['id'],
      summary: data['summary'],
      srcIp: data['src_ip'],
      dstIp: data['dst_ip'],
      protocol: data['protocol'],
      srcPort: data['src_port'],
      dstPort: data['dst_port'],
      length: data['length'],
      timestamp: DateTime.parse(data['timestamp']),
      status: data['status'],
      confidence: (data['confidence'] ?? 0.0).toDouble(),
      explanation: data['explanation'],
    );

    _packets.insert(0, packet); // Add to beginning for latest first
    _totalPackets++;

    if (packet.status == 'normal') {
      _normalCount++;
    } else if (packet.status == 'known_attack') {
      _attackCount++;
    } else if (packet.status == 'zero_day') {
      _zeroDayCount++;
    }

    notifyListeners();
  }

  // For demo - add a zero-day anomaly
  void addZeroDayPacket() {
    final packet = Packet(
      id: _totalPackets + 1,
      summary: 'UDP 192.168.1.99:12345 → 10.0.0.1:53',
      srcIp: '192.168.1.99',
      dstIp: '10.0.0.1',
      protocol: 'UDP',
      srcPort: 12345,
      dstPort: 53,
      length: 512,
      timestamp: DateTime.now(),
      status: 'zero_day',
      confidence: 0.92,
      explanation: {
        'type': 'Zero-Day Anomaly',
        'error': 2.45,
        'threshold': 1.0,
      },
    );

    _packets.insert(0, packet);
    _totalPackets++;
    _zeroDayCount++;
    notifyListeners();
  }

  // Clear all packets
  void clearPackets() {
    _packets.clear();
    _totalPackets = 0;
    _normalCount = 0;
    _attackCount = 0;
    _zeroDayCount = 0;
    notifyListeners();
  }

  @override
  void dispose() {
    _packetTimer?.cancel();
    super.dispose();
  }
}
