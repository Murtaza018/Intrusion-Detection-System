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
  Set<int> _processedPacketIds =
      {}; // ADDED: Track processed IDs to prevent duplicates

  bool get isRunning => _isRunning;
  List<Packet> get packets => _packets;
  int get totalPackets => _totalPackets;
  int get normalCount => _normalCount;
  int get attackCount => _attackCount;
  int get zeroDayCount => _zeroDayCount;

  static const String BASE_URL = "http://127.0.0.1:5001";

  Future<void> startPipeline() async {
    if (_isRunning) return; // ADDED: Prevent multiple starts

    _isRunning = true;
    notifyListeners();

    try {
      final response = await http.post(
        Uri.parse('$BASE_URL/api/pipeline/start'),
        headers: {'X-API-Key': 'MySuperSecretKey12345!'},
      );

      if (response.statusCode == 200) {
        print("[FLUTTER] Pipeline started successfully");
        _startRealPacketStream();
      } else {
        print("[FLUTTER] Failed to start pipeline: ${response.statusCode}");
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
    if (!_isRunning) return; // ADDED: Prevent multiple stops

    _isRunning = false;
    _packetTimer?.cancel();
    _packetTimer = null;
    notifyListeners();

    try {
      final response = await http.post(
        Uri.parse('$BASE_URL/api/pipeline/stop'),
        headers: {'X-API-Key': 'MySuperSecretKey12345!'},
      );

      if (response.statusCode == 200) {
        print("[FLUTTER] Pipeline stopped successfully");
      }
    } catch (e) {
      print('Error stopping pipeline: $e');
    }
  }

  void _startRealPacketStream() {
    _packetTimer?.cancel();

    // ADDED: Also fetch stats immediately when starting
    _fetchStatsFromBackend();

    _packetTimer = Timer.periodic(Duration(seconds: 2), (timer) async {
      if (!_isRunning) {
        timer.cancel();
        return;
      }

      try {
        // Fetch both packets and stats in parallel
        await Future.wait([
          _fetchRecentPackets(),
          _fetchStatsFromBackend(),
        ]);
      } catch (e) {
        print('Error in packet stream: $e');
      }
    });
  }

  // ADDED: Separate method for fetching packets
  Future<void> _fetchRecentPackets() async {
    try {
      final response = await http.get(
        Uri.parse('$BASE_URL/api/packets/recent?limit=10'), // Increased limit
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
  }

  // ADDED: Separate method for fetching stats from backend
  Future<void> _fetchStatsFromBackend() async {
    try {
      final response = await http.get(
        Uri.parse('$BASE_URL/api/stats'),
        headers: {'X-API-Key': 'MySuperSecretKey12345!'},
      );

      if (response.statusCode == 200) {
        final statsData = jsonDecode(response.body);
        _updateStatsFromBackend(statsData);
      }
    } catch (e) {
      print('Error fetching stats: $e');
    }
  }

  // ADDED: Update stats from backend to stay in sync
  void _updateStatsFromBackend(Map<String, dynamic> statsData) {
    // Only update if backend has more recent data
    if (statsData['total_packets'] > _totalPackets) {
      _totalPackets = statsData['total_packets'] ?? _totalPackets;
      _normalCount = statsData['normal_count'] ?? _normalCount;
      _attackCount = statsData['attack_count'] ?? _attackCount;
      _zeroDayCount = statsData['zero_day_count'] ?? _zeroDayCount;
      notifyListeners();
    }
  }

  void _addPacketFromApi(Map<String, dynamic> data) {
    final int packetId = data['id'];

    // IMPROVED: Use Set for faster duplicate checking
    if (_processedPacketIds.contains(packetId)) {
      return;
    }

    final packet = Packet(
      id: packetId,
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

    _packets.insert(0, packet);
    _processedPacketIds.add(packetId); // Track this ID

    // Note: We're letting backend stats be the source of truth
    // but we'll still update locally for immediate UI feedback
    _totalPackets++;

    if (packet.status == 'normal') {
      _normalCount++;
    } else if (packet.status == 'known_attack') {
      _attackCount++;
    } else if (packet.status == 'zero_day') {
      _zeroDayCount++;
    }

    // ADDED: Limit packets list to prevent memory issues
    if (_packets.length > 200) {
      _packets = _packets.sublist(0, 100);
      // Also clean up processed IDs to prevent memory leaks
      final recentIds = _packets.map((p) => p.id).toSet();
      _processedPacketIds = _processedPacketIds.intersection(recentIds);
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
    _processedPacketIds.add(packet.id);
    notifyListeners();
  }

  // Clear all packets
  void clearPackets() {
    _packets.clear();
    _processedPacketIds.clear();
    _totalPackets = 0;
    _normalCount = 0;
    _attackCount = 0;
    _zeroDayCount = 0;
    notifyListeners();
  }

  // ADDED: Get pipeline status from backend
  Future<Map<String, dynamic>?> getPipelineStatus() async {
    try {
      final response = await http.get(
        Uri.parse('$BASE_URL/api/pipeline/status'),
        headers: {'X-API-Key': 'MySuperSecretKey12345!'},
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
    } catch (e) {
      print('Error fetching pipeline status: $e');
    }
    return null;
  }

  @override
  void dispose() {
    _packetTimer?.cancel();
    super.dispose();
  }
}
