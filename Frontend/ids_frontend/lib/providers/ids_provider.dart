import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

// --- Packet Data Model ---
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
  final Map<String, dynamic>? explanation;
  String? userLabel; // Stores the label assigned by the user

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
    this.userLabel,
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
  Set<int> _processedPacketIds = {};

  String _currentFilter = 'all';
  bool _isLoadingMore = false;

  // --- ADAPTATION QUEUES ---
  final Set<int> _selectedPacketIds = {};
  final List<Packet> _ganQueue = [];
  final List<Packet> _jitterQueue = [];

  // --- BATCH LABELING & SAFETY STATE (NEW) ---
  String? _batchLabel;
  bool _isNewAttack = false;
  bool _consistencyChecked = false;
  bool _consistencyPassed = false;

  // Getters
  bool get isRunning => _isRunning;
  List<Packet> get packets => _packets;
  int get totalPackets => _totalPackets;
  int get normalCount => _normalCount;
  int get attackCount => _attackCount;
  int get zeroDayCount => _zeroDayCount;
  String get currentFilter => _currentFilter;
  bool get isLoadingMore => _isLoadingMore;

  List<Packet> get filteredPackets {
    if (_currentFilter == 'all') return _packets;
    return _packets.where((packet) => packet.status == _currentFilter).toList();
  }

  // Queue Getters
  List<Packet> get ganQueue => _ganQueue;
  List<Packet> get jitterQueue => _jitterQueue;
  int get totalSelected => _ganQueue.length + _jitterQueue.length;

  // Label State Getters
  String? get batchLabel => _batchLabel;
  bool get isNewAttack => _isNewAttack;
  bool get consistencyChecked => _consistencyChecked;
  bool get consistencyPassed => _consistencyPassed;

  // CHANGE IP HERE IF NEEDED (10.0.2.2 for Emulator)
  static const String BASE_URL = "http://127.0.0.1:5001";

  // --- SELECTION LOGIC ---
  bool isSelected(int packetId) => _selectedPacketIds.contains(packetId);

  void toggleSelection(Packet packet, String queueType) {
    if (_selectedPacketIds.contains(packet.id)) {
      // Deselect
      _selectedPacketIds.remove(packet.id);
      _ganQueue.removeWhere((p) => p.id == packet.id);
      _jitterQueue.removeWhere((p) => p.id == packet.id);
    } else {
      // Select
      _selectedPacketIds.add(packet.id);
      if (queueType == 'gan') {
        _ganQueue.add(packet);
      } else if (queueType == 'jitter') {
        _jitterQueue.add(packet);
      }
    }
    // Reset consistency checks when selection changes
    _consistencyChecked = false;
    _consistencyPassed = false;
    notifyListeners();
  }

  void clearQueues() {
    _selectedPacketIds.clear();
    _ganQueue.clear();
    _jitterQueue.clear();
    _batchLabel = null;
    _consistencyChecked = false;
    notifyListeners();
  }

  // --- BATCH LABEL SETTERS (NEW) ---
  void setBatchLabel(String? label, bool isNew) {
    _batchLabel = label;
    _isNewAttack = isNew;
    // Require re-check if label strategies change
    _consistencyChecked = false;
    _consistencyPassed = false;
    notifyListeners();
  }

  void setConsistencyStatus(bool checked, bool passed) {
    _consistencyChecked = checked;
    _consistencyPassed = passed;
    notifyListeners();
  }

  void setFilter(String filter) {
    if (_currentFilter != filter) {
      _currentFilter = filter;
      _packets = []; // Clear list to avoid mixing types
      _processedPacketIds.clear();
      notifyListeners();
      _fetchRecentPackets(); // Fetch new data from backend
    }
  }

  // --- Pipeline Control ---
  Future<void> startPipeline() async {
    if (_isRunning) return;
    _isRunning = true;
    notifyListeners();
    try {
      await http.post(Uri.parse('$BASE_URL/api/pipeline/start'),
          headers: {'X-API-Key': 'MySuperSecretKey12345!'});
      _startRealPacketStream();
    } catch (e) {
      _isRunning = false;
      notifyListeners();
    }
  }

  Future<void> stopPipeline() async {
    if (!_isRunning) return;
    _isRunning = false;
    _packetTimer?.cancel();
    notifyListeners();
    try {
      await http.post(Uri.parse('$BASE_URL/api/pipeline/stop'),
          headers: {'X-API-Key': 'MySuperSecretKey12345!'});
    } catch (e) {
      print('Error stopping: $e');
    }
  }

  void _startRealPacketStream() {
    _packetTimer?.cancel();
    _fetchRecentPackets();
    _fetchStatsFromBackend();
    _packetTimer = Timer.periodic(Duration(seconds: 2), (timer) async {
      if (!_isRunning) {
        timer.cancel();
        return;
      }
      await Future.wait([_fetchRecentPackets(), _fetchStatsFromBackend()]);
    });
  }

  Future<void> _fetchRecentPackets() async {
    try {
      String url = '$BASE_URL/api/packets/recent?limit=100';
      if (_currentFilter != 'all') url += '&status=$_currentFilter';
      final response = await http.get(Uri.parse(url),
          headers: {'X-API-Key': 'MySuperSecretKey12345!'});
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        for (var packetData in data['packets']) {
          _addPacketFromApi(packetData);
        }
      }
    } catch (e) {
      print('Error fetching: $e');
    }
  }

  List<String> _existingLabels = [];
  List<String> get existingLabels => _existingLabels;

  Future<void> fetchLabels() async {
    try {
      final response = await http.get(Uri.parse('$BASE_URL/api/labels'),
          headers: {'X-API-Key': 'MySuperSecretKey12345!'});
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _existingLabels = List<String>.from(data['labels']);
        notifyListeners();
      }
    } catch (e) {
      print("Error fetching labels: $e");
      _existingLabels = [
        "BENIGN",
        "DDoS",
        "PortScan",
        "Bot",
        "Infiltration",
        "Web Attack"
      ];
      notifyListeners();
    }
  }

  void _addPacketFromApi(Map<String, dynamic> data) {
    final int packetId = data['id'];
    final newPacket = _mapDataToPacket(data);

    if (_processedPacketIds.contains(packetId)) {
      final index = _packets.indexWhere((p) => p.id == packetId);
      if (index != -1) {
        if (jsonEncode(_packets[index].explanation) !=
            jsonEncode(newPacket.explanation)) {
          _packets[index] = newPacket;
          notifyListeners();
        }
      }
      return;
    }
    _packets.insert(0, newPacket);
    _processedPacketIds.add(packetId);

    if (_packets.length > 5000) {
      _packets = _packets.sublist(0, 4000);
      final recentIds = _packets.map((p) => p.id).toSet();
      _processedPacketIds = _processedPacketIds.intersection(recentIds);
    }
    notifyListeners();
  }

  Packet _mapDataToPacket(Map<String, dynamic> data) {
    return Packet(
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
  }

  Future<void> _fetchStatsFromBackend() async {
    try {
      final response = await http.get(Uri.parse('$BASE_URL/api/stats'),
          headers: {'X-API-Key': 'MySuperSecretKey12345!'});
      if (response.statusCode == 200) {
        final statsData = jsonDecode(response.body);
        if (statsData['total_packets'] > _totalPackets) {
          _totalPackets = statsData['total_packets'] ?? _totalPackets;
          _normalCount = statsData['normal_count'] ?? _normalCount;
          _attackCount = statsData['attack_count'] ?? _attackCount;
          _zeroDayCount = statsData['zero_day_count'] ?? _zeroDayCount;
          notifyListeners();
        }
      }
    } catch (e) {
      print('Error stats: $e');
    }
  }

  // --- SEND RETRAIN REQUEST (UPDATED) ---
  Future<bool> sendRetrainRequest() async {
    // Only block if GAN packets exist but no label is set
    if (_ganQueue.isNotEmpty && _batchLabel == null) return false;

    try {
      final body = {
        "gan_queue": _ganQueue
            .map((p) => {"id": p.id, "status": p.status, "summary": p.summary})
            .toList(),
        "jitter_queue": _jitterQueue
            .map((p) => {"id": p.id, "status": p.status, "summary": p.summary})
            .toList(),

        // Pass the Label Info
        "target_label": _batchLabel,
        "is_new_label": _isNewAttack
      };

      print("[FLUTTER] Sending Retrain Request...");

      final response = await http.post(
        Uri.parse('$BASE_URL/api/retrain'),
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'MySuperSecretKey12345!'
        },
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final respData = jsonDecode(response.body);
        print("[FLUTTER] Success: ${respData['message']}");
        return true;
      } else {
        print("[FLUTTER] Failed: ${response.statusCode} - ${response.body}");
        return false;
      }
    } catch (e) {
      print("[FLUTTER] Error sending retrain request: $e");
      return false;
    }
  }

  Future<Map<String, dynamic>> analyzeQueues() async {
    try {
      final body = {
        "gan_queue": _ganQueue.map((p) => {"id": p.id}).toList(),
        "jitter_queue": _jitterQueue.map((p) => {"id": p.id}).toList(),
      };

      final response = await http.post(
        Uri.parse('$BASE_URL/api/analyze_selection'),
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'MySuperSecretKey12345!'
        },
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
    } catch (e) {
      print("Analysis error: $e");
    }
    return {};
  }

  // Load More (History)
  Future<void> loadMorePackets() async {
    if (_isLoadingMore) return;
    _isLoadingMore = true;
    notifyListeners();
    try {
      int currentCount = _packets.length;
      String url = '$BASE_URL/api/packets/recent?limit=50&offset=$currentCount';
      if (_currentFilter != 'all') url += '&status=$_currentFilter';
      final response = await http.get(Uri.parse(url),
          headers: {'X-API-Key': 'MySuperSecretKey12345!'});
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        for (var packetData in data['packets']) {
          final int packetId = packetData['id'];
          if (!_processedPacketIds.contains(packetId)) {
            _packets.add(_mapDataToPacket(packetData));
            _processedPacketIds.add(packetId);
          }
        }
      }
    } catch (e) {
      print('Error loading more: $e');
    } finally {
      _isLoadingMore = false;
      notifyListeners();
    }
  }

  void addZeroDayPacket() {
    final packet = Packet(
        id: DateTime.now().millisecondsSinceEpoch,
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
          'threshold': 1.0
        });
    _packets.insert(0, packet);
    _processedPacketIds.add(packet.id);
    notifyListeners();
  }

  // Method to mark a packet as false positive (sends request to backend)
  Future<bool> markAsFalsePositive(int packetId) async {
    try {
      final response = await http.post(
        Uri.parse('$BASE_URL/api/feedback/false_positive'),
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'MySuperSecretKey12345!'
        },
        body: jsonEncode({'packet_id': packetId}),
      );
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}
