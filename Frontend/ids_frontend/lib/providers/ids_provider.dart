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
    this.confidence = 0.0,
    this.explanation,
    this.userLabel,
  });

  // Roadmap Point 3 Getters
  double get maeAnomaly => explanation?['mae_anomaly'] ?? 0.0;
  double get gnnAnomaly => explanation?['gnn_anomaly'] ?? 0.0;
}

class IdsProvider with ChangeNotifier {
  bool _isRunning = false;
  List<Packet> _packets = [];
  int _totalPackets = 0;
  int _normalCount = 0;
  int _attackCount = 0;
  int _zeroDayCount = 0;
  Timer? _packetTimer;
  Timer? _sensoryTimer; // High-speed loop for Gauges
  Set<int> _processedPacketIds = {};

  String _currentFilter = 'all';
  bool _isLoadingMore = false;

  // --- NEW: LIVE SENSORY STATE (ROADMAP POINT 3) ---
  double _liveGnnAnomaly = 0.0;
  double _liveMaeAnomaly = 0.0;
  String _liveStatus = 'unknown';

  // --- ADAPTATION QUEUES (RESTORED) ---
  final Set<int> _selectedPacketIds = {};
  final List<Packet> _ganQueue = [];
  final List<Packet> _jitterQueue = [];

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

  // Sensory Getters
  double get liveGnnAnomaly => _liveGnnAnomaly;
  double get liveMaeAnomaly => _liveMaeAnomaly;
  String get liveStatus => _liveStatus;

  List<Packet> get filteredPackets {
    if (_currentFilter == 'all') return _packets;
    return _packets.where((packet) => packet.status == _currentFilter).toList();
  }

  // Queue & Label Getters
  List<Packet> get ganQueue => _ganQueue;
  List<Packet> get jitterQueue => _jitterQueue;
  int get totalSelected => _ganQueue.length + _jitterQueue.length;
  String? get batchLabel => _batchLabel;
  bool get isNewAttack => _isNewAttack;
  bool get consistencyChecked => _consistencyChecked;
  bool get consistencyPassed => _consistencyPassed;

  static const String BASE_URL = "http://127.0.0.1:5001";
  static const Map<String, String> HEADERS = {
    'X-API-Key': 'MySuperSecretKey12345!'
  };

  // --- SELECTION & BATCH LOGIC (RESTORED) ---
  bool isSelected(int packetId) => _selectedPacketIds.contains(packetId);

  void toggleSelection(Packet packet, String queueType) {
    if (_selectedPacketIds.contains(packet.id)) {
      _selectedPacketIds.remove(packet.id);
      _ganQueue.removeWhere((p) => p.id == packet.id);
      _jitterQueue.removeWhere((p) => p.id == packet.id);
    } else {
      _selectedPacketIds.add(packet.id);
      if (queueType == 'gan')
        _ganQueue.add(packet);
      else if (queueType == 'jitter') _jitterQueue.add(packet);
    }
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

  void setBatchLabel(String? label, bool isNew) {
    _batchLabel = label;
    _isNewAttack = isNew;
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
      _packets = [];
      _processedPacketIds.clear();
      notifyListeners();
      _fetchRecentPackets();
    }
  }

  // --- PIPELINE CONTROL ---
  Future<void> startPipeline() async {
    if (_isRunning) return;
    _isRunning = true;
    notifyListeners();
    try {
      await http.post(Uri.parse('$BASE_URL/api/pipeline/start'),
          headers: HEADERS);
      _startDataStreams();
    } catch (e) {
      _isRunning = false;
      notifyListeners();
    }
  }

  Future<void> stopPipeline() async {
    _isRunning = false;
    _packetTimer?.cancel();
    _sensoryTimer?.cancel();
    notifyListeners();
    try {
      await http.post(Uri.parse('$BASE_URL/api/pipeline/stop'),
          headers: HEADERS);
    } catch (e) {
      debugPrint('Stop error: $e');
    }
  }

  void _startDataStreams() {
    _packetTimer?.cancel();
    _sensoryTimer?.cancel();

    _packetTimer = Timer.periodic(Duration(seconds: 2), (timer) {
      if (!_isRunning) {
        timer.cancel();
        return;
      }
      _fetchRecentPackets();
      _fetchStatsFromBackend();
    });

    _sensoryTimer = Timer.periodic(Duration(milliseconds: 500), (timer) {
      if (!_isRunning) {
        timer.cancel();
        return;
      }
      _fetchLiveSensoryData();
    });
  }

  // --- API CALLS (RESTORED & ENHANCED) ---
  Future<void> _fetchLiveSensoryData() async {
    try {
      final response = await http.get(Uri.parse('$BASE_URL/api/sensory/live'),
          headers: HEADERS);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _liveGnnAnomaly = (data['gnn_anomaly'] ?? 0.0).toDouble();
        _liveMaeAnomaly = (data['mae_anomaly'] ?? 0.0).toDouble();
        _liveStatus = data['status'] ?? 'unknown';
        notifyListeners();
      }
    } catch (e) {
      debugPrint('Sensory error: $e');
    }
  }

  Future<void> _fetchRecentPackets() async {
    try {
      String url = '$BASE_URL/api/packets/recent?limit=100';
      if (_currentFilter != 'all') url += '&status=$_currentFilter';
      final response = await http.get(Uri.parse(url), headers: HEADERS);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        for (var packetData in data['packets']) {
          _addPacketFromApi(packetData);
        }
      }
    } catch (e) {
      debugPrint('Fetch error: $e');
    }
  }

// --- PAGINATION LOGIC (LOAD OLDER PACKETS) ---
  Future<void> loadMorePackets() async {
    if (_isLoadingMore) return;
    _isLoadingMore = true;
    notifyListeners();

    try {
      // Offset is based on current list length to fetch the next batch
      int currentCount = _packets.length;
      String url = '$BASE_URL/api/packets/recent?limit=50&offset=$currentCount';

      if (_currentFilter != 'all') url += '&status=$_currentFilter';

      final response = await http.get(Uri.parse(url), headers: HEADERS);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final List<dynamic> fetchedPackets = data['packets'];

        for (var packetData in fetchedPackets) {
          final int packetId = packetData['id'];
          // Only add if we haven't seen it before
          if (!_processedPacketIds.contains(packetId)) {
            // .add() appends to the END of the list (older data)
            _packets.add(_mapDataToPacket(packetData));
            _processedPacketIds.add(packetId);
          }
        }
      }
    } catch (e) {
      debugPrint('Error loading more packets: $e');
    } finally {
      _isLoadingMore = false;
      notifyListeners();
    }
  }

  void _addPacketFromApi(Map<String, dynamic> data) {
    final int packetId = data['id'];
    final newPacket = _mapDataToPacket(data);

    if (_processedPacketIds.contains(packetId)) {
      final index = _packets.indexWhere((p) => p.id == packetId);
      if (index != -1 &&
          jsonEncode(_packets[index].explanation) !=
              jsonEncode(newPacket.explanation)) {
        _packets[index] = newPacket;
        notifyListeners();
      }
      return;
    }
    _packets.insert(0, newPacket);
    _processedPacketIds.add(packetId);
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

// --- LABEL MANAGEMENT (FOR ADAPTATION SCREEN) ---
  List<String> _existingLabels = [];
  List<String> get existingLabels => _existingLabels;

  Future<void> fetchLabels() async {
    try {
      final response =
          await http.get(Uri.parse('$BASE_URL/api/labels'), headers: HEADERS);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        // Extract the list of labels from the backend's label_encoder.pkl
        _existingLabels = List<String>.from(data['labels']);
        notifyListeners();
      }
    } catch (e) {
      debugPrint("Error fetching labels from backend: $e");
      // Fallback labels so the UI doesn't break if the backend is down
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

  Future<void> _fetchStatsFromBackend() async {
    try {
      final response =
          await http.get(Uri.parse('$BASE_URL/api/stats'), headers: HEADERS);
      if (response.statusCode == 200) {
        final statsData = jsonDecode(response.body);
        _totalPackets = statsData['total_packets'] ?? _totalPackets;
        _normalCount = statsData['normal_count'] ?? _normalCount;
        _attackCount = statsData['attack_count'] ?? _attackCount;
        _zeroDayCount = statsData['zero_day_count'] ?? _zeroDayCount;
        notifyListeners();
      }
    } catch (e) {
      debugPrint('Stats error: $e');
    }
  }

  Future<bool> sendRetrainRequest() async {
    if (_ganQueue.isEmpty || _batchLabel == null) return false;
    try {
      final body = {
        "gan_queue": _ganQueue
            .map((p) => {"id": p.id, "status": p.status, "summary": p.summary})
            .toList(),
        "jitter_queue": _jitterQueue
            .map((p) => {"id": p.id, "status": p.status, "summary": p.summary})
            .toList(),
        "target_label": _batchLabel,
        "is_new_label": _isNewAttack
      };
      final response = await http.post(Uri.parse('$BASE_URL/api/retrain'),
          headers: {...HEADERS, 'Content-Type': 'application/json'},
          body: jsonEncode(body));
      return response.statusCode == 200;
    } catch (e) {
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
          headers: {...HEADERS, 'Content-Type': 'application/json'},
          body: jsonEncode(body));
      if (response.statusCode == 200) return jsonDecode(response.body);
    } catch (e) {
      debugPrint("Analysis error: $e");
    }
    return {};
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
          'mae_anomaly': 0.85,
          'gnn_anomaly': 0.42
        });
    _packets.insert(0, packet);
    _processedPacketIds.add(packet.id);
    notifyListeners();
  }
}
