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
  Timer? _sensoryTimer;
  Set<int> _processedPacketIds = {};

  // --- NEW: SOC-GRADE MULTI-FILTER STATE ---
  Map<String, dynamic> _activeFilters = {
    'status': 'all', // 'all', 'normal', 'known_attack', 'zero_day'
    'search': '', // Matches IP or Protocol
  };

  bool _isLoadingMore = false;

  double _liveGnnAnomaly = 0.0;
  double _liveMaeAnomaly = 0.0;
  String _liveStatus = 'unknown';

  final Set<int> _selectedPacketIds = {};
  final List<Packet> _ganQueue = [];
  final List<Packet> _jitterQueue = [];

  String? _batchLabel;
  bool _isNewAttack = false;
  bool _consistencyChecked = false;
  bool _consistencyPassed = false;

  // Basic Getters
  bool get isRunning => _isRunning;
  int get totalPackets => _totalPackets;
  int get normalCount => _normalCount;
  int get attackCount => _attackCount;
  int get zeroDayCount => _zeroDayCount;
  bool get isLoadingMore => _isLoadingMore;

  // Sensory Getters
  double get liveGnnAnomaly => _liveGnnAnomaly;
  double get liveMaeAnomaly => _liveMaeAnomaly;
  String get liveStatus => _liveStatus;

  // Filter Getters
  String get currentFilter => _activeFilters['status'];
  Map<String, dynamic> get activeFilters => _activeFilters;

  // --- REWRITTEN: MULTI-FILTER LOGIC (IP, Protocol, Status) ---
  List<Packet> get filteredPackets {
    return _packets.where((p) {
      // 1. Status Check (Normal/Attack/ZeroDay)
      bool matchesStatus = _activeFilters['status'] == 'all' ||
          p.status == _activeFilters['status'];

      // 2. Search Check (IPs or Protocol)
      String query = _activeFilters['search'].toLowerCase();
      bool matchesSearch = query.isEmpty ||
          p.srcIp.contains(query) ||
          p.dstIp.contains(query) ||
          p.protocol.toLowerCase().contains(query);

      return matchesStatus && matchesSearch;
    }).toList();
  }

  // Selection Getters
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

  // --- FILTER & SEARCH SETTERS ---
  void setFilter(String statusFilter) {
    _activeFilters['status'] = statusFilter;
    // We don't clear packets anymore, just filter the view for a smoother experience
    notifyListeners();
  }

  void updateSearchQuery(String query) {
    _activeFilters['search'] = query;
    notifyListeners();
  }

  void clearAllFilters() {
    _activeFilters = {'status': 'all', 'search': ''};
    notifyListeners();
  }

  // --- SELECTION LOGIC ---
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

  List<String> _existingLabels = [];
  List<String> get existingLabels => _existingLabels;

  void setBatchLabel(String? label, bool isNew) {
    _batchLabel = label;
    _isNewAttack = isNew;
    _consistencyChecked = false;
    _consistencyPassed = false;
    notifyListeners();
  }

  // Ensure these are also present if not already:
  void setConsistencyStatus(bool checked, bool passed) {
    _consistencyChecked = checked;
    _consistencyPassed = passed;
    notifyListeners();
  }

  void clearQueues() {
    _selectedPacketIds.clear();
    _ganQueue.clear();
    _jitterQueue.clear();
    _batchLabel = null;
    _consistencyChecked = false;
    _consistencyPassed = false;
    notifyListeners();
  }

  /// Sends the selected packets and target labels to the backend /api/retrain endpoint.
  Future<bool> sendRetrainRequest() async {
    // Safety check: Don't allow empty training runs
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
        "is_new_label":
            _isNewAttack // Correctly flags WGAN to create a new neuron
      };

      final response = await http.post(
        Uri.parse('$BASE_URL/api/retrain'),
        headers: {...HEADERS, 'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );

      return response.statusCode == 200;
    } catch (e) {
      debugPrint("Retrain Request Failed: $e");
      return false;
    }
  }

  Future<Map<String, dynamic>> analyzeQueues() async {
    // Return empty result if nothing to analyze
    if (_ganQueue.isEmpty && _jitterQueue.isEmpty) return {};

    try {
      final body = {
        "gan_queue": _ganQueue.map((p) => {"id": p.id}).toList(),
        "jitter_queue": _jitterQueue.map((p) => {"id": p.id}).toList(),
      };

      final response = await http.post(
        Uri.parse('$BASE_URL/api/analyze_selection'),
        headers: {
          ...HEADERS,
          'Content-Type': 'application/json',
        },
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        // Result typically contains: {"passed": true/false, "variance": 0.045}
        return result;
      }
    } catch (e) {
      debugPrint("Analysis error: $e");
    }

    // Fallback if backend is unreachable or errors out
    return {"passed": false, "error": "Backend unreachable"};
  }

  // This fetches the labels used by your backend label_encoder.pkl
  Future<void> fetchLabels() async {
    try {
      final response =
          await http.get(Uri.parse('$BASE_URL/api/labels'), headers: HEADERS);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _existingLabels = List<String>.from(data['labels']);
        notifyListeners();
      }
    } catch (e) {
      debugPrint("Label fetch error: $e");
      _existingLabels = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration"];
      notifyListeners();
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

    _packetTimer = Timer.periodic(const Duration(seconds: 2), (timer) {
      if (!_isRunning) {
        timer.cancel();
        return;
      }
      _fetchRecentPackets();
      _fetchStatsFromBackend();
    });

    _sensoryTimer = Timer.periodic(const Duration(milliseconds: 500), (timer) {
      if (!_isRunning) {
        timer.cancel();
        return;
      }
      _fetchLiveSensoryData();
    });
  }

  // --- CORE API CALLS ---
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
      // We fetch all types during polling to keep counts accurate,
      // but the UI getter handles the filtering.
      String url = '$BASE_URL/api/packets/recent?limit=100';
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

  Future<void> loadMorePackets() async {
    if (_isLoadingMore) return;
    _isLoadingMore = true;
    notifyListeners();
    try {
      int offset = _packets.length;
      String url = '$BASE_URL/api/packets/recent?limit=50&offset=$offset';
      final response = await http.get(Uri.parse(url), headers: HEADERS);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        for (var packetData in data['packets']) {
          final int id = packetData['id'];
          if (!_processedPacketIds.contains(id)) {
            _packets.add(_mapDataToPacket(packetData));
            _processedPacketIds.add(id);
          }
        }
      }
    } catch (e) {
      debugPrint('Pagination error: $e');
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

  Future<void> _fetchStatsFromBackend() async {
    try {
      final response =
          await http.get(Uri.parse('$BASE_URL/api/stats'), headers: HEADERS);
      if (response.statusCode == 200) {
        final stats = jsonDecode(response.body);
        _totalPackets = stats['total_packets'] ?? _totalPackets;
        _normalCount = stats['normal_count'] ?? _normalCount;
        _attackCount = stats['attack_count'] ?? _attackCount;
        _zeroDayCount = stats['zero_day_count'] ?? _zeroDayCount;
        notifyListeners();
      }
    } catch (e) {
      debugPrint('Stats error: $e');
    }
  }

  void addZeroDayPacket() {
    final id = DateTime.now().millisecondsSinceEpoch;
    final packet = Packet(
        id: id,
        summary: 'INJECTED_NOVELTY_DATA',
        srcIp: '192.168.1.99',
        dstIp: '10.0.0.1',
        protocol: 'TCP',
        srcPort: 443,
        dstPort: 443,
        length: 1024,
        timestamp: DateTime.now(),
        status: 'zero_day',
        confidence: 0.95,
        explanation: {
          'description': 'Simulated anomaly for structural verification.',
          'mae_anomaly': 0.88,
          'gnn_anomaly': 0.15,
          'status': 'done'
        });
    _packets.insert(0, packet);
    _processedPacketIds.add(id);
    notifyListeners();
  }
}
