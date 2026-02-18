import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:cryptography/cryptography.dart';
import 'package:hex/hex.dart'; // Ensure you added 'hex: ^0.2.0' to pubspec.yaml

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
  IdsProvider() {
    runECCSaneCheck(); // Triggers the ECC validation on app launch
  }
  // --- 1. SECURE CONFIGURATION ---
  static const String BASE_URL = "http://127.0.0.1:5001";
  static const String apiKey =
      String.fromEnvironment('API_KEY', defaultValue: 'MySuperSecretKey12345!');

  // --- ECC PUBLIC KEY COORDINATES (Extracted from your cert.pem) ---
  static const String _pubXHex =
      "182110e3c4af92f5f2d2ae042be8dd44d67e51d1a4728986874e0fcd64829253";
  static const String _pubYHex =
      "66c676c3e1d01c05b5299c744060a1911598259fffa710925f1060bd23160970";

  // --- 2. PIPELINE STATE ---
  bool _isRunning = false;
  List<Packet> _packets = [];
  int _totalPackets = 0;
  int _normalCount = 0;
  int _attackCount = 0;
  int _zeroDayCount = 0;
  Timer? _packetTimer;
  Timer? _sensoryTimer;
  final Set<int> _processedPacketIds = {};

  // --- 3. FILTER & UI STATE ---
  Map<String, dynamic> _activeFilters = {
    'status': 'all', // 'all', 'normal', 'known_attack', 'zero_day'
    'search': '', // Matches IP or Protocol
  };
  bool _isLoadingMore = false;

  // --- 4. SENSORY (GAUGE) STATE ---
  double _liveGnnAnomaly = 0.0;
  double _liveMaeAnomaly = 0.0;
  String _liveStatus = 'unknown';

  // --- 5. SELECTION & CONTINUAL LEARNING QUEUES ---
  final Set<int> _selectedPacketIds = {};
  final List<Packet> _ganQueue = [];
  final List<Packet> _jitterQueue = [];

  String? _batchLabel;
  bool _isNewAttack = false;
  bool _consistencyChecked = false;
  bool _consistencyPassed = false;
  List<String> _existingLabels = [];

  // --- GETTERS ---
  bool get isRunning => _isRunning;
  int get totalPackets => _totalPackets;
  int get normalCount => _normalCount;
  int get attackCount => _attackCount;
  int get zeroDayCount => _zeroDayCount;
  bool get isLoadingMore => _isLoadingMore;
  double get liveGnnAnomaly => _liveGnnAnomaly;
  double get liveMaeAnomaly => _liveMaeAnomaly;
  String get liveStatus => _liveStatus;
  String get currentFilter => _activeFilters['status'];
  Map<String, dynamic> get activeFilters => _activeFilters;
  List<Packet> get ganQueue => _ganQueue;
  List<Packet> get jitterQueue => _jitterQueue;
  int get totalSelected => _ganQueue.length + _jitterQueue.length;
  String? get batchLabel => _batchLabel;
  bool get isNewAttack => _isNewAttack;
  bool get consistencyChecked => _consistencyChecked;
  bool get consistencyPassed => _consistencyPassed;
  List<String> get existingLabels => _existingLabels;

  Map<String, String> get headers => {
        'X-API-Key': apiKey,
        'Content-Type': 'application/json',
      };

  List<Packet> get filteredPackets {
    return _packets.where((p) {
      bool matchesStatus = _activeFilters['status'] == 'all' ||
          p.status == _activeFilters['status'];
      String query = _activeFilters['search'].toLowerCase();
      bool matchesSearch = query.isEmpty ||
          p.srcIp.contains(query) ||
          p.dstIp.contains(query) ||
          p.protocol.toLowerCase().contains(query);
      return matchesStatus && matchesSearch;
    }).toList();
  }

  // --- 6. ECC VERIFICATION LOGIC ---
  Future<bool> _verifyServerSignature(Map<String, dynamic> responseBody) async {
    final String? signatureHex = responseBody['signature'];
    final Map<String, dynamic>? payload = responseBody['payload'];

    if (signatureHex == null || payload == null) return false;

    try {
      final algorithm = Ecdsa.p256(Sha256());
      final signatureBytes = HEX.decode(signatureHex);
      final messageBytes = utf8.encode(jsonEncode(payload));

      final publicKey = EcPublicKey(
        x: HEX.decode(_pubXHex),
        y: HEX.decode(_pubYHex),
        type: KeyPairType.p256,
      );

      final isVerified = await algorithm.verify(
        messageBytes,
        signature: Signature(signatureBytes, publicKey: publicKey),
      );

      if (!isVerified)
        debugPrint("⚠️ SECURITY ALERT: Response signature mismatch!");
      return isVerified;
    } catch (e) {
      debugPrint("❌ ECC Verification Error: $e");
      return false;
    }
  }

  Future<void> runECCSaneCheck() async {
    debugPrint("🧪 Starting ECC Sanity Test...");
    try {
      final algorithm = Ecdsa.p256(Sha256());
      final keyPair = await algorithm.newKeyPair();
      final message = utf8.encode("sanity_test");
      final signature = await algorithm.sign(message, keyPair: keyPair);
      final isVerified = await algorithm.verify(message, signature: signature);
      debugPrint(isVerified ? "✅ ECC SANITY PASSED" : "❌ ECC SANITY FAILED");
    } catch (e) {
      debugPrint("❌ ECC ERROR: $e");
    }
  }

  // --- 7. UI CONTROL METHODS ---
  void setFilter(String statusFilter) {
    _activeFilters['status'] = statusFilter;
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

  void clearQueues() {
    _selectedPacketIds.clear();
    _ganQueue.clear();
    _jitterQueue.clear();
    _batchLabel = null;
    _consistencyChecked = false;
    _consistencyPassed = false;
    notifyListeners();
  }

  // --- 8. CORE API CALLS ---

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
          headers: headers, body: jsonEncode(body));
      final data = jsonDecode(response.body);
      return response.statusCode == 200 && await _verifyServerSignature(data);
    } catch (e) {
      return false;
    }
  }

  Future<Map<String, dynamic>> analyzeQueues() async {
    if (_ganQueue.isEmpty && _jitterQueue.isEmpty) return {};
    try {
      final body = {
        "gan_queue": _ganQueue.map((p) => {"id": p.id}).toList(),
        "jitter_queue": _jitterQueue.map((p) => {"id": p.id}).toList(),
      };
      final response = await http.post(
          Uri.parse('$BASE_URL/api/analyze_selection'),
          headers: headers,
          body: jsonEncode(body));
      final data = jsonDecode(response.body);
      if (response.statusCode == 200 && await _verifyServerSignature(data)) {
        return data['payload'];
      }
    } catch (e) {
      debugPrint("Analysis error: $e");
    }
    return {"passed": false, "error": "Security failure or offline"};
  }

  Future<void> fetchLabels() async {
    try {
      final response =
          await http.get(Uri.parse('$BASE_URL/api/labels'), headers: headers);
      final data = jsonDecode(response.body);
      if (response.statusCode == 200 && await _verifyServerSignature(data)) {
        _existingLabels = List<String>.from(data['payload']['labels']);
        notifyListeners();
      }
    } catch (e) {
      debugPrint("Label fetch error: $e");
      _existingLabels = ["BENIGN", "DDoS", "PortScan", "Bot", "Infiltration"];
      notifyListeners();
    }
  }

  // --- 9. PIPELINE CONTROL ---
  Future<void> startPipeline() async {
    if (_isRunning) return;
    _isRunning = true;
    notifyListeners();
    try {
      await http.post(Uri.parse('$BASE_URL/api/pipeline/start'),
          headers: headers);
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
          headers: headers);
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

  // --- 10. FETCHERS & UTILS ---

  Future<void> _fetchLiveSensoryData() async {
    try {
      final response = await http.get(Uri.parse('$BASE_URL/api/sensory/live'),
          headers: headers);
      final data = jsonDecode(response.body);
      if (response.statusCode == 200 && await _verifyServerSignature(data)) {
        final payload = data['payload'];
        _liveGnnAnomaly = (payload['gnn_anomaly'] ?? 0.0).toDouble();
        _liveMaeAnomaly = (payload['mae_anomaly'] ?? 0.0).toDouble();
        _liveStatus = payload['status'] ?? 'unknown';
        notifyListeners();
      }
    } catch (e) {
      debugPrint('Sensory error: $e');
    }
  }

  Future<void> _fetchRecentPackets() async {
    try {
      final response = await http.get(
          Uri.parse('$BASE_URL/api/packets/recent?limit=100'),
          headers: headers);
      final data = jsonDecode(response.body);
      if (response.statusCode == 200 && await _verifyServerSignature(data)) {
        for (var packetData in data['payload']['packets']) {
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
      final response = await http.get(
          Uri.parse('$BASE_URL/api/packets/recent?limit=50&offset=$offset'),
          headers: headers);
      final data = jsonDecode(response.body);
      if (response.statusCode == 200 && await _verifyServerSignature(data)) {
        for (var packetData in data['payload']['packets']) {
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
          await http.get(Uri.parse('$BASE_URL/api/stats'), headers: headers);
      final data = jsonDecode(response.body);
      if (response.statusCode == 200 && await _verifyServerSignature(data)) {
        final stats = data['payload'];
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
