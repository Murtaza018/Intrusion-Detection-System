import 'dart:collection';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'package:cryptography/cryptography.dart';
import 'package:hex/hex.dart'; // Ensure you added 'hex: ^0.2.0' to pubspec.yaml
import 'package:pointycastle/export.dart' as pc;
import 'package:pointycastle/ecc/curves/secp256r1.dart';
import 'package:crypto/crypto.dart';
import 'dart:convert';

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
    runRealDataTest();
  }
  // --- 1. SECURE CONFIGURATION ---
  static const String BASE_URL = "http://127.0.0.1:5001";
  static const String apiKey =
      String.fromEnvironment('API_KEY', defaultValue: 'MySuperSecretKey12345!');

  // --- ECC PUBLIC KEY COORDINATES (Extracted from your cert.pem) ---
  static const String _pubXHex =
      "582110e3c4af92f5f2d2ae042be8dd44d67e51d1a4728986874e0fcd64829253";
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
      // FIX: use bodyBytes
      final data = jsonDecode(utf8.decode(response.bodyBytes));
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
      // FIX: use bodyBytes
      final data = jsonDecode(utf8.decode(response.bodyBytes));
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
      // FIX: use bodyBytes
      final data = jsonDecode(utf8.decode(response.bodyBytes));
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
      // FIX: use bodyBytes
      final data = jsonDecode(utf8.decode(response.bodyBytes));
      if (response.statusCode == 200 && await _verifyServerSignature(data)) {
        final payload = data['payload'];
        // Convert the String from the backend back into a Double for the UI
        _liveGnnAnomaly =
            double.tryParse(payload['gnn_anomaly']?.toString() ?? '0.0') ?? 0.0;
        _liveMaeAnomaly =
            double.tryParse(payload['mae_anomaly']?.toString() ?? '0.0') ?? 0.0;
        _liveStatus = payload['status'] ?? 'unknown';
        notifyListeners();
      }
    } catch (e) {
      debugPrint('Sensory error: $e');
    }
  }

  bool _isProcessing = false; // Add this class variable

  Future<void> _fetchRecentPackets() async {
    if (_isProcessing) return; // Skip if we are already busy verifying
    _isProcessing = true;

    try {
      final response = await http.get(
          Uri.parse('$BASE_URL/api/packets/recent?limit=100'),
          headers: headers);

      if (response.statusCode == 200) {
        // 1. SEND RAW BYTES TO ISOLATE IMMEDIATELY
        // This stops the UI from lagging because jsonDecode happens in the background
        final result = await compute(_verifyAndParseInBackground, {
          'bodyBytes': response.bodyBytes,
          'pubX': _pubXHex,
          'pubY': _pubYHex,
        });

        if (result['success'] == true) {
          final List packets = result['packets'];
          if (result['success'] == true) {
            final List packets = result['packets'];

            for (var packetData in packets) {
              if (packetData['confidence'] is String) {
                packetData['confidence'] = double.tryParse(
                        packetData['confidence']?.toString() ?? '0.0') ??
                    0.0;
              }

              if (packetData['explanation'] != null) {
                var exp = packetData['explanation'];

                exp['gnn_anomaly'] =
                    double.tryParse(exp['gnn_anomaly'].toString()) ?? 0.0;
                exp['mae_anomaly'] =
                    double.tryParse(exp['mae_anomaly'].toString()) ?? 0.0;
              }
              // -----------------------

              _addPacketFromApi(packetData);
            }
            notifyListeners();
          }
        }
      }
    } catch (e) {
      print("🚨 Fetch Error: $e");
    } finally {
      _isProcessing = false; // Release the lock
    }
  }

  Future<Map<String, dynamic>> _verifyAndParseInBackground(
      Map<String, dynamic> params) async {
    try {
      // 1. Decode UTF8 and JSON in the background
      final String rawBody = utf8.decode(params['bodyBytes']);
      final Map<String, dynamic> data = jsonDecode(rawBody);

      final String? signatureHex = data['signature'];
      final Map<String, dynamic>? payload = data['payload'];

      if (signatureHex == null || payload == null) return {'success': false};

      // 2. ECC Verification logic
      // Create the string variable first so you can use it for the debug print
      final dynamic sortedPayload = _toSortedMap(payload);
      final String jsonString = jsonEncode(sortedPayload);

      final msgBytes = Uint8List.fromList(utf8.encode(jsonString));
      final sigBytes = Uint8List.fromList(HEX.decode(signatureHex));

      // --- ECC DEBUG START ---
      // print("--- ECC DEBUG START ---");
      // print("RAW BYTES TO VERIFY (HEX): ${HEX.encode(msgBytes)}");
      // print(
      //     "FIRST 50 CHARS: ${jsonString.substring(0, jsonString.length.clamp(0, 50))}");
      // print("--- ECC DEBUG END ---");
      // --- ECC DEBUG END ---

      bool isValid =
          _ecdsaVerifyRaw(msgBytes, sigBytes, params['pubX'], params['pubY']);

      return {
        'success': isValid,
        'packets': isValid ? payload['packets'] : [],
      };
    } catch (e) {
      print(
          "🛡️ ECC Isolate Error: $e"); // Added print to catch errors like missing variables
      return {'success': false};
    }
  }

  Future<bool> _verifyServerSignature(Map<String, dynamic> responseBody) async {
    final String? signatureHex = responseBody['signature'];

    // Extract ONLY the payload. Do not include server_time or signature_type!
    final dynamic payload = responseBody['payload'];

    if (signatureHex == null || payload == null) return false;

    try {
      // Match Python: sort_keys=True, separators=(',', ':')
      // jsonEncode in Dart defaults to no spaces (',',':') which matches Python's separators
      final sortedPayload = _toSortedMap(payload);
      final jsonString = jsonEncode(sortedPayload);

      // Calculate hash for debugging
      final msgBytes = utf8.encode(jsonString);
      final localHash = sha256.convert(msgBytes);

      // print("🛡️ ECC: JSON to Verify: $jsonString");
      print("🛡️ ECC: Local Hash: $localHash");

      final sigBytes = Uint8List.fromList(HEX.decode(signatureHex));

      // Pass the raw bytes of the JSON string to your verification function
      return _ecdsaVerify(Uint8List.fromList(msgBytes), sigBytes);
    } catch (e) {
      print("🛡️ ECC: Error: $e");
      return false;
    }
  }

// Helper to ensure keys are sorted alphabetically (matching Python's sort_keys=True)
  dynamic _toSortedMap(dynamic item) {
    if (item is Map) {
      final sortedKeys = item.keys.toList()..sort();
      final result = <String, dynamic>{};
      for (var key in sortedKeys) {
        result[key] = _toSortedMap(item[key]);
      }
      return result;
    } else if (item is List) {
      return item.map((e) => _toSortedMap(e)).toList();
    }
    return item;
  }

// Recursively sorts all map keys, matching Python's sort_keys=True
  String _toSortedJson(dynamic value) {
    if (value is Map) {
      final sorted = SplayTreeMap<String, dynamic>.from(
          value.map((k, v) => MapEntry(k.toString(), v)));
      final buffer = StringBuffer('{');
      var first = true;
      for (final entry in sorted.entries) {
        if (!first) buffer.write(',');
        buffer.write(jsonEncode(entry.key));
        buffer.write(':');
        buffer.write(_toSortedJson(entry.value));
        first = false;
      }
      buffer.write('}');
      return buffer.toString();
    } else if (value is List) {
      final buffer = StringBuffer('[');
      var first = true;
      for (final item in value) {
        if (!first) buffer.write(',');
        buffer.write(_toSortedJson(item));
        first = false;
      }
      buffer.write(']');
      return buffer.toString();
    } else {
      return jsonEncode(value);
    }
  }

  bool _ecdsaVerify(Uint8List message, Uint8List sigBytes) {
    try {
      // 1. Setup Curve (P-256)
      // PointyCastle uses 'prime256v1' for the SECP256R1 curve
      final domainParams = pc.ECDomainParameters('prime256v1');
      final x = BigInt.parse(_pubXHex, radix: 16);
      final y = BigInt.parse(_pubYHex, radix: 16);

      final point = domainParams.curve.createPoint(x, y);
      final pubKey = pc.ECPublicKey(point, domainParams);

      // 2. Hash the message exactly once
      final digest = pc.SHA256Digest();
      final hash = digest.process(message);

      // 3. Extract R and S (Big-endian 32-bytes each)
      final r = BigInt.parse(HEX.encode(sigBytes.sublist(0, 32)), radix: 16);
      BigInt s = BigInt.parse(HEX.encode(sigBytes.sublist(32, 64)), radix: 16);

      // 4. Normalize S value (Malleability Check)
      // Python's cryptography library often enforces Low-S signatures
      if (s > (domainParams.n >> 1)) {
        s = domainParams.n - s;
      }
      final ecSig = pc.ECSignature(r, s);

      // 5. Initialize Signer
      // We pass null for the digest algorithm because we are passing the
      // pre-computed 'hash' directly to verifySignature.
      final signer = pc.ECDSASigner(null, pc.HMac(pc.SHA256Digest(), 64));
      signer.init(false, pc.PublicKeyParameter(pubKey));

      // 6. Verify against the pre-calculated hash
      final isValid = signer.verifySignature(hash, ecSig);
      print("🛡️ ECC: Internal PointyCastle result: $isValid");

      return isValid;
    } catch (e) {
      print("🛡️ ECC: PointyCastle Error: $e");
      return false;
    }
  }

  Future<bool> _verifyInBackground(Map<String, dynamic> params) async {
    final responseBody = params['responseBody'];
    final String pubX = params['pubX'];
    final String pubY = params['pubY'];

    final String? signatureHex = responseBody['signature'];
    final dynamic payload = responseBody['payload'];

    if (signatureHex == null || payload == null) return false;

    try {
      final sortedPayload = _toSortedMap(payload);
      final jsonString = jsonEncode(sortedPayload);
      final msgBytes = Uint8List.fromList(utf8.encode(jsonString));

      print("--- ECC DEBUG START ---");
      print("RAW BYTES TO VERIFY (HEX): ${HEX.encode(msgBytes)}");
      print("FIRST 50 CHARS: ${jsonString.substring(0, 50)}");
      print("--- ECC DEBUG END ---");

      // FIX: Decode the hex string into Uint8List here
      final sigBytes = Uint8List.fromList(HEX.decode(signatureHex));

      // Now pass sigBytes (Uint8List) instead of signatureHex (String)
      return _ecdsaVerifyRaw(msgBytes, sigBytes, pubX, pubY);
    } catch (e) {
      print("🛡️ ECC Isolate Error: $e");
      return false;
    }
  }

  bool _ecdsaVerifyRaw(
      Uint8List message, Uint8List sigBytes, String pubXHex, String pubYHex) {
    try {
      final domainParams = pc.ECDomainParameters('prime256v1');
      final x = BigInt.parse(pubXHex, radix: 16);
      final y = BigInt.parse(pubYHex, radix: 16);

      final point = domainParams.curve.createPoint(x, y);
      final pubKey = pc.ECPublicKey(point, domainParams);

      // 1. Get the hash bytes directly from the SHA256 library
      final hash = sha256.convert(message).bytes;

      // 2. Extract R and S
      final r = BigInt.parse(HEX.encode(sigBytes.sublist(0, 32)), radix: 16);
      BigInt s = BigInt.parse(HEX.encode(sigBytes.sublist(32, 64)), radix: 16);

      // 3. Normalization (Low-S) is CRITICAL for Python compatibility
      if (s > (domainParams.n >> 1)) {
        s = domainParams.n - s;
      }

      // 4. Use a FRESH signer instance every time
      // We pass null for the digest because we are providing the 'hash' (the digest) ourselves
      final signer = pc.ECDSASigner(null, pc.HMac(pc.SHA256Digest(), 64));
      signer.init(false, pc.PublicKeyParameter(pubKey));

      // 5. Explicitly cast hash to Uint8List to avoid any List<int> vs Uint8List confusion
      final result = signer.verifySignature(
          Uint8List.fromList(hash), pc.ECSignature(r, s));

      print("🛡️ ECC Isolate: Internal Math Result: $result");
      return result;
    } catch (e) {
      print("🛡️ ECC Isolate: PointyCastle Math Error: $e");
      return false;
    }
  }

  Future<void> runRealDataTest() async {
    debugPrint("🧪 Starting Real Data ECC Test...");
    try {
      // Fresh pair from Python logs
      const String testSigHex =
          "a0666b19bbd5a724f82acdf490b62e35f5b490c5a21ac0a87f83726148033da3f02cc854c035a0f835352590a8e257ca2e9d01b337b94fec2a575162a4921c6c";
      const String testJsonStr =
          '{"gnn_anomaly":0.0,"mae_anomaly":0.062300905585289,"status":"normal"}';

      final sigBytes = Uint8List.fromList(HEX.decode(testSigHex));
      final msgBytes = Uint8List.fromList(utf8.encode(testJsonStr));

      final domainParams = pc.ECDomainParameters('prime256v1');
      final x = BigInt.parse(_pubXHex, radix: 16);
      final y = BigInt.parse(_pubYHex, radix: 16);
      final point = domainParams.curve.createPoint(x, y);
      final pubKey = pc.ECPublicKey(point, domainParams);

      final digest = pc.SHA256Digest();
      final hash = digest.process(msgBytes);
      debugPrint("🧪 Hash: ${HEX.encode(hash)}");

      final r = BigInt.parse(HEX.encode(sigBytes.sublist(0, 32)), radix: 16);
      final s = BigInt.parse(HEX.encode(sigBytes.sublist(32, 64)), radix: 16);

      final signer = pc.ECDSASigner(null);
      signer.init(false, pc.PublicKeyParameter(pubKey));
      final result = signer.verifySignature(hash, pc.ECSignature(r, s));

      debugPrint("🧪 Real Data Test Result: $result");
    } catch (e, stack) {
      debugPrint("🧪 Error: $e");
      debugPrint("🧪 Stack: $stack");
    }
  }

  Future<void> runECCSaneCheck() async {
    debugPrint("🧪 Starting ECC Sanity Test...");
    // Simple check - just verify the pointycastle import works
    try {
      final curve = pc.ECCurve_secp256r1();
      debugPrint(curve != null ? "✅ ECC SANITY PASSED" : "❌ ECC SANITY FAILED");
    } catch (e) {
      debugPrint("❌ ECC ERROR: $e");
    }
  }

  // Converts BigInt to exactly 32 bytes without sign padding
  Uint8List _bigIntToBytes(BigInt number) {
    final hex = number.toRadixString(16).padLeft(64, '0');
    return Uint8List.fromList(HEX.decode(hex));
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
      // FIX: use bodyBytes
      final data = jsonDecode(utf8.decode(response.bodyBytes));
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
      // FIX: use bodyBytes
      final data = jsonDecode(utf8.decode(response.bodyBytes));
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
