// providers/ids_state.dart
//
// All mutable state + getters.
//
// Performance rules enforced here:
//   - Batch packet inserts: call beginBatch() / endBatch() instead of
//     prependPacket() in a loop — only ONE notifyListeners() fires per batch.
//   - _cachedFiltered is invalidated lazily; filteredPackets is O(n) only
//     when the packet list or filters actually changed.
//   - isLoadingMore and isProcessing are internal guards — they do NOT call
//     notifyListeners() because the UI doesn't render a loading spinner that
//     needs to react to them frame-by-frame.

import 'dart:async';

import 'package:flutter/foundation.dart';

import '../models/packet.dart';

class IdsState with ChangeNotifier {
  // Inside IdsState class
  String? _currentSeverityFilter;
  String? get currentSeverityFilter => _currentSeverityFilter;

  set currentSeverityFilter(String? value) {
    _currentSeverityFilter = value;
    notifyListeners();
  }

  int get activeFilterCount {
    int count = 0;
    if (currentFilter != 'all') count++;
    if (_currentSeverityFilter != null) count++;
    final query = _activeFilters['search'] as String;
    if (query.isNotEmpty) count++;
    return count;
  }

  // 2. Add a specific setter for Search Query since it's inside a Map
  void setSearchQuery(String query) {
    _activeFilters['search'] = query;
    _invalidateFilterCache();
    notifyListeners();
  }
  // ---------------------------------------------------------------------------
  // Pipeline
  // ---------------------------------------------------------------------------

  bool _isRunning = false;
  bool get isRunning => _isRunning;
  set isRunning(bool v) {
    if (_isRunning == v) return;
    _isRunning = v;
    notifyListeners();
  }

  Timer? packetTimer;
  Timer? sensoryTimer;

  // ---------------------------------------------------------------------------
  // Packet list
  // ---------------------------------------------------------------------------

  final List<Packet> _packets = [];

  /// Unmodifiable view — never mutate this directly.
  List<Packet> get packets => List.unmodifiable(_packets);

  final Set<int> processedPacketIds = {};

  // Internal-only flags — no notifyListeners needed
  bool isProcessing = false;
  bool isLoadingMore = false;

  // ---------------------------------------------------------------------------
  // Batch insert API
  // (Use this instead of calling prependPacket in a loop)
  // ---------------------------------------------------------------------------

  bool _inBatch = false;
  bool _batchHadChanges = false;

  /// Call before adding multiple packets. Suppresses notifications until [endBatch].
  void beginBatch() {
    _inBatch = true;
    _batchHadChanges = false;
  }

  /// Call after batch inserts. Fires ONE notifyListeners if anything changed.
  void endBatch() {
    _inBatch = false;
    if (_batchHadChanges) {
      _invalidateFilterCache();
      notifyListeners();
    }
    _batchHadChanges = false;
  }

  /// Inserts at the front and trims to [maxSize].
  /// Inside a batch this is silent; outside it notifies immediately.
  void prependPacket(Packet packet, {int maxSize = 200}) {
    _packets.insert(0, packet);
    if (_packets.length > maxSize)
      _packets.removeRange(maxSize, _packets.length);
    _invalidateFilterCache();
    if (_inBatch) {
      _batchHadChanges = true;
    } else {
      notifyListeners();
    }
  }

  /// Appends to the end (pagination).
  void appendPacket(Packet packet) {
    _packets.add(packet);
    _invalidateFilterCache();
    notifyListeners();
  }

  /// In-place replacement — used by upsert.
  void updatePacketAt(int index, Packet packet) {
    _packets[index] = packet;
    _invalidateFilterCache();
    notifyListeners();
  }

  /// Fire a notification without changing data (use sparingly).
  void markDirty() {
    _invalidateFilterCache();
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Stats counters
  // ---------------------------------------------------------------------------

  int _totalPackets = 0;
  int _normalCount = 0;
  int _attackCount = 0;
  int _zeroDayCount = 0;

  int get totalPackets => _totalPackets;
  int get normalCount => _normalCount;
  int get attackCount => _attackCount;
  int get zeroDayCount => _zeroDayCount;
  // Add this alongside your other getters
  List<Packet> get allPackets => _packets;

  void updateStats({
    required int total,
    required int normal,
    required int attack,
    required int zeroDay,
  }) {
    // Skip the rebuild if nothing actually changed
    if (_totalPackets == total &&
        _normalCount == normal &&
        _attackCount == attack &&
        _zeroDayCount == zeroDay) return;

    _totalPackets = total;
    _normalCount = normal;
    _attackCount = attack;
    _zeroDayCount = zeroDay;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Sensory (gauge) state
  // ---------------------------------------------------------------------------

  double _liveGnnAnomaly = 0.0;
  double _liveMaeAnomaly = 0.0;
  String _liveStatus = 'unknown';

  double get liveGnnAnomaly => _liveGnnAnomaly;
  double get liveMaeAnomaly => _liveMaeAnomaly;
  String get liveStatus => _liveStatus;

  void updateSensory({
    required double gnn,
    required double mae,
    required String status,
  }) {
    if (_liveGnnAnomaly == gnn &&
        _liveMaeAnomaly == mae &&
        _liveStatus == status) return;

    _liveGnnAnomaly = gnn;
    _liveMaeAnomaly = mae;
    _liveStatus = status;
    notifyListeners();
  }

  // ---------------------------------------------------------------------------
  // Filter & search — with lazy cache
  // ---------------------------------------------------------------------------

  Map<String, dynamic> _activeFilters = {
    'status': 'all',
    'search': '',
  };

  Map<String, dynamic> get activeFilters => Map.unmodifiable(_activeFilters);
  String get currentFilter => _activeFilters['status'] as String;

  // Lazy filter cache — rebuilt only when packets or filters change
  List<Packet>? _cachedFiltered;

  void _invalidateFilterCache() => _cachedFiltered = null;

  void setActiveFilters(Map<String, dynamic> filters) {
    _activeFilters = filters;
    _invalidateFilterCache();
    notifyListeners();
  }

  List<Packet> get filteredPackets {
    _cachedFiltered ??= _buildFilteredList();
    return _cachedFiltered!;
  }

  // Inside IdsState class in ids_state.dart

  List<Packet> _buildFilteredList() {
    // 1. Get current status and query from your existing Map/fields
    final status = _activeFilters['status'] as String;
    final query = (_activeFilters['search'] as String).toLowerCase();

    return _packets.where((p) {
      // 2. Type/Status Filter (Using p.status from your Packet model)
      if (status != 'all' && p.status != status) return false;

      // 3. Severity Filter (Using p.gnnAnomaly from your Packet model)
      if (_currentSeverityFilter != null) {
        final score = p.gnnAnomaly;
        final passes = switch (_currentSeverityFilter) {
          'low' => score >= 0.1 && score < 0.3,
          'medium' => score >= 0.3 && score < 0.6,
          'high' => score >= 0.6 && score < 0.8,
          'critical' => score >= 0.8,
          _ => true,
        };
        if (!passes) return false;
      }

      // 4. Search Query logic
      if (query.isNotEmpty) {
        return _matchesSearch(p, query);
      }

      return true;
    }).toList(growable: false);
  }

  bool _matchesSearch(Packet packet, String query) {
    for (final token in query.split(' ')) {
      if (token.isEmpty) continue;

      if (token.startsWith('ip:')) {
        final val = token.substring(3);
        if (!packet.srcIp.contains(val) && !packet.dstIp.contains(val))
          return false;
      } else if (token.startsWith('port:')) {
        final p = token.substring(5);
        if (packet.srcPort.toString() != p && packet.dstPort.toString() != p)
          return false;
      } else if (token.startsWith('proto:')) {
        if (packet.protocol.toLowerCase() != token.substring(6)) return false;
      } else if (token.startsWith('anomaly:>')) {
        final threshold = double.tryParse(token.substring(9)) ?? 0;
        if (packet.gnnAnomaly <= threshold) return false;
      } else if (token.startsWith('anomaly:<')) {
        final threshold = double.tryParse(token.substring(9)) ?? 1;
        if (packet.gnnAnomaly >= threshold) return false;
      } else {
        if (!packet.srcIp.contains(query) &&
            !packet.dstIp.contains(query) &&
            !packet.protocol.toLowerCase().contains(query) &&
            !packet.summary.toLowerCase().contains(query)) return false;
      }
    }
    return true;
  }

  // ---------------------------------------------------------------------------
  // Selection & continual learning queues
  // ---------------------------------------------------------------------------

  final Set<int> selectedPacketIds = {};
  final List<Packet> ganQueue = [];
  final List<Packet> jitterQueue = [];

  String? _batchLabel;
  bool _isNewAttack = false;
  bool _consistencyChecked = false;
  bool _consistencyPassed = false;
  List<String> _existingLabels = [];

  String? get batchLabel => _batchLabel;
  bool get isNewAttack => _isNewAttack;
  bool get consistencyChecked => _consistencyChecked;
  bool get consistencyPassed => _consistencyPassed;
  List<String> get existingLabels => List.unmodifiable(_existingLabels);
  int get totalSelected => ganQueue.length + jitterQueue.length;

  void setBatchLabel(String? label, bool isNew) {
    _batchLabel = label;
    _isNewAttack = isNew;
    _consistencyChecked = false;
    _consistencyPassed = false;
    notifyListeners();
  }

  void setConsistencyStatus(bool checked, bool passed) {
    if (_consistencyChecked == checked && _consistencyPassed == passed) return;
    _consistencyChecked = checked;
    _consistencyPassed = passed;
    notifyListeners();
  }

  void setExistingLabels(List<String> labels) {
    _existingLabels = labels;
    notifyListeners();
  }

// ---------------------------------------------------------------------------
// Retrain job state  (updated by RetrainJobPoller)
// ---------------------------------------------------------------------------

  String? _retrainJobId = null;
  String _retrainStatus = 'idle';
  String _retrainPhase = '';
  int _retrainProgress = 0;
  String? _retrainError = null;
  Map<String, dynamic> _retrainResults = {};

  String? get retrainJobId => _retrainJobId;
  String get retrainStatus => _retrainStatus;
  String get retrainPhase => _retrainPhase;
  int get retrainProgress => _retrainProgress;
  String? get retrainError => _retrainError;
  Map<String, dynamic> get retrainResults => Map.unmodifiable(_retrainResults);

  bool get isRetraining =>
      _retrainStatus == 'queued' || _retrainStatus == 'running';

  void updateRetrainJob({
    String? jobId,
    required String status,
    required String phase,
    required int progress,
    String? error,
    Map<String, dynamic>? retrainResults,
  }) {
    _retrainJobId = jobId ?? _retrainJobId;
    _retrainStatus = status;
    _retrainPhase = phase;
    _retrainProgress = progress;
    _retrainError = error;
    if (retrainResults != null) _retrainResults = retrainResults;
    notifyListeners();
  }

  void clearRetrainJob() {
    _retrainJobId = null;
    _retrainStatus = 'idle';
    _retrainPhase = '';
    _retrainProgress = 0;
    _retrainError = null;
    _retrainResults = {};
    notifyListeners();
  }
}
