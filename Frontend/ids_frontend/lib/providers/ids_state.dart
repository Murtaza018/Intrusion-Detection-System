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

  List<Packet> _buildFilteredList() {
    final status = _activeFilters['status'] as String;
    final query = (_activeFilters['search'] as String).toLowerCase();
    return _packets.where((p) {
      if (status != 'all' && p.status != status) return false;
      if (query.isEmpty) return true;
      return p.srcIp.contains(query) ||
          p.dstIp.contains(query) ||
          p.protocol.toLowerCase().contains(query);
    }).toList(growable: false);
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
}
