// providers/ids_provider.dart
//
// Thin coordinator. Owns one instance of each helper class and exposes
// the public API that widgets consume via Provider/context.read/watch.
//
// File map:
//   ids_config.dart            — URL, API key, public key constants
//   ids_state.dart             — all mutable state + getters
//   ids_filter_controller.dart — filter / search / selection / queue mutations
//   ids_api_client.dart        — HTTP calls: retrain, analyze, labels, stats
//   ids_packet_fetcher.dart    — packet polling, mapping, pagination, sensory
//   ids_diagnostics.dart       — ECC sane-check + real data test

import 'package:flutter/foundation.dart';

import '../models/packet.dart';
import 'ids_api_client.dart';
import 'ids_diagnostics.dart';
import 'ids_filter_controller.dart';
import 'ids_packet_fetcher.dart';
import 'ids_state.dart';
import 'retrain_job_poller.dart';

class IdsProvider with ChangeNotifier {
  IdsProvider() {
    _state.addListener(notifyListeners);
    IdsDiagnostics.runECCSaneCheck();
    IdsDiagnostics.runRealDataTest();
    _initializeNotifications();
  }

  final IdsState _state = IdsState();
  final IdsApiClient _api = IdsApiClient();
  late final IdsPacketFetcher _fetcher = IdsPacketFetcher(_state);
  late final IdsFilterController _filters = IdsFilterController(_state);
  late final RetrainJobPoller _retrainPoller = RetrainJobPoller(_state);

  // ---------------------------------------------------------------------------
  // State passthrough getters
  // ---------------------------------------------------------------------------
  IdsApiClient get apiClient => _api;
  bool get isRunning => _state.isRunning;
  int get totalPackets => _state.totalPackets;
  int get normalCount => _state.normalCount;
  int get attackCount => _state.attackCount;
  int get zeroDayCount => _state.zeroDayCount;
  bool get isLoadingMore => _state.isLoadingMore;
  double get liveGnnAnomaly => _state.liveGnnAnomaly;
  double get liveMaeAnomaly => _state.liveMaeAnomaly;
  String get liveStatus => _state.liveStatus;
  String get currentFilter => _state.currentFilter;
  List<Packet> get filteredPackets => _state.filteredPackets;
  List<Packet> get ganQueue => _state.ganQueue;
  List<Packet> get jitterQueue => _state.jitterQueue;
  int get totalSelected => _state.totalSelected;
  String? get batchLabel => _state.batchLabel;
  bool get isNewAttack => _state.isNewAttack;
  bool get consistencyChecked => _state.consistencyChecked;
  bool get consistencyPassed => _state.consistencyPassed;
  List<String> get existingLabels => _state.existingLabels;
  bool isSelected(int id) => _filters.isSelected(id);
  String? get retrainJobId => _state.retrainJobId;
  String get retrainStatus => _state.retrainStatus;
  String get retrainPhase => _state.retrainPhase;
  int get retrainProgress => _state.retrainProgress;
  String? get retrainError => _state.retrainError;
  Map<String, dynamic> get retrainResults => _state.retrainResults;
  bool get isRetraining => _state.isRetraining;

  // ---------------------------------------------------------------------------
  // Filter & selection passthrough
  // ---------------------------------------------------------------------------

  void setFilter(String f) => _filters.setFilter(f);
  void updateSearchQuery(String q) => _filters.updateSearchQuery(q);
  void clearAllFilters() => _filters.clearAllFilters();
  void toggleSelection(Packet p, String q) => _filters.toggleSelection(p, q);
  void setBatchLabel(String? l, bool n) => _filters.setBatchLabel(l, n);
  void setConsistencyStatus(bool c, bool p) =>
      _filters.setConsistencyStatus(c, p);
  void clearQueues() => _filters.clearQueues();
  // Add to "State passthrough getters" section
  // State passthrough getters
  int get activeFilterCount => _state.activeFilterCount;
  String? get currentSeverityFilter => _state.currentSeverityFilter;

// Filter & selection passthrough
  void setSeverityFilter(String? s) => _filters.setSeverityFilter(s);

  Future<void> _initializeNotifications() async {
    await _api.setupNotifications();
  }
  // ---------------------------------------------------------------------------
  // Pipeline control
  // ---------------------------------------------------------------------------

  Future<void> startPipeline() async {
    if (_state.isRunning) return;
    _state.isRunning = true;
    final ok = await _api.startPipeline();
    if (ok) {
      _fetcher.startPolling();
    } else {
      _state.isRunning = false;
    }
  }

  Future<void> stopPipeline() async {
    _state.isRunning = false;
    _fetcher.stopPolling();
    await _api.stopPipeline();
  }

  List<String> _dmzIps = [];
  List<String> get dmzIps => _dmzIps;

  Future<void> loadDmzIps() async {
    _dmzIps = await apiClient.fetchDmzIps();
    notifyListeners();
  }

  Future<void> addDmzIp(String ip) async {
    if (ip.isEmpty || _dmzIps.contains(ip)) return;
    bool success = await apiClient.addDmzIp(ip);
    if (success) {
      _dmzIps.add(ip);
      notifyListeners();
    }
  }

  Future<void> removeDmzIp(String ip) async {
    bool success = await apiClient.removeDmzIp(ip);
    if (success) {
      _dmzIps.remove(ip);
      notifyListeners();
    }
  }

  // ---------------------------------------------------------------------------
  // Continual learning
  // ---------------------------------------------------------------------------

  Future<String?> sendRetrainRequest() async {
    final hasGan = _state.ganQueue.isNotEmpty;
    final hasJitter = _state.jitterQueue.isNotEmpty;

    if (!hasGan && !hasJitter) return null;
    if (_state.isRetraining) return _state.retrainJobId;

    // Jitter-only: no label needed, default to BENIGN
    // GAN: label must be set
    if (hasGan && _state.batchLabel == null) return null;

    return _retrainPoller.submitRetrainJob(
      ganQueue: _state.ganQueue
          .map((p) => {'id': p.id, 'status': p.status, 'summary': p.summary})
          .toList(),
      jitterQueue: _state.jitterQueue
          .map((p) => {'id': p.id, 'status': p.status, 'summary': p.summary})
          .toList(),
      targetLabel: hasGan ? _state.batchLabel! : 'BENIGN',
      isNewLabel: _state.isNewAttack,
    );
  }

  Future<void> cancelRetrain() => _retrainPoller.cancelJob();

  void clearRetrainJob() => _state.clearRetrainJob();

  Future<Map<String, dynamic>> analyzeQueues() async {
    if (_state.ganQueue.isEmpty && _state.jitterQueue.isEmpty) return {};
    return _api.analyzeQueues(
      ganQueue: _state.ganQueue,
      jitterQueue: _state.jitterQueue,
    );
  }

  Future<void> fetchLabels() async {
    _state.setExistingLabels(await _api.fetchLabels());
  }

  // ---------------------------------------------------------------------------
  // Packet helpers
  // ---------------------------------------------------------------------------

  Future<void> loadMorePackets() => _fetcher.loadMorePackets();
  void addZeroDayPacket() => _fetcher.addZeroDayPacket();

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------

  @override
  void dispose() {
    _fetcher.stopPolling();
    _state.removeListener(notifyListeners);
    _state.dispose();
    super.dispose();
    _retrainPoller.stopPolling();
  }
}
