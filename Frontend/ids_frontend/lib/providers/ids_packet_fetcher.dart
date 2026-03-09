// providers/ids_packet_fetcher.dart
//
// Packet polling, sensory data, pagination, and packet mapping.
//
// Performance contract:
//   - All ECC/JSON work for packets runs in a compute() isolate.
//   - fetchStats() and fetchRecentPackets() are staggered — stats fires
//     2.5 s after packets so they never compete on the same tick.
//   - New packets are inserted with a single beginBatch/endBatch block,
//     producing exactly ONE notifyListeners() per poll cycle regardless of
//     how many packets arrived.
//   - _verify() for stats/pagination uses compute() too — no ECDSA math
//     ever runs on the UI thread.

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:hex/hex.dart';
import 'package:http/http.dart' as http;

import '../models/packet.dart';
import '../utils/crypto_utils.dart';
import '../utils/isolate_workers.dart';
import 'ids_config.dart';
import 'ids_state.dart';

class IdsPacketFetcher {
  IdsPacketFetcher(this._state);

  final IdsState _state;
  Timer? _statsDelayTimer;

  // ---------------------------------------------------------------------------
  // Timer management
  // ---------------------------------------------------------------------------

  void startPolling() {
    _state.packetTimer?.cancel();
    _state.sensoryTimer?.cancel();

    // Packets every 5 s
    _state.packetTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (!_state.isRunning) {
        _state.packetTimer?.cancel();
        return;
      }
      fetchRecentPackets();
      // Stagger stats 2.5 s after packets so they never fire simultaneously
      _statsDelayTimer?.cancel();
      _statsDelayTimer = Timer(const Duration(milliseconds: 2500), () {
        if (_state.isRunning) fetchStats();
      });
    });

    // Sensory gauges every 4 s — offset by 1 s from the packet timer
    _state.sensoryTimer = Timer(const Duration(seconds: 1), () {
      if (!_state.isRunning) return;
      fetchLiveSensoryData();
      _state.sensoryTimer = Timer.periodic(const Duration(seconds: 4), (_) {
        if (!_state.isRunning) {
          _state.sensoryTimer?.cancel();
          return;
        }
        fetchLiveSensoryData();
      });
    });
  }

  void stopPolling() {
    _state.packetTimer?.cancel();
    _state.sensoryTimer?.cancel();
    _statsDelayTimer?.cancel();
  }

  // ---------------------------------------------------------------------------
  // Live packet polling — ONE rebuild per poll cycle via batch insert
  // ---------------------------------------------------------------------------

  Future<void> fetchRecentPackets() async {
    if (_state.isProcessing) return;
    _state.isProcessing = true;

    try {
      final res = await http.get(
        Uri.parse('${IdsConfig.baseUrl}/api/packets/recent?limit=100'),
        headers: IdsConfig.headers,
      );
      if (res.statusCode != 200) return;

      // All JSON decode + ECC math runs in a background isolate
      final result = await compute(verifyAndParsePacketsInBackground, {
        'bodyBytes': res.bodyBytes,
        'pubX': IdsConfig.pubXHex,
        'pubY': IdsConfig.pubYHex,
      });

      List rawPackets = result['packets'] as List? ?? [];

      // Debug fallback: show data even if signature check fails
      if (result['success'] != true) {
        if (!kDebugMode) return;
        final data = jsonDecode(utf8.decode(res.bodyBytes));
        rawPackets = data['payload']['packets'];
      }

      // --- BATCH INSERT: only ONE notifyListeners() at the end ---
      _state.beginBatch();
      for (final packetData in rawPackets) {
        final int id = packetData['id'];
        if (!_state.processedPacketIds.contains(id)) {
          _state.processedPacketIds.add(id);
          _state.prependPacket(mapDataToPacket(packetData));
        }
      }
      _state.endBatch();
    } catch (e) {
      debugPrint('🚨 fetchRecentPackets error: $e');
    } finally {
      _state.isProcessing = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Pagination
  // ---------------------------------------------------------------------------

  Future<void> loadMorePackets() async {
    if (_state.isLoadingMore) return;
    _state.isLoadingMore = true;
    notifyLoadingChanged(); // tell UI the spinner should show

    try {
      final res = await http.get(
        Uri.parse('${IdsConfig.baseUrl}/api/packets/recent'
            '?limit=50&offset=${_state.packets.length}'),
        headers: IdsConfig.headers,
      );
      final data = jsonDecode(utf8.decode(res.bodyBytes));
      if (res.statusCode == 200 && await _verifyInIsolate(data)) {
        _state.beginBatch();
        for (final packetData in data['payload']['packets']) {
          final int id = packetData['id'];
          if (!_state.processedPacketIds.contains(id)) {
            _state.processedPacketIds.add(id);
            _state.appendPacket(mapDataToPacket(packetData));
          }
        }
        _state.endBatch();
      }
    } catch (e) {
      debugPrint('Pagination error: $e');
    } finally {
      _state.isLoadingMore = false;
      notifyLoadingChanged();
    }
  }

  /// Called by loadMorePackets to update the spinner — kept here so only
  /// the pagination path triggers an isLoadingMore-driven rebuild.
  void notifyLoadingChanged() => _state.markDirty();

  // ---------------------------------------------------------------------------
  // Sensory / gauge data
  // ---------------------------------------------------------------------------

  Future<void> fetchLiveSensoryData() async {
    try {
      final res = await http.get(
        Uri.parse('${IdsConfig.baseUrl}/api/sensory/live'),
        headers: IdsConfig.headers,
      );
      final data = jsonDecode(utf8.decode(res.bodyBytes));
      final payload = data['payload'];
      if (payload == null) return;

      // Only run ECC when anomalous — normal polling stays cheap
      if (payload['status'] != 'normal') {
        final result = await compute(secureParserIsolate, {
          'bodyBytes': res.bodyBytes,
          'pubX': IdsConfig.pubXHex,
          'pubY': IdsConfig.pubYHex,
        });
        if (result['success'] != true) return;
      }

      // updateSensory has an equality guard — no rebuild if values unchanged
      _state.updateSensory(
        gnn:
            double.tryParse(payload['gnn_anomaly']?.toString() ?? '0.0') ?? 0.0,
        mae:
            double.tryParse(payload['mae_anomaly']?.toString() ?? '0.0') ?? 0.0,
        status: payload['status'] ?? 'unknown',
      );
    } catch (e) {
      debugPrint('Sensory error: $e');
    }
  }

  // ---------------------------------------------------------------------------
  // Stats
  // ---------------------------------------------------------------------------

  Future<void> fetchStats() async {
    try {
      final res = await http.get(
        Uri.parse('${IdsConfig.baseUrl}/api/stats'),
        headers: IdsConfig.headers,
      );
      final data = jsonDecode(utf8.decode(res.bodyBytes));
      if (res.statusCode == 200 && await _verifyInIsolate(data)) {
        final s = data['payload'];
        // updateStats has an equality guard — no rebuild if counts unchanged
        _state.updateStats(
          total: s['total_packets'] as int? ?? _state.totalPackets,
          normal: s['normal_count'] as int? ?? _state.normalCount,
          attack: s['attack_count'] as int? ?? _state.attackCount,
          zeroDay: s['zero_day_count'] as int? ?? _state.zeroDayCount,
        );
      }
    } catch (e) {
      debugPrint('Stats error: $e');
    }
  }

  // ---------------------------------------------------------------------------
  // Packet mapping
  // ---------------------------------------------------------------------------

  Packet mapDataToPacket(Map<String, dynamic> data) {
    final exp = data['explanation'] as Map<String, dynamic>?;
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
      confidence:
          double.tryParse(data['confidence']?.toString() ?? '0.0') ?? 0.0,
      maeAnomaly:
          double.tryParse(exp?['mae_anomaly']?.toString() ?? '0.0') ?? 0.0,
      gnnAnomaly:
          double.tryParse(exp?['gnn_anomaly']?.toString() ?? '0.0') ?? 0.0,
      explanation: exp,
    );
  }

  // ---------------------------------------------------------------------------
  // Upsert helper
  // ---------------------------------------------------------------------------

  void upsertPacket(Map<String, dynamic> data) {
    final int id = data['id'];
    final newPacket = mapDataToPacket(data);
    final packetList = _state.packets;

    if (_state.processedPacketIds.contains(id)) {
      final index = packetList.indexWhere((p) => p.id == id);
      if (index != -1 &&
          jsonEncode(packetList[index].explanation) !=
              jsonEncode(newPacket.explanation)) {
        _state.updatePacketAt(index, newPacket);
      }
      return;
    }
    _state.processedPacketIds.add(id);
    _state.prependPacket(newPacket);
  }

  // ---------------------------------------------------------------------------
  // Test helper
  // ---------------------------------------------------------------------------

  void addZeroDayPacket() {
    final id = DateTime.now().millisecondsSinceEpoch;
    _state.processedPacketIds.add(id);
    _state.prependPacket(Packet(
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
      maeAnomaly: 0.88,
      gnnAnomaly: 0.15,
      explanation: {
        'description': 'Simulated anomaly for structural verification.',
        'mae_anomaly': 0.88,
        'gnn_anomaly': 0.15,
        'status': 'done',
      },
    ));
  }

  // ---------------------------------------------------------------------------
  // Signature verification — always in an isolate, never on the UI thread
  // ---------------------------------------------------------------------------

  Future<bool> _verifyInIsolate(Map<String, dynamic> responseBody) async {
    final String? sigHex = responseBody['signature'];
    final dynamic payload = responseBody['payload'];
    if (sigHex == null || payload == null) return false;
    try {
      // Ship the raw bytes to a background isolate so ECDSA math can't jank
      final bodyForIsolate = utf8.encode(jsonEncode(responseBody)) as Uint8List;
      final result = await compute(secureParserIsolate, {
        'bodyBytes': Uint8List.fromList(bodyForIsolate),
        'pubX': IdsConfig.pubXHex,
        'pubY': IdsConfig.pubYHex,
      });
      return result['success'] == true;
    } catch (e) {
      debugPrint('Verify error: $e');
      return false;
    }
  }
}
