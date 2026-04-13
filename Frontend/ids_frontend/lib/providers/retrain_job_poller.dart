import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:hex/hex.dart';
import 'package:http/http.dart' as http;

import '../utils/crypto_utils.dart';
import '../utils/isolate_workers.dart';
import 'ids_config.dart';
import 'ids_state.dart';

/// Terminal statuses — polling stops when any of these is received.
const _terminalStatuses = {'done', 'failed', 'rolled_back', 'cancelled'};

class RetrainJobPoller {
  RetrainJobPoller(this._state);

  final IdsState _state;
  Timer? _pollTimer;

  Future<String?> submitRetrainJob({
    required List<Map<String, dynamic>> ganQueue,
    required List<Map<String, dynamic>> jitterQueue,
    required String targetLabel,
    required bool isNewLabel,
  }) async {
    try {
      final body = jsonEncode({
        'gan_queue': ganQueue,
        'jitter_queue': jitterQueue,
        'target_label': targetLabel,
        'is_new_label': isNewLabel,
      });

      final res = await http.post(
        Uri.parse('${IdsConfig.baseUrl}/api/retrain'),
        headers: IdsConfig.headers,
        body: body,
      );

      // 202 = accepted, 409 = already running
      final data = jsonDecode(utf8.decode(res.bodyBytes));
      final result = await _verifyInIsolate(res.bodyBytes);

      if (!result['success']) {
        debugPrint('[Retrain] Signature verification failed');
        _state.updateRetrainJob(
          status: 'failed',
          phase: 'signature_error',
          progress: 0,
          error: 'Server signature invalid',
        );
        return null;
      }

      final payload = data['payload'];

      if (res.statusCode == 409) {
        // Already running — start polling the existing job
        final current = payload['current'];
        _applyJobPayload(current);
        startPolling();
        return current['job_id'];
      }

      if (res.statusCode == 202) {
        _applyJobPayload(payload);
        startPolling();
        return payload['job_id'];
      }

      _state.updateRetrainJob(
        status: 'failed',
        phase: 'submit_error',
        progress: 0,
        error: payload['error'] ?? 'Unknown error',
      );
      return null;
    } catch (e) {
      debugPrint('[Retrain] Submit error: $e');
      _state.updateRetrainJob(
        status: 'failed',
        phase: 'submit_error',
        progress: 0,
        error: e.toString(),
      );
      return null;
    }
  }

  // ------------------------------------------------------------------
  // Polling
  // ------------------------------------------------------------------

  void startPolling() {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(seconds: 4), (_) => _poll());
    debugPrint('[Retrain] Polling started');
  }

  void stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
    debugPrint('[Retrain] Polling stopped');
  }

  Future<void> _poll() async {
    try {
      final res = await http.get(
        Uri.parse('${IdsConfig.baseUrl}/api/retrain/status'),
        headers: IdsConfig.headers,
      );

      final data = jsonDecode(utf8.decode(res.bodyBytes));
      final result = await _verifyInIsolate(res.bodyBytes);
      if (!result['success']) return;

      final payload = data['payload'];
      _applyJobPayload(payload);

      // Stop polling once the job reaches a terminal state
      if (_terminalStatuses.contains(payload['status'])) {
        stopPolling();
      }
    } catch (e) {
      debugPrint('[Retrain] Poll error: $e');
    }
  }

  // ------------------------------------------------------------------
  // Cancel
  // ------------------------------------------------------------------

  Future<void> cancelJob() async {
    stopPolling();
    try {
      await http.post(
        Uri.parse('${IdsConfig.baseUrl}/api/retrain/cancel'),
        headers: IdsConfig.headers,
      );
      _state.updateRetrainJob(
        status: 'cancelled',
        phase: 'cancelled',
        progress: 0,
      );
    } catch (e) {
      debugPrint('[Retrain] Cancel error: $e');
    }
  }

  // ------------------------------------------------------------------
  // Helpers
  // ------------------------------------------------------------------

  void _applyJobPayload(Map<String, dynamic> payload) {
    _state.updateRetrainJob(
      jobId: payload['job_id'],
      status: payload['status'] ?? 'unknown',
      phase: payload['phase'] ?? '',
      progress: (payload['progress'] as num?)?.toInt() ?? 0,
      error: payload['error'],
      retrainResults: payload['retrain_results'] != null
          ? Map<String, dynamic>.from(payload['retrain_results'])
          : null,
    );
  }

  Future<Map<String, dynamic>> _verifyInIsolate(Uint8List bodyBytes) {
    return compute(secureParserIsolate, {
      'bodyBytes': bodyBytes,
      'pubX': IdsConfig.pubXHex,
      'pubY': IdsConfig.pubYHex,
    });
  }
}
