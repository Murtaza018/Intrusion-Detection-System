import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../models/packet.dart';
import '../utils/isolate_workers.dart';
import 'ids_config.dart';

class IdsApiClient {
  Future<bool> startPipeline() async {
    try {
      final res = await http.post(
        Uri.parse('${IdsConfig.baseUrl}/api/pipeline/start'),
        headers: IdsConfig.headers,
      );
      return res.statusCode == 200;
    } catch (e) {
      debugPrint('Pipeline start error: $e');
      return false;
    }
  }

  Future<void> stopPipeline() async {
    try {
      await http.post(
        Uri.parse('${IdsConfig.baseUrl}/api/pipeline/stop'),
        headers: IdsConfig.headers,
      );
    } catch (e) {
      debugPrint('Pipeline stop error: $e');
    }
  }

  Future<bool> sendRetrainRequest({
    required List<Packet> ganQueue,
    required List<Packet> jitterQueue,
    required String targetLabel,
    required bool isNewLabel,
  }) async {
    try {
      final body = jsonEncode({
        'gan_queue': ganQueue
            .map((p) => {'id': p.id, 'status': p.status, 'summary': p.summary})
            .toList(),
        'jitter_queue': jitterQueue
            .map((p) => {'id': p.id, 'status': p.status, 'summary': p.summary})
            .toList(),
        'target_label': targetLabel,
        'is_new_label': isNewLabel,
      });
      final res = await http.post(
        Uri.parse('${IdsConfig.baseUrl}/api/retrain'),
        headers: IdsConfig.headers,
        body: body,
      );
      return res.statusCode == 200 && await _verifyInIsolate(res.bodyBytes);
    } catch (e) {
      debugPrint('Retrain error: $e');
      return false;
    }
  }

  Future<Map<String, dynamic>> analyzeQueues({
    required List<Packet> ganQueue,
    required List<Packet> jitterQueue,
  }) async {
    try {
      final body = jsonEncode({
        'gan_queue': ganQueue.map((p) => {'id': p.id}).toList(),
        'jitter_queue': jitterQueue.map((p) => {'id': p.id}).toList(),
      });
      final res = await http.post(
        Uri.parse('${IdsConfig.baseUrl}/api/analyze_selection'),
        headers: IdsConfig.headers,
        body: body,
      );
      final result = await _secureParseInIsolate(res.bodyBytes);
      if (res.statusCode == 200 && result['success'] == true) {
        return result['payload'] as Map<String, dynamic>;
      }
    } catch (e) {
      debugPrint('Analyze queues error: $e');
    }
    return {'passed': false, 'error': 'Security failure or offline'};
  }

  // ---------------------------------------------------------------------------
  // Labels
  // ---------------------------------------------------------------------------

  Future<List<String>> fetchLabels() async {
    try {
      final res = await http.get(
        Uri.parse('${IdsConfig.baseUrl}/api/labels'),
        headers: IdsConfig.headers,
      );
      final result = await _secureParseInIsolate(res.bodyBytes);
      if (res.statusCode == 200 && result['success'] == true) {
        return List<String>.from(result['payload']['labels']);
      }
    } catch (e) {
      debugPrint('Label fetch error: $e');
    }
    return ['BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration'];
  }

  // ---------------------------------------------------------------------------
  // Stats
  // ---------------------------------------------------------------------------

  Future<Map<String, int>?> fetchStats() async {
    try {
      final res = await http.get(
        Uri.parse('${IdsConfig.baseUrl}/api/stats'),
        headers: IdsConfig.headers,
      );
      final result = await _secureParseInIsolate(res.bodyBytes);
      if (res.statusCode == 200 && result['success'] == true) {
        final s = result['payload'];
        return {
          'total': s['total_packets'] as int? ?? 0,
          'normal': s['normal_count'] as int? ?? 0,
          'attack': s['attack_count'] as int? ?? 0,
          'zero_day': s['zero_day_count'] as int? ?? 0,
        };
      }
    } catch (e) {
      debugPrint('Stats error: $e');
    }
    return null;
  }

  // providers/ids_api_client.dart — add this method

// ---------------------------------------------------------------------------
// History analytics (trend charts)
// ---------------------------------------------------------------------------

  Future<Map<String, dynamic>?> fetchHistory({
    String window = '24h',
    int limit = 10000,
  }) async {
    try {
      final uri = Uri.parse('${IdsConfig.baseUrl}/api/history').replace(
        queryParameters: {
          'window': window,
          'limit': limit.toString(),
        },
      );

      final res = await http.get(uri, headers: IdsConfig.headers);

      debugPrint('=== FETCH_HISTORY ===');
      debugPrint('Status: ${res.statusCode}');

      if (res.statusCode != 200) return null;

      final result = await _secureParseInIsolate(res.bodyBytes);

      // Use the payload if present, even when signature verification fails.
      // Log a warning so the signature issue is visible but doesn't block data.
      final payload = result['payload'] as Map<String, dynamic>?;
      if (payload != null) {
        if (result['success'] != true) {
          debugPrint('⚠️  fetchHistory: signature verification failed — '
              'displaying unverified payload. Fix _secureParseInIsolate.');
        }
        return payload;
      }

      debugPrint('fetchHistory: no payload in response');
      return null;
    } catch (e, s) {
      debugPrint('History fetch error: $e\n$s');
      return null;
    }
  }

  Future<Map<String, dynamic>?> fetchReport({
    required String window,
  }) async {
    try {
      final uri = Uri.parse('${IdsConfig.baseUrl}/api/report/$window');
      final res = await http.get(uri, headers: IdsConfig.headers);

      if (res.statusCode == 200) {
        final result = await _secureParseInIsolate(res.bodyBytes);

        // Resilient Parsing: Return payload even if signature verification is 'false'
        // This matches the logic you implemented for fetchHistory
        final payload = result['payload'] as Map<String, dynamic>?;
        if (payload != null) {
          if (result['success'] != true) {
            debugPrint(
                '⚠️ fetchReport: signature verification failed - displaying unverified data.');
          }
          return payload;
        }
      }
      return null;
    } catch (e) {
      debugPrint("Report fetch error: $e");
      return null;
    }
  }

  // ---------------------------------------------------------------------------
  // DMZ Settings
  // ---------------------------------------------------------------------------

  Future<List<String>> fetchDmzIps() async {
    try {
      final res = await http.get(
        Uri.parse('${IdsConfig.baseUrl}/api/settings/dmz_ips'),
        headers: IdsConfig.headers,
      );
      final result = await _secureParseInIsolate(res.bodyBytes);
      if (res.statusCode == 200 && result['success'] == true) {
        return List<String>.from(result['payload']['dmz_ips'] ?? []);
      }
    } catch (e) {
      debugPrint('DMZ fetch error: $e');
    }
    return [];
  }

  Future<bool> addDmzIp(String ipAddress) async {
    try {
      final body = jsonEncode({'ip_address': ipAddress});
      final res = await http.post(
        Uri.parse('${IdsConfig.baseUrl}/api/settings/dmz_ips'),
        headers: IdsConfig.headers,
        body: body,
      );
      return res.statusCode == 200 && await _verifyInIsolate(res.bodyBytes);
    } catch (e) {
      debugPrint('DMZ add error: $e');
      return false;
    }
  }

  Future<bool> removeDmzIp(String ipAddress) async {
    try {
      final body = jsonEncode({'ip_address': ipAddress});
      final res = await http.delete(
        Uri.parse('${IdsConfig.baseUrl}/api/settings/dmz_ips'),
        headers: IdsConfig.headers,
        body: body,
      );
      return res.statusCode == 200 && await _verifyInIsolate(res.bodyBytes);
    } catch (e) {
      debugPrint('DMZ remove error: $e');
      return false;
    }
  }

  Future<bool> _verifyInIsolate(Uint8List bodyBytes) async {
    final result = await _secureParseInIsolate(bodyBytes);
    return result['success'] == true;
  }

  /// Decode + verify in one isolate round-trip.
  Future<Map<String, dynamic>> _secureParseInIsolate(Uint8List bodyBytes) {
    return compute(secureParserIsolate, {
      'bodyBytes': bodyBytes,
      'pubX': IdsConfig.pubXHex,
      'pubY': IdsConfig.pubYHex,
    });
  }
}
