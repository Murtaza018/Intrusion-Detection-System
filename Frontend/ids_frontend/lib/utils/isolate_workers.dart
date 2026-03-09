// utils/isolate_workers.dart
//
// Top-level functions passed to Flutter's compute().
// MUST live outside any class — the isolate spawner requires top-level symbols.
//
// Both workers do ALL heavy work (JSON decode, key sort, SHA-256, ECDSA) inside
// the isolate so the UI thread is never touched.

import 'dart:convert';
import 'dart:typed_data';

import 'package:hex/hex.dart';

import 'crypto_utils.dart';

// ---------------------------------------------------------------------------
// Unified secure parser
// Used for: /api/sensory/live (anomalous), /api/stats, /api/retrain, etc.
// ---------------------------------------------------------------------------

/// Entry point for `compute(secureParserIsolate, params)`.
///
/// params: { 'bodyBytes': Uint8List, 'pubX': String, 'pubY': String }
/// returns: { 'success': bool, 'payload': dynamic }
Future<Map<String, dynamic>> secureParserIsolate(
    Map<String, dynamic> params) async {
  try {
    final data = jsonDecode(utf8.decode(params['bodyBytes'] as Uint8List));
    final sigHex = data['signature'] as String?;
    final payload = data['payload'];

    if (sigHex == null || payload == null) return {'success': false};

    final msgBytes =
        Uint8List.fromList(utf8.encode(jsonEncode(toSortedMap(payload))));
    final sigBytes = Uint8List.fromList(HEX.decode(sigHex));

    return {
      'success':
          ecdsaVerifyRaw(msgBytes, sigBytes, params['pubX'], params['pubY']),
      'payload': payload,
    };
  } catch (e) {
    return {'success': false, 'error': e.toString()};
  }
}

// ---------------------------------------------------------------------------
// Packet-list verifier
// Used for: /api/packets/recent
// Returns the packets list alongside the success flag so the caller doesn't
// need to re-decode the body a second time.
// ---------------------------------------------------------------------------

/// Entry point for `compute(verifyAndParsePacketsInBackground, params)`.
///
/// params: { 'bodyBytes': Uint8List, 'pubX': String, 'pubY': String }
/// returns: { 'success': bool, 'packets': List }
Future<Map<String, dynamic>> verifyAndParsePacketsInBackground(
    Map<String, dynamic> params) async {
  try {
    final data = jsonDecode(utf8.decode(params['bodyBytes'] as Uint8List));
    final sigHex = data['signature'] as String?;
    final payload = data['payload'] as Map<String, dynamic>?;

    if (sigHex == null || payload == null)
      return {'success': false, 'packets': []};

    final msgBytes =
        Uint8List.fromList(utf8.encode(jsonEncode(toSortedMap(payload))));
    final sigBytes = Uint8List.fromList(HEX.decode(sigHex));

    final ok =
        ecdsaVerifyRaw(msgBytes, sigBytes, params['pubX'], params['pubY']);
    return {
      'success': ok,
      'packets': ok ? (payload['packets'] ?? []) : [],
    };
  } catch (e) {
    return {'success': false, 'packets': [], 'error': e.toString()};
  }
}
