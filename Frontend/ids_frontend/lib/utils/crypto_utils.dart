// utils/crypto_utils.dart
//
// ECDSA/P-256 signature verification.
// Key optimisation: domain params and public key are built ONCE and cached.

import 'dart:collection';
import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:hex/hex.dart';
import 'package:pointycastle/export.dart' as pc;

// ---------------------------------------------------------------------------
// Cached domain params — ECDomainParameters('prime256v1') is expensive.
// Building it once at startup saves ~2-5 ms on every verify call.
// ---------------------------------------------------------------------------
final _domainParams = pc.ECDomainParameters('prime256v1');

/// Cache the parsed public key so BigInt.parse + createPoint run only once.
pc.ECPublicKey? _cachedPubKey;
String? _cachedPubXHex;
String? _cachedPubYHex;

pc.ECPublicKey _getOrBuildPubKey(String pubXHex, String pubYHex) {
  if (_cachedPubKey != null &&
      _cachedPubXHex == pubXHex &&
      _cachedPubYHex == pubYHex) {
    return _cachedPubKey!;
  }
  final x = BigInt.parse(pubXHex, radix: 16);
  final y = BigInt.parse(pubYHex, radix: 16);
  _cachedPubKey =
      pc.ECPublicKey(_domainParams.curve.createPoint(x, y), _domainParams);
  _cachedPubXHex = pubXHex;
  _cachedPubYHex = pubYHex;
  return _cachedPubKey!;
}

// ---------------------------------------------------------------------------
// Map sorting — needed to reproduce Python's sort_keys=True JSON hash
// ---------------------------------------------------------------------------

/// Sync recursive sort. Safe to call inside an isolate.
dynamic toSortedMap(dynamic item) {
  if (item is Map) {
    final keys = item.keys.toList()..sort();
    return {for (final k in keys) k: toSortedMap(item[k])};
  }
  if (item is List) return item.map(toSortedMap).toList();
  return item;
}

/// Async variant — yields between map iterations so the main isolate
/// never blocks longer than one microtask at a time.
Future<dynamic> toSortedMapAsync(dynamic item) async {
  if (item is Map) {
    final keys = item.keys.toList()..sort();
    final result = <String, dynamic>{};
    for (final k in keys) {
      result[k as String] = await toSortedMapAsync(item[k]);
    }
    return result;
  }
  if (item is List) return Future.wait(item.map(toSortedMapAsync));
  return item;
}

/// Canonical JSON with sorted keys — matches Python json.dumps(sort_keys=True).
String toSortedJson(dynamic value) {
  if (value is Map) {
    final sorted = SplayTreeMap<String, dynamic>.from(
        value.map((k, v) => MapEntry(k.toString(), v)));
    final buf = StringBuffer('{');
    var first = true;
    for (final e in sorted.entries) {
      if (!first) buf.write(',');
      buf
        ..write(jsonEncode(e.key))
        ..write(':')
        ..write(toSortedJson(e.value));
      first = false;
    }
    buf.write('}');
    return buf.toString();
  }
  if (value is List) {
    final buf = StringBuffer('[');
    var first = true;
    for (final item in value) {
      if (!first) buf.write(',');
      buf.write(toSortedJson(item));
      first = false;
    }
    buf.write(']');
    return buf.toString();
  }
  return jsonEncode(value);
}

/// Converts a BigInt to exactly 32 big-endian bytes.
Uint8List bigIntToBytes(BigInt n) =>
    Uint8List.fromList(HEX.decode(n.toRadixString(16).padLeft(64, '0')));

// ---------------------------------------------------------------------------
// Core verify — used both on the main isolate and inside compute() workers
// ---------------------------------------------------------------------------

/// Verifies a raw ECDSA/P-256 signature.
///
/// [message]  — unencoded bytes that were signed (NOT pre-hashed).
/// [sigBytes] — 64-byte compact signature: r (32 bytes) ++ s (32 bytes).
/// [pubXHex]  — server public key X coordinate (hex string).
/// [pubYHex]  — server public key Y coordinate (hex string).
///
/// Low-S normalisation is applied for Python `cryptography` lib compatibility.
bool ecdsaVerifyRaw(
    Uint8List message, Uint8List sigBytes, String pubXHex, String pubYHex) {
  try {
    final pubKey = _getOrBuildPubKey(pubXHex, pubYHex);
    final hash = sha256.convert(message).bytes;

    final r = BigInt.parse(HEX.encode(sigBytes.sublist(0, 32)), radix: 16);
    BigInt s = BigInt.parse(HEX.encode(sigBytes.sublist(32, 64)), radix: 16);

    // Low-S normalisation
    if (s > (_domainParams.n >> 1)) s = _domainParams.n - s;

    final signer = pc.ECDSASigner(null, pc.HMac(pc.SHA256Digest(), 64));
    signer.init(false, pc.PublicKeyParameter(pubKey));
    return signer.verifySignature(
        Uint8List.fromList(hash), pc.ECSignature(r, s));
  } catch (e) {
    print('🛡️ ECC verify error: $e');
    return false;
  }
}
