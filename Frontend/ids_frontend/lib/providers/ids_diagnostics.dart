// providers/ids_diagnostics.dart
//
// Startup self-tests that run once when IdsProvider is constructed.
// Keep these separate so they are easy to comment out or extend
// without touching any production logic.

import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:hex/hex.dart';
import 'package:pointycastle/export.dart' as pc;

import '../utils/crypto_utils.dart';
import 'ids_config.dart';

class IdsDiagnostics {
  /// Verifies PointyCastle can instantiate the curve and load the server's
  /// public key. A failure here means the ECC dependency is broken.
  static Future<void> runECCSaneCheck() async {
    try {
      final domainParams = pc.ECDomainParameters('prime256v1');
      final x = BigInt.parse(IdsConfig.pubXHex, radix: 16);
      final y = BigInt.parse(IdsConfig.pubYHex, radix: 16);
      domainParams.curve.createPoint(x, y);
      debugPrint('✅ ECC SANITY PASSED — public key loaded successfully');
    } catch (e) {
      debugPrint('❌ ECC SANITY FAILED: $e');
    }
  }

  /// End-to-end test using a known-good signature captured from server logs.
  ///
  /// HOW TO USE:
  ///   1. Grab a fresh `signature` + `payload` pair from a server log.
  ///   2. Paste the values into the constants below.
  ///   3. Re-run the app — you should see "✅ Real Data Test PASSED".
  ///   4. Comment the constants back out when done.
  static Future<void> runRealDataTest() async {
    // --- Paste fresh values from server logs here ---
    // const String testSigHex =
    //     'a0666b19bbd5a724f82acdf490b62e35f5b490c5a21ac0a87f83726148033da'
    //     '3f02cc854c035a0f835352590a8e257ca2e9d01b337b94fec2a575162a4921c6c';
    // const String testJsonStr =
    //     '{"gnn_anomaly":0.0,"mae_anomaly":0.062300905585289,"status":"normal"}';
    // ------------------------------------------------

    // ignore: dead_code
    const bool testEnabled = false; // flip to true after pasting values above

    if (!testEnabled) {
      debugPrint(
          '🧪 Real data test skipped — set testEnabled=true in ids_diagnostics.dart');
      return;
    }

    // Unreachable until testEnabled is flipped — kept here for easy copy-paste.
    // final sigBytes = Uint8List.fromList(HEX.decode(testSigHex));
    // final msgBytes = Uint8List.fromList(utf8.encode(testJsonStr));
    // final result   = ecdsaVerifyRaw(
    //     msgBytes, sigBytes, IdsConfig.pubXHex, IdsConfig.pubYHex);
    // debugPrint(result
    //     ? '✅ Real Data Test PASSED'
    //     : '❌ Real Data Test FAILED — check key coords or sig bytes');
  }
}
