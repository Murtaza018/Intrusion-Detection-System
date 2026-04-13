import 'package:flutter/foundation.dart';
import 'package:pointycastle/export.dart' as pc;

import 'ids_config.dart';

class IdsDiagnostics {
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

  static Future<void> runRealDataTest() async {
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
