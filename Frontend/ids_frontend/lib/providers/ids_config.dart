// providers/ids_config.dart
//
// Central place for all configuration constants.
// Import this wherever BASE_URL, apiKey, or public key coords are needed.

class IdsConfig {
  IdsConfig._(); // prevent instantiation

  static const String baseUrl = 'http://127.0.0.1:5001';

  static const String apiKey = String.fromEnvironment(
    'API_KEY',
    defaultValue: 'MySuperSecretKey12345!',
  );
  static const String vapidKey = String.fromEnvironment(
    'Vapid_Key',
    defaultValue: '',
  );

  /// ECC P-256 public key X coordinate (hex) — extracted from cert.pem
  static const String pubXHex =
      '582110e3c4af92f5f2d2ae042be8dd44d67e51d1a4728986874e0fcd64829253';

  /// ECC P-256 public key Y coordinate (hex) — extracted from cert.pem
  static const String pubYHex =
      '66c676c3e1d01c05b5299c744060a1911598259fffa710925f1060bd23160970';

  static Map<String, String> get headers => {
        'X-API-Key': apiKey,
        'Content-Type': 'application/json',
      };
}
