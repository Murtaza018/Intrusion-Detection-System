// // // lib/services/network_graph_service.dart
// // import 'dart:convert';
// // import 'dart:async';
// // import 'package:http/http.dart' as http;
// // import '../models/graph_data.dart';
// // import '../providers/ids_config.dart';

// // class NetworkGraphService {
// //   static final NetworkGraphService _instance = NetworkGraphService._internal();
// //   factory NetworkGraphService() => _instance;
// //   NetworkGraphService._internal();

// //   final String _baseUrl = IdsConfig.baseUrl;
// //   // Replace with your backend IP
// //   final String _apiKey = IdsConfig.apiKey;

// //   Future<GraphData> fetchGraph() async {
// //     final uri = Uri.parse('$_baseUrl/api/graph');
// //     final response = await http.get(
// //       uri,
// //       headers: {'X-API-Key': _apiKey},
// //     );

// //     if (response.statusCode != 200) {
// //       throw Exception('Graph fetch failed: ${response.statusCode}');
// //     }
// //     final body = jsonDecode(response.body);
// //     final payload = body['payload'] as Map<String, dynamic>; // ECC wrapper

// //     return GraphData.fromJson(payload);
// //   }

// //   Stream<GraphData> graphStream() {
// //     return Stream.periodic(const Duration(seconds: 2), (i) => i).asyncMap(
// //       (_) async {
// //         try {
// //           return await fetchGraph();
// //         } catch (e) {
// //           throw Exception('Graph stream error: $e');
// //         }
// //       },
// //     );
// //   }
// // }
// // lib/services/network_graph_service.dart
// import 'dart:convert';
// import 'dart:async';
// import 'dart:isolate'; // Added Isolate for background processing
// import 'package:http/http.dart' as http;
// import '../models/graph_data.dart';
// import '../providers/ids_config.dart';

// class NetworkGraphService {
//   static final NetworkGraphService _instance = NetworkGraphService._internal();
//   factory NetworkGraphService() => _instance;
//   NetworkGraphService._internal();

//   final String _baseUrl = IdsConfig.baseUrl;
//   // Replace with your backend IP
//   final String _apiKey = IdsConfig.apiKey;

//   Future<GraphData> fetchGraph() async {
//     final uri = Uri.parse('$_baseUrl/api/graph');
//     final response = await http.get(
//       uri,
//       headers: {'X-API-Key': _apiKey},
//     );

//     if (response.statusCode != 200) {
//       throw Exception('Graph fetch failed: ${response.statusCode}');
//     }

//     // --- FIXED BOTTLENECK ---
//     // Extract the string first, then send it to the background thread.
//     // The background thread decodes the massive JSON and maps the Dart objects.
//     // Isolate.run returns the finished GraphData object instantly without blocking the UI.
//     final stringBody = response.body;
//     return await Isolate.run(() {
//       final body = jsonDecode(stringBody);
//       final payload = body['payload'] as Map<String, dynamic>; // ECC wrapper
//       return GraphData.fromJson(payload);
//     });
//   }

//   Stream<GraphData> graphStream() {
//     return Stream.periodic(const Duration(seconds: 2), (i) => i).asyncMap(
//       (_) async {
//         try {
//           return await fetchGraph();
//         } catch (e) {
//           throw Exception('Graph stream error: $e');
//         }
//       },
//     );
//   }
// }

// lib/services/network_graph_service.dart
import 'dart:convert';
import 'dart:async';
import 'package:flutter/foundation.dart'; // Web-safe compute
import 'package:http/http.dart' as http;
import '../models/graph_data.dart';
import '../providers/ids_config.dart';

// TOP-LEVEL FUNCTION: Required for compute() to work cross-platform
GraphData _parseGraphData(String stringBody) {
  final body = jsonDecode(stringBody);
  final payload = body['payload'] as Map<String, dynamic>;
  return GraphData.fromJson(payload);
}

class NetworkGraphService {
  static final NetworkGraphService _instance = NetworkGraphService._internal();
  factory NetworkGraphService() => _instance;
  NetworkGraphService._internal();

  final String _baseUrl = IdsConfig.baseUrl;
  final String _apiKey = IdsConfig.apiKey;

  Future<GraphData> fetchGraph() async {
    final uri = Uri.parse('$_baseUrl/api/graph');
    final response = await http.get(
      uri,
      headers: {'X-API-Key': _apiKey},
    );

    if (response.statusCode != 200) {
      throw Exception('Graph fetch failed: ${response.statusCode}');
    }

    // compute() is web-safe. It won't crash your browser.
    return await compute(_parseGraphData, response.body);
  }

  Stream<GraphData> graphStream() {
    return Stream.periodic(const Duration(seconds: 2), (i) => i).asyncMap(
      (_) async {
        try {
          return await fetchGraph();
        } catch (e) {
          throw Exception('Graph stream error: $e');
        }
      },
    );
  }
}
