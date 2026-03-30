// lib/screens/graph_screen.dart
import 'package:flutter/material.dart';
import '../services/network_graph_service.dart';
import '../widgets/graph_view_widget.dart';
import '../models/graph_data.dart';

class GraphScreen extends StatefulWidget {
  const GraphScreen({Key? key}) : super(key: key);

  @override
  State<GraphScreen> createState() => _GraphScreenState();
}

class _GraphScreenState extends State<GraphScreen> {
  final NetworkGraphService _graphService = NetworkGraphService();

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF0A0E12),
        foregroundColor: Colors.white,
        title: const Text(
          "Live Network Graph",
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 18,
            letterSpacing: -0.5,
          ),
        ),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(height: 1, color: Colors.white12),
        ),
      ),
      backgroundColor: const Color(0xFF0A0E12),
      body: StreamBuilder<GraphData>(
        stream: _graphService.graphStream(),
        builder: (context, snapshot) {
          if (snapshot.hasData) {
            return GraphViewWidget(
              graphData: snapshot.data!,
              size: size,
            );
          }
          return const Center(
            child: CircularProgressIndicator(),
          );
        },
      ),
    );
  }
}
