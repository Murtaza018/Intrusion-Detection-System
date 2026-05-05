// lib/screens/graph_screen.dart
import 'package:flutter/material.dart';
import '../services/network_graph_service.dart';
import '../widgets/graph_view_widget.dart';
import '../models/graph_data.dart';

class GraphScreen extends StatefulWidget {
  const GraphScreen({super.key}); // Added super.key for best practice

  @override
  State<GraphScreen> createState() => _GraphScreenState();
}

class _GraphScreenState extends State<GraphScreen> {
  late final NetworkGraphService _graphService;

  @override
  void initState() {
    super.initState();
    // Initialize stream services here to ensure they tie to the state lifecycle safely
    _graphService = NetworkGraphService();
  }

  @override
  Widget build(BuildContext context) {
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
            // LayoutBuilder gives you the exact pixels available AFTER
            // the AppBar and BottomNav have taken their share.
            return LayoutBuilder(
              builder: (context, constraints) {
                return GraphViewWidget(
                  graphData: snapshot.data!,
                  // Pass the constrained size, preventing nodes from rendering off-screen
                  size: Size(constraints.maxWidth, constraints.maxHeight),
                );
              },
            );
          }
          if (snapshot.hasError) {
            return Center(
              child: Text(
                'DATA STREAM ERROR',
                style: const TextStyle(
                  color: Colors.redAccent,
                  fontFamily: 'monospace',
                  fontWeight: FontWeight.bold,
                ),
              ),
            );
          }
          return const Center(
            child: CircularProgressIndicator(color: Color(0xFF00E5FF)),
          );
        },
      ),
    );
  }
}
