import 'package:flutter/material.dart';
import '../providers/ids_provider.dart';

class MaeVisualizerDialog extends StatelessWidget {
  final Packet packet;

  MaeVisualizerDialog({required this.packet, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final explanation = packet.explanation ?? {};
    final List<dynamic> factors = explanation['top_contributing_factors'] ?? [];

    // Check if the backend is still generating SHAP values (Roadmap Point 3)
    final String processStatus = explanation['status'] ?? 'done';

    // 1. HANDLE LOADING STATE
    // Show spinner ONLY if the backend specifically says it's still 'analyzing'
    if (processStatus == 'analyzing' && factors.isEmpty) {
      return AlertDialog(
        backgroundColor: Colors.grey[900],
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(height: 20),
            CircularProgressIndicator(color: Colors.orangeAccent),
            SizedBox(height: 24),
            Text("AI EXPLAINER ACTIVE",
                style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.1)),
            SizedBox(height: 8),
            Text("Reconstructing 9x9 structural grid...",
                textAlign: TextAlign.center,
                style: TextStyle(color: Colors.white70, fontSize: 12)),
            SizedBox(height: 20),
          ],
        ),
      );
    }

    // 2. HANDLE ERROR / NO DATA STATE
    // If it's an anomaly but factors is still empty and it's NOT analyzing, it failed.
    if (factors.isEmpty && packet.status != 'normal') {
      return AlertDialog(
        backgroundColor: Colors.grey[900],
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
        title: Text("VISUAL DATA UNAVAILABLE",
            style: TextStyle(
                color: Colors.redAccent,
                fontSize: 16,
                fontWeight: FontWeight.bold)),
        content: Text(
            "The XAI worker encountered an error (IndexError) or the background data is still being summarized.",
            style: TextStyle(color: Colors.white70, fontSize: 13)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text("CLOSE", style: TextStyle(color: Colors.white38)),
          )
        ],
      );
    }

    // 3. PREPARE GRID DATA (Populate from available features)
    List<double> gridData = List.filled(81, 0.0);
    if (factors.isNotEmpty) {
      for (int i = 0; i < factors.length; i++) {
        if (i < 81) {
          // Extract normalized observed values from the SHAP payload
          double val = double.tryParse(
                  factors[i]['observed_value']?.toString() ?? '0.0') ??
              0.0;
          gridData[i] = val.clamp(0.0, 1.0);
        }
      }
    }

    return AlertDialog(
      backgroundColor: Colors.grey[900],
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      title: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("MAE STRUCTURAL RECONSTRUCTION",
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold)),
          Text("9x9 Tabular-to-Image Mapping",
              style: TextStyle(
                  color: Colors.white38, fontSize: 10, letterSpacing: 0.5)),
        ],
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // The Visual Reconstruction Grid (SAFE Paper Implementation)
          Container(
            width: 220,
            height: 220,
            padding: EdgeInsets.all(4),
            decoration: BoxDecoration(
              color: Colors.black,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.white10),
            ),
            child: GridView.builder(
              gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 9,
                crossAxisSpacing: 1.5,
                mainAxisSpacing: 1.5,
              ),
              itemCount: 81,
              itemBuilder: (context, index) {
                double val = gridData[index];
                return Container(
                  decoration: BoxDecoration(
                    color: Color.lerp(
                        Colors.blueGrey[900],
                        packet.status == 'normal'
                            ? Colors.cyanAccent
                            : Colors.orangeAccent,
                        val),
                    borderRadius: BorderRadius.circular(1),
                  ),
                );
              },
            ),
          ),
          SizedBox(height: 24),

          // Technical Insights (Point 3 Dashboard Context)
          _buildMetricRow("Reconstruction Error",
              "${packet.maeAnomaly.toStringAsFixed(5)}"),
          _buildMetricRow("Anomalous Index",
              packet.maeAnomaly > 0.15 ? "CRITICAL" : "STABLE"),
          _buildMetricRow("Protocol Context", packet.protocol),

          SizedBox(height: 12),
          Text(
            "High intensity cells represent features contributing most to the structural anomaly.",
            textAlign: TextAlign.center,
            style: TextStyle(
                color: Colors.white24,
                fontSize: 9,
                fontStyle: FontStyle.italic),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text("DISMISS",
              style: TextStyle(
                  color: Colors.white70, fontWeight: FontWeight.bold)),
        )
      ],
    );
  }

  Widget _buildMetricRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(color: Colors.white38, fontSize: 11)),
          Text(value,
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontFamily: 'RobotoMono',
                  fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
