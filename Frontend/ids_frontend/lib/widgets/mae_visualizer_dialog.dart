import 'package:flutter/material.dart';
import '../providers/ids_provider.dart';

class MaeVisualizerDialog extends StatelessWidget {
  final Packet packet;

  MaeVisualizerDialog({required this.packet});

  @override
  Widget build(BuildContext context) {
    // --- REAL DATA EXTRACTION (ROADMAP POINT 3) ---
    // We attempt to extract values from the 'top_contributing_factors' or features.
    // If the full vector isn't available, we fall back to a zeroed list.
    final explanation = packet.explanation ?? {};
    final List<dynamic> factors = explanation['top_contributing_factors'] ?? [];

    // Initialize an 81-cell grid (9x9)
    List<double> gridData = List.filled(81, 0.0);

    // Populate grid with actual observed values from the XAI payload
    // Note: In a production environment, you might pass the full 78-dim vector
    // in the 'explanation' object under a 'raw_scaled' key.
    if (factors.isNotEmpty) {
      for (int i = 0; i < factors.length; i++) {
        if (i < 81) {
          // Extract the magnitude of the feature's impact
          double val = double.tryParse(
                  factors[i]['observed_value']?.toString() ?? '0.0') ??
              0.0;
          gridData[i] = val.clamp(0.0, 1.0); // Normalize for heat-map coloring
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
          // The Visual Reconstruction Grid
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
                    // Lighter blue for low values, bright cyan/magenta for high values
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

          // Technical Insights
          _buildMetricRow("Reconstruction Error",
              "${packet.maeAnomaly.toStringAsFixed(5)}"),
          _buildMetricRow("Anomalous Index",
              packet.maeAnomaly > 0.15 ? "CRITICAL" : "STABLE"),
          _buildMetricRow("Protocol Context", packet.protocol),

          SizedBox(height: 12),
          Text(
            "High intensity cells represent features contributing most to the anomaly detection.",
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
