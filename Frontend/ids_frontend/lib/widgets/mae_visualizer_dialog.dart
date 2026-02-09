import 'package:flutter/material.dart';
import '../providers/ids_provider.dart';

class MaeVisualizerDialog extends StatefulWidget {
  final Packet packet;

  const MaeVisualizerDialog({required this.packet, Key? key}) : super(key: key);

  @override
  _MaeVisualizerDialogState createState() => _MaeVisualizerDialogState();
}

class _MaeVisualizerDialogState extends State<MaeVisualizerDialog> {
  // --- STATE: Tracks which cell is currently "Pinned" ---
  int? pinnedIndex;

  // --- FEATURE MAPPING (Preserved 78-feature list) ---
  final List<String> featureNames = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Inbound",
    "Label Index"
  ];

  @override
  Widget build(BuildContext context) {
    final explanation = widget.packet.explanation ?? {};
    final List<dynamic> factors = explanation['top_contributing_factors'] ?? [];

    // Prepare Grid Data
    List<double> gridData = List.filled(81, 0.0);
    for (int i = 0; i < factors.length; i++) {
      if (i < 81) {
        double val = double.tryParse(
                factors[i]['observed_value']?.toString() ?? '0.0') ??
            0.0;
        gridData[i] = val.clamp(0.0, 1.0);
      }
    }

    return AlertDialog(
      backgroundColor: const Color(0xFF0D1117),
      insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: const BorderSide(color: Colors.white10),
      ),
      content: SizedBox(
        width: 600, // Wider to accommodate the inspector panel
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildHeader(),
            const SizedBox(height: 24),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // LEFT: THE INTERACTIVE GRID
                _buildGrid(gridData),
                const SizedBox(width: 24),
                // RIGHT: THE FEATURE INSPECTOR
                Expanded(child: _buildInspector(gridData)),
              ],
            ),
            const SizedBox(height: 24),
            _buildFooterActions(),
          ],
        ),
      ),
    );
  }

  Widget _buildGrid(List<double> gridData) {
    return Container(
      width: 280,
      height: 280,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white10),
      ),
      child: GridView.builder(
        physics: const NeverScrollableScrollPhysics(),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 9,
          crossAxisSpacing: 3,
          mainAxisSpacing: 3,
        ),
        itemCount: 81,
        itemBuilder: (context, index) {
          double val = gridData[index];
          bool isPinned = pinnedIndex == index;

          return GestureDetector(
            onTap: () => setState(() => pinnedIndex = index),
            child: MouseRegion(
              cursor: SystemMouseCursors.click,
              child: Container(
                decoration: BoxDecoration(
                  color: _getHeatmapColor(val, widget.packet.status),
                  borderRadius: BorderRadius.circular(1),
                  border: isPinned
                      ? Border.all(color: const Color(0xFF00E5FF), width: 1.5)
                      : null,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildInspector(List<double> gridData) {
    if (pinnedIndex == null) {
      return Container(
        height: 280,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.02),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.white.withOpacity(0.05)),
        ),
        child: const Text(
          "SELECT A CELL TO INSPECT\nSTRUCTURAL FEATURE",
          textAlign: TextAlign.center,
          style:
              TextStyle(color: Colors.white24, fontSize: 10, letterSpacing: 1),
        ),
      );
    }

    final String name = pinnedIndex! < featureNames.length
        ? featureNames[pinnedIndex!]
        : "Padding Segment";
    final double value = gridData[pinnedIndex!];

    return Container(
      height: 280,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF15191C),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF00E5FF).withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("FEATURE INSPECTOR",
              style: TextStyle(
                  color: Color(0xFF00E5FF),
                  fontSize: 9,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.5)),
          const SizedBox(height: 16),
          Text(name.toUpperCase(),
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'monospace')),
          const Divider(color: Colors.white10, height: 24),
          _inspectorRow("Grid Coordinate",
              "Row ${pinnedIndex! ~/ 9 + 1}, Col ${pinnedIndex! % 9 + 1}"),
          _inspectorRow(
              "Anomaly Intensity", "${(value * 100).toStringAsFixed(2)}%"),
          const SizedBox(height: 16),
          LinearProgressIndicator(
            value: value,
            backgroundColor: Colors.white.withOpacity(0.05),
            color: _getHeatmapColor(value, widget.packet.status),
            minHeight: 4,
          ),
          const Spacer(),
          TextButton.icon(
            onPressed: () => setState(() => pinnedIndex = null),
            icon: const Icon(Icons.close, size: 14, color: Colors.white24),
            label: const Text("CLEAR PIN",
                style: TextStyle(color: Colors.white24, fontSize: 10)),
          )
        ],
      ),
    );
  }

  Widget _inspectorRow(String label, String val) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label,
              style: const TextStyle(color: Colors.white30, fontSize: 9)),
          const SizedBox(height: 4),
          Text(val,
              style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 11,
                  fontFamily: 'monospace')),
        ],
      ),
    );
  }

  // --- Helper UI Builders ---
  Widget _buildHeader() {
    return Row(
      children: [
        const Icon(Icons.grid_view_rounded, color: Color(0xFF00E5FF), size: 20),
        const SizedBox(width: 12),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Text("MAE STRUCTURAL RECONSTRUCTION",
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold)),
            Text("9x9 Tabular-to-Image Feature Mapping",
                style: TextStyle(color: Colors.white24, fontSize: 10)),
          ],
        ),
      ],
    );
  }

  Widget _buildFooterActions() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        const Text(
          "Click cells to pin features for detailed analysis.",
          style: TextStyle(
              color: Colors.white24, fontSize: 10, fontStyle: FontStyle.italic),
        ),
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text("DISMISS",
              style: TextStyle(
                  color: Colors.white70, fontWeight: FontWeight.bold)),
        ),
      ],
    );
  }

  Color _getHeatmapColor(double value, String status) {
    if (value < 0.05) return Colors.white.withOpacity(0.05);
    if (status == 'normal') {
      return Color.lerp(
          Colors.cyan.withOpacity(0.1), Colors.cyanAccent, value)!;
    } else {
      return Color.lerp(
          Colors.orange.withOpacity(0.2), Colors.redAccent, value)!;
    }
  }
}
