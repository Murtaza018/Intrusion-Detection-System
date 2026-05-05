import 'package:flutter/material.dart';
import '../providers/ids_provider.dart';
import '../models/packet.dart';

class MaeVisualizerDialog extends StatefulWidget {
  final Packet packet;
  const MaeVisualizerDialog({required this.packet, Key? key}) : super(key: key);

  @override
  _MaeVisualizerDialogState createState() => _MaeVisualizerDialogState();
}

class _MaeVisualizerDialogState extends State<MaeVisualizerDialog> {
  // --- STATE: Track two comparison subjects ---
  int? pinnedA;
  int? pinnedB;

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

  void _handleCellTap(int index) {
    setState(() {
      if (pinnedA == index) {
        pinnedA = null;
      } else if (pinnedB == index) {
        pinnedB = null;
      } else if (pinnedA == null) {
        pinnedA = index;
      } else {
        pinnedB = index; // Overwrites B if both are full, or fills B if empty
      }
    });
  }

  Widget _buildGrid(List<double> gridData) {
    return Container(
      width: 300,
      height: 300,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
          color: Colors.black,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.white10)),
      child: GridView.builder(
        physics: const NeverScrollableScrollPhysics(),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 9, crossAxisSpacing: 4, mainAxisSpacing: 4),
        itemCount: 81,
        itemBuilder: (context, index) {
          bool isA = pinnedA == index;
          bool isB = pinnedB == index;
          return GestureDetector(
            onTap: () => _handleCellTap(index),
            child: Container(
              decoration: BoxDecoration(
                color: _getHeatmapColor(gridData[index], widget.packet.status),
                borderRadius: BorderRadius.circular(1),
                border: isA
                    ? Border.all(color: const Color(0xFF00E5FF), width: 2)
                    : (isB
                        ? Border.all(color: Colors.pinkAccent, width: 2)
                        : null),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildDualInspector(List<double> gridData) {
    return Column(
      children: [
        _buildInspectorPane(
            "SUBJECT ALPHA", pinnedA, gridData, const Color(0xFF00E5FF)),
        const SizedBox(height: 12),
        _buildInspectorPane(
            "SUBJECT BETA", pinnedB, gridData, Colors.pinkAccent),
      ],
    );
  }

  Widget _buildInspectorPane(
      String title, int? index, List<double> gridData, Color themeColor) {
    bool hasData = index != null;
    String name = hasData
        ? (index < featureNames.length
            ? featureNames[index]
            : "Auxiliary Segment")
        : "---";
    double val = hasData ? gridData[index] : 0.0;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: hasData
            ? themeColor.withOpacity(0.03)
            : Colors.white.withOpacity(0.01),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
            color: hasData ? themeColor.withOpacity(0.4) : Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(title,
                  style: TextStyle(
                      color: themeColor,
                      fontSize: 8,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2)),
              if (hasData)
                Icon(Icons.check_circle, size: 10, color: themeColor),
            ],
          ),
          const SizedBox(height: 8),
          Text(name,
              maxLines: 1, // Prevent overflow if name is long
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'monospace')),
          const SizedBox(height: 12),
          LinearProgressIndicator(
              value: val,
              color: themeColor,
              backgroundColor: Colors.white.withOpacity(0.05),
              minHeight: 2),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                  hasData
                      ? "COORD: R${index ~/ 9 + 1} C${index % 9 + 1}"
                      : "NO SELECTION",
                  style: const TextStyle(color: Colors.white24, fontSize: 9)),
              Text("${(val * 100).toStringAsFixed(1)}%",
                  style: TextStyle(
                      color: hasData ? Colors.white70 : Colors.white10,
                      fontSize: 10,
                      fontFamily: 'monospace')),
            ],
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final explanation = widget.packet.explanation ?? {};
    bool isZeroDay = widget.packet.status == 'zero_day';
    final isDesktop = MediaQuery.of(context).size.width > 600; // Viewport check

    List<double> gridData = List.filled(81, 0.0);

    if (isZeroDay &&
        explanation['sensory_analysis'] != null &&
        explanation['sensory_analysis']['original_grid'] != null &&
        explanation['sensory_analysis']['reconstructed_grid'] != null) {
      List<dynamic> orig = explanation['sensory_analysis']['original_grid'];
      List<dynamic> recon =
          explanation['sensory_analysis']['reconstructed_grid'];

      for (int i = 0; i < orig.length && i < 81; i++) {
        double oVal = double.tryParse(orig[i].toString()) ?? 0.0;
        double rVal = double.tryParse(recon[i].toString()) ?? 0.0;
        gridData[i] = (oVal - rVal).abs().clamp(0.0, 1.0);
      }
    } else {
      final List<dynamic> rawFeatures = explanation['raw_features'] ?? [];
      for (int i = 0; i < rawFeatures.length && i < 81; i++) {
        gridData[i] =
            (double.tryParse(rawFeatures[i]?.toString() ?? '0.0') ?? 0.0)
                .clamp(0.0, 1.0);
      }
    }

    return AlertDialog(
      backgroundColor: const Color(0xFF0D1117),
      // Tighter padding on mobile so the 300px grid fits
      insetPadding: EdgeInsets.all(isDesktop ? 20 : 10),
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: const BorderSide(color: Colors.white10)),
      content: ConstrainedBox(
        // Fixed the 800px hardcoded death box
        constraints: const BoxConstraints(maxWidth: 800),
        child: SingleChildScrollView(
          // Added scrolling for mobile stacking
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildHeader(isZeroDay),
              const SizedBox(height: 24),
              // Use Flex to stack vertically on mobile and side-by-side on desktop
              Flex(
                direction: isDesktop ? Axis.horizontal : Axis.vertical,
                crossAxisAlignment: isDesktop
                    ? CrossAxisAlignment.start
                    : CrossAxisAlignment.center,
                children: [
                  _buildGrid(gridData),
                  SizedBox(
                      height: isDesktop ? 0 : 24, width: isDesktop ? 24 : 0),
                  // Only use Expanded if horizontal, otherwise it causes infinite height errors in Column
                  isDesktop
                      ? Expanded(child: _buildDualInspector(gridData))
                      : _buildDualInspector(gridData),
                ],
              ),
              const SizedBox(height: 24),
              _buildFooter(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(bool isZeroDay) {
    return Row(
      children: [
        Icon(isZeroDay ? Icons.warning_amber_rounded : Icons.blur_on_rounded,
            color: isZeroDay ? Colors.orangeAccent : const Color(0xFF00E5FF),
            size: 24),
        const SizedBox(width: 12),
        Expanded(
          // Wrapped in Expanded to prevent text overflow
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                  isZeroDay
                      ? "MAE RECONSTRUCTION ERROR"
                      : "RAW FEATURE HEATMAP",
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14, // Slightly smaller for safety
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1)),
              Text(
                  isZeroDay
                      ? "Highlighting structural anomalies with high reconstruction error"
                      : "Select two network features to cross-reference their scaled intensity",
                  style: const TextStyle(color: Colors.white24, fontSize: 10)),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildFooter() {
    return Wrap(
      // Swapped Row for Wrap to prevent button clipping
      alignment: WrapAlignment.spaceBetween,
      crossAxisAlignment: WrapCrossAlignment.center,
      spacing: 12,
      runSpacing: 12,
      children: [
        const Text("Alpha (Cyan) vs Beta (Pink) Correlation Mode",
            style: TextStyle(
                color: Colors.white24,
                fontSize: 9,
                fontStyle: FontStyle.italic)),
        TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("CLOSE ANALYSIS",
                style: TextStyle(
                    color: Colors.white70, fontWeight: FontWeight.bold))),
      ],
    );
  }

  Color _getHeatmapColor(double value, String status) {
    if (value < 0.05) return Colors.white.withOpacity(0.05);
    return status == 'normal'
        ? Color.lerp(Colors.cyan.withOpacity(0.1), Colors.cyanAccent, value)!
        : Color.lerp(Colors.orange.withOpacity(0.2), Colors.redAccent, value)!;
  }
}
