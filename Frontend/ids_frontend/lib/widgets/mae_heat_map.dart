import 'package:flutter/material.dart';

// ═══════════════════════════════════════════════════════════════════════════
// MAE Heatmap Grid Widget
// ═══════════════════════════════════════════════════════════════════════════
class MaeHeatmapGrid extends StatelessWidget {
  final List<double> gridValues;
  final String title;

  const MaeHeatmapGrid({
    super.key,
    required this.gridValues,
    required this.title,
  });

  Color _getValueColor(double value) {
    return Color.lerp(Colors.blue.shade900, Colors.redAccent, value) ??
        Colors.transparent;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.bold,
              color: Colors.white54,
              letterSpacing: 1),
        ),
        const SizedBox(height: 8),
        Container(
          width: 140,
          height: 140,
          decoration: BoxDecoration(
            border: Border.all(color: Colors.white.withOpacity(0.1)),
          ),
          child: GridView.builder(
            physics: const NeverScrollableScrollPhysics(),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 9,
              crossAxisSpacing: 1,
              mainAxisSpacing: 1,
            ),
            itemCount: 81,
            itemBuilder: (context, index) {
              double val =
                  (index < gridValues.length) ? gridValues[index] : 0.0;
              return Container(
                decoration: BoxDecoration(
                  color: _getValueColor(val),
                  borderRadius: BorderRadius.circular(1),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
