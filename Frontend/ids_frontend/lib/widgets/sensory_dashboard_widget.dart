import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:syncfusion_flutter_gauges/gauges.dart';
import '../providers/ids_provider.dart';

class SensoryDashboard extends StatelessWidget {
  const SensoryDashboard({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final idsProvider = Provider.of<IdsProvider>(context);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF0D1117),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: Colors.white.withOpacity(0.05)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min, // Prevents expanding vertically
        children: [
          Row(
            children: [
              const Icon(Icons.hub_outlined,
                  color: Color(0xFF00E5FF), size: 12),
              const SizedBox(width: 8),
              const Text("HYBRID SENSORY ENGINES",
                  style: TextStyle(
                      color: Colors.white30,
                      fontSize: 9,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2)),
              const Spacer(),
              _buildLiveIndicator(idsProvider.liveStatus),
            ],
          ),
          const SizedBox(height: 8),
          // FIX: Use a Row with Flexible to prevent overflow
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              Flexible(
                  child: _buildSensorGauge("GNN CONTEXT", "Topological Shift",
                      idsProvider.liveGnnAnomaly, const Color(0xFF00E5FF))),
              Container(
                  width: 1,
                  height: 40,
                  color: Colors.white.withOpacity(0.05),
                  margin: const EdgeInsets.symmetric(horizontal: 10)),
              Flexible(
                  child: _buildSensorGauge("MAE VISUAL", "Structure Error",
                      idsProvider.liveMaeAnomaly, Colors.pinkAccent)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSensorGauge(
      String title, String subtitle, double value, Color color) {
    double displayValue = (value * 100).clamp(0, 100);
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        SizedBox(
          height: 70, // Reduced height to fit dashboard
          width: 120,
          child: SfRadialGauge(
            axes: <RadialAxis>[
              RadialAxis(
                minimum: 0,
                maximum: 100,
                showLabels: false,
                showTicks: false,
                startAngle: 180,
                endAngle: 0,
                radiusFactor: 0.9,
                axisLineStyle: AxisLineStyle(
                    thickness: 0.15,
                    color: Colors.white.withOpacity(0.05),
                    thicknessUnit: GaugeSizeUnit.factor),
                pointers: <GaugePointer>[
                  RangePointer(
                      value: displayValue,
                      width: 0.15,
                      sizeUnit: GaugeSizeUnit.factor,
                      color: color,
                      enableAnimation: true),
                  MarkerPointer(
                      value: displayValue,
                      markerType: MarkerType.invertedTriangle,
                      color: Colors.white,
                      markerHeight: 4,
                      markerWidth: 4)
                ],
                annotations: <GaugeAnnotation>[
                  GaugeAnnotation(
                    widget: Text("${displayValue.toStringAsFixed(1)}%",
                        style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w900,
                            color: color,
                            fontFamily: 'monospace')),
                    angle: 90,
                    positionFactor: 0.5,
                  )
                ],
              )
            ],
          ),
        ),
        Text(title,
            style: const TextStyle(
                color: Colors.white70,
                fontSize: 8,
                fontWeight: FontWeight.bold)),
        const SizedBox(height: 2),
        Text(subtitle,
            style: const TextStyle(color: Colors.white10, fontSize: 7)),
      ],
    );
  }

  Widget _buildLiveIndicator(String status) {
    Color color = status == 'normal'
        ? Colors.greenAccent
        : (status == 'known_attack' ? Colors.orangeAccent : Colors.redAccent);
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
            width: 4,
            height: 4,
            decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 6),
        Text(status.toUpperCase(),
            style: TextStyle(
                color: color.withOpacity(0.7),
                fontSize: 8,
                fontWeight: FontWeight.bold,
                letterSpacing: 1)),
      ],
    );
  }
}
