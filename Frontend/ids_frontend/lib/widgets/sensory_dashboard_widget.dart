import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:syncfusion_flutter_gauges/gauges.dart';
import '../providers/ids_provider.dart';

class SensoryDashboard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final idsProvider = Provider.of<IdsProvider>(context);

    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[900],
        borderRadius: BorderRadius.circular(15),
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("SENSORY ENGINES",
                  style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2)),
              _buildStatusIndicator(idsProvider.liveStatus),
            ],
          ),
          SizedBox(height: 20),
          Row(
            children: [
              // GNN Gauge (Topological)
              Expanded(
                child: _buildSensorGauge(
                  title: "GNN CONTEXT",
                  subtitle: "Topological Shift",
                  value: idsProvider.liveGnnAnomaly,
                  color: Colors.cyanAccent,
                ),
              ),
              // MAE Gauge (Structural)
              Expanded(
                child: _buildSensorGauge(
                  title: "MAE VISUAL",
                  subtitle: "Structure Error",
                  value: idsProvider.liveMaeAnomaly,
                  color: Colors.pinkAccent,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSensorGauge(
      {required String title,
      required String subtitle,
      required double value,
      required Color color}) {
    // Value mapping: normalize sensory scores for the gauge 0-1.0
    double displayValue = (value * 100).clamp(0, 100);

    return Column(
      children: [
        Text(title,
            style: TextStyle(
                color: Colors.white70,
                fontSize: 12,
                fontWeight: FontWeight.bold)),
        Container(
          height: 140,
          child: SfRadialGauge(
            axes: <RadialAxis>[
              RadialAxis(
                minimum: 0,
                maximum: 100,
                showLabels: false,
                showTicks: false,
                startAngle: 180,
                endAngle: 0,
                radiusFactor: 0.8,
                canScaleToFit: true,
                axisLineStyle: AxisLineStyle(
                  thickness: 0.2,
                  color: Colors.white10,
                  thicknessUnit: GaugeSizeUnit.factor,
                ),
                pointers: <GaugePointer>[
                  RangePointer(
                    value: displayValue,
                    width: 0.2,
                    sizeUnit: GaugeSizeUnit.factor,
                    color: color,
                    enableAnimation: true,
                    animationDuration: 1000,
                  ),
                  MarkerPointer(
                    value: displayValue,
                    markerType: MarkerType.circle,
                    color: Colors.white,
                    markerHeight: 10,
                    markerWidth: 10,
                  )
                ],
                annotations: <GaugeAnnotation>[
                  GaugeAnnotation(
                    widget: Text("${displayValue.toStringAsFixed(1)}%",
                        style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.bold,
                            color: Colors.white)),
                    angle: 90,
                    positionFactor: 0.5,
                  )
                ],
              )
            ],
          ),
        ),
        Text(subtitle, style: TextStyle(color: Colors.white38, fontSize: 10)),
      ],
    );
  }

  Widget _buildStatusIndicator(String status) {
    Color color = status == 'normal'
        ? Colors.green
        : (status == 'known_attack' ? Colors.orange : Colors.red);
    return Container(
      padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
          color: color.withOpacity(0.2),
          borderRadius: BorderRadius.circular(5)),
      child: Text(status.toUpperCase(),
          style: TextStyle(
              color: color, fontSize: 10, fontWeight: FontWeight.bold)),
    );
  }
}
