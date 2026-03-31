// lib/screens/alert_history_screen.dart
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import '../providers/ids_api_client.dart';

class AlertHistoryScreen extends StatefulWidget {
  const AlertHistoryScreen({Key? key}) : super(key: key);

  @override
  State<AlertHistoryScreen> createState() => _AlertHistoryScreenState();
}

class _AlertHistoryScreenState extends State<AlertHistoryScreen> {
  late IdsProvider _provider;
  late IdsApiClient _client;

  bool _loading = true;
  Map<String, dynamic> _historyData = {};
  String _selectedWindow = '24h'; // window selector

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _provider = Provider.of<IdsProvider>(context, listen: false);
      _client = _provider.apiClient; // use public getter, not _api
      _fetchHistory();
    });
  }

  Future<void> _fetchHistory() async {
    try {
      final data = await _client.fetchHistory(
        window: _selectedWindow,
        limit: 10000,
      );
      if (data != null) {
        _historyData = data;
      } else {
        _historyData = {};
      }
    } catch (e) {
      debugPrint("History fetch error: $e");
      _historyData = {};
    } finally {
      if (mounted) {
        setState(() => _loading = false);
      }
    }
  }

  List<Widget> _buildKpiCards() {
    final perf = _historyData["performance"] as Map<String, dynamic>? ?? {};

    final total = perf["total_alerts"] as int? ?? 0;
    final detRate = perf["detection_rate"] as num? ?? 0.0;
    final avgAnomaly = perf["avg_anomaly"] as num? ?? 0.0;

    return [
      _KpiCard(
        title: "Total Alerts",
        value: total.toString(),
        color: Colors.blue,
      ),
      _KpiCard(
        title: "Det. Rate",
        value: "${(detRate * 100).toStringAsFixed(1)}%",
        color: Colors.green,
      ),
      _KpiCard(
        title: "Avg Anomaly",
        value: avgAnomaly.toStringAsFixed(3),
        color: Colors.orange,
      ),
    ];
  }

  Widget _timeSeriesChart() {
    final series = (_historyData["alerts_volume"] as List?) ?? [];
    if (series.isEmpty) {
      return const Center(child: Text("No data"));
    }

    final List<FlSpot> spots = series.map((e) {
      final ts = (e["timestamp"] as num?)?.toDouble() ?? 0.0;
      final count = (e["count"] as num?)?.toDouble() ?? 0.0;
      return FlSpot(ts, count);
    }).toList();

    if (spots.isEmpty) {
      return const Center(child: Text("No data"));
    }

    final minT = spots.first.x;
    final maxT = spots.last.x;
    final dt = (maxT - minT).clamp(1.0, double.infinity);

    final spotsNorm = spots.map((s) {
      final x = dt > 0 ? (s.x - minT) / dt : 0.0;
      return FlSpot(x, s.y);
    }).toList();

    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.white12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: LineChart(
        LineChartData(
          lineTouchData: LineTouchData(enabled: false),
          gridData: FlGridData(
            show: true,
            horizontalInterval: 10.0,
            drawVerticalLine: true,
            getDrawingHorizontalLine: (value) {
              return FlLine(
                color: Colors.white.withOpacity(0.1),
                strokeWidth: 0.5,
              );
            },
            getDrawingVerticalLine: (value) {
              return FlLine(
                color: Colors.white.withOpacity(0.1),
                strokeWidth: 0.5,
              );
            },
          ),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                interval: 10.0,
                getTitlesWidget: (val, meta) {
                  return Text(
                    val.toStringAsFixed(0),
                    style: const TextStyle(
                      fontSize: 10,
                      color: Colors.white70,
                    ),
                  );
                },
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: false,
              ),
            ),
          ),
          borderData: FlBorderData(
            show: false,
          ),
          lineBarsData: [
            LineChartBarData(
              spots: spotsNorm,
              isCurved: true,
              barWidth: 2,
              color: const Color(0xFF00E5FF),
              belowBarData: BarAreaData(
                show: true,
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    const Color(0xFF00E5FF).withOpacity(0.3),
                    const Color(0xFF00E5FF).withOpacity(0.05),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _intrusionTypeChart() {
    final series = (_historyData["by_type"] as List?) ?? [];
    if (series.isEmpty) {
      return const Center(child: Text("No data"));
    }

    final List<BarChartGroupData> bars = series.map((e) {
      final label = e["label"] as String? ?? "unknown";
      final count = (e["count"] as num?)?.toDouble() ?? 0.0;

      return BarChartGroupData(
        x: series.indexOf(e),
        barRods: [
          BarChartRodData(
            toY: count,
            color: const Color(0xFF7C4DFF),
            width: 16,
          ),
        ],
      );
    }).toList();

    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: Colors.white12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: BarChart(
        BarChartData(
          barTouchData: BarTouchData(enabled: false),
          gridData: FlGridData(
            show: false,
          ),
          titlesData: FlTitlesData(
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (value, meta) {
                  final index = value.toInt();
                  if (index < 0 || index >= series.length) {
                    return Container();
                  }
                  final e = series[index];
                  final label = (e["label"] as String?) ?? "N/A";
                  return RotatedBox(
                    quarterTurns: 1,
                    child: Text(
                      label,
                      style: const TextStyle(
                        fontSize: 9,
                        color: Colors.white70,
                      ),
                    ),
                  );
                },
              ),
            ),
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                interval: 10.0,
                getTitlesWidget: (value, meta) {
                  return Text(
                    value.toStringAsFixed(0),
                    style: const TextStyle(
                      fontSize: 9,
                      color: Colors.white70,
                    ),
                  );
                },
              ),
            ),
          ),
          borderData: FlBorderData(
            show: false,
          ),
          barGroups: bars,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Historical Alert Analytics"),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(48),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(
              children: [
                Text(
                  "Window:",
                  style: const TextStyle(
                    fontSize: 12,
                    color: Colors.white54,
                  ),
                ),
                const SizedBox(width: 8),
                DropdownButton<String>(
                  value: _selectedWindow,
                  items: const [
                    DropdownMenuItem(value: '1h', child: Text("1h")),
                    DropdownMenuItem(value: '24h', child: Text("24h")),
                    DropdownMenuItem(value: '7d', child: Text("7d")),
                    DropdownMenuItem(value: '30d', child: Text("30d")),
                  ],
                  onChanged: (value) {
                    if (value != null) {
                      setState(() {
                        _selectedWindow = value;
                      });
                      _fetchHistory();
                    }
                  },
                  style: const TextStyle(
                    fontSize: 12,
                    color: Colors.white,
                  ),
                  dropdownColor: const Color(0xFF1A1F24),
                  icon: const Icon(
                    Icons.arrow_drop_down,
                    color: Colors.white54,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : SingleChildScrollView(
              padding: const EdgeInsets.all(12),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: _buildKpiCards(),
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    "Alerts Over Time",
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF00E5FF),
                    ),
                  ),
                  const SizedBox(height: 8),
                  SizedBox(
                    height: 180,
                    child: _timeSeriesChart(),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    "Intrusion Types",
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF00E5FF),
                    ),
                  ),
                  const SizedBox(height: 8),
                  SizedBox(
                    height: 160,
                    child: _intrusionTypeChart(),
                  ),
                ],
              ),
            ),
    );
  }
}

class _KpiCard extends StatelessWidget {
  final String title;
  final String value;
  final Color color;

  const _KpiCard({
    Key? key,
    required this.title,
    required this.value,
    required this.color,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: color.withOpacity(0.15),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              value,
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              title,
              style: TextStyle(
                fontSize: 10,
                color: Colors.white.withOpacity(0.7),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
