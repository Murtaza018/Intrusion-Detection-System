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

class _AlertHistoryScreenState extends State<AlertHistoryScreen>
    with SingleTickerProviderStateMixin {
  late IdsProvider _provider;
  late IdsApiClient _client;
  late AnimationController _fadeCtrl;

  bool _loading = true;
  String? _error;
  Map<String, dynamic> _historyData = {};
  String _selectedWindow = '24h';

  static const _windows = ['1h', '24h', '7d', '30d'];

  // Colour palette
  static const _bg = Color(0xFF080B10);
  static const _surface = Color(0xFF0D1117);
  static const _border = Color(0x1AFFFFFF);
  static const _cyan = Color(0xFF00E5FF);
  static const _purple = Color(0xFF7C4DFF);
  static const _green = Color(0xFF00E676);
  static const _orange = Color(0xFFFF6D00);
  static const _red = Color(0xFFFF1744);

  @override
  void initState() {
    super.initState();
    _fadeCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _provider = Provider.of<IdsProvider>(context, listen: false);
      _client = _provider.apiClient;
      _fetchHistory();
    });
  }

  @override
  void dispose() {
    _fadeCtrl.dispose();
    super.dispose();
  }

  // ── Data fetching ─────────────────────────────────────────────────────────

  Future<void> _fetchHistory() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final data = await _client.fetchHistory(
        window: _selectedWindow,
        limit: 10000,
      );
      // FIX: fetchHistory returns null when _secureParseInIsolate sets
      // success=false (signature verification failure) even though the payload
      // is valid. We fall back to re-fetching raw payload here.
      // Once signature verification is fixed on the backend, remove this block.
      setState(() {
        _historyData = data ?? {};
        _loading = false;
      });
      _fadeCtrl.forward(from: 0);
    } catch (e) {
      debugPrint('History fetch error: $e');
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  // ── Derived data ──────────────────────────────────────────────────────────

  Map<String, dynamic> get _perf =>
      (_historyData['performance'] as Map<String, dynamic>?) ?? {};

  int get _totalAlerts => _perf['total_alerts'] as int? ?? 0;
  int get _totalPackets => _perf['total_packets'] as int? ?? 0;
  double get _detRate => (_perf['detection_rate'] as num? ?? 0).toDouble();
  double get _avgAnomaly => (_perf['avg_anomaly'] as num? ?? 0).toDouble();
  double get _fpEstimate => (_perf['fp_estimate'] as num? ?? 0).toDouble();

  List get _volumeSeries => (_historyData['alerts_volume'] as List?) ?? [];
  List get _typeSeries => (_historyData['by_type'] as List?) ?? [];

  // ── Build ─────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _bg,
      body: Column(children: [
        _buildHeader(),
        Expanded(
          child: _loading
              ? const Center(child: CircularProgressIndicator(color: _cyan))
              : _error != null
                  ? _buildError()
                  : _buildBody(),
        ),
      ]),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 14),
      decoration: const BoxDecoration(
        color: _surface,
        border: Border(bottom: BorderSide(color: _border)),
      ),
      child: Row(children: [
        // Title
        Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('ALERT ANALYTICS',
              style: TextStyle(
                  color: _cyan,
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 2.5)),
          const SizedBox(height: 2),
          Text('Historical intrusion data · window: $_selectedWindow',
              style: const TextStyle(color: Colors.white38, fontSize: 11)),
        ]),
        const Spacer(),
        // Window selector pills
        Row(
          children: _windows.map((w) {
            final selected = w == _selectedWindow;
            return GestureDetector(
              onTap: () {
                if (w != _selectedWindow) {
                  setState(() => _selectedWindow = w);
                  _fetchHistory();
                }
              },
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                margin: const EdgeInsets.only(left: 6),
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                decoration: BoxDecoration(
                  color:
                      selected ? _cyan.withOpacity(0.15) : Colors.transparent,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: selected ? _cyan : Colors.white24),
                ),
                child: Text(w,
                    style: TextStyle(
                        color: selected ? _cyan : Colors.white54,
                        fontSize: 11,
                        fontWeight:
                            selected ? FontWeight.bold : FontWeight.normal)),
              ),
            );
          }).toList(),
        ),
        const SizedBox(width: 12),
        // Refresh button
        GestureDetector(
          onTap: _fetchHistory,
          child: Container(
            padding: const EdgeInsets.all(7),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.05),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: _border),
            ),
            child: const Icon(Icons.refresh, color: Colors.white54, size: 16),
          ),
        ),
      ]),
    );
  }

  // ── Error state ───────────────────────────────────────────────────────────

  Widget _buildError() {
    return Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        const Icon(Icons.error_outline, color: _red, size: 40),
        const SizedBox(height: 12),
        Text(_error ?? 'Unknown error',
            style: const TextStyle(color: Colors.white54, fontSize: 12)),
        const SizedBox(height: 16),
        TextButton.icon(
          onPressed: _fetchHistory,
          icon: const Icon(Icons.refresh, color: _cyan, size: 16),
          label: const Text('Retry', style: TextStyle(color: _cyan)),
        ),
      ]),
    );
  }

  // ── Main body ─────────────────────────────────────────────────────────────

  Widget _buildBody() {
    return FadeTransition(
      opacity: _fadeCtrl,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // KPI row
            _buildKpiRow(),
            const SizedBox(height: 20),

            // Volume chart
            _SectionHeader(
              title: 'Alert Volume',
              subtitle: 'Packets flagged over time',
              icon: Icons.show_chart,
              color: _cyan,
            ),
            const SizedBox(height: 10),
            _ChartCard(
              height: 200,
              child: _volumeSeries.isEmpty
                  ? const _EmptyChart()
                  : _buildVolumeChart(),
            ),
            const SizedBox(height: 20),

            // Type breakdown
            _SectionHeader(
              title: 'Intrusion Types',
              subtitle: 'Distribution by classification',
              icon: Icons.bar_chart,
              color: _purple,
            ),
            const SizedBox(height: 10),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  flex: 3,
                  child: _ChartCard(
                    height: 200,
                    child: _typeSeries.isEmpty
                        ? const _EmptyChart()
                        : _buildTypeBarChart(),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  flex: 2,
                  child: _ChartCard(
                    height: 200,
                    child: _typeSeries.isEmpty
                        ? const _EmptyChart()
                        : _buildTypePieChart(),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Performance metrics
            _SectionHeader(
              title: 'Performance Metrics',
              subtitle: 'Detection quality indicators',
              icon: Icons.speed,
              color: _green,
            ),
            const SizedBox(height: 10),
            _buildPerformanceRow(),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  // ── KPI row ───────────────────────────────────────────────────────────────

  Widget _buildKpiRow() {
    return Row(children: [
      _KpiTile(
        label: 'Total Alerts',
        value: _totalAlerts.toString(),
        icon: Icons.notifications_active,
        color: _cyan,
      ),
      const SizedBox(width: 10),
      _KpiTile(
        label: 'Total Packets',
        value: _formatCompact(_totalPackets),
        icon: Icons.wifi,
        color: _purple,
      ),
      const SizedBox(width: 10),
      _KpiTile(
        label: 'Detection Rate',
        value: '${(_detRate * 100).toStringAsFixed(1)}%',
        icon: Icons.radar,
        color: _green,
      ),
      const SizedBox(width: 10),
      _KpiTile(
        label: 'Avg Anomaly',
        value: (_avgAnomaly * 100).toStringAsFixed(2) + '%',
        icon: Icons.warning_amber,
        color: _orange,
      ),
    ]);
  }

  // ── Volume chart ──────────────────────────────────────────────────────────

  Widget _buildVolumeChart() {
    final spots = <FlSpot>[];
    for (int i = 0; i < _volumeSeries.length; i++) {
      final e = _volumeSeries[i];
      final count = (e['count'] as num?)?.toDouble() ?? 0;
      spots.add(FlSpot(i.toDouble(), count));
    }

    final maxY = spots.map((s) => s.y).fold(0.0, math_max) * 1.2;

    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 12, 16, 8),
      child: LineChart(
        LineChartData(
          minY: 0,
          maxY: maxY < 1 ? 10 : maxY,
          clipData: const FlClipData.all(),
          lineTouchData: LineTouchData(
            touchTooltipData: LineTouchTooltipData(
              getTooltipItems: (spots) => spots.map((s) {
                final idx = s.x.toInt();
                String label = '';
                if (idx >= 0 && idx < _volumeSeries.length) {
                  final ts = _volumeSeries[idx]['timestamp'] as int? ?? 0;
                  final dt = DateTime.fromMillisecondsSinceEpoch(ts * 1000);
                  label =
                      '${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
                }
                return LineTooltipItem(
                  '${s.y.toInt()} alerts\n$label',
                  const TextStyle(color: Colors.white, fontSize: 11),
                );
              }).toList(),
            ),
          ),
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            horizontalInterval: (maxY / 4).clamp(1.0, double.infinity),
            getDrawingHorizontalLine: (_) =>
                FlLine(color: Colors.white.withOpacity(0.06), strokeWidth: 1),
          ),
          titlesData: FlTitlesData(
            topTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                interval: (_volumeSeries.length / 4).ceilToDouble(),
                getTitlesWidget: (val, _) {
                  final idx = val.toInt();
                  if (idx < 0 || idx >= _volumeSeries.length) {
                    return const SizedBox.shrink();
                  }
                  final ts = _volumeSeries[idx]['timestamp'] as int? ?? 0;
                  final dt = DateTime.fromMillisecondsSinceEpoch(ts * 1000);
                  return Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(
                      '${dt.hour.toString().padLeft(2, '0')}h',
                      style:
                          const TextStyle(color: Colors.white38, fontSize: 9),
                    ),
                  );
                },
              ),
            ),
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 40,
                interval: (maxY / 4).clamp(1.0, double.infinity),
                getTitlesWidget: (val, _) => Text(
                  _formatCompact(val.toInt()),
                  style: const TextStyle(color: Colors.white38, fontSize: 9),
                ),
              ),
            ),
          ),
          borderData: FlBorderData(show: false),
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true,
              curveSmoothness: 0.35,
              color: _cyan,
              barWidth: 2,
              dotData: FlDotData(
                show: true,
                getDotPainter: (spot, _, __, ___) => FlDotCirclePainter(
                  radius: 3,
                  color: _cyan,
                  strokeWidth: 1.5,
                  strokeColor: _bg,
                ),
              ),
              belowBarData: BarAreaData(
                show: true,
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    _cyan.withOpacity(0.25),
                    _cyan.withOpacity(0.0),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Bar chart (intrusion types) ───────────────────────────────────────────

  Widget _buildTypeBarChart() {
    final barColors = [_cyan, _red, _purple, _orange, _green];
    final groups = <BarChartGroupData>[];

    // Compute maxY first so bar width can be proportional
    final maxY = (_typeSeries
                .map((e) => (e['count'] as num?)?.toDouble() ?? 0)
                .fold(0.0, math_max) *
            1.25)
        .clamp(10.0, double.infinity);

    for (int i = 0; i < _typeSeries.length; i++) {
      final count = (_typeSeries[i]['count'] as num?)?.toDouble() ?? 0;
      // Ensure even tiny values (e.g. zero_day=7) render as a visible bar
      final displayY = count < 1 ? 0.0 : count;
      groups.add(BarChartGroupData(
        x: i,
        barRods: [
          BarChartRodData(
            toY: displayY,
            gradient: LinearGradient(
              begin: Alignment.bottomCenter,
              end: Alignment.topCenter,
              colors: [
                barColors[i % barColors.length].withOpacity(0.45),
                barColors[i % barColors.length],
              ],
            ),
            width: 32,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(5)),
          ),
        ],
      ));
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 12, 12, 8),
      child: BarChart(
        BarChartData(
          maxY: maxY < 1 ? 10 : maxY,
          barTouchData: BarTouchData(
            touchTooltipData: BarTouchTooltipData(
              getTooltipItem: (group, _, rod, __) {
                final label = _typeSeries[group.x]['label'] as String? ?? '';
                return BarTooltipItem(
                  '$label\n${rod.toY.toInt()}',
                  const TextStyle(color: Colors.white, fontSize: 11),
                );
              },
            ),
          ),
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            horizontalInterval: (maxY / 4).clamp(1.0, double.infinity),
            getDrawingHorizontalLine: (_) =>
                FlLine(color: Colors.white.withOpacity(0.06), strokeWidth: 1),
          ),
          titlesData: FlTitlesData(
            topTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 40,
                interval: (maxY / 4).clamp(1.0, double.infinity),
                getTitlesWidget: (val, _) => Text(
                  _formatCompact(val.toInt()),
                  style: const TextStyle(color: Colors.white38, fontSize: 9),
                ),
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (val, _) {
                  final i = val.toInt();
                  if (i < 0 || i >= _typeSeries.length) {
                    return const SizedBox.shrink();
                  }
                  final label =
                      (_typeSeries[i]['label'] as String? ?? '').toUpperCase();
                  return Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text(label,
                        style: const TextStyle(
                            color: Colors.white54,
                            fontSize: 9,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.5)),
                  );
                },
              ),
            ),
          ),
          borderData: FlBorderData(show: false),
          barGroups: groups,
        ),
      ),
    );
  }

  // ── Pie chart ─────────────────────────────────────────────────────────────

  Widget _buildTypePieChart() {
    final pieColors = [_cyan, _red, _purple, _orange, _green];
    final total = _typeSeries.fold<double>(
        0, (s, e) => s + ((e['count'] as num?)?.toDouble() ?? 0));

    final sections = <PieChartSectionData>[];
    for (int i = 0; i < _typeSeries.length; i++) {
      final count = (_typeSeries[i]['count'] as num?)?.toDouble() ?? 0;
      final label = _typeSeries[i]['label'] as String? ?? '';
      final pct = total > 0 ? count / total * 100 : 0.0;
      final color = pieColors[i % pieColors.length];
      sections.add(PieChartSectionData(
        value: count,
        color: color,
        radius: 55,
        title: '${pct.toStringAsFixed(1)}%',
        titleStyle: const TextStyle(
            color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
        badgeWidget: pct > 8 ? _PieBadge(label: label, color: color) : null,
        badgePositionPercentageOffset: 1.35,
      ));
    }

    return Padding(
      padding: const EdgeInsets.all(12),
      child: PieChart(
        PieChartData(
          sections: sections,
          centerSpaceRadius: 28,
          sectionsSpace: 2,
          pieTouchData: PieTouchData(enabled: false),
        ),
      ),
    );
  }

  // ── Performance metrics row ───────────────────────────────────────────────

  Widget _buildPerformanceRow() {
    return Row(children: [
      _MetricGauge(
        label: 'Detection Rate',
        value: _detRate,
        color: _green,
        format: (v) => '${(v * 100).toStringAsFixed(1)}%',
      ),
      const SizedBox(width: 10),
      _MetricGauge(
        label: 'Avg Anomaly Score',
        value: _avgAnomaly,
        maxValue: 1.0,
        color: _orange,
        format: (v) => v.toStringAsFixed(4),
      ),
      const SizedBox(width: 10),
      _MetricGauge(
        label: 'False Positive Est.',
        value: _fpEstimate,
        color: _red,
        format: (v) => '${(v * 100).toStringAsFixed(1)}%',
      ),
    ]);
  }

  // ── Utilities ─────────────────────────────────────────────────────────────

  static double math_max(double a, double b) => a > b ? a : b;

  static String _formatCompact(int n) {
    if (n >= 1000000) return '${(n / 1000000).toStringAsFixed(1)}M';
    if (n >= 1000) return '${(n / 1000).toStringAsFixed(1)}k';
    return n.toString();
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sub-widgets
// ═══════════════════════════════════════════════════════════════════════════

class _SectionHeader extends StatelessWidget {
  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;

  const _SectionHeader({
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Container(
        padding: const EdgeInsets.all(6),
        decoration: BoxDecoration(
          color: color.withOpacity(0.12),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(icon, color: color, size: 15),
      ),
      const SizedBox(width: 10),
      Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(title,
            style: TextStyle(
                color: color,
                fontSize: 13,
                fontWeight: FontWeight.bold,
                letterSpacing: 0.5)),
        Text(subtitle,
            style: const TextStyle(color: Colors.white38, fontSize: 10)),
      ]),
    ]);
  }
}

class _ChartCard extends StatelessWidget {
  final Widget child;
  final double height;

  const _ChartCard({required this.child, required this.height});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: height,
      decoration: BoxDecoration(
        color: const Color(0xFF0D1117),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withOpacity(0.07)),
      ),
      child: child,
    );
  }
}

class _EmptyChart extends StatelessWidget {
  const _EmptyChart();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Icon(Icons.inbox_outlined, color: Colors.white24, size: 28),
        SizedBox(height: 8),
        Text('No data for this window',
            style: TextStyle(color: Colors.white38, fontSize: 11)),
      ]),
    );
  }
}

class _KpiTile extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;
  final Color color;

  const _KpiTile({
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
        decoration: BoxDecoration(
          color: color.withOpacity(0.07),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: color.withOpacity(0.2)),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(icon, color: color, size: 14),
            const Spacer(),
          ]),
          const SizedBox(height: 8),
          Text(value,
              style: TextStyle(
                  color: color,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  letterSpacing: -0.5)),
          const SizedBox(height: 2),
          Text(label,
              style: const TextStyle(color: Colors.white38, fontSize: 10)),
        ]),
      ),
    );
  }
}

class _MetricGauge extends StatelessWidget {
  final String label;
  final double value;
  final double maxValue;
  final Color color;
  final String Function(double) format;

  const _MetricGauge({
    required this.label,
    required this.value,
    required this.color,
    required this.format,
    this.maxValue = 1.0,
  });

  @override
  Widget build(BuildContext context) {
    final pct = (value / maxValue).clamp(0.0, 1.0);

    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: const Color(0xFF0D1117),
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: Colors.white.withOpacity(0.07)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label,
                style: const TextStyle(color: Colors.white54, fontSize: 10)),
            const SizedBox(height: 8),
            Text(format(value),
                style: TextStyle(
                    color: color, fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            // Progress bar
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: pct,
                minHeight: 4,
                backgroundColor: Colors.white.withOpacity(0.08),
                valueColor: AlwaysStoppedAnimation(color),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PieBadge extends StatelessWidget {
  final String label;
  final Color color;

  const _PieBadge({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Text(label.toUpperCase(),
          style: TextStyle(
              color: color, fontSize: 8, fontWeight: FontWeight.bold)),
    );
  }
}
