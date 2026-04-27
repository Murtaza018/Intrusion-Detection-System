import 'package:flutter/material.dart';
import '../providers/ids_api_client.dart';

class ReportSummaryScreen extends StatefulWidget {
  const ReportSummaryScreen({Key? key}) : super(key: key);

  @override
  State<ReportSummaryScreen> createState() => _ReportSummaryScreenState();
}

class _ReportSummaryScreenState extends State<ReportSummaryScreen> {
  final _client = IdsApiClient();

  bool _loading = true;
  Map<String, dynamic>? _report;
  String _window = "1d";

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetch());
  }

  Future<void> _fetch() async {
    if (!mounted) return;
    setState(() => _loading = true);

    final data = await _client.fetchReport(window: _window);
    if (mounted) {
      setState(() {
        _report = data;
        _loading = false;
      });
    }
  }

  void _updateWindow(String window) {
    setState(() => _window = window);
    _fetch();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    if (_loading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    if (_report == null) {
      return Scaffold(
        appBar: AppBar(title: const Text("Security Report")),
        body: Center(
          child: Text(
            "No report available for window: $_window",
            style: theme.textTheme.bodyMedium,
          ),
        ),
      );
    }

    final stats = _report!["stats"] as Map<String, dynamic>? ?? {};
    final explainables =
        _report!["explainables"] as Map<String, dynamic>? ?? {};
    final mae = explainables["mae_anomaly"] as Map<String, dynamic>? ?? {};
    final summary = _report!["summary_text"] as String;
    final start = _report!["start_time"] as String;
    final end = _report!["end_time"] as String;

    final detectionRate = (stats["detection_rate"] ?? 0.0) * 100;
    final totalPackets = stats["total_packets"] ?? 0;

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          SliverAppBar.large(
            title: const Text("Security Summary"),
            actions: [
              IconButton(icon: const Icon(Icons.refresh), onPressed: _fetch),
            ],
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Row(
                children: [
                  FilterChip(
                    label: const Text("Daily"),
                    selected: _window == "1d",
                    onSelected: (v) => v ? _updateWindow("1d") : null,
                  ),
                  const SizedBox(width: 8),
                  FilterChip(
                    label: const Text("Weekly"),
                    selected: _window == "1w",
                    onSelected: (v) => v ? _updateWindow("1w") : null,
                  ),
                  const Spacer(),
                  Text(
                    "Traffic Ending: ${end.split('T')[0]}",
                    style: theme.textTheme.labelSmall,
                  ),
                ],
              ),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                _buildHeroSection(theme, detectionRate, stats),
                const SizedBox(height: 24),
                Text("Network Metrics", style: theme.textTheme.titleMedium),
                const SizedBox(height: 12),
                GridView.count(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  crossAxisCount: 2,
                  childAspectRatio: 2.8, // Compact boxes
                  crossAxisSpacing: 12,
                  mainAxisSpacing: 12,
                  children: [
                    _buildMetricTile(theme, "Total Traffic",
                        totalPackets.toString(), Icons.analytics_outlined),
                    _buildMetricTile(
                        theme,
                        "Normal",
                        stats["normal"].toString(),
                        Icons.check_circle_outline,
                        Colors.green),
                    _buildMetricTile(
                        theme,
                        "Attacks",
                        stats["attack"].toString(),
                        Icons.warning_amber_rounded,
                        Colors.orange),
                    _buildMetricTile(
                        theme,
                        "Zero-Day",
                        stats["zero_day"].toString(),
                        Icons.security_outlined,
                        Colors.red),
                  ],
                ),
                const SizedBox(height: 24),
                Text("Explainability Insights (XAI)",
                    style: theme.textTheme.titleMedium),
                const SizedBox(height: 12),
                _buildExplainabilityCard(theme, mae),
                const SizedBox(height: 24),
                Text("Executive Summary", style: theme.textTheme.titleMedium),
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: colorScheme.surfaceVariant.withOpacity(0.3),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                        color: colorScheme.outlineVariant.withOpacity(0.5)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("$start — $end", style: theme.textTheme.labelSmall),
                      const SizedBox(height: 8),
                      Text(summary,
                          style:
                              theme.textTheme.bodyMedium?.copyWith(height: 1.5),
                          textAlign: TextAlign.justify),
                    ],
                  ),
                ),
                const SizedBox(height: 40),
              ]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeroSection(ThemeData theme, double rate, Map stats) {
    final isCritical = rate > 2.0;
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isCritical
              ? [Colors.red[900]!, Colors.red[600]!]
              : [theme.colorScheme.primaryContainer, theme.colorScheme.primary],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
              color: (isCritical ? Colors.red : theme.colorScheme.primary)
                  .withOpacity(0.3),
              blurRadius: 12,
              offset: const Offset(0, 6))
        ],
      ),
      child: Column(
        children: [
          Text("${rate.toStringAsFixed(1)}%",
              style: theme.textTheme.displayLarge
                  ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold)),
          const Text("THREAT DETECTION RATE",
              style: TextStyle(
                  color: Colors.white70,
                  letterSpacing: 1.5,
                  fontSize: 10,
                  fontWeight: FontWeight.bold)),
          const SizedBox(height: 20),
          _buildTrafficDistributionBar(stats),
        ],
      ),
    );
  }

  Widget _buildTrafficDistributionBar(Map stats) {
    final total = (stats["total_packets"] ?? 1).toDouble();
    final n = (stats["normal"] ?? 0) / total;
    final a = (stats["attack"] ?? 0) / total;
    final z = (stats["zero_day"] ?? 0) / total;

    return Column(
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: SizedBox(
            height: 8,
            child: Row(
              children: [
                Expanded(
                    flex: (n * 100).toInt(),
                    child: Container(color: Colors.greenAccent)),
                Expanded(
                    flex: (a * 100).toInt(),
                    child: Container(color: Colors.orangeAccent)),
                Expanded(
                    flex: (z * 100).toInt(),
                    child: Container(color: Colors.white)),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        const Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text("Safe", style: TextStyle(color: Colors.white, fontSize: 10)),
            Text("Known", style: TextStyle(color: Colors.white, fontSize: 10)),
            Text("Zero-Day",
                style: TextStyle(color: Colors.white, fontSize: 10)),
          ],
        )
      ],
    );
  }

  Widget _buildMetricTile(
      ThemeData theme, String label, String value, IconData icon,
      [Color? color]) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: theme.colorScheme.outlineVariant),
      ),
      child: Row(
        children: [
          Icon(icon, size: 18, color: color ?? theme.colorScheme.primary),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(value,
                    style: theme.textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.bold)),
                Text(label,
                    style: theme.textTheme.labelSmall,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildExplainabilityCard(ThemeData theme, Map mae) {
    final avg = mae["avg"] ?? 0.0;
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: theme.colorScheme.outlineVariant)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Row(
              children: [
                const CircleAvatar(
                    radius: 18, child: Icon(Icons.psychology, size: 18)),
                const SizedBox(width: 12),
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text("MAE Anomaly Analysis",
                      style: theme.textTheme.titleSmall),
                  Text("Baseline Deviations",
                      style: theme.textTheme.labelSmall),
                ]),
                const Spacer(),
                _buildAnomalyBadge(avg),
              ],
            ),
            const Divider(height: 24),
            _buildDetailRow("Mean Error", "$avg"),
            _buildDetailRow("Novelty Hits", "${mae["high_mae_count"]} alerts"),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        Text(label, style: const TextStyle(fontSize: 12, color: Colors.grey)),
        Text(value,
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
      ]),
    );
  }

  Widget _buildAnomalyBadge(double score) {
    final color = score > 0.3 ? Colors.red : Colors.green;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: BorderRadius.circular(8)),
      child: Text(score > 0.3 ? "HIGH RISK" : "STABLE",
          style: TextStyle(
              color: color, fontSize: 10, fontWeight: FontWeight.bold)),
    );
  }
}
