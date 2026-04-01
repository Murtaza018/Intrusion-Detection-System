import 'package:flutter/material.dart';
import '../providers/ids_config.dart';
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

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_report == null) {
      return Center(
        child: Text(
          "No report available for window: $_window",
          style: theme.textTheme.bodyMedium,
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

    return Scaffold(
      appBar: AppBar(
        title: const Text("Security Summary Report"),
        bottom: PreferredSize(
          preferredSize: const Size(double.infinity, 48),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                const Text("Window:", style: TextStyle(fontSize: 12)),
                const SizedBox(width: 8),
                FilterChip(
                  label: const Text("Daily"),
                  selected: _window == "1d",
                  onSelected: (v) {
                    if (v) {
                      setState(() => _window = "1d");
                      _fetch();
                    }
                  },
                ),
                const SizedBox(width: 8),
                FilterChip(
                  label: const Text("Weekly"),
                  selected: _window == "1w",
                  onSelected: (v) {
                    if (v) {
                      setState(() => _window = "1w");
                      _fetch();
                    }
                  },
                ),
              ],
            ),
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "Summary",
                      style: theme.textTheme.titleSmall,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      "$start – $end",
                      style: theme.textTheme.bodySmall,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      summary,
                      style: theme.textTheme.bodyMedium,
                      textAlign: TextAlign.justify,
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // KPI cards
            Text(
              "Metrics",
              style: theme.textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                _buildStatCard(
                    "Total", (stats["total_packets"] ?? 0).toString()),
                const SizedBox(width: 8),
                _buildStatCard(
                  "Normal",
                  (stats["normal"] ?? 0).toString(),
                  color: Colors.green[400],
                ),
                const SizedBox(width: 8),
                _buildStatCard(
                  "Attack",
                  (stats["attack"] ?? 0).toString(),
                  color: Colors.orange[400],
                ),
                const SizedBox(width: 8),
                _buildStatCard(
                  "Zero‑Day",
                  (stats["zero_day"] ?? 0).toString(),
                  color: Colors.red[400],
                ),
              ],
            ),

            const SizedBox(height: 16),

            // Explainables section
            Text(
              "Explainability Insights",
              style: theme.textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "MAE Anomaly",
                      style: theme.textTheme.titleSmall,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      "Average anomaly score: ${mae["avg"]}",
                      style: theme.textTheme.bodySmall,
                    ),
                    Text(
                      "High‑anomaly patterns: ${mae["high_mae_count"]} samples",
                      style: theme.textTheme.bodySmall,
                    ),
                    if (mae["high_mae_count"] as int > 0)
                      Text(
                        "For example, one packet had MAE ≈ ${mae["high_mae_example"]}",
                        style: theme.textTheme.bodySmall,
                      ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatCard(String label, String value, {Color? color}) {
    final theme = Theme.of(context);
    return Expanded(
      child: Card(
        color: color ?? theme.cardColor,
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: Colors.white,
                ),
              ),
              Text(
                value,
                style: theme.textTheme.titleMedium?.copyWith(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
