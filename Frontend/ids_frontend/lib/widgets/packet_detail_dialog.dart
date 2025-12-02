import 'package:flutter/material.dart';
import '../providers/ids_provider.dart';

class PacketDetailDialog extends StatelessWidget {
  final Packet packet;

  const PacketDetailDialog({required this.packet});

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.transparent,
      insetPadding: EdgeInsets.all(16),
      child: Container(
        width: double.infinity,
        constraints: BoxConstraints(maxWidth: 600),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black26,
              blurRadius: 10,
              offset: Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _buildHeader(context),
            Flexible(
              child: SingleChildScrollView(
                padding: EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildSectionTitle('Packet Information'),
                    _buildInfoGrid(),
                    if (packet.confidence > 0) ...[
                      SizedBox(height: 24),
                      _buildSectionTitle('Detection Results'),
                      _buildDetectionInfo(),
                    ],
                    if (packet.explanation != null) ...[
                      SizedBox(height: 24),
                      _buildSectionTitle('AI-Powered Explanation'),
                      _buildExplanationSection(context),
                    ],
                  ],
                ),
              ),
            ),
            _buildFooter(context),
          ],
        ),
      ),
    );
  }

  // ... (_buildHeader, _buildSectionTitle, _buildInfoGrid, _buildInfoRow, _buildDetectionInfo, _buildMetricCard remain the same) ...
  // (Paste those methods here if copying partially, otherwise use the full block below)

  Widget _buildHeader(BuildContext context) {
    Color statusColor = _getStatusColor(packet.status);
    IconData statusIcon = _getStatusIcon(packet.status);

    return Container(
      padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      decoration: BoxDecoration(
        color: statusColor.withOpacity(0.1),
        borderRadius: BorderRadius.vertical(top: Radius.circular(12)),
      ),
      child: Row(
        children: [
          Icon(statusIcon, color: statusColor, size: 28),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Packet #${packet.id}',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.grey[800],
                  ),
                ),
                Text(
                  _formatTime(packet.timestamp),
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey[600],
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: statusColor,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              _getStatusText(packet.status).toUpperCase(),
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
          ),
          SizedBox(width: 8),
          IconButton(
            icon: Icon(Icons.close, color: Colors.grey[600]),
            onPressed: () => Navigator.pop(context),
            padding: EdgeInsets.zero,
            constraints: BoxConstraints(),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: EdgeInsets.only(bottom: 16),
      child: Text(
        title,
        style: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.bold,
          color: Colors.grey[800],
        ),
      ),
    );
  }

  Widget _buildInfoGrid() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildInfoRow('Source IP', packet.srcIp),
              _buildInfoRow('Source Port', packet.srcPort.toString()),
              _buildInfoRow('Protocol', packet.protocol),
            ],
          ),
        ),
        SizedBox(width: 24),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildInfoRow('Destination IP', packet.dstIp),
              _buildInfoRow('Destination Port', packet.dstPort.toString()),
              _buildInfoRow('Length', '${packet.length} bytes'),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label.toUpperCase(),
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: Colors.grey[600],
            ),
          ),
          SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              fontSize: 16,
              fontFamily: 'RobotoMono',
              color: Colors.grey[900],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetectionInfo() {
    Color statusColor = _getStatusColor(packet.status);
    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: statusColor.withOpacity(0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: statusColor.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          _buildMetricCard(
            title: 'Confidence',
            value: '${(packet.confidence * 100).toStringAsFixed(1)}%',
            icon: Icons.verified,
            color: statusColor,
          ),
        ],
      ),
    );
  }

  Widget _buildMetricCard({
    required String title,
    required String value,
    required IconData icon,
    required Color color,
  }) {
    return Expanded(
      child: Row(
        children: [
          Container(
            padding: EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: color, size: 24),
          ),
          SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title.toUpperCase(),
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: Colors.grey[600],
                ),
              ),
              Text(
                value,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: color,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildExplanationSection(BuildContext context) {
    final explanation = packet.explanation!;
    final String type = explanation['type'] ?? 'UNKNOWN';

    if (type == 'INITIAL_DETECTION' || type.contains('FALLBACK')) {
      return _buildFallbackExplanation(explanation);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildExplanationHeader(explanation),
        SizedBox(height: 16),
        Text(
          explanation['description'] ?? 'No description available.',
          style: TextStyle(fontSize: 16, height: 1.5, color: Colors.grey[800]),
        ),
        SizedBox(height: 24),

        // --- ADDED: Key Indicators Section ---
        if (explanation['key_indicators'] != null)
          _buildKeyIndicators(explanation['key_indicators']),
        // -------------------------------------

        SizedBox(height: 24),
        if (explanation['top_contributing_factors'] != null)
          _buildContributingFactors(explanation['top_contributing_factors']),
        SizedBox(height: 24),
        if (explanation['recommended_actions'] != null)
          _buildRecommendedActions(explanation['recommended_actions']),
      ],
    );
  }

  Widget _buildExplanationHeader(Map<String, dynamic> explanation) {
    String riskLevel = explanation['risk_level'] ?? 'UNKNOWN';
    Color riskColor = _getRiskColor(riskLevel);

    return Row(
      children: [
        Icon(Icons.analytics, color: Colors.purple, size: 28),
        SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                explanation['title']
                        ?.toString()
                        .replaceAll('⚠️ ', '')
                        .replaceAll('🚨 ', '')
                        .replaceAll('🌐 ', '')
                        .replaceAll('🆕 ', '') ??
                    'Analysis',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.purple[800],
                ),
              ),
              if (explanation['attack_classification'] != null)
                Text(
                  explanation['attack_classification'],
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.purple[600],
                    fontStyle: FontStyle.italic,
                  ),
                ),
            ],
          ),
        ),
        Container(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
            color: riskColor,
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Colors.white, size: 16),
              SizedBox(width: 4),
              Text(
                '$riskLevel RISK',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // --- NEW: Key Indicators Widget ---
  Widget _buildKeyIndicators(dynamic indicatorsData) {
    List<String> indicators = [];
    if (indicatorsData is String) {
      indicators = indicatorsData
          .replaceAll('[', '')
          .replaceAll(']', '')
          .split(',')
          .map((e) => e.trim())
          .toList();
    } else if (indicatorsData is List) {
      indicators = indicatorsData.map((e) => e.toString()).toList();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Key Indicators',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.grey[800],
          ),
        ),
        SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: indicators
              .map((indicator) => Chip(
                    avatar:
                        Icon(Icons.search, size: 16, color: Colors.blue[800]),
                    label: Text(indicator),
                    backgroundColor: Colors.blue[50],
                    labelStyle:
                        TextStyle(color: Colors.blue[900], fontSize: 12),
                    side: BorderSide(color: Colors.blue.withOpacity(0.2)),
                  ))
              .toList(),
        ),
      ],
    );
  }
  // ----------------------------------

  Widget _buildContributingFactors(dynamic factorsData) {
    List<dynamic> factors = [];
    if (factorsData is String) {
      String cleaned = factorsData.replaceAll('[', '').replaceAll(']', '');
      List<String> items = cleaned.split('}, {');
      for (String item in items) {
        String jsonItem = item.startsWith('{') ? item : '{$item';
        jsonItem = jsonItem.endsWith('}') ? jsonItem : '$jsonItem}';

        jsonItem = jsonItem
            .replaceAll('factor:', '"factor":')
            .replaceAll('impact:', '"impact":')
            .replaceAll('magnitude:', '"magnitude":')
            .replaceAll('observed_value:', '"observed_value":')
            .replaceAll(': Increased risk', ': "Increased risk"')
            .replaceAll(': Decreased risk', ': "Decreased risk"');

        try {
          Map<String, String> factorMap = {};
          jsonItem
              .replaceAll('{', '')
              .replaceAll('}', '')
              .split(',')
              .forEach((pair) {
            List<String> kv = pair.split(':');
            if (kv.length == 2) {
              String key = kv[0].trim().replaceAll('"', '');
              String value = kv[1].trim().replaceAll('"', '');
              factorMap[key] = value;
            }
          });
          factors.add(factorMap);
        } catch (e) {
          print('Error parsing factor: $e');
        }
      }
    } else if (factorsData is List) {
      factors = factorsData;
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Top Contributing Factors',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.grey[800],
          ),
        ),
        SizedBox(height: 12),
        ...factors
            .map((factor) => _buildFactorTile(factor as Map<String, dynamic>)),
      ],
    );
  }

  Widget _buildFactorTile(Map<String, dynamic> factor) {
    bool isRisk = factor['impact'].toString().contains('Increased');
    Color color = isRisk ? Colors.red : Colors.green;
    IconData icon = isRisk ? Icons.arrow_upward : Icons.arrow_downward;

    return Container(
      margin: EdgeInsets.only(bottom: 8),
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Container(
            padding: EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: color, size: 20),
          ),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  factor['factor'],
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                Text(
                  'Observed Value: ${factor['observed_value']}',
                  style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                factor['magnitude'],
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                  color: color,
                ),
              ),
              Text(
                factor['impact'],
                style: TextStyle(fontSize: 12, color: color),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendedActions(dynamic actionsData) {
    List<String> actions = [];
    if (actionsData is String) {
      actions = actionsData
          .replaceAll('[', '')
          .replaceAll(']', '')
          .split(',')
          .map((e) => e.trim())
          .toList();
    } else if (actionsData is List) {
      actions = actionsData.map((e) => e.toString()).toList();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Recommended Actions',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.grey[800],
          ),
        ),
        SizedBox(height: 12),
        Container(
          decoration: BoxDecoration(
            color: Colors.blue.withOpacity(0.05),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.blue.withOpacity(0.2)),
          ),
          child: Column(
            children: actions
                .map((action) => ListTile(
                      leading:
                          Icon(Icons.check_circle_outline, color: Colors.blue),
                      title: Text(action, style: TextStyle(fontSize: 14)),
                      dense: true,
                    ))
                .toList(),
          ),
        ),
      ],
    );
  }

  Widget _buildFallbackExplanation(Map<String, dynamic> explanation) {
    bool isAnalyzing = explanation['status'] == 'analyzing';
    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.orange.withOpacity(0.1),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.orange.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          isAnalyzing
              ? SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : Icon(Icons.info_outline, color: Colors.orange, size: 28),
          SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  explanation['title'] ?? 'Analysis In Progress',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.orange[800],
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  explanation['description'] ?? 'Please wait...',
                  style: TextStyle(color: Colors.orange[900]),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFooter(BuildContext context) {
    return Padding(
      padding: EdgeInsets.all(16),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Close', style: TextStyle(fontSize: 16)),
          ),
        ],
      ),
    );
  }

  // Helper functions
  String _formatTime(DateTime timestamp) {
    return '${timestamp.year}-${timestamp.month.toString().padLeft(2, '0')}-${timestamp.day.toString().padLeft(2, '0')} '
        '${timestamp.hour.toString().padLeft(2, '0')}:${timestamp.minute.toString().padLeft(2, '0')}:${timestamp.second.toString().padLeft(2, '0')}';
  }

  Color _getStatusColor(String status) {
    switch (status) {
      case 'known_attack':
        return Colors.red;
      case 'zero_day':
        return Colors.orange;
      default:
        return Colors.green;
    }
  }

  IconData _getStatusIcon(String status) {
    switch (status) {
      case 'known_attack':
        return Icons.warning_rounded;
      case 'zero_day':
        return Icons.new_releases_rounded;
      default:
        return Icons.check_circle_rounded;
    }
  }

  String _getStatusText(String status) {
    switch (status) {
      case 'known_attack':
        return 'Known Attack';
      case 'zero_day':
        return 'Zero-Day';
      default:
        return 'Normal';
    }
  }

  Color _getRiskColor(String level) {
    switch (level.toUpperCase()) {
      case 'HIGH':
      case 'CRITICAL':
        return Colors.red;
      case 'MEDIUM':
        return Colors.orange;
      case 'LOW':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }
}
