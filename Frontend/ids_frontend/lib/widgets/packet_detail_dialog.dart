import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import 'label_selection_dialog.dart';

class PacketDetailDialog extends StatelessWidget {
  final Packet packet;

  const PacketDetailDialog({required this.packet});

  @override
  Widget build(BuildContext context) {
    // Determine width based on screen size to be responsive
    double width = MediaQuery.of(context).size.width * 0.85;
    double height = MediaQuery.of(context).size.height * 0.85;

    return Dialog(
      backgroundColor: Colors.transparent,
      insetPadding: EdgeInsets.all(16),
      child: Container(
        width: width,
        height: height,
        constraints: BoxConstraints(maxWidth: 1100, maxHeight: 800),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
                color: Colors.black26, blurRadius: 20, offset: Offset(0, 10)),
          ],
        ),
        child: Column(
          children: [
            // --- HEADER ---
            _buildHeader(context),

            // --- CONTENT BODY ---
            Expanded(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // LEFT COLUMN: Technical Specs & Detection
                  Expanded(
                    flex: 4,
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.grey[50],
                        border:
                            Border(right: BorderSide(color: Colors.grey[200]!)),
                      ),
                      child: SingleChildScrollView(
                        padding: EdgeInsets.all(24),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _buildSectionTitle('Network Details'),
                            _buildInfoGrid(),
                            if (packet.confidence > 0) ...[
                              SizedBox(height: 32),
                              _buildSectionTitle('AI Detection Score'),
                              _buildDetectionInfo(),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ),

                  // RIGHT COLUMN: XAI Explanation
                  Expanded(
                    flex: 6,
                    child: Container(
                      color: Colors.white,
                      child: SingleChildScrollView(
                        padding: EdgeInsets.all(24),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            if (packet.explanation != null) ...[
                              _buildSectionTitle('AI Analysis & Reasoning'),
                              _buildExplanationSection(context),
                            ] else
                              Center(
                                child: Padding(
                                  padding: const EdgeInsets.only(top: 100.0),
                                  child: Text(
                                      "No explanation available for this packet.",
                                      style: TextStyle(color: Colors.grey)),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // --- FOOTER ---
            _buildFooter(context),
          ],
        ),
      ),
    );
  }

// ... (Imports and Build method identical to previous wide version) ...
// Replace only the _buildFooter method with this logic:

  Widget _buildFooter(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context, listen: false);
    bool isSelected = provider.isSelected(packet.id);

    // Logic for Buttons based on packet status
    List<Widget> buttons = [];

    if (packet.status == 'normal') {
      // Normal Packet Options
      buttons.add(_buildActionButton(context, provider,
          icon: Icons.bug_report,
          label: "Report Missed Attack",
          color: Colors.red,
          queueType: 'gan' // False Negative -> GAN
          ));
    } else {
      // Attack / Zero-Day Options
      buttons.add(_buildActionButton(context, provider,
          icon: Icons.thumb_up_alt_outlined,
          label: "Report False Alarm",
          color: Colors.green,
          queueType: 'jitter' // False Positive -> Jittering
          ));

      if (packet.status == 'zero_day') {
        buttons.add(SizedBox(width: 12));
        buttons.add(_buildActionButton(context, provider,
            icon: Icons.science,
            label: "Confirm Zero-Day",
            color: Colors.deepPurple,
            queueType: 'gan' // True Zero-Day -> GAN
            ));
      }
    }

    // Toggle Button (If already selected)
    if (isSelected) {
      buttons = [
        OutlinedButton.icon(
          onPressed: () {
            provider.toggleSelection(packet, ''); // Removes it
            Navigator.pop(context);
          },
          icon: Icon(Icons.remove_circle_outline),
          label: Text("Remove from List"),
          style: OutlinedButton.styleFrom(foregroundColor: Colors.grey),
        )
      ];
    }

    return Container(
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
          color: Colors.grey[50],
          borderRadius: BorderRadius.vertical(bottom: Radius.circular(16)),
          border: Border(top: BorderSide(color: Colors.grey[200]!))),
      child: Row(mainAxisAlignment: MainAxisAlignment.end, children: buttons),
    );
  }

  Widget _buildActionButton(BuildContext context, IdsProvider provider,
      {required IconData icon,
      required String label,
      required Color color,
      required String queueType}) {
    return ElevatedButton.icon(
      onPressed: () {
        // Simple Toggle for everything. Labeling happens later on Adaptation Screen.
        provider.toggleSelection(packet, queueType);

        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(
              'Added to ${queueType.toUpperCase()} Queue. Label it in the Adapt tab.'),
          backgroundColor: color,
          duration: Duration(seconds: 1),
        ));
      },
      icon: Icon(icon, size: 18),
      label: Text(label),
      style: ElevatedButton.styleFrom(
          backgroundColor: color, foregroundColor: Colors.white),
    );
  }

  Widget _buildHeader(BuildContext context) {
    Color statusColor = _getStatusColor(packet.status);
    IconData statusIcon = _getStatusIcon(packet.status);

    return Container(
      padding: EdgeInsets.symmetric(horizontal: 24, vertical: 20),
      decoration: BoxDecoration(
        color: statusColor.withOpacity(0.08),
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
        border: Border(bottom: BorderSide(color: statusColor.withOpacity(0.2))),
      ),
      child: Row(
        children: [
          Container(
            padding: EdgeInsets.all(8),
            decoration:
                BoxDecoration(color: Colors.white, shape: BoxShape.circle),
            child: Icon(statusIcon, color: statusColor, size: 32),
          ),
          SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text('Packet #${packet.id}',
                        style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            color: Colors.grey[900])),
                    SizedBox(width: 12),
                    Container(
                      padding:
                          EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                          color: statusColor,
                          borderRadius: BorderRadius.circular(6)),
                      child: Text(_getStatusText(packet.status).toUpperCase(),
                          style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 11,
                              letterSpacing: 0.5)),
                    ),
                  ],
                ),
                SizedBox(height: 4),
                Text(_formatTime(packet.timestamp),
                    style: TextStyle(fontSize: 13, color: Colors.grey[600])),
              ],
            ),
          ),
          IconButton(
              icon: Icon(Icons.close, color: Colors.grey[500]),
              onPressed: () => Navigator.pop(context)),
        ],
      ),
    );
  }

  // ... (Paste _buildSectionTitle, _buildInfoGrid, _buildDetectionInfo, _buildExplanationSection and its children here from the previous version) ...
  // Assuming you have the code for these widgets from the previous turn. If you need them repeated, let me know!

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: EdgeInsets.only(bottom: 16),
      child: Text(title,
          style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: Colors.grey[500],
              letterSpacing: 1.0)),
    );
  }

  Widget _buildInfoGrid() {
    return Column(
      children: [
        _buildDetailRow(
            Icons.upload, 'Source', '${packet.srcIp}:${packet.srcPort}'),
        Divider(height: 24),
        _buildDetailRow(
            Icons.download, 'Destination', '${packet.dstIp}:${packet.dstPort}'),
        Divider(height: 24),
        Row(
          children: [
            Expanded(
                child:
                    _buildDetailRow(Icons.router, 'Protocol', packet.protocol)),
            Expanded(
                child: _buildDetailRow(
                    Icons.data_usage, 'Length', '${packet.length} bytes')),
          ],
        ),
      ],
    );
  }

  Widget _buildDetailRow(IconData icon, String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 20, color: Colors.grey[400]),
        SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label,
                  style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: Colors.grey[500])),
              SizedBox(height: 2),
              Text(value,
                  style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w500,
                      color: Colors.grey[800],
                      fontFamily: 'RobotoMono')),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildDetectionInfo() {
    Color statusColor = _getStatusColor(packet.status);
    return Container(
      padding: EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: statusColor.withOpacity(0.3), width: 1.5),
        boxShadow: [
          BoxShadow(color: statusColor.withOpacity(0.05), blurRadius: 10)
        ],
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("Model Confidence",
                  style: TextStyle(fontSize: 14, color: Colors.grey[700])),
              Text('${(packet.confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: statusColor)),
            ],
          ),
          SizedBox(height: 8),
          LinearProgressIndicator(
            value: packet.confidence,
            backgroundColor: Colors.grey[200],
            color: statusColor,
            minHeight: 6,
            borderRadius: BorderRadius.circular(3),
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

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _buildExplanationHeader(explanation),
      SizedBox(height: 20),

      // Description Box
      Container(
        padding: EdgeInsets.all(16),
        decoration: BoxDecoration(
            color: Colors.blue[50]!.withOpacity(0.5),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: Colors.blue[100]!)),
        child: Text(explanation['description'] ?? 'No description.',
            style:
                TextStyle(fontSize: 14, height: 1.5, color: Colors.blue[900])),
      ),
      SizedBox(height: 24),

      if (explanation['key_indicators'] != null) ...[
        _buildKeyIndicators(explanation['key_indicators']),
        SizedBox(height: 24),
      ],

      if (explanation['top_contributing_factors'] != null)
        _buildContributingFactors(explanation['top_contributing_factors']),

      SizedBox(height: 24),

      if (explanation['recommended_actions'] != null)
        _buildRecommendedActions(explanation['recommended_actions']),
    ]);
  }

  Widget _buildKeyIndicators(dynamic indicatorsData) {
    List<String> indicators = (indicatorsData is List)
        ? indicatorsData.map((e) => e.toString()).toList()
        : [];
    if (indicators.isEmpty) return SizedBox.shrink();

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('Key Indicators',
          style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.bold,
              color: Colors.grey[700])),
      SizedBox(height: 8),
      Wrap(
          spacing: 8,
          runSpacing: 8,
          children: indicators
              .map((indicator) => Chip(
                    label: Text(indicator),
                    backgroundColor: Colors.white,
                    labelStyle:
                        TextStyle(color: Colors.grey[800], fontSize: 12),
                    side: BorderSide(color: Colors.grey[300]!),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(6)),
                  ))
              .toList()),
    ]);
  }

  Widget _buildExplanationHeader(Map<String, dynamic> explanation) {
    String riskLevel = explanation['risk_level'] ?? 'UNKNOWN';
    Color riskColor = _getRiskColor(riskLevel);
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Expanded(
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(
                explanation['title']
                        ?.toString()
                        .replaceAll('⚠️ ', '')
                        .replaceAll('🚨 ', '') ??
                    'Analysis',
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.grey[800])),
            if (explanation['attack_classification'] != null)
              Text(explanation['attack_classification'],
                  style: TextStyle(fontSize: 13, color: Colors.grey[600])),
          ]),
        ),
        Container(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
              color: riskColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: riskColor)),
          child: Text('$riskLevel RISK',
              style: TextStyle(
                  color: riskColor, fontWeight: FontWeight.bold, fontSize: 12)),
        ),
      ],
    );
  }

  Widget _buildContributingFactors(dynamic factorsData) {
    List<dynamic> factors = factorsData is List ? factorsData : [];

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('Top Contributing Factors',
          style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.bold,
              color: Colors.grey[700])),
      SizedBox(height: 12),
      ...factors
          .map((factor) => _buildFactorTile(factor as Map<String, dynamic>)),
    ]);
  }

  Widget _buildFactorTile(Map<String, dynamic> factor) {
    bool isRisk = factor['impact'].toString().contains('Increased');
    Color color = isRisk ? Colors.red : Colors.green;
    return Container(
      margin: EdgeInsets.only(bottom: 8),
      padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: Colors.grey[200]!)),
      child: Row(children: [
        Icon(isRisk ? Icons.arrow_upward : Icons.arrow_downward,
            color: color, size: 16),
        SizedBox(width: 12),
        Expanded(
            child: Text(factor['factor'],
                style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13))),
        Text(factor['magnitude'].toString(),
            style: TextStyle(
                fontWeight: FontWeight.bold, fontSize: 13, color: color)),
      ]),
    );
  }

  Widget _buildRecommendedActions(dynamic actionsData) {
    List<String> actions =
        (actionsData as List).map((e) => e.toString()).toList();
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('Recommended Actions',
          style: TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.bold,
              color: Colors.grey[700])),
      SizedBox(height: 8),
      ...actions
          .map((action) => Padding(
                padding: EdgeInsets.only(bottom: 6),
                child: Row(children: [
                  Icon(Icons.check_circle, size: 16, color: Colors.green),
                  SizedBox(width: 8),
                  Expanded(child: Text(action, style: TextStyle(fontSize: 13))),
                ]),
              ))
          .toList(),
    ]);
  }

  Widget _buildFallbackExplanation(Map<String, dynamic> explanation) {
    return Center(
        child: Column(children: [
      CircularProgressIndicator(),
      SizedBox(height: 16),
      Text("Analyzing...", style: TextStyle(color: Colors.grey))
    ]));
  }

  // Helpers
  String _formatTime(DateTime timestamp) =>
      '${timestamp.hour}:${timestamp.minute}:${timestamp.second}';
  Color _getStatusColor(String status) => status == 'known_attack'
      ? Colors.red
      : (status == 'zero_day' ? Colors.orange : Colors.green);
  IconData _getStatusIcon(String status) => status == 'known_attack'
      ? Icons.warning_rounded
      : (status == 'zero_day'
          ? Icons.new_releases_rounded
          : Icons.check_circle_rounded);
  String _getStatusText(String status) => status == 'known_attack'
      ? 'Known Attack'
      : (status == 'zero_day' ? 'Zero-Day' : 'Normal');
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
