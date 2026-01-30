import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import 'label_selection_dialog.dart';

class PacketDetailDialog extends StatelessWidget {
  final Packet packet;

  const PacketDetailDialog({required this.packet});

  @override
  Widget build(BuildContext context) {
    // Determine width based on screen size to be responsive - user's original layout
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
                              _buildSectionTitle('Hybrid Ensemble Score'),
                              _buildDetectionInfo(),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ),

                  // RIGHT COLUMN: XAI Explanation (Updated for Point 2 & 3)
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

  // --- HEADER WIDGET ---
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

  // --- AI ANALYSIS SECTION (ROADMAP UPDATE) ---
  Widget _buildExplanationSection(BuildContext context) {
    final explanation = packet.explanation!;
    final String type = explanation['type'] ?? 'UNKNOWN';

    // If still in the background processing queue
    if (type == 'INITIAL_DETECTION' || type.contains('FALLBACK')) {
      return _buildFallbackExplanation(explanation);
    }

    final List<dynamic> factors = explanation['top_contributing_factors'] ?? [];
    final Map<String, dynamic> sensory = explanation['sensory_analysis'] ?? {};

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

      // POINT 3: SENSORY INSIGHTS TILE
      if (sensory.isNotEmpty) ...[
        _buildSectionTitle('Sensory Engine Insights'),
        Row(
          children: [
            Expanded(
                child: _buildSensorySmallTile("Topological (GNN)",
                    sensory['topological_shift'] ?? "Stable", Icons.hub)),
            SizedBox(width: 12),
            Expanded(
                child: _buildSensorySmallTile(
                    "Visual (MAE)",
                    sensory['visual_anomaly'] ?? "Consistent",
                    Icons.grid_view)),
          ],
        ),
        SizedBox(height: 24),
      ],

      // POINT 2: SHAP FEATURE CONTRIBUTION
      if (factors.isNotEmpty) ...[
        _buildSectionTitle('Hybrid Feature Impact (SHAP)'),
        ...factors
            .map((f) => _buildFactorTile(f as Map<String, dynamic>))
            .toList(),
        SizedBox(height: 24),
      ],

      if (explanation['recommended_actions'] != null)
        _buildRecommendedActions(explanation['recommended_actions']),
    ]);
  }

  // --- SUB-WIDGETS FOR EXPLANATION ---
  Widget _buildSensorySmallTile(String label, String value, IconData icon) {
    bool isAnomalous = value != "Stable" && value != "Consistent";
    Color color = isAnomalous ? Colors.orange : Colors.green;
    return Container(
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 16, color: color),
          SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: TextStyle(fontSize: 10, color: Colors.grey[600])),
                Text(value,
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: color)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFactorTile(Map<String, dynamic> factor) {
    bool isRisk = factor['impact'].toString().contains('Increased');
    Color color = isRisk ? Colors.red : Colors.green;
    double magnitude = double.tryParse(factor['magnitude'] ?? '0.0') ?? 0.0;

    // Feature classification for GNN/MAE icons
    IconData factorIcon = Icons.bar_chart;
    if (factor['factor'].toString().contains('GNN')) factorIcon = Icons.hub;
    if (factor['factor'].toString().contains('MAE'))
      factorIcon = Icons.grid_view;

    return Container(
      margin: EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(factorIcon, size: 14, color: Colors.grey),
              SizedBox(width: 8),
              Expanded(
                  child: Text(factor['factor'],
                      style: TextStyle(
                          fontWeight: FontWeight.w600, fontSize: 13))),
              Text(factor['observed_value'],
                  style: TextStyle(fontSize: 11, color: Colors.grey)),
            ],
          ),
          SizedBox(height: 4),
          Row(
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: (magnitude * 2).clamp(0.05, 1.0),
                    backgroundColor: Colors.grey[100],
                    color: color,
                    minHeight: 6,
                  ),
                ),
              ),
              SizedBox(width: 12),
              Text(isRisk ? "+$magnitude" : "-$magnitude",
                  style: TextStyle(
                      fontWeight: FontWeight.bold, fontSize: 12, color: color)),
            ],
          ),
        ],
      ),
    );
  }

  // --- FOOTER BUTTONS (RESTORED LOGIC) ---
  Widget _buildFooter(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context, listen: false);
    bool isSelected = provider.isSelected(packet.id);

    List<Widget> buttons = [];

    if (packet.status == 'normal') {
      buttons.add(_buildActionButton(context, provider,
          icon: Icons.bug_report,
          label: "Report Missed Attack",
          color: Colors.red,
          queueType: 'gan'));
    } else {
      buttons.add(_buildActionButton(context, provider,
          icon: Icons.thumb_up_alt_outlined,
          label: "Report False Alarm",
          color: Colors.green,
          queueType: 'jitter'));

      if (packet.status == 'zero_day') {
        buttons.add(SizedBox(width: 12));
        buttons.add(_buildActionButton(context, provider,
            icon: Icons.science,
            label: "Confirm Zero-Day",
            color: Colors.deepPurple,
            queueType: 'gan'));
      }
    }

    if (isSelected) {
      buttons = [
        OutlinedButton.icon(
          onPressed: () {
            provider.toggleSelection(packet, '');
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

  // --- TECHNICAL SPEC HELPERS ---
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
                      fontSize: 14,
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
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("Detection Confidence",
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

  // --- GENERAL HELPERS ---
  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: EdgeInsets.only(bottom: 16),
      child: Text(title,
          style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: Colors.grey[500],
              letterSpacing: 1.2)),
    );
  }

  Widget _buildActionButton(BuildContext context, IdsProvider provider,
      {required IconData icon,
      required String label,
      required Color color,
      required String queueType}) {
    return ElevatedButton.icon(
      onPressed: () {
        provider.toggleSelection(packet, queueType);
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(
              'Packet #${packet.id} added to ${queueType.toUpperCase()} queue for adaptation.'),
          backgroundColor: color,
        ));
      },
      icon: Icon(icon, size: 18),
      label: Text(label),
      style: ElevatedButton.styleFrom(
          backgroundColor: color, foregroundColor: Colors.white),
    );
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
        ])),
        Container(
          padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(
              color: riskColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: riskColor)),
          child: Text('$riskLevel RISK',
              style: TextStyle(
                  color: riskColor, fontWeight: FontWeight.bold, fontSize: 11)),
        ),
      ],
    );
  }

  Widget _buildRecommendedActions(dynamic actionsData) {
    List<String> actions =
        (actionsData as List).map((e) => e.toString()).toList();
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      _buildSectionTitle('Recommended Actions'),
      ...actions
          .map((action) => Padding(
                padding: EdgeInsets.only(bottom: 8),
                child: Row(children: [
                  Icon(Icons.check_circle, size: 16, color: Colors.green),
                  SizedBox(width: 8),
                  Expanded(
                      child: Text(action,
                          style: TextStyle(
                              fontSize: 13, color: Colors.grey[700]))),
                ]),
              ))
          .toList(),
    ]);
  }

  Widget _buildFallbackExplanation(Map<String, dynamic> explanation) {
    return Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      SizedBox(height: 100),
      CircularProgressIndicator(strokeWidth: 3, color: Colors.deepPurple),
      SizedBox(height: 24),
      Text("AI Explainer is generating SHAP values...",
          style:
              TextStyle(color: Colors.grey[600], fontWeight: FontWeight.w500)),
      SizedBox(height: 8),
      Text("This takes 2-3 seconds per anomaly.",
          style: TextStyle(color: Colors.grey[400], fontSize: 12)),
    ]));
  }

  String _formatTime(DateTime timestamp) =>
      '${timestamp.hour}:${timestamp.minute.toString().padLeft(2, '0')}:${timestamp.second.toString().padLeft(2, '0')}';
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
