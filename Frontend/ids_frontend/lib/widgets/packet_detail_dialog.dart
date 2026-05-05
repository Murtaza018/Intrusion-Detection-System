import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import '../models/packet.dart';
import 'mae_heat_map.dart';

class PacketDetailDialog extends StatelessWidget {
  final Packet packet;
  const PacketDetailDialog({required this.packet, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Determine screen size to dictate dialog layout
    final isDesktop = MediaQuery.of(context).size.width > 800;

    return Dialog(
      backgroundColor: const Color(0xFF0D1117), // Deep SOC Background
      insetPadding:
          EdgeInsets.all(isDesktop ? 20 : 10), // Tighter padding on mobile
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(color: Colors.white.withOpacity(0.05))),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 1000, maxHeight: 800),
        child: Column(
          children: [
            _buildHeader(context),
            Expanded(
              // Flex dynamically switches between Row (desktop) and Column (mobile)
              child: Flex(
                direction: isDesktop ? Axis.horizontal : Axis.vertical,
                children: [
                  _buildTechnicalColumn(isDesktop),
                  // Divider changes orientation based on layout
                  Container(
                      width: isDesktop ? 1 : double.infinity,
                      height: isDesktop ? double.infinity : 1,
                      color: Colors.white.withOpacity(0.05)),
                  _buildAIColumn(context, isDesktop),
                ],
              ),
            ),
            _buildFooter(context),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context) {
    Color statusColor = packet.status == 'known_attack'
        ? Colors.redAccent
        : (packet.status == 'zero_day'
            ? Colors.orangeAccent
            : Colors.greenAccent);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
          border: Border(
              bottom: BorderSide(color: Colors.white.withOpacity(0.05)))),
      child: Row(
        children: [
          Icon(Icons.terminal, color: statusColor, size: 24),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("DPI_STREAM_ID_${packet.id}",
                    style: const TextStyle(
                        fontFamily: 'monospace',
                        fontWeight: FontWeight.bold,
                        fontSize: 14, // Slightly reduced for mobile safety
                        letterSpacing: 1,
                        color: Color(0xFF00E5FF))),
                Text(packet.summary,
                    style:
                        const TextStyle(color: Colors.white38, fontSize: 11)),
              ],
            ),
          ),
          IconButton(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.close, color: Colors.white24, size: 20)),
        ],
      ),
    );
  }

  Widget _buildTechnicalColumn(bool isDesktop) {
    return Expanded(
      flex:
          isDesktop ? 4 : 3, // Takes less vertical space when stacked on mobile
      child: Container(
        padding: const EdgeInsets.all(20),
        color: Colors.black.withOpacity(0.1),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _sectionTitle("NETWORK SPECIFICATIONS"),
              _detail("SOURCE", "${packet.srcIp}:${packet.srcPort}"),
              _detail("TARGET", "${packet.dstIp}:${packet.dstPort}"),
              _detail("PROTOCOL", packet.protocol),
              _detail("SIZE", "${packet.length} BYTES"),
              _detail(
                  "TIME", packet.timestamp.toIso8601String().substring(11, 19)),
              const SizedBox(height: 24),
              if (packet.confidence > 0) ...[
                _sectionTitle("DETECTION CONFIDENCE"),
                const SizedBox(height: 12),
                Text("${(packet.confidence * 100).toStringAsFixed(2)}%",
                    style: TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.w900,
                        color: packet.confidence > 0.4
                            ? Colors.redAccent
                            : Colors.greenAccent,
                        fontFamily: 'monospace')),
                const SizedBox(height: 8),
                LinearProgressIndicator(
                    value: packet.confidence,
                    color: packet.confidence > 0.4
                        ? Colors.redAccent
                        : Colors.greenAccent,
                    backgroundColor: Colors.white10,
                    minHeight: 2),
              ]
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAIColumn(BuildContext context, bool isDesktop) {
    final explanation = packet.explanation;
    final List<dynamic> recommendations =
        explanation?['recommended_actions'] ?? [];
    final ScrollController _scrollController = ScrollController();

    return Expanded(
      flex:
          isDesktop ? 6 : 7, // Takes more vertical space when stacked on mobile
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: explanation == null
            ? const Center(
                child: Text("NO ANALYSIS DATA",
                    style: TextStyle(color: Colors.white12)))
            : Scrollbar(
                controller: _scrollController,
                thumbVisibility: true,
                child: SingleChildScrollView(
                  controller: _scrollController,
                  padding: const EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionTitle("AI REASONING & XAI"),
                      Container(
                        padding: const EdgeInsets.all(16),
                        width: double.infinity,
                        decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.02),
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(
                                color: Colors.white.withOpacity(0.05))),
                        child: Text(
                            explanation['description'] ??
                                'Establishing deep feature correlation...',
                            style: const TextStyle(
                                fontSize: 12,
                                color: Colors.white70,
                                height: 1.6)),
                      ),
                      if (recommendations.isNotEmpty) ...[
                        const SizedBox(height: 24),
                        _sectionTitle("RECOMMENDED MITIGATION"),
                        ...recommendations
                            .map((action) => _buildActionCard(action))
                            .toList(),
                      ],
                      const SizedBox(height: 24),
                      if (explanation['sensory_analysis'] != null)
                        _buildSensorySmallTiles(
                            explanation['sensory_analysis']),
                      if (packet.status == 'zero_day' &&
                          explanation['sensory_analysis'] != null &&
                          explanation['sensory_analysis']['original_grid'] !=
                              null) ...[
                        const SizedBox(height: 24),
                        _sectionTitle("MAE STRUCTURAL X-RAY (ZERO-DAY)"),
                        // Changed from Row to Wrap to prevent overflow on mobile
                        Wrap(
                          alignment: WrapAlignment.spaceEvenly,
                          spacing: 16,
                          runSpacing: 16,
                          children: [
                            MaeHeatmapGrid(
                              title: "Original Traffic",
                              gridValues: List<double>.from(
                                  explanation['sensory_analysis']
                                          ['original_grid']
                                      .map((x) => x.toDouble())),
                            ),
                            MaeHeatmapGrid(
                              title: "MAE Reconstruction",
                              gridValues: List<double>.from(
                                  explanation['sensory_analysis']
                                          ['reconstructed_grid']
                                      .map((x) => x.toDouble())),
                            ),
                          ],
                        ),
                      ],
                      const SizedBox(height: 24),
                      if (explanation['top_contributing_factors'] != null) ...[
                        _sectionTitle("SHAP FEATURE ANALYSIS"),
                        ...(explanation['top_contributing_factors'] as List)
                            .map((f) => _buildFeatureBar(f))
                            .toList(),
                      ],
                    ],
                  ),
                ),
              ),
      ),
    );
  }

  Widget _buildActionCard(String action) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.blueAccent.withOpacity(0.05),
        border: Border.all(color: Colors.blueAccent.withOpacity(0.1)),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Row(
        children: [
          const Icon(Icons.gavel_rounded, color: Colors.blueAccent, size: 14),
          const SizedBox(width: 12),
          Expanded(
              child: Text(action,
                  style: const TextStyle(color: Colors.white70, fontSize: 11))),
        ],
      ),
    );
  }

  Widget _buildSensorySmallTiles(Map<String, dynamic> sensory) {
    // Changed Row to Flex/Expanded to handle small screens gracefully
    return Row(
      children: [
        _miniTile("GNN TOPOLOGY", sensory['topological_shift'] ?? "STABLE",
            Icons.hub_outlined),
        const SizedBox(width: 8),
        _miniTile("MAE STRUCTURAL", sensory['visual_anomaly'] ?? "STABLE",
            Icons.grid_view_rounded),
      ],
    );
  }

  Widget _miniTile(String label, String val, IconData icon) {
    bool alert = val != "Stable" && val != "Consistent";
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.02),
            border: Border.all(
                color: alert
                    ? Colors.orangeAccent.withOpacity(0.2)
                    : Colors.greenAccent.withOpacity(0.1))),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(icon, size: 12, color: Colors.white24),
              const SizedBox(width: 6),
              Expanded(
                child: Text(label,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                        fontSize: 9,
                        color: Colors.white24,
                        fontWeight: FontWeight.bold)),
              )
            ]),
            const SizedBox(height: 6),
            Text(val,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    fontFamily: 'monospace',
                    color: alert ? Colors.orangeAccent : Colors.greenAccent)),
          ],
        ),
      ),
    );
  }

  Widget _buildFeatureBar(dynamic factor) {
    double mag = double.tryParse(factor['magnitude']?.toString() ?? '0') ?? 0;
    bool risk = factor['impact'].toString().contains('Increased');
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        children: [
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            Expanded(
              child: Text(factor['factor'],
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                      fontSize: 10,
                      color: Colors.white70,
                      fontFamily: 'monospace')),
            ),
            const SizedBox(width: 8),
            Text(factor['observed_value'],
                style: const TextStyle(
                    fontSize: 10,
                    color: Colors.white38,
                    fontFamily: 'monospace')),
          ]),
          const SizedBox(height: 6),
          LinearProgressIndicator(
              value: (mag * 2).clamp(0.05, 1.0),
              color: risk
                  ? Colors.redAccent.withOpacity(0.8)
                  : Colors.greenAccent.withOpacity(0.8),
              backgroundColor: Colors.white.withOpacity(0.05),
              minHeight: 2),
        ],
      ),
    );
  }

  Widget _buildFooter(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context, listen: false);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.2),
          border:
              Border(top: BorderSide(color: Colors.white.withOpacity(0.05)))),
      // Replaced Row with Wrap so buttons stack neatly on mobile
      child: Wrap(
        alignment: WrapAlignment.end,
        spacing: 12,
        runSpacing: 12,
        children: [
          if (packet.status == 'normal')
            _actionBtn(context, provider, "REPORT MISS",
                Icons.bug_report_outlined, Colors.redAccent, 'gan')
          else ...[
            _actionBtn(context, provider, "FALSE ALARM",
                Icons.check_circle_outline, Colors.greenAccent, 'jitter'),
            if (packet.status == 'zero_day')
              _actionBtn(context, provider, "CONFIRM NOVELTY",
                  Icons.biotech_outlined, const Color(0xFF7C4DFF), 'gan'),
          ]
        ],
      ),
    );
  }

  Widget _actionBtn(BuildContext context, IdsProvider prov, String lbl,
      IconData ico, Color clr, String type) {
    return OutlinedButton.icon(
      onPressed: () {
        prov.toggleSelection(packet, type);
        Navigator.pop(context);
      },
      icon: Icon(ico, size: 14),
      label: Text(lbl,
          style: const TextStyle(
              fontSize: 10, fontWeight: FontWeight.bold, letterSpacing: 1)),
      style: OutlinedButton.styleFrom(
          foregroundColor: clr,
          side: BorderSide(color: clr.withOpacity(0.4)),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(4))),
    );
  }

  Widget _sectionTitle(String t) => Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(t,
          style: const TextStyle(
              color: Colors.white24,
              fontSize: 9,
              fontWeight: FontWeight.bold,
              letterSpacing: 2)));

  Widget _detail(String l, String v) => Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        Text(l,
            style: const TextStyle(
                color: Colors.white30, fontSize: 10, letterSpacing: 1)),
        const SizedBox(width: 8),
        Expanded(
          child: Text(v,
              textAlign: TextAlign.right,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 11,
                  fontFamily: 'monospace')),
        )
      ]));
}
