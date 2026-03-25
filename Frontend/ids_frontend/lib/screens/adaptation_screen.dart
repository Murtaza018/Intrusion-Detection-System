// import 'package:flutter/material.dart';
// import 'package:provider/provider.dart';
// import '../providers/ids_provider.dart';
// import '../models/packet.dart';

// class AdaptationScreen extends StatefulWidget {
//   @override
//   _AdaptationScreenState createState() => _AdaptationScreenState();
// }

// class _AdaptationScreenState extends State<AdaptationScreen> {
//   @override
//   void initState() {
//     super.initState();
//     Future.microtask(
//         () => Provider.of<IdsProvider>(context, listen: false).fetchLabels());
//   }

//   @override
//   Widget build(BuildContext context) {
//     final provider = Provider.of<IdsProvider>(context);

//     return Scaffold(
//       backgroundColor: Colors.grey[100],
//       appBar: AppBar(
//         title: Text("Model Adaptation",
//             style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20)),
//         backgroundColor: Colors.white,
//         foregroundColor: Colors.black87,
//         elevation: 0,
//       ),
//       body: Column(
//         children: [
//           _buildCompactLabelingSection(context, provider),
//           Divider(height: 1, thickness: 1),
//           Expanded(
//             child: Row(
//               crossAxisAlignment: CrossAxisAlignment.stretch,
//               children: [
//                 Expanded(
//                   flex: 1,
//                   child: _buildQueueList(context, "False Positives (Jitter)",
//                       provider.jitterQueue, Colors.green),
//                 ),
//                 VerticalDivider(
//                     width: 1, thickness: 1, color: Colors.grey[300]),
//                 Expanded(
//                   flex: 1,
//                   child: _buildQueueList(context, "New Attacks (GAN)",
//                       provider.ganQueue, Colors.red),
//                 ),
//               ],
//             ),
//           ),
//         ],
//       ),
//       bottomNavigationBar: _buildBottomBar(context, provider),
//     );
//   }

//   // --- COMPACT LABELING UI ---
//   Widget _buildCompactLabelingSection(
//       BuildContext context, IdsProvider provider) {
//     if (provider.ganQueue.isEmpty) return SizedBox.shrink();

//     return Container(
//       padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
//       color: Colors.white,
//       child: Column(
//         mainAxisSize: MainAxisSize.min,
//         children: [
//           Row(
//             mainAxisAlignment: MainAxisAlignment.spaceBetween,
//             children: [
//               Text("Assign Label:",
//                   style: TextStyle(
//                       fontWeight: FontWeight.bold, color: Colors.grey[800])),
//               Row(
//                 children: [
//                   _buildCompactChip("Existing", !provider.isNewAttack,
//                       () => provider.setBatchLabel(null, false)),
//                   SizedBox(width: 8),
//                   _buildCompactChip("New Zero-Day", provider.isNewAttack,
//                       () => provider.setBatchLabel(null, true)),
//                 ],
//               ),
//             ],
//           ),
//           SizedBox(height: 8),
//           SizedBox(
//             height: 48,
//             child: provider.isNewAttack
//                 ? TextField(
//                     decoration: _compactInputDecoration("Enter New Attack Name",
//                         Icons.new_releases, Colors.orange),
//                     onChanged: (val) => provider.setBatchLabel(val, true),
//                     style: TextStyle(fontSize: 14),
//                   )
//                 : DropdownButtonFormField<String>(
//                     decoration: _compactInputDecoration(
//                         "Select Existing Attack Type",
//                         Icons.category,
//                         Colors.blue),
//                     value: provider.batchLabel,
//                     style: TextStyle(fontSize: 14, color: Colors.black),
//                     items: provider.existingLabels.isEmpty
//                         ? [
//                             DropdownMenuItem(
//                                 value: "Loading...", child: Text("Loading..."))
//                           ]
//                         : provider.existingLabels
//                             .map((l) =>
//                                 DropdownMenuItem(value: l, child: Text(l)))
//                             .toList(),
//                     onChanged: (val) => provider.setBatchLabel(val, false),
//                   ),
//           ),
//         ],
//       ),
//     );
//   }

//   InputDecoration _compactInputDecoration(
//       String label, IconData icon, Color color) {
//     return InputDecoration(
//       labelText: label,
//       prefixIcon: Icon(icon, color: color, size: 20),
//       border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
//       contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 0),
//       labelStyle: TextStyle(fontSize: 13),
//     );
//   }

//   Widget _buildCompactChip(String label, bool isSelected, VoidCallback onTap) {
//     return InkWell(
//       onTap: onTap,
//       borderRadius: BorderRadius.circular(16),
//       child: Container(
//         padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
//         decoration: BoxDecoration(
//           color: isSelected
//               ? Colors.deepPurple.withOpacity(0.1)
//               : Colors.grey[100],
//           borderRadius: BorderRadius.circular(16),
//           border: Border.all(
//               color: isSelected ? Colors.deepPurple : Colors.grey[300]!),
//         ),
//         child: Text(
//           label,
//           style: TextStyle(
//             fontSize: 12,
//             fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
//             color: isSelected ? Colors.deepPurple : Colors.grey[700],
//           ),
//         ),
//       ),
//     );
//   }

//   // --- BOTTOM BAR ---
//   Widget _buildBottomBar(BuildContext context, IdsProvider provider) {
//     bool hasLabel =
//         provider.batchLabel != null && provider.batchLabel!.isNotEmpty;
//     bool hasGanPackets = provider.ganQueue.isNotEmpty;
//     bool hasJitterPackets = provider.jitterQueue.isNotEmpty;

//     bool consistencyReady =
//         (hasGanPackets || hasJitterPackets) && (!hasGanPackets || hasLabel);
//     bool trainingReady = consistencyReady &&
//         (provider.ganQueue.isEmpty || provider.consistencyChecked);

//     return Container(
//       padding: EdgeInsets.fromLTRB(16, 8, 16, 16),
//       decoration: BoxDecoration(
//         color: Colors.white,
//         boxShadow: [
//           BoxShadow(color: Colors.black12, blurRadius: 5, offset: Offset(0, -2))
//         ],
//       ),
//       child: Column(
//         mainAxisSize: MainAxisSize.min,
//         children: [
//           if ((hasGanPackets || hasJitterPackets) && !trainingReady)
//             Padding(
//               padding: const EdgeInsets.only(bottom: 8.0),
//               child: Text(
//                 (hasGanPackets && !hasLabel)
//                     ? "* Select a label above to proceed"
//                     : (hasGanPackets && !provider.consistencyChecked)
//                         ? "* Check consistency before training"
//                         : "", // jitter-only — no warning needed
//                 style: TextStyle(
//                     color: Colors.orange[800],
//                     fontSize: 12,
//                     fontWeight: FontWeight.w500),
//               ),
//             ),
//           Row(
//             children: [
//               // CHECK CONSISTENCY
//               if (hasGanPackets)
//                 Expanded(
//                   child: OutlinedButton.icon(
//                     onPressed:
//                         (!consistencyReady || provider.totalSelected <= 1)
//                             ? null
//                             : () => _showAnalysisDialog(context, provider),
//                     icon: Icon(Icons.analytics_outlined, size: 18),
//                     label: Text(
//                         provider.consistencyChecked
//                             ? "Re-Check"
//                             : "Check Consistency",
//                         style: TextStyle(fontSize: 13)),
//                     style: OutlinedButton.styleFrom(
//                       padding: EdgeInsets.symmetric(vertical: 14),
//                       shape: RoundedRectangleBorder(
//                           borderRadius: BorderRadius.circular(8)),
//                     ),
//                   ),
//                 ),
//               if (hasGanPackets) SizedBox(width: 12),

//               // START TRAINING
//               Expanded(
//                 child: ElevatedButton.icon(
//                   onPressed: !trainingReady
//                       ? null
//                       : () async {
//                           ScaffoldMessenger.of(context).showSnackBar(SnackBar(
//                               content: Text("Sending data..."),
//                               duration: Duration(seconds: 1)));
//                           final jobId = await provider.sendRetrainRequest();
//                           bool success = jobId != null;
//                           if (success) {
//                             provider.clearQueues();
//                             if (context.mounted) {
//                               Navigator.pop(context);
//                               ScaffoldMessenger.of(context).showSnackBar(
//                                   SnackBar(
//                                       content: Text("Training Queued!"),
//                                       backgroundColor: Colors.green));
//                             }
//                           }
//                         },
//                   icon: Icon(Icons.model_training, size: 18),
//                   label:
//                       Text("Start Retraining", style: TextStyle(fontSize: 13)),
//                   style: ElevatedButton.styleFrom(
//                     backgroundColor: Colors.deepPurple,
//                     foregroundColor: Colors.white,
//                     padding: EdgeInsets.symmetric(vertical: 14),
//                     shape: RoundedRectangleBorder(
//                         borderRadius: BorderRadius.circular(8)),
//                   ),
//                 ),
//               ),
//             ],
//           ),
//         ],
//       ),
//     );
//   }

//   // --- QUEUE LIST ---
//   Widget _buildQueueList(
//       BuildContext context, String title, List<Packet> queue, Color color) {
//     return Column(
//       children: [
//         Container(
//           padding: EdgeInsets.symmetric(vertical: 10, horizontal: 4),
//           width: double.infinity,
//           color: color.withOpacity(0.08),
//           child: Text(title,
//               style: TextStyle(
//                   fontWeight: FontWeight.bold, color: color, fontSize: 13),
//               textAlign: TextAlign.center),
//         ),
//         Expanded(
//           child: queue.isEmpty
//               ? Center(
//                   child:
//                       Text("Empty", style: TextStyle(color: Colors.grey[300])))
//               : ListView.builder(
//                   itemCount: queue.length,
//                   padding: EdgeInsets.all(0),
//                   itemBuilder: (ctx, i) => Container(
//                     decoration: BoxDecoration(
//                         border: Border(
//                             bottom: BorderSide(color: Colors.grey[200]!))),
//                     child: ListTile(
//                       dense: true,
//                       contentPadding:
//                           EdgeInsets.symmetric(horizontal: 12, vertical: 0),
//                       leading: CircleAvatar(
//                         backgroundColor: color.withOpacity(0.1),
//                         radius: 14,
//                         child: Text("${queue[i].id}",
//                             style: TextStyle(
//                                 fontSize: 9,
//                                 fontWeight: FontWeight.bold,
//                                 color: color)),
//                       ),
//                       title: Text(queue[i].summary,
//                           maxLines: 1,
//                           overflow: TextOverflow.ellipsis,
//                           style: TextStyle(fontSize: 12)),
//                       trailing: InkWell(
//                         child: Icon(Icons.close,
//                             size: 16, color: Colors.grey[400]),
//                         onTap: () =>
//                             Provider.of<IdsProvider>(context, listen: false)
//                                 .toggleSelection(queue[i], ''),
//                       ),
//                     ),
//                   ),
//                 ),
//         ),
//       ],
//     );
//   }

//   // --- ANALYSIS DIALOG ---
//   void _showAnalysisDialog(BuildContext context, IdsProvider provider) async {
//     showDialog(
//         context: context,
//         barrierDismissible: false,
//         builder: (_) => Center(child: CircularProgressIndicator()));

//     final result = await provider.analyzeQueues();

//     if (context.mounted) Navigator.pop(context);
//     provider.setConsistencyStatus(true, true);

//     if (context.mounted && result.isNotEmpty) {
//       showDialog(
//         context: context,
//         builder: (_) => AlertDialog(
//           title: Text("Consistency Report",
//               style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
//           contentPadding: EdgeInsets.fromLTRB(24, 20, 24, 0),
//           content: Column(
//             mainAxisSize: MainAxisSize.min,
//             children: [
//               _buildScoreTile(
//                 "GAN Queue",
//                 result['gan_score'],
//                 result['gan_status'] as String? ?? 'No data',
//               ),
//               SizedBox(height: 16),
//               _buildScoreTile(
//                 "Jitter Queue",
//                 result['jitter_score'],
//                 result['jitter_status'] as String? ?? 'Not analysed',
//               ),
//               SizedBox(height: 10),
//             ],
//           ),
//           actions: [
//             TextButton(
//                 onPressed: () => Navigator.pop(context), child: Text("OK"))
//           ],
//         ),
//       );
//     }
//   }

//   // --- SCORE TILE ---
//   Widget _buildScoreTile(String title, dynamic score, String? status) {
//     double s = (score is num) ? score.toDouble() : 0.0;
//     String safeStatus = status ?? 'No data';
//     bool hasData = s > 0.0 ||
//         (safeStatus != "No Features Found" && safeStatus != "Not enough data");
//     Color color = s > 0.7 ? Colors.green : Colors.red;

//     return Column(
//       crossAxisAlignment: CrossAxisAlignment.start,
//       children: [
//         Row(
//           mainAxisAlignment: MainAxisAlignment.spaceBetween,
//           children: [
//             Text(title,
//                 style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
//             if (hasData)
//               Text("${(s * 100).toStringAsFixed(1)}%",
//                   style: TextStyle(fontWeight: FontWeight.bold, color: color)),
//           ],
//         ),
//         SizedBox(height: 6),
//         ClipRRect(
//           borderRadius: BorderRadius.circular(4),
//           child: LinearProgressIndicator(
//               value: hasData ? s : 0.0,
//               backgroundColor: Colors.grey[200],
//               color: color,
//               minHeight: 8),
//         ),
//         SizedBox(height: 4),
//         Text(safeStatus,
//             style: TextStyle(color: Colors.grey[600], fontSize: 11)),
//       ],
//     );
//   }
// }

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import '../models/packet.dart';

class AdaptationScreen extends StatefulWidget {
  @override
  _AdaptationScreenState createState() => _AdaptationScreenState();
}

class _AdaptationScreenState extends State<AdaptationScreen> {
  @override
  void initState() {
    super.initState();
    Future.microtask(
        () => Provider.of<IdsProvider>(context, listen: false).fetchLabels());
  }

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context);

    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        title: const Text(
          "Model Adaptation",
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 18,
            letterSpacing: -0.5,
          ),
        ),
        backgroundColor: const Color(0xFF0D1117),
        foregroundColor: Colors.white,
        elevation: 0,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(
            height: 1,
            color: Colors.white12,
          ),
        ),
      ),
      body: Column(
        children: [
          _buildCompactLabelingSection(context, provider),
          const Divider(
            height: 1,
            thickness: 1,
            color: Colors.white10,
          ),
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(
                  child: _buildQueueList(
                    context,
                    "False Positives (Jitter)",
                    provider.jitterQueue,
                    Colors.greenAccent,
                  ),
                ),
                const VerticalDivider(
                  width: 1,
                  thickness: 1,
                  color: Colors.white10,
                ),
                Expanded(
                  child: _buildQueueList(
                    context,
                    "New Attacks (GAN)",
                    provider.ganQueue,
                    Colors.redAccent,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      bottomNavigationBar: _buildBottomBar(context, provider),
    );
  }

  // --- COMPACT LABELING UI ---
  Widget _buildCompactLabelingSection(
      BuildContext context, IdsProvider provider) {
    if (provider.ganQueue.isEmpty) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
      decoration: BoxDecoration(
        color: const Color(0xFF15191C),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                "Assign Label:",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 13,
                  color: Colors.white70,
                ),
              ),
              Row(
                children: [
                  _buildCompactChip(
                    "Existing",
                    !provider.isNewAttack,
                    () => provider.setBatchLabel(null, false),
                  ),
                  const SizedBox(width: 8),
                  _buildCompactChip(
                    "New Zero-Day",
                    provider.isNewAttack,
                    () => provider.setBatchLabel(null, true),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 12),
          SizedBox(
            height: 52,
            child: provider.isNewAttack
                ? TextField(
                    decoration: _compactInputDecoration(
                      "Enter New Attack Name",
                      Icons.new_releases,
                      Colors.orangeAccent,
                    ),
                    onChanged: (val) => provider.setBatchLabel(val, true),
                    style: const TextStyle(
                      fontSize: 13,
                      color: Colors.white,
                    ),
                    cursorColor: Colors.orangeAccent,
                  )
                : DropdownButtonFormField<String>(
                    decoration: _compactInputDecoration(
                      "Select Existing Attack Type",
                      Icons.category,
                      Colors.blueAccent,
                    ),
                    value: provider.batchLabel,
                    style: const TextStyle(
                      fontSize: 13,
                      color: Colors.white,
                    ),
                    items: provider.existingLabels.isEmpty
                        ? [
                            const DropdownMenuItem(
                              value: "Loading...",
                              child: Text(
                                "Loading...",
                                style: TextStyle(color: Colors.white54),
                              ),
                            )
                          ]
                        : provider.existingLabels
                            .map((l) => DropdownMenuItem(
                                  value: l,
                                  child: Text(
                                    l,
                                    style: const TextStyle(color: Colors.white),
                                  ),
                                ))
                            .toList(),
                    onChanged: (val) => provider.setBatchLabel(val, false),
                    dropdownColor: const Color(0xFF15191C),
                    borderRadius: BorderRadius.circular(4),
                  ),
          ),
        ],
      ),
    );
  }

  InputDecoration _compactInputDecoration(
      String label, IconData icon, Color color) {
    return InputDecoration(
      labelText: label,
      labelStyle: const TextStyle(
        fontSize: 11,
        fontWeight: FontWeight.normal,
        color: Colors.white30,
      ),
      prefixIcon: Icon(icon, color: color, size: 18),
      fillColor: const Color(0xFF1E252B),
      filled: true,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(4),
        borderSide: BorderSide.none,
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(4),
        borderSide: BorderSide(color: color.withOpacity(0.4)),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
    );
  }

  Widget _buildCompactChip(String label, bool isSelected, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected
              ? const Color(0xFF00E5FF).withOpacity(0.12)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isSelected
                ? const Color(0xFF00E5FF).withOpacity(0.6)
                : Colors.white12,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            color: isSelected ? const Color(0xFF00E5FF) : Colors.white54,
          ),
        ),
      ),
    );
  }

  // --- BOTTOM BAR ---
  Widget _buildBottomBar(BuildContext context, IdsProvider provider) {
    bool hasLabel =
        provider.batchLabel != null && provider.batchLabel!.isNotEmpty;
    bool hasGanPackets = provider.ganQueue.isNotEmpty;
    bool hasJitterPackets = provider.jitterQueue.isNotEmpty;

    bool consistencyReady =
        (hasGanPackets || hasJitterPackets) && (!hasGanPackets || hasLabel);
    bool trainingReady = consistencyReady &&
        (provider.ganQueue.isEmpty || provider.consistencyChecked);

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.05),
        border: const Border(
          top: BorderSide(color: Colors.white10),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 6,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if ((hasGanPackets || hasJitterPackets) && !trainingReady)
            Padding(
              padding: const EdgeInsets.only(bottom: 8.0),
              child: Text(
                (hasGanPackets && !hasLabel)
                    ? "* Select a label above to proceed"
                    : (hasGanPackets && !provider.consistencyChecked)
                        ? "* Check consistency before training"
                        : "", // jitter-only — no warning needed
                style: const TextStyle(
                  color: Colors.orangeAccent,
                  fontSize: 11,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          Row(
            children: [
              // CHECK CONSISTENCY
              if (hasGanPackets)
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed:
                        (!consistencyReady || provider.totalSelected <= 1)
                            ? null
                            : () => _showAnalysisDialog(context, provider),
                    icon: const Icon(
                      Icons.analytics_outlined,
                      size: 18,
                      color: Colors.white70,
                    ),
                    label: Text(
                      provider.consistencyChecked
                          ? "Re-Check"
                          : "Check Consistency",
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                        color: Colors.white,
                      ),
                    ),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: Color(0xFF7C4DFF)),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                ),
              if (hasGanPackets) const SizedBox(width: 12),

              // START TRAINING
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: !trainingReady
                      ? null
                      : () async {
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text("Sending data..."),
                              duration: Duration(seconds: 1),
                              backgroundColor: Color(0xFF15191C),
                            ),
                          );
                          final jobId = await provider.sendRetrainRequest();
                          bool success = jobId != null;
                          if (success) {
                            provider.clearQueues();
                            if (context.mounted) {
                              Navigator.pop(context);
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text("Training Queued!"),
                                  backgroundColor: Color(0xFF15191C),
                                ),
                              );
                            }
                          }
                        },
                  icon: const Icon(
                    Icons.model_training,
                    size: 18,
                    color: Colors.white,
                  ),
                  label: Text(
                    "Start Retraining",
                    style: const TextStyle(fontSize: 12, color: Colors.white),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF7C4DFF),
                    foregroundColor: Colors.white,
                    elevation: 2,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // --- QUEUE LIST ---
  Widget _buildQueueList(
      BuildContext context, String title, List<Packet> queue, Color color) {
    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
          width: double.infinity,
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: const BorderRadius.vertical(
              top: Radius.circular(4),
            ),
          ),
          child: Text(
            title,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: color,
              fontSize: 12,
            ),
            textAlign: TextAlign.center,
          ),
        ),
        Expanded(
          child: queue.isEmpty
              ? Center(
                  child: Text(
                    "Empty",
                    style: TextStyle(color: Colors.white30, fontSize: 11),
                  ),
                )
              : ListView.builder(
                  itemCount: queue.length,
                  padding: EdgeInsets.zero,
                  itemBuilder: (ctx, i) => Container(
                    decoration: BoxDecoration(
                      border: const Border(
                          bottom: BorderSide(color: Colors.white10)),
                    ),
                    child: ListTile(
                      dense: true,
                      contentPadding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      leading: CircleAvatar(
                        backgroundColor: color.withOpacity(0.15),
                        radius: 13,
                        child: Text(
                          "${queue[i].id}",
                          style: TextStyle(
                            fontSize: 9,
                            fontWeight: FontWeight.bold,
                            color: color.withOpacity(0.9),
                          ),
                        ),
                      ),
                      title: Text(
                        queue[i].summary,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 12,
                          color: Colors.white,
                        ),
                      ),
                      trailing: InkWell(
                        borderRadius: BorderRadius.circular(12),
                        onTap: () =>
                            Provider.of<IdsProvider>(context, listen: false)
                                .toggleSelection(queue[i], ''),
                        child: const Icon(
                          Icons.close,
                          size: 16,
                          color: Colors.white24,
                        ),
                      ),
                    ),
                  ),
                ),
        ),
      ],
    );
  }

  // --- ANALYSIS DIALOG ---
  void _showAnalysisDialog(BuildContext context, IdsProvider provider) async {
    showDialog(
        context: context,
        barrierDismissible: false,
        builder: (_) => const Center(
              child: SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Color(0xFF7C4DFF),
                ),
              ),
            ));

    final result = await provider.analyzeQueues();

    if (context.mounted) Navigator.pop(context);
    provider.setConsistencyStatus(true, true);

    if (context.mounted && result.isNotEmpty) {
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          backgroundColor: const Color(0xFF15191C),
          title: Text(
            "Consistency Report",
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              letterSpacing: 0.5,
              color: Colors.white,
            ),
          ),
          contentPadding: const EdgeInsets.fromLTRB(24, 16, 24, 16),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildScoreTile(
                "GAN Queue",
                result['gan_score'],
                result['gan_status'] as String? ?? 'No data',
              ),
              const SizedBox(height: 16),
              _buildScoreTile(
                "Jitter Queue",
                result['jitter_score'],
                result['jitter_status'] as String? ?? 'Not analysed',
              ),
            ],
          ),
          actionsPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text(
                "OK",
                style: TextStyle(
                  color: Color(0xFF00E5FF),
                  fontSize: 12,
                ),
              ),
            ),
          ],
        ),
      );
    }
  }

  // --- SCORE TILE ---
  Widget _buildScoreTile(String title, dynamic score, String? status) {
    double s = (score is num) ? score.toDouble() : 0.0;
    String safeStatus = status ?? 'No data';
    bool hasData = s > 0.0 ||
        (safeStatus != "No Features Found" && safeStatus != "Not enough data");
    Color color = s > 0.7 ? Colors.greenAccent : Colors.redAccent;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              title,
              style: const TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 13,
                color: Colors.white,
              ),
            ),
            if (hasData)
              Text(
                "${(s * 100).toStringAsFixed(1)}%",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: color,
                  fontSize: 12,
                ),
              ),
          ],
        ),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: hasData ? s : 0.0,
            backgroundColor: const Color(0xFF2D333B),
            color: color,
            minHeight: 6,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          safeStatus,
          style: const TextStyle(
            color: Colors.white54,
            fontSize: 11,
          ),
        ),
      ],
    );
  }
}
