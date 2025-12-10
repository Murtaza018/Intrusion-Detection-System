import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';

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
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: Text("Model Adaptation",
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20)),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
        elevation: 0,
      ),
      body: Column(
        children: [
          // 1. Compact Labeling Section
          _buildCompactLabelingSection(context, provider),
          Divider(height: 1, thickness: 1),

          // 2. Main Content (Two Columns)
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // LEFT: JITTER QUEUE
                Expanded(
                  flex: 1,
                  child: _buildQueueList(context, "False Positives (Jitter)",
                      provider.jitterQueue, Colors.green),
                ),
                VerticalDivider(
                    width: 1, thickness: 1, color: Colors.grey[300]),

                // RIGHT: GAN QUEUE
                Expanded(
                  flex: 1,
                  child: _buildQueueList(context, "New Attacks (GAN)",
                      provider.ganQueue, Colors.red),
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
    if (provider.ganQueue.isEmpty) return SizedBox.shrink();

    return Container(
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      color: Colors.white,
      child: Column(
        mainAxisSize: MainAxisSize.min, // Shrink to fit
        children: [
          // Row 1: Title + Toggle Chips (Side by Side to save space)
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("Assign Label:",
                  style: TextStyle(
                      fontWeight: FontWeight.bold, color: Colors.grey[800])),
              Row(
                children: [
                  _buildCompactChip("Existing", !provider.isNewAttack,
                      () => provider.setBatchLabel(null, false)),
                  SizedBox(width: 8),
                  _buildCompactChip("New Zero-Day", provider.isNewAttack,
                      () => provider.setBatchLabel(null, true)),
                ],
              ),
            ],
          ),
          SizedBox(height: 8),

          // Row 2: Input Field (Thinner)
          SizedBox(
            height: 48, // Fixed height for compactness
            child: provider.isNewAttack
                ? TextField(
                    decoration: _compactInputDecoration("Enter New Attack Name",
                        Icons.new_releases, Colors.orange),
                    onChanged: (val) => provider.setBatchLabel(val, true),
                    style: TextStyle(fontSize: 14),
                  )
                : DropdownButtonFormField<String>(
                    decoration: _compactInputDecoration(
                        "Select Existing Attack Type",
                        Icons.category,
                        Colors.blue),
                    value: provider.batchLabel,
                    style: TextStyle(fontSize: 14, color: Colors.black),
                    items: provider.existingLabels.isEmpty
                        ? [
                            DropdownMenuItem(
                                value: "Loading...", child: Text("Loading..."))
                          ]
                        : provider.existingLabels
                            .map((l) =>
                                DropdownMenuItem(value: l, child: Text(l)))
                            .toList(),
                    onChanged: (val) => provider.setBatchLabel(val, false),
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
      prefixIcon: Icon(icon, color: color, size: 20),
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
      contentPadding:
          EdgeInsets.symmetric(horizontal: 12, vertical: 0), // Minimal padding
      labelStyle: TextStyle(fontSize: 13),
    );
  }

  Widget _buildCompactChip(String label, bool isSelected, VoidCallback onTap) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected
              ? Colors.deepPurple.withOpacity(0.1)
              : Colors.grey[100],
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
              color: isSelected ? Colors.deepPurple : Colors.grey[300]!),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 12,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            color: isSelected ? Colors.deepPurple : Colors.grey[700],
          ),
        ),
      ),
    );
  }

  // --- BOTTOM BAR (ACTIONS) ---
  Widget _buildBottomBar(BuildContext context, IdsProvider provider) {
    bool hasLabel =
        provider.batchLabel != null && provider.batchLabel!.isNotEmpty;
    bool hasGanPackets = provider.ganQueue.isNotEmpty;
    bool hasJitterPackets = provider.jitterQueue.isNotEmpty;

    bool consistencyReady =
        (hasGanPackets || hasJitterPackets) && (!hasGanPackets || hasLabel);
    bool trainingReady = provider.consistencyChecked && consistencyReady;

    return Container(
      padding: EdgeInsets.fromLTRB(16, 8, 16, 16), // Reduced top padding
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(color: Colors.black12, blurRadius: 5, offset: Offset(0, -2))
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Warning Text (Only shows if needed)
          if ((hasGanPackets || hasJitterPackets) && !trainingReady)
            Padding(
              padding: const EdgeInsets.only(bottom: 8.0),
              child: Text(
                (hasGanPackets && !hasLabel)
                    ? "* Select a label above to proceed"
                    : "* Check consistency before training",
                style: TextStyle(
                    color: Colors.orange[800],
                    fontSize: 12,
                    fontWeight: FontWeight.w500),
              ),
            ),

          // Two Buttons Side-by-Side
          Row(
            children: [
              // 2. CHECK CONSISTENCY
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: (!consistencyReady || provider.totalSelected <= 1)
                      ? null
                      : () => _showAnalysisDialog(context, provider),
                  icon: Icon(Icons.analytics_outlined, size: 18),
                  label: Text(
                      provider.consistencyChecked
                          ? "Re-Check"
                          : "Check Consistency",
                      style: TextStyle(fontSize: 13)),
                  style: OutlinedButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8)),
                  ),
                ),
              ),
              SizedBox(width: 12),

              // 3. START TRAINING
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: !trainingReady
                      ? null
                      : () async {
                          // (Same logic as before)
                          ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                              content: Text("Sending data..."),
                              duration: Duration(seconds: 1)));
                          bool success = await provider.sendRetrainRequest();
                          if (success) {
                            provider.clearQueues();
                            if (context.mounted) {
                              Navigator.pop(context);
                              ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(
                                      content: Text("Training Queued!"),
                                      backgroundColor: Colors.green));
                            }
                          }
                        },
                  icon: Icon(Icons.model_training, size: 18),
                  label:
                      Text("Start Retraining", style: TextStyle(fontSize: 13)),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.deepPurple,
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8)),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  // --- QUEUE LIST (No Changes needed here) ---
  Widget _buildQueueList(
      BuildContext context, String title, List<Packet> queue, Color color) {
    return Column(
      children: [
        Container(
          padding: EdgeInsets.symmetric(vertical: 10, horizontal: 4),
          width: double.infinity,
          color: color.withOpacity(0.08),
          child: Text(title,
              style: TextStyle(
                  fontWeight: FontWeight.bold, color: color, fontSize: 13),
              textAlign: TextAlign.center),
        ),
        Expanded(
          child: queue.isEmpty
              ? Center(
                  child:
                      Text("Empty", style: TextStyle(color: Colors.grey[300])))
              : ListView.builder(
                  itemCount: queue.length,
                  padding: EdgeInsets.all(0),
                  itemBuilder: (ctx, i) => Container(
                    decoration: BoxDecoration(
                        border: Border(
                            bottom: BorderSide(color: Colors.grey[200]!))),
                    child: ListTile(
                      dense: true, // Make list tile compact
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 12, vertical: 0),
                      leading: CircleAvatar(
                        backgroundColor: color.withOpacity(0.1),
                        child: Text("${queue[i].id}",
                            style: TextStyle(
                                fontSize: 9,
                                fontWeight: FontWeight.bold,
                                color: color)),
                        radius: 14,
                      ),
                      title: Text(queue[i].summary,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(fontSize: 12)),
                      trailing: InkWell(
                        child: Icon(Icons.close,
                            size: 16, color: Colors.grey[400]),
                        onTap: () =>
                            Provider.of<IdsProvider>(context, listen: false)
                                .toggleSelection(queue[i], ''),
                      ),
                    ),
                  ),
                ),
        ),
      ],
    );
  }

  // --- ANALYSIS DIALOG (Same logic, just keeping it here) ---
  void _showAnalysisDialog(BuildContext context, IdsProvider provider) async {
    showDialog(
        context: context,
        barrierDismissible: false,
        builder: (_) => Center(child: CircularProgressIndicator()));
    final result = await provider.analyzeQueues();
    if (context.mounted) Navigator.pop(context);
    provider.setConsistencyStatus(true, true);

    if (context.mounted && result.isNotEmpty) {
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: Text("Consistency Report",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          contentPadding: EdgeInsets.fromLTRB(24, 20, 24, 0),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildScoreTile(
                  "GAN Queue", result['gan_score'], result['gan_status']),
              SizedBox(height: 16),
              _buildScoreTile("Jitter Queue", result['jitter_score'],
                  result['jitter_status']),
              SizedBox(height: 10),
            ],
          ),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(context), child: Text("OK"))
          ],
        ),
      );
    }
  }

  Widget _buildScoreTile(String title, dynamic score, String status) {
    double s = (score is num) ? score.toDouble() : 0.0;
    bool hasData = s > 0.0 ||
        (status != "No Features Found" && status != "Not enough data");
    Color color = s > 0.7 ? Colors.green : Colors.red;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(title,
                style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
            if (hasData)
              Text("${(s * 100).toStringAsFixed(1)}%",
                  style: TextStyle(fontWeight: FontWeight.bold, color: color)),
          ],
        ),
        SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
              value: hasData ? s : 0.0,
              backgroundColor: Colors.grey[200],
              color: color,
              minHeight: 8),
        ),
        SizedBox(height: 4),
        Text(status, style: TextStyle(color: Colors.grey[600], fontSize: 11)),
      ],
    );
  }
}
