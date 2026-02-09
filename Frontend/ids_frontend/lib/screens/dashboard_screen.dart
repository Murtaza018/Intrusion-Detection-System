import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import '../widgets/packet_list.dart';
import '../widgets/sensory_dashboard_widget.dart';
import 'adaptation_screen.dart';

class DashboardScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context);

    return Scaffold(
      body: Row(
        children: [
          _buildSidebar(context, provider),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildTopBar(provider),
                _buildCompactSearchFilterBar(provider),

                // Optimized Sensory Layer
                if (provider.isRunning)
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 24, vertical: 4),
                    child: SensoryDashboard(),
                  ),

                Padding(
                  padding: const EdgeInsets.fromLTRB(24, 12, 24, 4),
                  child: Row(
                    children: [
                      const Icon(Icons.analytics_outlined,
                          color: Colors.white24, size: 12),
                      const SizedBox(width: 8),
                      Text(
                          "LIVE DATA STREAM [${provider.filteredPackets.length} PACKETS]",
                          style: const TextStyle(
                              color: Colors.white38,
                              fontSize: 9,
                              fontWeight: FontWeight.bold,
                              letterSpacing: 1.5)),
                      const Spacer(),
                      const Text("REAL-TIME UPDATES ACTIVE",
                          style: TextStyle(
                              color: Colors.greenAccent,
                              fontSize: 8,
                              fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),

                Expanded(child: PacketList()),
              ],
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton.small(
        onPressed: () => _showAddPacketDialog(context),
        backgroundColor: Colors.orangeAccent,
        child: const Icon(Icons.bug_report, size: 18),
      ),
    );
  }

  Widget _buildCompactSearchFilterBar(IdsProvider provider) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 4),
      padding: const EdgeInsets.symmetric(
          horizontal: 12, vertical: 2), // Tightened padding
      decoration: BoxDecoration(
        color: const Color(0xFF15191C),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: Colors.white.withOpacity(0.03)),
      ),
      child: Row(
        children: [
          Expanded(
            flex: 3,
            child: TextField(
              onChanged: (val) => provider.updateSearchQuery(val),
              textAlignVertical:
                  TextAlignVertical.center, // FIX: Centers the placeholder text
              style: const TextStyle(
                  fontSize: 11, color: Colors.white, fontFamily: 'monospace'),
              decoration: const InputDecoration(
                isCollapsed: true, // FIX: Removes default internal padding
                hintText: "FILTER BY IP, PORT, OR PROTOCOL...",
                hintStyle: TextStyle(color: Colors.white10, fontSize: 11),
                prefixIcon: Icon(Icons.search, size: 14, color: Colors.white24),
                border: InputBorder.none,
                contentPadding: EdgeInsets.symmetric(
                    vertical: 12), // Centers text vertically
              ),
            ),
          ),
          Container(
              width: 1,
              height: 20,
              color: Colors.white10,
              margin: const EdgeInsets.symmetric(horizontal: 12)),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildFilterChip(provider, 'ALL', 'all'),
              _buildFilterChip(provider, 'NORMAL', 'normal'),
              _buildFilterChip(provider, 'THREATS', 'known_attack'),
              _buildFilterChip(provider, 'ZERO-DAY', 'zero_day'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSidebar(BuildContext context, IdsProvider provider) {
    return Container(
      width: 220,
      color: const Color(0xFF0D1117),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text("COMMAND CENTER",
              style: TextStyle(
                  color: Color(0xFF00E5FF),
                  fontSize: 9,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.5)),
          const SizedBox(height: 12),
          _buildPowerControl(provider),
          const SizedBox(height: 28),
          const Divider(color: Colors.white10, height: 1),
          const SizedBox(height: 20),
          const Text("ENGINE TELEMETRY",
              style: TextStyle(
                  color: Colors.white24,
                  fontSize: 9,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.2)),
          const SizedBox(height: 16),
          _buildSidebarStat(
              "TOTAL SCANS", provider.totalPackets, const Color(0xFF00E5FF)),
          _buildSidebarStat("BENIGN", provider.normalCount, Colors.greenAccent),
          _buildSidebarStat("THREATS", provider.attackCount, Colors.redAccent),
          _buildSidebarStat(
              "NOVELTY", provider.zeroDayCount, Colors.orangeAccent),
          const Spacer(),
          if (provider.totalSelected > 0)
            _buildAdaptiveButton(context, provider),
        ],
      ),
    );
  }

  Widget _buildSidebarStat(String label, int value, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label,
              style: const TextStyle(
                  fontSize: 9, color: Colors.white54, fontFamily: 'monospace')),
          Text(value.toString().padLeft(5, '0'),
              style: TextStyle(
                  fontSize: 12,
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontFamily: 'monospace')),
        ],
      ),
    );
  }

  Widget _buildPowerControl(IdsProvider provider) {
    bool isRunning = provider.isRunning;
    return InkWell(
      onTap: () =>
          isRunning ? provider.stopPipeline() : provider.startPipeline(),
      borderRadius: BorderRadius.circular(4),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
        decoration: BoxDecoration(
          color: isRunning
              ? Colors.red.withOpacity(0.05)
              : Colors.green.withOpacity(0.05),
          borderRadius: BorderRadius.circular(4),
          border: Border.all(
              color: isRunning
                  ? Colors.redAccent.withOpacity(0.3)
                  : Colors.greenAccent.withOpacity(0.3)),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(isRunning ? Icons.power_settings_new : Icons.bolt,
                color: isRunning ? Colors.redAccent : Colors.greenAccent,
                size: 14),
            const SizedBox(width: 8),
            Text(isRunning ? "STOP ENGINE" : "START CORE",
                style: TextStyle(
                    color: isRunning ? Colors.redAccent : Colors.greenAccent,
                    fontWeight: FontWeight.bold,
                    fontSize: 10,
                    letterSpacing: 1)),
          ],
        ),
      ),
    );
  }

  Widget _buildAdaptiveButton(BuildContext context, IdsProvider provider) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(boxShadow: [
        BoxShadow(
            color: const Color(0xFF7C4DFF).withOpacity(0.1),
            blurRadius: 4,
            offset: const Offset(0, 2))
      ]),
      child: ElevatedButton(
        onPressed: () => Navigator.push(
            context, MaterialPageRoute(builder: (_) => AdaptationScreen())),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF7C4DFF),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 10),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(4)),
          elevation: 0,
        ),
        child: Text("ADAPTIVE FEEDBACK (${provider.totalSelected})",
            style: const TextStyle(
                fontSize: 9, fontWeight: FontWeight.bold, letterSpacing: 0.5)),
      ),
    );
  }

  Widget _buildTopBar(IdsProvider provider) {
    return Container(
      padding: const EdgeInsets.fromLTRB(24, 20, 24, 4),
      child: Row(
        children: [
          const Text("NEURAL-IDS CONTROL",
              style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w900,
                  letterSpacing: -0.5)),
          const SizedBox(width: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: provider.isRunning
                  ? Colors.greenAccent.withOpacity(0.1)
                  : Colors.redAccent.withOpacity(0.1),
              borderRadius: BorderRadius.circular(2),
            ),
            child: Text(provider.isRunning ? "CORE ONLINE" : "OFFLINE",
                style: TextStyle(
                    color: provider.isRunning
                        ? Colors.greenAccent
                        : Colors.redAccent,
                    fontSize: 8,
                    fontWeight: FontWeight.bold)),
          ),
          const Spacer(),
          const Icon(Icons.notifications_none, color: Colors.white12, size: 18),
        ],
      ),
    );
  }

  Widget _buildFilterChip(IdsProvider provider, String label, String filter) {
    bool selected = provider.currentFilter == filter;
    return Padding(
      padding: const EdgeInsets.only(left: 6),
      child: InkWell(
        onTap: () => provider.setFilter(filter),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: selected
                ? const Color(0xFF00E5FF).withOpacity(0.15)
                : Colors.transparent,
            borderRadius: BorderRadius.circular(2),
            border: Border.all(
                color: selected
                    ? const Color(0xFF00E5FF).withOpacity(0.5)
                    : Colors.white10),
          ),
          child: Text(label,
              style: TextStyle(
                  color: selected ? const Color(0xFF00E5FF) : Colors.white38,
                  fontSize: 8,
                  fontWeight: FontWeight.bold)),
        ),
      ),
    );
  }

  void _showAddPacketDialog(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context, listen: false);
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF15191C),
        shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(4),
            side: BorderSide(color: Colors.orangeAccent.withOpacity(0.1))),
        title: const Text("STRUCTURAL NOVELTY TEST",
            style: TextStyle(
                color: Colors.orangeAccent,
                fontSize: 13,
                fontWeight: FontWeight.bold)),
        content: const Text(
            "Inject a high-entropy simulated 0-day packet for engine validation?",
            style: TextStyle(fontSize: 11, color: Colors.white70)),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text("ABORT",
                  style: TextStyle(color: Colors.white24, fontSize: 10))),
          FilledButton(
            onPressed: () {
              provider.addZeroDayPacket();
              Navigator.pop(context);
            },
            style: FilledButton.styleFrom(
                backgroundColor: Colors.orangeAccent,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(2))),
            child: const Text("EXECUTE",
                style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }
}
