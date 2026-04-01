// lib/screens/dashboard_screen.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import '../widgets/packet_list.dart';
import '../widgets/sensory_dashboard_widget.dart';
import 'adaptation_screen.dart';

class DashboardScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context, listen: false);

    return Scaffold(
      body: Row(
        children: [
          _buildSidebar(context, provider),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildTopBar(provider),
                _buildSearchFilterBar(context, provider),
                Selector<IdsProvider, bool>(
                  selector: (_, p) => p.isRunning,
                  builder: (context, isRunning, child) {
                    return isRunning
                        ? Padding(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 24, vertical: 4),
                            child: SensoryDashboard(),
                          )
                        : const SizedBox.shrink();
                  },
                ),
                _buildStreamHeader(),
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

  // ── Stream header ─────────────────────────────────────────────────────────

  Widget _buildStreamHeader() {
    return Consumer<IdsProvider>(
      builder: (context, provider, _) => Padding(
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
            if (provider.isRunning)
              const Text("REAL-TIME UPDATES ACTIVE",
                  style: TextStyle(
                      color: Colors.greenAccent,
                      fontSize: 8,
                      fontWeight: FontWeight.bold)),
          ],
        ),
      ),
    );
  }

  // ── Search + Filter bar ───────────────────────────────────────────────────

  Widget _buildSearchFilterBar(BuildContext context, IdsProvider provider) {
    return Container(
      margin: const EdgeInsets.fromLTRB(24, 6, 24, 4),
      decoration: BoxDecoration(
        color: const Color(0xFF0D1117),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Top row: search field ───────────────────────────────────────
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 2, 8, 2),
            child: Row(children: [
              const Icon(Icons.search, size: 14, color: Colors.white24),
              const SizedBox(width: 8),
              Expanded(
                child: TextField(
                  onChanged: (val) => provider.updateSearchQuery(val),
                  textAlignVertical: TextAlignVertical.center,
                  style: const TextStyle(
                      fontSize: 11,
                      color: Colors.white,
                      fontFamily: 'monospace'),
                  decoration: const InputDecoration(
                    isCollapsed: true,
                    hintText:
                        'ip:192.168  port:443  severity:high  proto:TCP  …',
                    hintStyle: TextStyle(color: Colors.white12, fontSize: 11),
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.symmetric(vertical: 10),
                  ),
                ),
              ),
              // Help tooltip
              _SearchHelpButton(),
            ]),
          ),

          // Divider
          Container(height: 1, color: Colors.white.withOpacity(0.04)),

          // ── Bottom row: filter chips + severity slider ──────────────────
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 5, 12, 5),
            child: Row(children: [
              // Type filter chips
              _buildFilterChip(provider, 'ALL', 'all'),
              _buildFilterChip(provider, 'NORMAL', 'normal'),
              _buildFilterChip(provider, 'THREATS', 'known_attack'),
              _buildFilterChip(provider, 'ZERO-DAY', 'zero_day'),

              Container(
                  width: 1,
                  height: 16,
                  color: Colors.white10,
                  margin: const EdgeInsets.symmetric(horizontal: 12)),

              // Severity filter chips
              const Text('SEVERITY',
                  style: TextStyle(
                      color: Colors.white24, fontSize: 8, letterSpacing: 1.2)),
              const SizedBox(width: 8),
              _buildSeverityChip(provider, 'ANY', null),
              _buildSeverityChip(provider, 'LOW', 'low'),
              _buildSeverityChip(provider, 'MED', 'medium'),
              _buildSeverityChip(provider, 'HIGH', 'high'),
              _buildSeverityChip(provider, 'CRIT', 'critical'),

              const Spacer(),

              // Active filter indicator
              Consumer<IdsProvider>(
                builder: (_, p, __) {
                  final active = p.activeFilterCount;
                  if (active == 0) return const SizedBox.shrink();
                  return GestureDetector(
                    onTap: () => p.clearAllFilters(),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: Colors.redAccent.withOpacity(0.1),
                        borderRadius: BorderRadius.circular(3),
                        border: Border.all(
                            color: Colors.redAccent.withOpacity(0.3)),
                      ),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        const Icon(Icons.filter_list,
                            color: Colors.redAccent, size: 10),
                        const SizedBox(width: 4),
                        Text('$active ACTIVE · CLEAR',
                            style: const TextStyle(
                                color: Colors.redAccent,
                                fontSize: 8,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 0.5)),
                      ]),
                    ),
                  );
                },
              ),
            ]),
          ),
        ],
      ),
    );
  }

  Widget _buildFilterChip(IdsProvider provider, String label, String filter) {
    return Consumer<IdsProvider>(
      builder: (context, p, _) {
        final selected = p.currentFilter == filter;
        return Padding(
          padding: const EdgeInsets.only(right: 6),
          child: InkWell(
            onTap: () => p.setFilter(filter),
            borderRadius: BorderRadius.circular(3),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: selected
                    ? const Color(0xFF00E5FF).withOpacity(0.12)
                    : Colors.transparent,
                borderRadius: BorderRadius.circular(3),
                border: Border.all(
                    color: selected
                        ? const Color(0xFF00E5FF).withOpacity(0.5)
                        : Colors.white10),
              ),
              child: Text(label,
                  style: TextStyle(
                      color:
                          selected ? const Color(0xFF00E5FF) : Colors.white38,
                      fontSize: 8,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.5)),
            ),
          ),
        );
      },
    );
  }

  Widget _buildSeverityChip(
      IdsProvider provider, String label, String? severity) {
    final chipColors = {
      'low': Colors.greenAccent,
      'medium': Colors.yellowAccent,
      'high': Colors.orange,
      'critical': Colors.redAccent,
    };
    final color = severity != null
        ? (chipColors[severity] ?? Colors.white54)
        : Colors.white54;

    return Consumer<IdsProvider>(
      builder: (_, p, __) {
        final selected = p.currentSeverityFilter == severity;
        return Padding(
          padding: const EdgeInsets.only(right: 5),
          child: InkWell(
            onTap: () => p.setSeverityFilter(severity),
            borderRadius: BorderRadius.circular(3),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 4),
              decoration: BoxDecoration(
                color: selected ? color.withOpacity(0.12) : Colors.transparent,
                borderRadius: BorderRadius.circular(3),
                border: Border.all(
                    color: selected ? color.withOpacity(0.5) : Colors.white10),
              ),
              child: Text(label,
                  style: TextStyle(
                      color: selected ? color : Colors.white24,
                      fontSize: 8,
                      fontWeight: FontWeight.bold)),
            ),
          ),
        );
      },
    );
  }

  // ── Sidebar ───────────────────────────────────────────────────────────────

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
          Selector<IdsProvider, bool>(
            selector: (_, p) => p.isRunning,
            builder: (context, isRunning, _) => _buildPowerControl(provider),
          ),
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
          Selector<IdsProvider, int>(
            selector: (_, p) => p.totalPackets,
            builder: (_, val, __) =>
                _buildSidebarStat("TOTAL SCANS", val, const Color(0xFF00E5FF)),
          ),
          Selector<IdsProvider, int>(
            selector: (_, p) => p.normalCount,
            builder: (_, val, __) =>
                _buildSidebarStat("BENIGN", val, Colors.greenAccent),
          ),
          Selector<IdsProvider, int>(
            selector: (_, p) => p.attackCount,
            builder: (_, val, __) =>
                _buildSidebarStat("THREATS", val, Colors.redAccent),
          ),
          Selector<IdsProvider, int>(
            selector: (_, p) => p.zeroDayCount,
            builder: (_, val, __) =>
                _buildSidebarStat("NOVELTY", val, Colors.orangeAccent),
          ),
          const Spacer(),
          Selector<IdsProvider, int>(
            selector: (_, p) => p.totalSelected,
            builder: (context, total, _) => total > 0
                ? _buildAdaptiveButton(context, provider)
                : const SizedBox.shrink(),
          ),
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
          const SizedBox(width: 8),
          Selector<IdsProvider, bool>(
            selector: (_, p) => p.isRunning,
            builder: (_, isRunning, __) => Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: isRunning
                    ? Colors.greenAccent.withOpacity(0.1)
                    : Colors.redAccent.withOpacity(0.1),
                borderRadius: BorderRadius.circular(2),
              ),
              child: Text(isRunning ? "CORE ONLINE" : "OFFLINE",
                  style: TextStyle(
                      color: isRunning ? Colors.greenAccent : Colors.redAccent,
                      fontSize: 8,
                      fontWeight: FontWeight.bold)),
            ),
          ),
          const Spacer(),
          const Icon(Icons.notifications_none, color: Colors.white12, size: 18),
        ],
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

// ═══════════════════════════════════════════════════════════════════════════
// Search help tooltip button
// ═══════════════════════════════════════════════════════════════════════════

class _SearchHelpButton extends StatelessWidget {
  const _SearchHelpButton();

  static const _helpItems = [
    _HelpEntry('ip: ', '192.168', 'Filter by source or destination IP'),
    _HelpEntry('port: ', '443', 'Filter by port number'),
    _HelpEntry('proto: ', 'TCP', 'Filter by protocol (TCP/UDP/ICMP)'),
    _HelpEntry('severity: ', 'high', 'low · medium · high · critical'),
    _HelpEntry('type: ', 'attack', 'normal · known_attack · zero_day'),
    _HelpEntry('anomaly: ', '> 0.5', 'Anromaly score threshold (e.g. >0.3)'),
    _HelpEntry('', '', ''), // spacer
    _HelpEntry('Example:', '', 'ip:192.168 severity:high proto:TCP'),
  ];

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      richMessage: WidgetSpan(
        child: Container(
          width: 320,
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('SEARCH SYNTAX',
                  style: TextStyle(
                      color: Color(0xFF00E5FF),
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.5)),
              const SizedBox(height: 10),
              ..._helpItems.map((e) {
                if (e.prefix.isEmpty && e.example.isEmpty) {
                  return const Divider(color: Colors.white12, height: 16);
                }
                return Padding(
                  padding: const EdgeInsets.only(bottom: 7),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      SizedBox(
                        width: 80,
                        child: RichText(
                          text: TextSpan(children: [
                            TextSpan(
                                text: e.prefix,
                                style: const TextStyle(
                                    color: Color(0xFF00E5FF),
                                    fontSize: 10,
                                    fontFamily: 'monospace',
                                    fontWeight: FontWeight.bold)),
                            TextSpan(
                                text: e.example.isNotEmpty ? e.example : '',
                                style: const TextStyle(
                                    color: Colors.white54,
                                    fontSize: 10,
                                    fontFamily: 'monospace')),
                          ]),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(e.description,
                            style: const TextStyle(
                                color: Colors.white54, fontSize: 10)),
                      ),
                    ],
                  ),
                );
              }),
            ],
          ),
        ),
      ),
      decoration: BoxDecoration(
        color: const Color(0xFF0D1117),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
        boxShadow: [
          BoxShadow(
              color: Colors.black.withOpacity(0.5),
              blurRadius: 16,
              offset: const Offset(0, 4)),
        ],
      ),
      preferBelow: true,
      triggerMode: TooltipTriggerMode.tap,
      showDuration: const Duration(seconds: 8),
      child: Container(
        padding: const EdgeInsets.all(6),
        child: const Icon(
          Icons.help_outline_rounded,
          color: Colors.white24,
          size: 15,
        ),
      ),
    );
  }
}

class _HelpEntry {
  final String prefix;
  final String example;
  final String description;
  const _HelpEntry(this.prefix, this.example, this.description);
}
