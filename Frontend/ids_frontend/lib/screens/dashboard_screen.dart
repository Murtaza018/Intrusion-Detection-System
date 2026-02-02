import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import '../widgets/packet_list.dart';
import '../widgets/sensory_dashboard_widget.dart'; // Import the new Sensory Gauges
import 'adaptation_screen.dart';

class DashboardScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // We listen here to rebuild the counts/buttons when data changes
    final provider = Provider.of<IdsProvider>(context);

    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        elevation: 0,
        title: Text(
          'IDS Monitor',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 22),
        ),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black87,
        actions: [
          // ADAPT BUTTON (Shows when packets are selected for continual learning)
          if (provider.totalSelected > 0)
            Padding(
              padding: const EdgeInsets.only(right: 8.0),
              child: FilledButton.icon(
                onPressed: () => Navigator.push(context,
                    MaterialPageRoute(builder: (_) => AdaptationScreen())),
                icon: Icon(Icons.build, size: 18),
                label: Text("Adapt (${provider.totalSelected})"),
                style: FilledButton.styleFrom(
                  backgroundColor: Colors.deepPurple,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8)),
                ),
              ),
            ),

          // POWER BUTTON
          _buildPowerButton(context, provider),
          SizedBox(width: 16),
        ],
      ),
      body: Column(
        children: [
          // 1. Compact Stats Row
          Container(
            color: Colors.white,
            padding: EdgeInsets.symmetric(vertical: 12, horizontal: 16),
            child: Row(
              children: [
                _buildCompactStat(context, 'Total', provider.totalPackets,
                    Colors.blue, 'all'),
                SizedBox(width: 12),
                _buildCompactStat(context, 'Normal', provider.normalCount,
                    Colors.green, 'normal'),
                SizedBox(width: 12),
                _buildCompactStat(context, 'Attacks', provider.attackCount,
                    Colors.red, 'known_attack'),
                SizedBox(width: 12),
                _buildCompactStat(context, 'Zero-Day', provider.zeroDayCount,
                    Colors.orange, 'zero_day'),
              ],
            ),
          ),

          Divider(height: 1),

          // 2. NEW: SENSORY VISUALIZATION (Roadmap Point 3)
          // Displays GNN (Topological) and MAE (Visual) gauges
          if (provider.isRunning)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: SensoryDashboard(),
            ),

          // 3. Packet List (Scrollable Logs)
          Expanded(
            child: PacketList(),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _showAddPacketDialog(context),
        backgroundColor: Colors.blue[700],
        foregroundColor: Colors.white,
        child: Icon(Icons.bug_report),
        tooltip: 'Simulate Zero-Day',
      ),
    );
  }

  // A sleek Power Button
  Widget _buildPowerButton(BuildContext context, IdsProvider provider) {
    bool isRunning = provider.isRunning;
    return Container(
      margin: EdgeInsets.symmetric(vertical: 8),
      child: ElevatedButton.icon(
        onPressed: () {
          if (isRunning) {
            provider.stopPipeline();
          } else {
            provider.startPipeline();
          }
        },
        icon: Icon(isRunning ? Icons.stop : Icons.play_arrow),
        label: Text(isRunning ? 'STOP' : 'START'),
        style: ElevatedButton.styleFrom(
          backgroundColor: isRunning ? Colors.red[50] : Colors.green[50],
          foregroundColor: isRunning ? Colors.red : Colors.green,
          elevation: 0,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          padding: EdgeInsets.symmetric(horizontal: 16),
        ),
      ),
    );
  }

  // Compact Stat Widget (Self-Contained Provider Logic)
  Widget _buildCompactStat(BuildContext context, String label, int value,
      Color color, String filter) {
    final currentFilter = Provider.of<IdsProvider>(context).currentFilter;
    bool isSelected = currentFilter == filter;

    return Expanded(
      child: InkWell(
        onTap: () =>
            Provider.of<IdsProvider>(context, listen: false).setFilter(filter),
        borderRadius: BorderRadius.circular(12),
        child: AnimatedContainer(
          duration: Duration(milliseconds: 200),
          padding: EdgeInsets.symmetric(vertical: 12, horizontal: 4),
          decoration: BoxDecoration(
            color: isSelected ? color.withOpacity(0.1) : Colors.grey[50],
            border: Border.all(
                color: isSelected ? color : Colors.transparent, width: 2),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                value.toString(),
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                  color: color,
                ),
              ),
              SizedBox(height: 4),
              Text(
                label.toUpperCase(),
                style: TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  color: Colors.grey[600],
                  letterSpacing: 0.5,
                ),
                textAlign: TextAlign.center,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showAddPacketDialog(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context, listen: false);

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Simulate Alert'),
        content: Text('Add a simulated zero-day attack packet for testing?'),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel', style: TextStyle(color: Colors.grey)),
          ),
          FilledButton(
            onPressed: () {
              provider.addZeroDayPacket();
              Navigator.pop(context);
            },
            style: FilledButton.styleFrom(backgroundColor: Colors.orange),
            child: Text('Simulate'),
          ),
        ],
      ),
    );
  }
}
//hhhh