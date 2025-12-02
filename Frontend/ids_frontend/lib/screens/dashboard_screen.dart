import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import '../widgets/packet_list.dart';
import '../widgets/stats_card.dart';

class DashboardScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('IDS Pipeline Monitor'),
        backgroundColor: Colors.blue[800],
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // Control Panel
          _buildControlPanel(context),

          // Statistics with ZERO-DAY COUNTER
          _buildStatsGrid(context),

          // Packet List
          Expanded(
            child: PacketList(),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () => _showAddPacketDialog(context),
        child: Icon(Icons.add_alert),
        tooltip: 'Simulate Alert',
      ),
    );
  }

  Widget _buildControlPanel(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context);

    return Container(
      padding: EdgeInsets.all(16),
      color: Colors.grey[50],
      child: Row(
        children: [
          Expanded(
            child: ElevatedButton(
              onPressed:
                  provider.isRunning ? null : () => provider.startPipeline(),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.play_arrow),
                  SizedBox(width: 8),
                  Text('START PIPELINE'),
                ],
              ),
            ),
          ),
          SizedBox(width: 12),
          Expanded(
            child: ElevatedButton(
              onPressed:
                  provider.isRunning ? () => provider.stopPipeline() : null,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red,
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.stop),
                  SizedBox(width: 8),
                  Text('STOP PIPELINE'),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // UPDATED: _buildStatsGrid with clickable cards and selection highlight
  Widget _buildStatsGrid(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context);
    final currentFilter = provider.currentFilter;

    return Container(
      padding: EdgeInsets.all(16),
      child: Row(
        children: [
          Expanded(
            child: _buildClickableStatsCard(
              context: context,
              title: 'TOTAL',
              value: provider.totalPackets.toString(),
              color: Colors.blue,
              icon: Icons.lan,
              filter: 'all',
              isSelected: currentFilter == 'all',
            ),
          ),
          SizedBox(width: 8),
          Expanded(
            child: _buildClickableStatsCard(
              context: context,
              title: 'NORMAL',
              value: provider.normalCount.toString(),
              color: Colors.green,
              icon: Icons.check_circle,
              filter: 'normal',
              isSelected: currentFilter == 'normal',
            ),
          ),
          SizedBox(width: 8),
          Expanded(
            child: _buildClickableStatsCard(
              context: context,
              title: 'ATTACKS',
              value: provider.attackCount.toString(),
              color: Colors.red,
              icon: Icons.warning,
              filter: 'known_attack',
              isSelected: currentFilter == 'known_attack',
            ),
          ),
          SizedBox(width: 8),
          Expanded(
            child: _buildClickableStatsCard(
              context: context,
              title: 'ZERO-DAY',
              value: provider.zeroDayCount.toString(),
              color: Colors.orange,
              icon: Icons.new_releases,
              filter: 'zero_day',
              isSelected: currentFilter == 'zero_day',
            ),
          ),
        ],
      ),
    );
  }

  // NEW: Helper widget for clickable StatsCard
  Widget _buildClickableStatsCard({
    required BuildContext context,
    required String title,
    required String value,
    required Color color,
    required IconData icon,
    required String filter,
    required bool isSelected,
  }) {
    return InkWell(
      onTap: () {
        Provider.of<IdsProvider>(context, listen: false).setFilter(filter);
      },
      child: Container(
        decoration: isSelected
            ? BoxDecoration(
                border: Border.all(color: color, width: 2),
                borderRadius: BorderRadius.circular(8),
              )
            : null,
        child: StatsCard(
          title: title,
          value: value,
          color: color,
          icon: icon,
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
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              provider.addZeroDayPacket();
              Navigator.pop(context);
            },
            child: Text('Add'),
          ),
        ],
      ),
    );
  }
}
