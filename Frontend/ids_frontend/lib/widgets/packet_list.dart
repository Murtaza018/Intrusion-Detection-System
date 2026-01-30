import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import 'packet_detail_dialog.dart';
import 'mae_visualizer_dialog.dart'; // Import the new dialog

class PacketList extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context);
    final packets = provider.filteredPackets;

    if (packets.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.lan, size: 64, color: Colors.grey[300]),
            SizedBox(height: 16),
            Text('No packets found', style: TextStyle(color: Colors.grey[500])),
          ],
        ),
      );
    }

    return ListView.builder(
      itemCount: packets.length + 1,
      itemBuilder: (context, index) {
        if (index == packets.length) {
          return Padding(
            padding: EdgeInsets.all(16.0),
            child: provider.isLoadingMore
                ? Center(child: CircularProgressIndicator())
                : TextButton(
                    onPressed: () => provider.loadMorePackets(),
                    child: Text("Load Older Packets")),
          );
        }
        return PacketListItem(packet: packets[index]);
      },
    );
  }
}

class PacketListItem extends StatelessWidget {
  final Packet packet;

  const PacketListItem({required this.packet});

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context);
    bool isSelected = provider.isSelected(packet.id);

    Color statusColor;
    IconData statusIcon;
    String statusText;

    switch (packet.status) {
      case 'known_attack':
        statusColor = Colors.red;
        statusIcon = Icons.warning;
        statusText = 'ATTACK';
        break;
      case 'zero_day':
        statusColor = Colors.orange;
        statusIcon = Icons.new_releases;
        statusText = 'ZERO-DAY';
        break;
      default:
        statusColor = Colors.green;
        statusIcon = Icons.check_circle;
        statusText = 'NORMAL';
    }

    return Card(
      elevation: isSelected ? 4 : 1,
      color: isSelected ? Colors.deepPurple.shade50 : Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: isSelected
            ? BorderSide(color: Colors.deepPurple, width: 2)
            : BorderSide.none,
      ),
      margin: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: ListTile(
        leading: Icon(statusIcon, color: statusColor),
        title: Text(packet.summary,
            style: TextStyle(fontWeight: FontWeight.w500, fontSize: 13)),
        subtitle: Text('${packet.srcIp} → ${packet.dstIp} • ${packet.protocol}',
            style: TextStyle(fontSize: 11)),

        // --- UPDATED TRAILING SECTION (ROADMAP POINT 3) ---
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Only show visualizer for anomalies or zero-days
            if (packet.status == 'zero_day' || packet.maeAnomaly > 0.1)
              IconButton(
                icon: Icon(Icons.grid_view_rounded,
                    color: Colors.orangeAccent, size: 20),
                tooltip: 'View MAE Structure',
                onPressed: () => showDialog(
                  context: context,
                  builder: (_) => MaeVisualizerDialog(packet: packet),
                ),
              ),

            // Status Chip
            Chip(
              label: Text(statusText,
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 9,
                      fontWeight: FontWeight.bold)),
              backgroundColor: statusColor,
              padding: EdgeInsets.zero,
              visualDensity: VisualDensity.compact,
            ),
          ],
        ),

        onTap: () => showDialog(
            context: context,
            builder: (_) => PacketDetailDialog(packet: packet)),
      ),
    );
  }
}
