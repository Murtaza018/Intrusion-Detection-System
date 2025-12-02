import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import 'packet_detail_dialog.dart';

class PacketList extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<IdsProvider>(context);
    // UPDATED: Use filteredPackets
    final packets = provider.filteredPackets;

    if (packets.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.lan, size: 64, color: Colors.grey[400]),
            SizedBox(height: 16),
            Text(
              'No packets found', // Changed text
              style: TextStyle(color: Colors.grey[600]),
            ),
            Text(
              provider.isRunning
                  ? 'Waiting for packets...'
                  : 'Start the pipeline to begin monitoring',
              style: TextStyle(color: Colors.grey[500], fontSize: 12),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      // UPDATED: Use packets.length
      itemCount: packets.length,
      itemBuilder: (context, index) {
        // UPDATED: Use packets[index]
        final packet = packets[index];
        return PacketListItem(packet: packet);
      },
    );
  }
}

class PacketListItem extends StatelessWidget {
  final Packet packet;

  const PacketListItem({required this.packet});

  @override
  Widget build(BuildContext context) {
    Color statusColor;
    IconData statusIcon;
    String statusText;

    switch (packet.status) {
      case 'known_attack':
        statusColor = Colors.red;
        statusIcon = Icons.warning;
        statusText = 'KNOWN ATTACK';
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
      margin: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: ListTile(
        leading: Icon(statusIcon, color: statusColor),
        title: Text(
          packet.summary,
          style: TextStyle(fontWeight: FontWeight.w500),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('${packet.srcIp} → ${packet.dstIp}'),
            Text(
              '${packet.protocol} • ${packet.length} bytes • ${_formatTime(packet.timestamp)}',
              style: TextStyle(fontSize: 12, color: Colors.grey[600]),
            ),
          ],
        ),
        trailing: Chip(
          label: Text(
            statusText,
            style: TextStyle(color: Colors.white, fontSize: 10),
          ),
          backgroundColor: statusColor,
        ),
        onTap: () => _showPacketDetails(context, packet),
      ),
    );
  }

  String _formatTime(DateTime timestamp) {
    return '${timestamp.hour.toString().padLeft(2, '0')}:${timestamp.minute.toString().padLeft(2, '0')}:${timestamp.second.toString().padLeft(2, '0')}';
  }

  void _showPacketDetails(BuildContext context, Packet packet) {
    showDialog(
      context: context,
      builder: (context) => PacketDetailDialog(packet: packet),
    );
  }
}
