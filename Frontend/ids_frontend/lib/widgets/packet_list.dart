import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import 'packet_detail_dialog.dart';
import 'mae_visualizer_dialog.dart';

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
            Icon(Icons.radar, size: 40, color: Colors.white10),
            const SizedBox(height: 16),
            const Text('SCANNING NETWORK INTERFACE...',
                style: TextStyle(
                    color: Colors.white24, letterSpacing: 2, fontSize: 10)),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      itemCount: packets.length + 1,
      itemBuilder: (context, index) {
        if (index == packets.length) {
          return Padding(
            padding: const EdgeInsets.all(24.0),
            child: provider.isLoadingMore
                ? const Center(child: CircularProgressIndicator(strokeWidth: 2))
                : TextButton(
                    onPressed: () => provider.loadMorePackets(),
                    child: const Text("FETCH OLDER SEQUENCES",
                        style: TextStyle(fontSize: 10, color: Colors.white30))),
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
    String tag;
    switch (packet.status) {
      case 'known_attack':
        statusColor = const Color(0xFFFF5252);
        tag = "CRIT";
        break;
      case 'zero_day':
        statusColor = Colors.orangeAccent;
        tag = "WARN";
        break;
      default:
        statusColor = Colors.greenAccent;
        tag = "INFO";
    }

    return GestureDetector(
      onTap: () => showDialog(
          context: context, builder: (_) => PacketDetailDialog(packet: packet)),
      child: Container(
        margin: const EdgeInsets.only(bottom: 6),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFF232931) : const Color(0xFF15191C),
          border: Border(left: BorderSide(color: statusColor, width: 3)),
        ),
        child: ListTile(
          dense: true,
          visualDensity: VisualDensity.compact,
          title: Text(packet.summary,
              style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: Colors.white)),
          subtitle: Text(
              "${packet.srcIp} -> ${packet.dstIp} | ${packet.protocol} | ${packet.length}B",
              style: const TextStyle(
                  color: Colors.white38,
                  fontSize: 10,
                  fontFamily: 'monospace')),
          trailing: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (packet.status == 'zero_day' || packet.maeAnomaly > 0.1)
                IconButton(
                  icon: const Icon(Icons.grid_3x3,
                      color: Colors.orangeAccent, size: 16),
                  onPressed: () => showDialog(
                      context: context,
                      builder: (_) => MaeVisualizerDialog(packet: packet)),
                ),
              Text(tag,
                  style: TextStyle(
                      color: statusColor,
                      fontWeight: FontWeight.w900,
                      fontSize: 9,
                      fontFamily: 'monospace')),
            ],
          ),
        ),
      ),
    );
  }
}
