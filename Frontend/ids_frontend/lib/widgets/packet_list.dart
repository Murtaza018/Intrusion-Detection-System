import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';
import 'packet_detail_dialog.dart';
import 'mae_visualizer_dialog.dart';
import '../models/packet.dart';

class PacketList extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // Using Selector to only rebuild when the filtered list changes
    return Selector<IdsProvider, List<Packet>>(
      selector: (_, provider) => provider.filteredPackets,
      builder: (context, packets, child) {
        final provider = Provider.of<IdsProvider>(context, listen: false);

        if (packets.isEmpty) {
          return _buildEmptyState(provider);
        }

        return ListView.builder(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          // Always add one for the "Load More" section
          itemCount: packets.length + 1,
          itemBuilder: (context, index) {
            if (index == packets.length) {
              return _buildLoadMoreButton(provider);
            }
            return PacketListItem(packet: packets[index]);
          },
        );
      },
    );
  }

  Widget _buildEmptyState(IdsProvider provider) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.radar, size: 40, color: Colors.white10),
          const SizedBox(height: 16),
          const Text('NO SEQUENCES MATCHING FILTER',
              style: TextStyle(
                  color: Colors.white24, letterSpacing: 2, fontSize: 10)),
          const SizedBox(height: 24),
          // FIX: Add the button here so users can load data even if the current view is empty
          _buildLoadMoreButton(provider),
        ],
      ),
    );
  }

  Widget _buildLoadMoreButton(IdsProvider provider) {
    return Selector<IdsProvider, bool>(
      selector: (_, p) => p.isLoadingMore,
      builder: (context, isLoading, _) {
        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: isLoading
              ? const Center(child: CircularProgressIndicator(strokeWidth: 2))
              : TextButton.icon(
                  onPressed: () => provider.loadMorePackets(),
                  icon: const Icon(Icons.download_rounded,
                      size: 14, color: Colors.white30),
                  label: const Text("FETCH OLDER SEQUENCES",
                      style: TextStyle(fontSize: 10, color: Colors.white30)),
                ),
        );
      },
    );
  }
}

class PacketListItem extends StatelessWidget {
  final Packet packet;
  const PacketListItem({required this.packet});

  @override
  Widget build(BuildContext context) {
    // Using listen: false because we already know if it's selected from the IdsProvider state
    final provider = Provider.of<IdsProvider>(context, listen: false);

    // We use a Selector here to handle the selection state changes efficiently
    return Selector<IdsProvider, bool>(
      selector: (_, p) => p.isSelected(packet.id),
      builder: (context, isSelected, child) {
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
              context: context,
              builder: (_) => PacketDetailDialog(packet: packet)),
          child: Container(
            margin: const EdgeInsets.only(bottom: 6),
            decoration: BoxDecoration(
              color: isSelected
                  ? const Color(0xFF232931)
                  : const Color(0xFF15191C),
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
      },
    );
  }
}
