import 'package:flutter/material.dart';
import '../providers/ids_provider.dart';

class PacketDetailDialog extends StatelessWidget {
  final Packet packet;

  const PacketDetailDialog({required this.packet});

  @override
  Widget build(BuildContext context) {
    return Dialog(
      child: Container(
        padding: EdgeInsets.all(16),
        width: MediaQuery.of(context).size.width * 0.9,
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  Icon(Icons.info, color: Colors.blue),
                  SizedBox(width: 8),
                  Text(
                    'Packet Details',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  Spacer(),
                  IconButton(
                    icon: Icon(Icons.close),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
              SizedBox(height: 16),
              _buildDetailRow('Packet ID', '#${packet.id}'),
              _buildDetailRow('Timestamp', packet.timestamp.toString()),
              _buildDetailRow('Status', _getStatusText(packet.status)),
              Divider(height: 24),
              Text('Network Information',
                  style: TextStyle(fontWeight: FontWeight.bold)),
              SizedBox(height: 8),
              _buildDetailRow('Source IP', packet.srcIp),
              _buildDetailRow('Destination IP', packet.dstIp),
              _buildDetailRow('Protocol', packet.protocol),
              _buildDetailRow('Source Port', packet.srcPort.toString()),
              _buildDetailRow('Destination Port', packet.dstPort.toString()),
              _buildDetailRow('Packet Length', '${packet.length} bytes'),
              if (packet.confidence > 0) ...[
                Divider(height: 24),
                Text('Detection Information',
                    style: TextStyle(fontWeight: FontWeight.bold)),
                SizedBox(height: 8),
                _buildDetailRow('Confidence',
                    '${(packet.confidence * 100).toStringAsFixed(1)}%'),
              ],
              if (packet.explanation != null) ...[
                Divider(height: 24),
                Text('Explanation',
                    style: TextStyle(fontWeight: FontWeight.bold)),
                SizedBox(height: 8),
                Container(
                  padding: EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.grey[50],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    _formatExplanation(packet.explanation!),
                    style: TextStyle(fontSize: 12),
                  ),
                ),
              ],
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: () => Navigator.pop(context),
                child: Text('Close'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text('$label:', style: TextStyle(fontWeight: FontWeight.w500)),
          SizedBox(width: 8),
          Text(value),
        ],
      ),
    );
  }

  String _getStatusText(String status) {
    switch (status) {
      case 'known_attack':
        return 'KNOWN ATTACK 🚨';
      case 'zero_day':
        return 'ZERO-DAY ANOMALY ⚠️';
      default:
        return 'NORMAL ✅';
    }
  }

  String _formatExplanation(Map<String, dynamic> explanation) {
    if (explanation['type'] == 'Zero-Day Anomaly') {
      return 'Zero-Day Detection\nReconstruction Error: ${explanation['error']}\nThreshold: ${explanation['threshold']}';
    }
    return explanation.toString();
  }
}
