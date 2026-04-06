// lib/screens/settings_screen.dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';

class SettingsScreen extends StatefulWidget {
  @override
  _SettingsScreenState createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final TextEditingController _ipController = TextEditingController();

  @override
  void initState() {
    super.initState();
    // Load IPs when screen initializes
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<IdsProvider>(context, listen: false).loadDmzIps();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D1117),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon:
              const Icon(Icons.arrow_back, color: Color(0xFF00E5FF), size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text("SYSTEM CONFIGURATION",
            style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.w900,
                letterSpacing: 1.0)),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("NETWORK TOPOLOGY SETTINGS",
                style: TextStyle(
                    color: Color(0xFF00E5FF),
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1.5)),
            const SizedBox(height: 20),
            _buildDmzPanel(),
          ],
        ),
      ),
    );
  }

  Widget _buildDmzPanel() {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF15191C),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Colors.white.withOpacity(0.06)),
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Icon(Icons.security, color: Colors.orangeAccent, size: 16),
              const SizedBox(width: 8),
              const Text("DMZ IDENTIFICATION NODES",
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 6),
          const Text(
              "IPs or subnet prefixes (e.g., '192.168.1.10' or '172.16.') that will be visually tagged as DMZ in the live graph.",
              style: TextStyle(color: Colors.white38, fontSize: 10)),
          const SizedBox(height: 20),
          _buildAddIpRow(),
          const SizedBox(height: 16),
          const Divider(color: Colors.white10, height: 1),
          const SizedBox(height: 16),
          _buildIpList(),
        ],
      ),
    );
  }

  Widget _buildAddIpRow() {
    return Row(
      children: [
        Expanded(
          child: Container(
            height: 36,
            decoration: BoxDecoration(
              color: const Color(0xFF0D1117),
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: Colors.white12),
            ),
            child: TextField(
              controller: _ipController,
              style: const TextStyle(
                  color: Colors.white, fontSize: 12, fontFamily: 'monospace'),
              decoration: const InputDecoration(
                hintText: "Enter IP or Prefix (e.g., 10.0.0.5)",
                hintStyle: TextStyle(color: Colors.white24, fontSize: 11),
                border: InputBorder.none,
                contentPadding:
                    EdgeInsets.symmetric(horizontal: 12, vertical: 12),
                isCollapsed: true,
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        ElevatedButton(
          onPressed: () {
            if (_ipController.text.isNotEmpty) {
              Provider.of<IdsProvider>(context, listen: false)
                  .addDmzIp(_ipController.text.trim());
              _ipController.clear();
            }
          },
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF00E5FF).withOpacity(0.15),
            foregroundColor: const Color(0xFF00E5FF),
            elevation: 0,
            side: BorderSide(color: const Color(0xFF00E5FF).withOpacity(0.5)),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          child: const Text("REGISTER IP",
              style: TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.0)),
        )
      ],
    );
  }

  Widget _buildIpList() {
    return Consumer<IdsProvider>(
      builder: (context, provider, _) {
        if (provider.dmzIps.isEmpty) {
          return const Padding(
            padding: EdgeInsets.symmetric(vertical: 16),
            child: Text("NO DMZ RULES CONFIGURED.",
                style: TextStyle(color: Colors.white24, fontSize: 10)),
          );
        }

        return ListView.builder(
          shrinkWrap: true,
          itemCount: provider.dmzIps.length,
          itemBuilder: (context, index) {
            final ip = provider.dmzIps[index];
            return Container(
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.02),
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: Colors.white.withOpacity(0.05)),
              ),
              child: Row(
                children: [
                  const Icon(Icons.lan, color: Colors.white24, size: 14),
                  const SizedBox(width: 12),
                  Text(ip,
                      style: const TextStyle(
                          color: Colors.white70,
                          fontSize: 12,
                          fontFamily: 'monospace')),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.delete_outline,
                        color: Colors.redAccent, size: 16),
                    constraints: const BoxConstraints(),
                    padding: EdgeInsets.zero,
                    onPressed: () => provider.removeDmzIp(ip),
                    tooltip: "Remove Protocol",
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}
