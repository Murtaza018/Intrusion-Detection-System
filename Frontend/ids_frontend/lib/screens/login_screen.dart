import 'package:flutter/material.dart';
import '../main.dart'; // To access BottomNavRoot
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({Key? key}) : super(key: key);

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _userController = TextEditingController();
  final _passController = TextEditingController();

  bool _isBooting = false;
  final List<String> _bootLogs = [];

  void _login() async {
    final user = _userController.text.trim();
    final pass = _passController.text.trim();

    // Show initial loading state
    setState(() {
      _isBooting = true;
      _bootLogs.clear();
      _bootLogs.add("> CONTACTING NEURAL CORE API...");
    });

    final provider = Provider.of<IdsProvider>(context, listen: false);

    // 1. SERVER-SIDE AUTHENTICATION
    final isAuthenticated = await provider.apiClient.authenticate(user, pass);

    if (!isAuthenticated) {
      setState(() {
        _isBooting = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("ACCESS DENIED: Invalid Credentials",
              style:
                  TextStyle(fontFamily: 'monospace', color: Colors.redAccent)),
          backgroundColor: Color(0xFF15191C),
        ),
      );
      return;
    }

    // 2. FAKE ENTERPRISE BOOT SEQUENCE (Only runs if backend says yes)
    setState(() => _bootLogs.add("> CREDENTIALS VERIFIED. TOKEN ISSUED [OK]"));
    await Future.delayed(const Duration(milliseconds: 600));

    setState(() => _bootLogs.add("> ALLOCATING RAM FOR GNN & MAE MODELS [OK]"));
    await Future.delayed(const Duration(milliseconds: 800));

    setState(() => _bootLogs.add("> BINDING TO NETWORK INTERFACE [OK]"));
    await Future.delayed(const Duration(milliseconds: 600));

    setState(
        () => _bootLogs.add("> SYSTEM ARMED. TRANSFERRING TO DASHBOARD..."));
    await Future.delayed(const Duration(milliseconds: 500));

    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const BottomNavRoot()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E12),
      body: Center(
        child: _isBooting ? _buildTerminalBoot() : _buildLoginForm(),
      ),
    );
  }

  Widget _buildLoginForm() {
    return Container(
      width: 380,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: const Color(0xFF15191C),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF00E5FF).withOpacity(0.3)),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF00E5FF).withOpacity(0.05),
            blurRadius: 20,
            spreadRadius: 2,
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const Icon(Icons.shield_outlined, size: 48, color: Color(0xFF00E5FF)),
          const SizedBox(height: 16),
          const Text(
            "NEURAL-IDS",
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w900,
              letterSpacing: 2,
              color: Colors.white,
            ),
          ),
          const Text(
            "AUTHORIZED PERSONNEL ONLY",
            style: TextStyle(
              fontSize: 10,
              fontFamily: 'monospace',
              color: Colors.white54,
              letterSpacing: 1.5,
            ),
          ),
          const SizedBox(height: 40),
          TextField(
            controller: _userController,
            style:
                const TextStyle(color: Colors.white, fontFamily: 'monospace'),
            decoration: InputDecoration(
              labelText: "USERNAME",
              labelStyle: const TextStyle(color: Colors.white54, fontSize: 12),
              enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.white12)),
              focusedBorder: const OutlineInputBorder(
                  borderSide: BorderSide(color: Color(0xFF00E5FF))),
              prefixIcon: const Icon(Icons.person_outline,
                  color: Colors.white54, size: 20),
            ),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _passController,
            obscureText: true,
            style:
                const TextStyle(color: Colors.white, fontFamily: 'monospace'),
            decoration: InputDecoration(
              labelText: "PASSWORD",
              labelStyle: const TextStyle(color: Colors.white54, fontSize: 12),
              enabledBorder: OutlineInputBorder(
                  borderSide: BorderSide(color: Colors.white12)),
              focusedBorder: const OutlineInputBorder(
                  borderSide: BorderSide(color: Color(0xFF00E5FF))),
              prefixIcon: const Icon(Icons.lock_outline,
                  color: Colors.white54, size: 20),
            ),
            onSubmitted: (_) => _login(),
          ),
          const SizedBox(height: 32),
          SizedBox(
            width: double.infinity,
            height: 48,
            child: ElevatedButton(
              onPressed: _login,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF00E5FF),
                foregroundColor: Colors.black,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(4)),
              ),
              child: const Text(
                "INITIALIZE CORE",
                style:
                    TextStyle(fontWeight: FontWeight.bold, letterSpacing: 1.5),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTerminalBoot() {
    return Container(
      width: 500,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.black,
        border: Border.all(color: Colors.white12),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.terminal, color: Colors.white54, size: 16),
              SizedBox(width: 8),
              Text("ids_boot_sequence.sh",
                  style: TextStyle(
                      color: Colors.white54,
                      fontFamily: 'monospace',
                      fontSize: 12)),
            ],
          ),
          const Divider(color: Colors.white12, height: 24),
          ..._bootLogs.map((log) => Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: Text(
                  log,
                  style: const TextStyle(
                    color: Color(0xFF00E5FF),
                    fontFamily: 'monospace',
                    fontSize: 13,
                  ),
                ),
              )),
        ],
      ),
    );
  }
}
