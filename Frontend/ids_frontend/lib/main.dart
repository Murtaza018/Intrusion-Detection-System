import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'screens/dashboard_screen.dart';
import 'providers/ids_provider.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => IdsProvider(),
      child: MaterialApp(
        title: 'NEURAL-IDS CONTROL',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          useMaterial3: true,
          brightness: Brightness.dark,
          scaffoldBackgroundColor: const Color(0xFF0A0E12),
          colorScheme: ColorScheme.dark(
            primary: const Color(0xFF00E5FF), // Cyber Cyan
            secondary: const Color(0xFF7C4DFF), // Deep Purple
            surface: const Color(0xFF15191C),
            error: const Color(0xFFFF5252),
          ),
          cardTheme: CardTheme(
            color: const Color(0xFF1A1F24),
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
              side: BorderSide(color: Colors.white.withOpacity(0.05)),
            ),
          ),
          appBarTheme: const AppBarTheme(
            backgroundColor: Color(0xFF0A0E12),
            elevation: 0,
            centerTitle: false,
            titleTextStyle: TextStyle(
              fontFamily: 'monospace',
              fontSize: 20,
              fontWeight: FontWeight.bold,
              letterSpacing: 2,
              color: Color(0xFF00E5FF),
            ),
          ),
        ),
        home: DashboardScreen(),
      ),
    );
  }
}
