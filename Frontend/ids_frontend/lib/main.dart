import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'firebase_options.dart';
import 'package:ids_monitor/screens/report_summary_screen.dart';
import 'package:ids_monitor/screens/settings_screen.dart';
import 'package:provider/provider.dart';
import './screens/dashboard_screen.dart';
import './screens/graph_screen.dart';
import './screens/adaptation_screen.dart';
import './providers/ids_provider.dart';
import './screens/alert_history_screen.dart';

@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  // Ensure Firebase is initialized for background processing
  await Firebase.initializeApp();
  print("Handling a background message: ${message.messageId}");
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Firebase services
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  // Register the background message handler
  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
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
              side: BorderSide(
                color: Colors.white.withOpacity(0.05),
              ),
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
        home: const _BottomNavRoot(),
      ),
    );
  }
}

class _BottomNavRoot extends StatefulWidget {
  const _BottomNavRoot();

  @override
  State<_BottomNavRoot> createState() => _BottomNavRootState();
}

class _BottomNavRootState extends State<_BottomNavRoot> {
  int _currentIndex = 0;

  final List<Widget> _screens = <Widget>[
    DashboardScreen(),
    GraphScreen(),
    AdaptationScreen(),
    AlertHistoryScreen(),
    ReportSummaryScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        backgroundColor: const Color(0xFF0D1117),
        selectedItemColor: const Color(0xFF00E5FF),
        unselectedItemColor: Colors.white54,
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.bug_report),
            label: "Live",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.graphic_eq),
            label: "Graph",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.model_training),
            label: "Adapt",
          ),
          BottomNavigationBarItem(
            // <-- new item
            icon: Icon(Icons.analytics),
            label: "History",
          ),
          BottomNavigationBarItem(
            // <-- new item
            icon: Icon(Icons.report),
            label: "Summary",
          ),
          BottomNavigationBarItem(
            // <-- new item
            icon: Icon(Icons.settings),
            label: "Settings",
          ),
        ],
      ),
    );
  }
}
