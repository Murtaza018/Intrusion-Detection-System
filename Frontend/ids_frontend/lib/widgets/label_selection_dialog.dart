import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/ids_provider.dart';

class LabelSelectionDialog extends StatefulWidget {
  final Packet packet;
  final IdsProvider provider;
  final Function(String) onConfirmed;

  const LabelSelectionDialog(
      {required this.packet,
      required this.provider,
      required this.onConfirmed});

  @override
  _LabelSelectionDialogState createState() => _LabelSelectionDialogState();
}

class _LabelSelectionDialogState extends State<LabelSelectionDialog> {
  String? _selectedLabel;
  bool _isNew = false;
  final TextEditingController _newLabelController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    // Filter out "BENIGN" from attack options usually, but let's keep all
    final labels = widget.provider.existingLabels;

    return AlertDialog(
      title: Text("Label This Attack"),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("What type of attack is Packet #${widget.packet.id}?",
              style: TextStyle(color: Colors.grey[700])),
          SizedBox(height: 16),

          // Toggle: Existing vs New
          Row(
            children: [
              ChoiceChip(
                label: Text("Existing Attack"),
                selected: !_isNew,
                onSelected: (val) => setState(() {
                  _isNew = false;
                  _selectedLabel = null;
                }),
              ),
              SizedBox(width: 12),
              ChoiceChip(
                label: Text("New / Zero-Day"),
                selected: _isNew,
                onSelected: (val) => setState(() {
                  _isNew = true;
                  _selectedLabel = null;
                }),
              ),
            ],
          ),
          SizedBox(height: 16),

          // Input Field
          if (!_isNew)
            DropdownButtonFormField<String>(
              decoration: InputDecoration(
                labelText: "Select Attack Type",
                border: OutlineInputBorder(),
                contentPadding:
                    EdgeInsets.symmetric(horizontal: 12, vertical: 12),
              ),
              items: labels
                  .map((l) => DropdownMenuItem(value: l, child: Text(l)))
                  .toList(),
              onChanged: (val) => setState(() => _selectedLabel = val),
            )
          else
            TextField(
              controller: _newLabelController,
              decoration: InputDecoration(
                labelText: "Name New Attack",
                hintText: "e.g., Exploit_CVE_2025",
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.new_releases),
              ),
              onChanged: (val) => setState(() => _selectedLabel = val),
            ),
        ],
      ),
      actions: [
        TextButton(
            onPressed: () => Navigator.pop(context), child: Text("Cancel")),
        FilledButton(
          onPressed: (_selectedLabel == null || _selectedLabel!.isEmpty)
              ? null
              : () => widget.onConfirmed(_selectedLabel!),
          child: Text("Confirm Label"),
          style: FilledButton.styleFrom(backgroundColor: Colors.deepPurple),
        ),
      ],
    );
  }
}
