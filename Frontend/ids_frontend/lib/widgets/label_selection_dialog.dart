import 'package:flutter/material.dart';
import '../providers/ids_provider.dart';
import '../models/packet.dart';

class LabelSelectionDialog extends StatefulWidget {
  final Packet packet;
  final IdsProvider provider;
  final Function(String) onConfirmed;

  const LabelSelectionDialog({
    super.key,
    required this.packet,
    required this.provider,
    required this.onConfirmed,
  });

  @override
  _LabelSelectionDialogState createState() => _LabelSelectionDialogState();
}

class _LabelSelectionDialogState extends State<LabelSelectionDialog> {
  String? _selectedLabel;
  bool _isNew = false;
  final TextEditingController _newLabelController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    final labels = widget.provider.existingLabels;

    return AlertDialog(
      title: const Text("Label This Attack"),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("What type of attack is Packet #${widget.packet.id}?",
              style: TextStyle(color: Colors.grey[700])),
          const SizedBox(height: 16),

          // CHANGED: Row to Wrap to prevent chip overflow on mobile
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ChoiceChip(
                label: const Text("Existing Attack"),
                selected: !_isNew,
                onSelected: (val) => setState(() {
                  _isNew = false;
                  _selectedLabel = null;
                }),
              ),
              ChoiceChip(
                label: const Text("New / Zero-Day"),
                selected: _isNew,
                onSelected: (val) => setState(() {
                  _isNew = true;
                  _selectedLabel = null;
                }),
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Input Field
          if (!_isNew)
            DropdownButtonFormField<String>(
              isExpanded: true, // Prevents text overflow inside dropdown
              decoration: const InputDecoration(
                labelText: "Select Attack Type",
                border: OutlineInputBorder(),
                contentPadding:
                    EdgeInsets.symmetric(horizontal: 12, vertical: 12),
              ),
              items: labels
                  .map((l) => DropdownMenuItem(
                        value: l,
                        child: Text(l, overflow: TextOverflow.ellipsis),
                      ))
                  .toList(),
              onChanged: (val) => setState(() => _selectedLabel = val),
            )
          else
            TextField(
              controller: _newLabelController,
              decoration: const InputDecoration(
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
            onPressed: () => Navigator.pop(context),
            child: const Text("Cancel")),
        FilledButton(
          onPressed: (_selectedLabel == null || _selectedLabel!.isEmpty)
              ? null
              : () => widget.onConfirmed(_selectedLabel!),
          style: FilledButton.styleFrom(backgroundColor: Colors.deepPurple),
          child: const Text("Confirm Label"),
        ),
      ],
    );
  }
}
