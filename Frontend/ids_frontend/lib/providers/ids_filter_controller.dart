// providers/ids_filter_controller.dart
//
// All UI-driven mutations: filter/search changes, packet selection,
// GAN/jitter queue management, batch label, and consistency flags.
// Reads and writes through IdsState — never touches HTTP.

import '../models/packet.dart';
import 'ids_state.dart';

class IdsFilterController {
  IdsFilterController(this._state);

  final IdsState _state;

  // ---------------------------------------------------------------------------
  // Filter & search
  // ---------------------------------------------------------------------------

  void setFilter(String statusFilter) {
    _state.setActiveFilters({
      ..._state.activeFilters,
      'status': statusFilter,
    });
  }

  void updateSearchQuery(String query) {
    _state.setActiveFilters({
      ..._state.activeFilters,
      'search': query,
    });
  }

  void clearAllFilters() {
    _state.setActiveFilters({'status': 'all', 'search': ''});
  }

  // ---------------------------------------------------------------------------
  // Packet selection
  // ---------------------------------------------------------------------------

  bool isSelected(int packetId) => _state.selectedPacketIds.contains(packetId);

  /// Toggles [packet] in/out of the appropriate queue.
  /// [queueType] must be `'gan'` or `'jitter'`.
  void toggleSelection(Packet packet, String queueType) {
    if (_state.selectedPacketIds.contains(packet.id)) {
      _state.selectedPacketIds.remove(packet.id);
      _state.ganQueue.removeWhere((p) => p.id == packet.id);
      _state.jitterQueue.removeWhere((p) => p.id == packet.id);
    } else {
      _state.selectedPacketIds.add(packet.id);
      if (queueType == 'gan') {
        _state.ganQueue.add(packet);
      } else if (queueType == 'jitter') {
        _state.jitterQueue.add(packet);
      }
    }
    // Reset consistency whenever the selection changes
    _state.setConsistencyStatus(false, false);
    _state.markDirty();
  }

  // ---------------------------------------------------------------------------
  // Batch label & consistency
  // ---------------------------------------------------------------------------

  void setBatchLabel(String? label, bool isNew) {
    _state.setBatchLabel(label, isNew);
  }

  void setConsistencyStatus(bool checked, bool passed) {
    _state.setConsistencyStatus(checked, passed);
  }

  // ---------------------------------------------------------------------------
  // Queue management
  // ---------------------------------------------------------------------------

  void clearQueues() {
    _state.selectedPacketIds.clear();
    _state.ganQueue.clear();
    _state.jitterQueue.clear();
    _state.setBatchLabel(null, false);
    _state.setConsistencyStatus(false, false);
    _state.markDirty();
  }
}
