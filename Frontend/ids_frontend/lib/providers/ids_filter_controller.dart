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

// Inside IdsFilterController class

  void setSeverityFilter(String? severity) {
    _state.currentSeverityFilter = severity;
    // Note: _buildFilteredList is called automatically by state.filteredPackets getter
  }

  void updateSearchQuery(String query) {
    _state.setSearchQuery(query);
  }

  void _applyFilters() {
    // Since IdsState uses a lazy getter (filteredPackets),
    // we just need to tell it that the data is 'dirty'.
    _state.markDirty();
  }

  bool _matchesSearch(Packet packet, String query) {
    for (final token in query.split(' ')) {
      if (token.isEmpty) continue;

      if (token.startsWith('ip:')) {
        final val = token.substring(3);
        if (!packet.srcIp.contains(val) && !packet.dstIp.contains(val))
          return false;
      } else if (token.startsWith('port:')) {
        final p = token.substring(5);
        if (packet.srcPort.toString() != p && packet.dstPort.toString() != p)
          return false;
      } else if (token.startsWith('proto:')) {
        if (packet.protocol.toLowerCase() != token.substring(6)) return false;
      } else if (token.startsWith('anomaly:>')) {
        final threshold = double.tryParse(token.substring(9)) ?? 0;
        if (packet.gnnAnomaly <= threshold) return false;
      } else if (token.startsWith('anomaly:<')) {
        final threshold = double.tryParse(token.substring(9)) ?? 1;
        if (packet.gnnAnomaly >= threshold) return false;
      } else {
        // General text match across available fields
        final lowerToken = token.toLowerCase();
        if (!packet.summary.toLowerCase().contains(lowerToken) &&
            !packet.srcIp.contains(lowerToken) &&
            !packet.dstIp.contains(lowerToken) &&
            !packet.protocol.toLowerCase().contains(lowerToken)) return false;
      }
    }
    return true;
  }
  // ---------------------------------------------------------------------------
  // Filter & search
  // ---------------------------------------------------------------------------

  void setFilter(String statusFilter) {
    _state.setActiveFilters({
      ..._state.activeFilters,
      'status': statusFilter,
    });
  }

  void clearAllFilters() {
    _state.currentSeverityFilter = null;
    _state.setSearchQuery('');
    _state.setActiveFilters({
      'status': 'all',
      'search': '',
    });
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
