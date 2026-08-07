import json
import logging
from typing import List

from app.core.config import settings

logger = logging.getLogger(__name__)

def prune_debug_list(
    debug_list: List[dict],
    max_entries: int = settings.DEBUG_LOG_MAX_ENTRIES,
    max_size_mb: float = settings.DEBUG_LOG_MAX_SIZE_MB
) -> List[dict]:
    """
    Enforces maximum entry count and file size bounds on a document debug log list.
    Prunes the oldest entries first, preserving the newest entries at the end.
    
    Args:
        debug_list (List[dict]): The current list of debug entries.
        max_entries (int): Maximum entries to retain.
        max_size_mb (float): Maximum allowed file size in MB.
        
    Returns:
        List[dict]: The pruned list of debug entries.
    """
    if not isinstance(debug_list, list):
        return []

    # 1. Prune by maximum entries count limit
    if len(debug_list) > max_entries:
        old_count = len(debug_list)
        debug_list = debug_list[-max_entries:]
        logger.info(
            "Pruned debug log entries due to count limit (Max: %d). Removed %d oldest entries.",
            max_entries,
            old_count - len(debug_list)
        )

    # 2. Prune by maximum file size limit (in bytes)
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    
    # We serialize the list to estimate size and remove oldest entries (from index 0)
    # until the JSON string representation is under the limit or only 1 entry remains.
    initial_entry_count = len(debug_list)
    while len(debug_list) > 1:
        try:
            serialized = json.dumps(debug_list, indent=4)
            if len(serialized.encode("utf-8")) <= max_size_bytes:
                break
        except Exception as err:
            logger.error("Failed to estimate debug log size during pruning: %s", str(err))
            break
        # Remove oldest entry
        debug_list.pop(0)

    pruned_count = initial_entry_count - len(debug_list)
    if pruned_count > 0:
        logger.info(
            "Pruned %d oldest debug log entries due to size limit (Max: %.2f MB).",
            pruned_count,
            max_size_mb
        )

    # 3. If a single remaining entry itself exceeds the limit, log a warning but preserve it validly
    if len(debug_list) == 1:
        try:
            serialized = json.dumps(debug_list, indent=4)
            size_bytes = len(serialized.encode("utf-8"))
            if size_bytes > max_size_bytes:
                logger.warning(
                    "Single remaining debug log entry size (%d bytes) exceeds the maximum allowed file size limits (Max: %d bytes). Preserving entry in valid form.",
                    size_bytes,
                    max_size_bytes
                )
        except Exception:
            pass

    return debug_list
