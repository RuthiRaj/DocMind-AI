"""
Formatter Helper Utilities.

Provides formatting utility functions for file sizes, timestamps, and metadata.
"""


def format_file_size(size_in_bytes: int) -> str:
    """
    Converts a file size in bytes into a human-readable string representation.

    Args:
        size_in_bytes (int): Total size in bytes.

    Returns:
        str: Formatted file size string (e.g., "5.93 MB", "512.00 KB", "1024 B").
    """
    if size_in_bytes < 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_in_bytes)
    unit_index = 0

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"
