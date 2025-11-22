"""
Text to Markdown Converter

Converts PDF text blocks to properly formatted Markdown with:
- Special character escaping
- Heading formatting
- List formatting
- Table formatting
- Inline formatting (bold, italic)
- Safety sanitization
"""

import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class TextStyle:
    """Text styling information."""
    is_bold: bool = False
    is_italic: bool = False
    is_heading: bool = False
    heading_level: Optional[int] = None
    is_list_item: bool = False
    list_type: Optional[str] = None  # "bullet" or "numbered"
    list_marker: Optional[str] = None


class TextToMarkdownConverter:
    """
    Converts text blocks to Markdown format with proper escaping and formatting.

    Features:
    - Escapes special Markdown characters
    - Formats headings (# ## ### etc.)
    - Formats lists (bullets and numbered)
    - Formats tables
    - Preserves inline formatting (bold, italic)
    - Sanitizes dangerous content
    """

    # Markdown special characters that need escaping in regular text
    # (not in code blocks or already-formatted sections)
    MARKDOWN_SPECIAL_CHARS = {
        '\\': '\\\\',  # Backslash first!
        '`': '\\`',
        '*': '\\*',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '[': '\\[',
        ']': '\\]',
        '(': '\\(',
        ')': '\\)',
        '#': '\\#',
        '+': '\\+',
        '-': '\\-',
        '.': '\\.',
        '!': '\\!',
        '|': '\\|',
    }

    # Characters that need escaping in different contexts
    HEADING_ESCAPE_CHARS = {'#', '*', '_', '`', '[', ']'}
    LIST_ESCAPE_CHARS = {'*', '-', '+', '.', '`', '[', ']'}

    def __init__(self, escape_mode: str = "smart"):
        """
        Initialize converter.

        Args:
            escape_mode: How to escape special characters:
                - "smart": Escape based on context (headings, lists, etc.)
                - "full": Escape all special characters
                - "minimal": Only escape absolutely necessary characters
        """
        self.escape_mode = escape_mode

    def convert(
        self,
        text: str,
        style: Optional[TextStyle] = None,
        preserve_whitespace: bool = False
    ) -> str:
        """
        Convert text to Markdown format.

        Args:
            text: Raw text to convert
            style: Optional styling information
            preserve_whitespace: Whether to preserve exact whitespace

        Returns:
            Markdown-formatted text
        """
        if not text or not text.strip():
            return ""

        # Sanitize input
        text = self._sanitize(text)

        # Apply style if provided
        if style:
            return self._format_styled_text(text, style, preserve_whitespace)

        # Default: escape and return
        return self._escape_text(text, preserve_whitespace)

    def convert_heading(
        self,
        text: str,
        level: int = 1,
        preserve_formatting: bool = True
    ) -> str:
        """
        Convert text to Markdown heading.

        Args:
            text: Heading text
            level: Heading level (1-6)
            preserve_formatting: Whether to preserve bold/italic

        Returns:
            Markdown heading (e.g., "# Heading")
        """
        level = max(1, min(6, level))  # Clamp to 1-6

        # Clean and escape heading text
        text = text.strip()
        if not preserve_formatting:
            text = self._strip_formatting(text)

        # Escape special characters (but allow some formatting)
        text = self._escape_heading_text(text)

        # Build heading
        prefix = '#' * level
        return f"{prefix} {text}"

    def convert_list_item(
        self,
        text: str,
        list_type: str = "bullet",
        marker: Optional[str] = None,
        indent_level: int = 0
    ) -> str:
        """
        Convert text to Markdown list item.

        Args:
            text: List item text
            list_type: "bullet" or "numbered"
            marker: Custom marker (e.g., "1.", "-", "*")
            indent_level: Indentation level (0 = top level)

        Returns:
            Markdown list item
        """
        text = text.strip()
        text = self._escape_list_text(text)

        # Determine marker
        if marker:
            list_marker = marker
        elif list_type == "numbered":
            list_marker = "1."
        else:
            list_marker = "-"

        # Build indentation
        indent = "  " * indent_level

        return f"{indent}{list_marker} {text}"

    def convert_table(
        self,
        rows: List[List[str]],
        has_header: bool = True
    ) -> str:
        """
        Convert table data to Markdown table format.

        Args:
            rows: List of rows, each row is list of cell texts
            has_header: Whether first row is header

        Returns:
            Markdown table
        """
        if not rows:
            return ""

        # Escape cell contents
        escaped_rows = []
        for row in rows:
            escaped_row = [self._escape_table_cell(cell) for cell in row]
            escaped_rows.append(escaped_row)

        # Build table
        lines = []

        # Header row
        if has_header and escaped_rows:
            header = escaped_rows[0]
            lines.append("| " + " | ".join(header) + " |")

            # Separator
            separator = ["---"] * len(header)
            lines.append("| " + " | ".join(separator) + " |")

            # Data rows
            for row in escaped_rows[1:]:
                lines.append("| " + " | ".join(row) + " |")
        else:
            # All data rows
            for row in escaped_rows:
                lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def convert_code_block(
        self,
        code: str,
        language: Optional[str] = None
    ) -> str:
        """
        Convert code to Markdown code block.

        Args:
            code: Code content
            language: Programming language (for syntax highlighting)

        Returns:
            Markdown code block
        """
        # Code blocks don't need escaping
        lang = language or ""
        return f"```{lang}\n{code}\n```"

    def convert_inline_code(self, code: str) -> str:
        """
        Convert to inline code.

        Args:
            code: Code content

        Returns:
            Inline code (e.g., `code`)
        """
        # Escape backticks in code
        code = code.replace('`', '\\`')
        return f"`{code}`"

    def apply_bold(self, text: str) -> str:
        """Apply bold formatting."""
        return f"**{text}**"

    def apply_italic(self, text: str) -> str:
        """Apply italic formatting."""
        return f"*{text}*"

    def apply_bold_italic(self, text: str) -> str:
        """Apply bold and italic formatting."""
        return f"***{text}***"

    def _format_styled_text(
        self,
        text: str,
        style: TextStyle,
        preserve_whitespace: bool
    ) -> str:
        """Format text based on style information."""

        # Handle headings
        if style.is_heading and style.heading_level:
            return self.convert_heading(text, style.heading_level)

        # Handle list items
        if style.is_list_item:
            return self.convert_list_item(
                text,
                list_type=style.list_type or "bullet",
                marker=style.list_marker
            )

        # Escape text
        text = self._escape_text(text, preserve_whitespace)

        # Apply inline formatting
        if style.is_bold and style.is_italic:
            text = self.apply_bold_italic(text)
        elif style.is_bold:
            text = self.apply_bold(text)
        elif style.is_italic:
            text = self.apply_italic(text)

        return text

    def _escape_text(self, text: str, preserve_whitespace: bool = False) -> str:
        """
        Escape Markdown special characters in text.

        Args:
            text: Text to escape
            preserve_whitespace: Whether to preserve exact whitespace

        Returns:
            Escaped text
        """
        if self.escape_mode == "minimal":
            # Only escape absolutely necessary characters
            text = text.replace('\\', '\\\\')
            text = text.replace('`', '\\`')
            return text

        elif self.escape_mode == "full":
            # Escape all special characters
            for char, escaped in self.MARKDOWN_SPECIAL_CHARS.items():
                text = text.replace(char, escaped)
            return text

        else:  # "smart" mode
            # Context-aware escaping
            # Don't escape characters that are likely intentional formatting

            # Escape backslashes first
            text = text.replace('\\', '\\\\')

            # Escape backticks (code)
            text = text.replace('`', '\\`')

            # Escape brackets (links)
            text = text.replace('[', '\\[')
            text = text.replace(']', '\\]')

            # Conditionally escape asterisks and underscores
            # (only if they look like accidental formatting)
            text = self._smart_escape_formatting_chars(text)

            # Escape hash at line start (headings)
            text = re.sub(r'^(#+)', r'\\\1', text, flags=re.MULTILINE)

            return text

    def _smart_escape_formatting_chars(self, text: str) -> str:
        """
        Smart escaping of * and _ that might be formatting.

        Only escape if they don't look like intentional bold/italic.
        """
        # Pattern: Escape single * or _ not part of formatting
        # This is a heuristic - may need refinement

        # Escape * and _ that are:
        # - At word boundaries without matching pair
        # - In the middle of words (likely not formatting)

        # For now, escape all to be safe
        # TODO: Implement smarter detection
        text = text.replace('*', '\\*')
        text = text.replace('_', '\\_')

        return text

    def _escape_heading_text(self, text: str) -> str:
        """Escape text for use in headings."""
        # Escape special chars that would break heading
        for char in self.HEADING_ESCAPE_CHARS:
            if char == '#':
                # Escape # in heading text
                text = text.replace('#', '\\#')
            elif char in '*_':
                # Keep bold/italic in headings
                continue
            else:
                text = text.replace(char, f'\\{char}')
        return text

    def _escape_list_text(self, text: str) -> str:
        """Escape text for use in list items."""
        # Escape chars that would break list formatting
        for char in self.LIST_ESCAPE_CHARS:
            if char in '*-+.':
                # Only escape at start of line
                text = re.sub(f'^\\{char}', f'\\\\{char}', text)
            else:
                text = text.replace(char, f'\\{char}')
        return text

    def _escape_table_cell(self, text: str) -> str:
        """Escape text for use in table cells."""
        # Escape pipes and newlines
        text = text.replace('|', '\\|')
        text = text.replace('\n', ' ')  # Tables can't have newlines
        text = text.replace('\r', '')
        return text.strip()

    def _strip_formatting(self, text: str) -> str:
        """Remove Markdown formatting from text."""
        # Remove bold
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)

        # Remove italic
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)

        # Remove code
        text = re.sub(r'`(.+?)`', r'\1', text)

        return text

    def _sanitize(self, text: str) -> str:
        """
        Sanitize text for safety.

        - Remove null bytes
        - Normalize unicode
        - Remove control characters (except newlines/tabs)
        - Limit line length
        """
        # Remove null bytes
        text = text.replace('\x00', '')

        # Remove other control characters (except \n, \r, \t)
        text = ''.join(
            char for char in text
            if char in '\n\r\t' or (ord(char) >= 32 and ord(char) != 127)
        )

        # Normalize unicode (NFC - canonical composition)
        try:
            import unicodedata
            text = unicodedata.normalize('NFC', text)
        except Exception:
            pass  # If normalization fails, continue

        # Limit extremely long lines (potential DoS)
        lines = text.split('\n')
        max_line_length = 10000
        sanitized_lines = []
        for line in lines:
            if len(line) > max_line_length:
                # Split long lines at word boundaries
                words = line.split()
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 > max_line_length:
                        sanitized_lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1

                if current_line:
                    sanitized_lines.append(' '.join(current_line))
            else:
                sanitized_lines.append(line)

        text = '\n'.join(sanitized_lines)

        return text


# Convenience functions
def text_to_markdown(
    text: str,
    style: Optional[TextStyle] = None,
    escape_mode: str = "smart"
) -> str:
    """
    Convert text to Markdown format.

    Args:
        text: Raw text
        style: Optional styling information
        escape_mode: Escaping mode ("smart", "full", "minimal")

    Returns:
        Markdown-formatted text
    """
    converter = TextToMarkdownConverter(escape_mode=escape_mode)
    return converter.convert(text, style=style)


def heading_to_markdown(text: str, level: int = 1) -> str:
    """
    Convert text to Markdown heading.

    Args:
        text: Heading text
        level: Heading level (1-6)

    Returns:
        Markdown heading
    """
    converter = TextToMarkdownConverter()
    return converter.convert_heading(text, level=level)


def list_to_markdown(
    items: List[str],
    list_type: str = "bullet",
    indent_level: int = 0
) -> str:
    """
    Convert list of items to Markdown list.

    Args:
        items: List item texts
        list_type: "bullet" or "numbered"
        indent_level: Indentation level

    Returns:
        Markdown list
    """
    converter = TextToMarkdownConverter()
    lines = []

    for i, item in enumerate(items):
        if list_type == "numbered":
            marker = f"{i+1}."
        else:
            marker = "-"

        line = converter.convert_list_item(
            item,
            list_type=list_type,
            marker=marker,
            indent_level=indent_level
        )
        lines.append(line)

    return "\n".join(lines)


def table_to_markdown(
    rows: List[List[str]],
    has_header: bool = True
) -> str:
    """
    Convert table to Markdown format.

    Args:
        rows: Table rows (list of lists)
        has_header: Whether first row is header

    Returns:
        Markdown table
    """
    converter = TextToMarkdownConverter()
    return converter.convert_table(rows, has_header=has_header)
