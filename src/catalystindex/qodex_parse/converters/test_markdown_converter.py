"""
Tests for Text to Markdown Converter

Tests cover:
- Special character escaping
- Heading formatting
- List formatting
- Table formatting
- Inline formatting
- Safety sanitization
- Edge cases
"""

import pytest
from text_to_markdown import (
    TextToMarkdownConverter,
    TextStyle,
    text_to_markdown,
    heading_to_markdown,
    list_to_markdown,
    table_to_markdown
)


class TestBasicEscaping:
    """Test special character escaping."""

    def test_escape_backslash(self):
        converter = TextToMarkdownConverter(escape_mode="full")
        text = "C:\\Users\\Documents\\file.txt"
        result = converter.convert(text)
        assert "\\\\" in result

    def test_escape_backticks(self):
        converter = TextToMarkdownConverter(escape_mode="full")
        text = "Use `code` here"
        result = converter.convert(text)
        assert "\\`" in result

    def test_escape_asterisks(self):
        converter = TextToMarkdownConverter(escape_mode="full")
        text = "Math: 5 * 3 = 15"
        result = converter.convert(text)
        assert "\\*" in result

    def test_escape_underscores(self):
        converter = TextToMarkdownConverter(escape_mode="full")
        text = "variable_name_here"
        result = converter.convert(text)
        assert "\\_" in result

    def test_escape_brackets(self):
        converter = TextToMarkdownConverter(escape_mode="full")
        text = "Array[index]"
        result = converter.convert(text)
        assert "\\[" in result and "\\]" in result

    def test_escape_hash(self):
        converter = TextToMarkdownConverter(escape_mode="full")
        text = "#hashtag #trending"
        result = converter.convert(text)
        assert "\\#" in result

    def test_escape_pipes(self):
        converter = TextToMarkdownConverter(escape_mode="full")
        text = "value1 | value2 | value3"
        result = converter.convert(text)
        assert "\\|" in result


class TestHeadingConversion:
    """Test heading formatting."""

    def test_heading_level_1(self):
        result = heading_to_markdown("Introduction", level=1)
        assert result == "# Introduction"

    def test_heading_level_2(self):
        result = heading_to_markdown("Chapter 1", level=2)
        assert result == "## Chapter 1"

    def test_heading_level_6(self):
        result = heading_to_markdown("Subsection", level=6)
        assert result == "###### Subsection"

    def test_heading_level_clamping(self):
        # Level too high
        result = heading_to_markdown("Test", level=10)
        assert result.startswith("######")  # Max level 6

        # Level too low
        result = heading_to_markdown("Test", level=0)
        assert result.startswith("#")  # Min level 1

    def test_heading_with_special_chars(self):
        result = heading_to_markdown("What is C++?", level=2)
        assert "C++" in result
        # + should be escaped or preserved
        assert "##" in result

    def test_heading_with_hash(self):
        result = heading_to_markdown("Chapter #1", level=2)
        assert "\\#" in result  # Hash should be escaped

    def test_heading_strips_whitespace(self):
        result = heading_to_markdown("   Intro   ", level=1)
        assert result == "# Intro"


class TestListConversion:
    """Test list formatting."""

    def test_bullet_list(self):
        items = ["Item 1", "Item 2", "Item 3"]
        result = list_to_markdown(items, list_type="bullet")
        lines = result.split('\n')
        assert all(line.startswith("-") for line in lines)

    def test_numbered_list(self):
        items = ["First", "Second", "Third"]
        result = list_to_markdown(items, list_type="numbered")
        lines = result.split('\n')
        assert "1." in lines[0]
        assert "2." in lines[1]
        assert "3." in lines[2]

    def test_indented_list(self):
        converter = TextToMarkdownConverter()
        result = converter.convert_list_item("Nested item", indent_level=1)
        assert result.startswith("  ")  # 2 spaces per level

    def test_list_with_special_chars(self):
        items = ["Use * for emphasis", "Use - for bullets"]
        result = list_to_markdown(items)
        # Should still be valid list
        assert all(line.startswith("-") for line in result.split('\n'))


class TestTableConversion:
    """Test table formatting."""

    def test_simple_table(self):
        rows = [
            ["Header 1", "Header 2"],
            ["Cell 1", "Cell 2"],
            ["Cell 3", "Cell 4"]
        ]
        result = table_to_markdown(rows, has_header=True)
        lines = result.split('\n')

        # Should have header, separator, and data rows
        assert len(lines) == 4
        assert "---" in lines[1]  # Separator row
        assert all("|" in line for line in lines)

    def test_table_with_pipes(self):
        rows = [
            ["Name", "Value"],
            ["A | B", "C | D"]
        ]
        result = table_to_markdown(rows)
        # Pipes in cells should be escaped
        assert "\\|" in result

    def test_table_with_newlines(self):
        rows = [
            ["Header"],
            ["Line 1\nLine 2"]
        ]
        result = table_to_markdown(rows)
        # Newlines should be converted to spaces
        assert "Line 1 Line 2" in result

    def test_table_no_header(self):
        rows = [
            ["Cell 1", "Cell 2"],
            ["Cell 3", "Cell 4"]
        ]
        result = table_to_markdown(rows, has_header=False)
        lines = result.split('\n')

        # Should just be data rows (no separator)
        assert len(lines) == 2
        assert all("|" in line for line in lines)


class TestInlineFormatting:
    """Test inline formatting."""

    def test_bold(self):
        converter = TextToMarkdownConverter()
        result = converter.apply_bold("Important")
        assert result == "**Important**"

    def test_italic(self):
        converter = TextToMarkdownConverter()
        result = converter.apply_italic("Emphasis")
        assert result == "*Emphasis*"

    def test_bold_italic(self):
        converter = TextToMarkdownConverter()
        result = converter.apply_bold_italic("Very Important")
        assert result == "***Very Important***"

    def test_inline_code(self):
        converter = TextToMarkdownConverter()
        result = converter.convert_inline_code("variable_name")
        assert result == "`variable_name`"

    def test_inline_code_with_backticks(self):
        converter = TextToMarkdownConverter()
        result = converter.convert_inline_code("Use `backticks`")
        # Backticks should be escaped
        assert "\\`" in result


class TestCodeBlocks:
    """Test code block formatting."""

    def test_code_block_no_language(self):
        converter = TextToMarkdownConverter()
        code = "def hello():\n    print('world')"
        result = converter.convert_code_block(code)
        assert result.startswith("```\n")
        assert result.endswith("\n```")
        assert code in result

    def test_code_block_with_language(self):
        converter = TextToMarkdownConverter()
        code = "const x = 42;"
        result = converter.convert_code_block(code, language="javascript")
        assert result.startswith("```javascript\n")
        assert code in result


class TestStyledText:
    """Test text with style information."""

    def test_heading_style(self):
        style = TextStyle(is_heading=True, heading_level=2)
        result = text_to_markdown("Chapter 1", style=style)
        assert result.startswith("##")

    def test_list_item_style(self):
        style = TextStyle(
            is_list_item=True,
            list_type="bullet",
            list_marker="-"
        )
        result = text_to_markdown("Item text", style=style)
        assert result.startswith("-")

    def test_bold_style(self):
        style = TextStyle(is_bold=True)
        result = text_to_markdown("Bold text", style=style)
        assert result.startswith("**")
        assert result.endswith("**")

    def test_italic_style(self):
        style = TextStyle(is_italic=True)
        result = text_to_markdown("Italic text", style=style)
        assert result.startswith("*")
        assert result.endswith("*")

    def test_bold_italic_style(self):
        style = TextStyle(is_bold=True, is_italic=True)
        result = text_to_markdown("Bold italic", style=style)
        assert result.startswith("***")
        assert result.endswith("***")


class TestSanitization:
    """Test input sanitization."""

    def test_remove_null_bytes(self):
        converter = TextToMarkdownConverter()
        text = "Hello\x00World"
        result = converter.convert(text)
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_remove_control_chars(self):
        converter = TextToMarkdownConverter()
        # Include various control characters
        text = "Hello\x01\x02\x03World"
        result = converter.convert(text)
        # Control chars should be removed
        assert "\x01" not in result
        assert "HelloWorld" in result

    def test_preserve_newlines_tabs(self):
        converter = TextToMarkdownConverter()
        text = "Line 1\nLine 2\tTabbed"
        result = converter.convert(text)
        assert "\n" in result
        assert "\t" in result

    def test_long_line_handling(self):
        converter = TextToMarkdownConverter()
        # Create extremely long line
        long_text = "word " * 10000
        result = converter.convert(long_text)
        # Should not crash, lines should be limited
        lines = result.split('\n')
        for line in lines:
            assert len(line) <= 10100  # With some buffer


class TestEscapeModes:
    """Test different escape modes."""

    def test_minimal_escape_mode(self):
        converter = TextToMarkdownConverter(escape_mode="minimal")
        text = "C:\\path\\to\\file with *asterisks* and _underscores_"
        result = converter.convert(text)
        # Should only escape backslashes and backticks
        assert "\\\\" in result
        # Asterisks and underscores might not be escaped
        # (depending on minimal implementation)

    def test_full_escape_mode(self):
        converter = TextToMarkdownConverter(escape_mode="full")
        text = "Text with *all* _special_ #chars # [brackets]"
        result = converter.convert(text)
        # All special chars should be escaped
        assert "\\*" in result
        assert "\\_" in result
        assert "\\#" in result
        assert "\\[" in result
        assert "\\]" in result

    def test_smart_escape_mode(self):
        converter = TextToMarkdownConverter(escape_mode="smart")
        text = "Regular text with some *potential* formatting"
        result = converter.convert(text)
        # Should intelligently escape based on context


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_text(self):
        result = text_to_markdown("")
        assert result == ""

    def test_whitespace_only(self):
        result = text_to_markdown("   \n   \t   ")
        assert result == ""

    def test_unicode_text(self):
        text = "Hello 世界 🌍 Привет"
        result = text_to_markdown(text)
        # Unicode should be preserved
        assert "世界" in result
        assert "🌍" in result
        assert "Привет" in result

    def test_mixed_content(self):
        # Text with headings, lists, code, etc.
        converter = TextToMarkdownConverter()

        # This should not crash
        text = "# Heading\n- List item\n`code` **bold** *italic*"
        result = converter.convert(text)
        assert isinstance(result, str)

    def test_malformed_markdown(self):
        # Input that's already partially Markdown
        text = "**Unclosed bold and _mixed_ italics*"
        result = text_to_markdown(text)
        # Should escape to prevent interpretation
        assert "\\" in result


class TestRealWorldExamples:
    """Test real-world PDF extraction scenarios."""

    def test_file_path_extraction(self):
        text = "File located at: C:\\Users\\Documents\\Report.pdf"
        result = text_to_markdown(text)
        # Backslashes should be escaped
        assert "\\\\" in result

    def test_code_snippet_extraction(self):
        text = "Use the function like this: array[index * 2]"
        result = text_to_markdown(text)
        # Should escape brackets and asterisk
        assert "\\[" in result or "`" in result

    def test_mathematical_expression(self):
        text = "Calculate: (a + b) * c / d"
        result = text_to_markdown(text)
        # Should handle math symbols
        assert "+" in result or "\\+" in result

    def test_url_extraction(self):
        text = "Visit: https://example.com/path?param=value&other=123"
        result = text_to_markdown(text)
        # URL should be preserved
        assert "https://example.com" in result

    def test_email_extraction(self):
        text = "Contact: user@example.com"
        result = text_to_markdown(text)
        # Email should be preserved
        assert "@" in result

    def test_numbered_section(self):
        text = "1.2.3 Section Title"
        result = text_to_markdown(text)
        # Should not be confused with list
        assert "1.2.3" in result


def test_convenience_functions():
    """Test convenience wrapper functions."""

    # heading_to_markdown
    heading = heading_to_markdown("Test Heading", level=2)
    assert heading.startswith("##")

    # list_to_markdown
    items = ["Item 1", "Item 2"]
    list_md = list_to_markdown(items)
    assert "-" in list_md

    # table_to_markdown
    table = [["H1", "H2"], ["C1", "C2"]]
    table_md = table_to_markdown(table)
    assert "|" in table_md


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
