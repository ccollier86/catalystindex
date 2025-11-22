# Qodex-Parse

**Universal Document Parsing Library for RAG Applications**

A modern, production-ready Python library for parsing PDF, DOCX, HTML, and Markdown documents with rich metadata extraction. Built for RAG applications, document viewers, research tools, and knowledge bases.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

✅ **Multi-Format Support**: PDF, DOCX, HTML, Markdown
✅ **Markdown Conversion**: Structure-preserving text extraction
✅ **Smart Heading Detection**: Multi-signal algorithm (7 signals)
✅ **Table Extraction**: Cell-level structure with Markdown export
✅ **Image Extraction**: With bounding box metadata
✅ **Spatial Metadata**: Reading order, bounding boxes, pointers
✅ **Semantic Layer**: Embeddings and nearest neighbors (requires OpenAI)
✅ **Enhanced Metadata**: LLM keywords and questions (requires OpenAI)
✅ **Layout Detection**: Multi-column, academic papers, complex layouts
✅ **Chunking Control**: Semantic or fixed, with size enforcement
✅ **FREE Basic Mode**: No API key required for structure extraction

---

## 🚀 Quick Start

### Installation

```bash
pip install qodex-parse
```

### Basic Usage (Zero Config)

```python
from qodex_parse import parse

# Works immediately - no configuration required!
doc = parse("document.pdf")  # Auto-detects FREE mode

# Access parsed content
for chunk in doc.chunks:
    print(chunk.text)  # Markdown-formatted text
    if chunk.type == "table":
        print(chunk.table.markdown)  # Table as Markdown
```

### With OpenAI (Premium Features)

```python
# Explicit API key
doc = parse("document.pdf", openai_key="sk-...")

# Or from environment (recommended)
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
doc = parse("document.pdf")  # Auto-upgrades to FULL mode

# Access semantic features
for chunk in doc.chunks:
    print(chunk.semantic.keywords)  # LLM-generated keywords
    print(chunk.semantic.neighbors)  # Similar chunks
```

---

## 📊 Processing Modes

Qodex-parse supports multiple processing modes to balance features and cost:

| Mode | Features | OpenAI Required | Use Case |
|------|----------|-----------------|----------|
| **auto** (default) | Auto-detects based on API key | No* | Recommended - works everywhere |
| **basic** (FREE) | Text, tables, images, spatial metadata | No | Document viewers, basic extraction |
| **full** | Everything! All metadata layers | Yes | RAG, knowledge bases, research tools |
| **spatial** | Structure + embeddings | Yes | Vector search without LLM enhancement |
| **semantic** | Structure + semantic layer | Yes | Embeddings + neighbors, no LLM keywords |

\* Auto mode detects OpenAI key from parameter or environment variable

---

## 💡 Usage Examples

### Batch Processing

```python
from qodex_parse import parse_batch

docs = parse_batch(
    ["doc1.pdf", "doc2.pdf", "doc3.pdf"],
    openai_key="sk-..."
)

print(f"Processed {len(docs)} documents")
print(f"Total chunks: {sum(len(doc.chunks) for doc in docs)}")
```

### Custom Configuration

```python
# Academic paper with 2-column layout
doc = parse(
    "paper.pdf",
    openai_key="sk-...",
    layout_mode="academic",      # Optimized for 2-column papers
    max_chunk_tokens=2000,       # Larger chunks for more context
    chunking_strategy="semantic"
)

# Size-constrained (mobile/embedded)
doc = parse(
    "document.pdf",
    qr_compatible=True,  # Enforces 800 token max, 2900 byte limit
    chunking_strategy="semantic"
)

# Speed-optimized (basic mode)
doc = parse(
    "simple_doc.pdf",
    mode="basic",          # No embeddings/LLM
    layout_mode="simple",  # Skip multi-column detection
    tables=False,          # Skip table extraction
    images=False           # Skip image extraction
)
```

### RAG Application

```python
from qodex_parse import parse_batch
import chromadb

# Parse knowledge base documents
docs = parse_batch(
    ["kb_doc1.pdf", "kb_doc2.pdf"],
    openai_key="sk-...",
    max_chunk_tokens=1200,
    chunking_strategy="semantic"
)

# Feed to vector database
client = chromadb.Client()
collection = client.create_collection("knowledge_base")

for doc in docs:
    for chunk in doc.chunks:
        collection.add(
            documents=[chunk.text],
            embeddings=[chunk.semantic.embedding],
            metadatas=[{
                "doc_id": doc.id,
                "chunk_id": chunk.id,
                "type": chunk.type
            }],
            ids=[chunk.id]
        )
```

---

## 🏗️ Architecture

Qodex-parse uses a layered architecture for robust document processing:

```
┌─────────────────────────────────────────────────────┐
│                    User API                         │
│              parse() / parse_batch()                │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                  Processing Layers                   │
├─────────────────────────────────────────────────────┤
│  1. Extraction   → Docling + OpenParse             │
│  2. Conversion   → Text-to-Markdown                 │
│  3. Chunking     → Semantic/Fixed chunking          │
│  4. Spatial      → Bounding boxes, pointers         │
│  5. Semantic     → Embeddings, neighbors (OpenAI)   │
│  6. Enhanced     → LLM keywords/questions (OpenAI)  │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                  Output Models                       │
│           QodexDocument + QodexChunks               │
└─────────────────────────────────────────────────────┘
```

### Key Components

- **Extraction Layer**: Docling for robust PDF parsing with table/image extraction
- **Conversion Layer**: Custom text-to-Markdown converter with 7-signal heading detection
- **Chunking Layer**: Semantic or fixed chunking with size control
- **Spatial Layer**: Bounding boxes, reading order, navigation pointers
- **Semantic Layer**: OpenAI embeddings, nearest neighbor search
- **Enhanced Layer**: LLM-generated keywords and questions

---

## ⚙️ Configuration Reference

### Main Parameters

```python
doc = parse(
    file_path: str,                    # Path to document
    openai_key: Optional[str] = None,  # OpenAI API key (or from env)
    mode: str = "auto",                # Processing mode
    layout_mode: str = "auto",         # Layout detection mode
    tables: bool = True,               # Extract tables
    images: bool = True,               # Extract images
    headings: bool = True,             # Detect headings

    # Chunking parameters
    max_chunk_tokens: int = 1200,       # Hard maximum
    target_chunk_tokens: int = 800,     # Target size
    chunk_overlap: int = 100,           # Token overlap
    chunking_strategy: str = "semantic", # "semantic" or "fixed"
    qr_compatible: bool = False         # Enforce strict size limits
)
```

### Layout Modes

- `"auto"` - Auto-detect layout complexity
- `"simple"` - Single column, top-to-bottom
- `"multi_column"` - Multi-column detection (newspapers)
- `"academic"` - Academic papers (2-column with abstract)

### Chunking Strategies

- `"semantic"` (recommended) - Respects paragraph/sentence boundaries, preserves structure
- `"fixed"` - Fixed-size chunks with overlap, faster but may split mid-thought

---

## 🛠️ Development

### Installation for Development

```bash
# Clone repository
git clone https://github.com/yourusername/qodex-parse.git
cd qodex-parse

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check .
black --check .

# Type checking
mypy qodex_parse/
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_api.py

# With coverage
pytest --cov=qodex_parse tests/
```

---

## 📋 Requirements

- Python 3.8+
- tiktoken (for token counting)
- docling (for PDF extraction)
- openai (for semantic/enhanced layers)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built on [Docling](https://github.com/DS4SD/docling) for PDF extraction
- Inspired by [OpenParse](https://github.com/Filimoa/open-parse) architecture
- Token counting via [tiktoken](https://github.com/openai/tiktoken)

---

## 💬 Support

- Documentation: https://qodex-parse.readthedocs.io
- Issues: https://github.com/yourusername/qodex-parse/issues
- Discussions: https://github.com/yourusername/qodex-parse/discussions

---

## 🗺️ Roadmap

- [ ] Advanced pipeline API for custom workflows
- [ ] Additional format support (RTF, ODT, etc.)
- [ ] Streaming API for large documents
- [ ] Cloud-native deployment (AWS Lambda, Cloud Run)
- [ ] CLI tool for batch processing
- [ ] Pre-built Docker images

---

**Made with ❤️ for the RAG community**
