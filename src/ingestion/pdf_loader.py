"""
pdf_loader.py
-------------
Loads a PDF textbook and splits it into overlapping chunks.
Uses pdfplumber with word-based column snapping to extract
borderless tables that pdfplumber's default table detector misses.

Each chunk is a LangChain Document with:
    - page_content: the raw text (or reconstructed table)
    - metadata: {"source": filename, "page": page_number, "type": "text"|"table"}

Run standalone to preview chunks:
    python -m src.ingestion.pdf_loader
"""

import os
import re
import pdfplumber
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() # loads .env variables

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Tolerance in points for snapping words to the same column
# Increase if columns are merging; decrease if columns are splitting
SNAP_TOLERANCE = 3
# Minimum number of rows that look tabular to treat a region as a table
MIN_TABLE_ROWS = 6
# Minimum number of columns to consider a row tabular
MIN_TABLE_COLS = 3

def _is_real_table(grid: list[dict]) -> bool:

    if not grid:
        return False

    col_counts = [len(row) for row in grid]
    avg_cols   = sum(col_counts) / len(col_counts)

    # 1. Column consistency
    consistent_rows = sum(1 for c in col_counts if abs(c - avg_cols) <= 1)
    if consistent_rows / len(col_counts) < 0.6:
        return False

    # 2. Minimum columns
    if avg_cols < MIN_TABLE_COLS:
        return False

    # 3. Must contain a number
    import re
    all_text = " ".join(text for row in grid for text in row.values())
    if not re.search(r"\d", all_text):
        return False

    # 4. NEW — reject tables where most cells are single characters
    # Diagram labels are single letters: "r", "b", "c", "e", "w"
    all_cells  = [text for row in grid for text in row.values() if text.strip()]
    short_cells = sum(1 for c in all_cells if len(c.strip()) <= 2)
    if short_cells / max(len(all_cells), 1) > 0.5:
        return False

    # 5. NEW — minimum average cell length
    # Real table cells have meaningful text, not single letters
    avg_cell_len = sum(len(c.strip()) for c in all_cells) / max(len(all_cells), 1)
    if avg_cell_len < 4:
        return False

    # 6. NEW — reject if too few rows relative to columns
    # Real tables are taller than they are wide
    # Diagrams tend to be wide and short
    if len(grid) < avg_cols:
        return False

    # 7. NEW — reject if content looks like diagram labels
    # Diagram labels are short fragments: single words, abbreviations
    # Real table cells repeat structural patterns
    unique_cells = {text.strip().lower() 
                    for row in grid 
                    for text in row.values() 
                    if text.strip()}
    # If almost every cell is unique and short, it's diagram labels
    avg_unique_len = sum(len(c) for c in unique_cells) / max(len(unique_cells), 1)
    if len(unique_cells) == len(all_cells) and avg_unique_len < 8:
        return False

    return True


def _extract_borderless_tables(page) -> list[str]:
    words = page.extract_words(
        x_tolerance=SNAP_TOLERANCE,
        y_tolerance=SNAP_TOLERANCE,
        keep_blank_chars=False,
    )
    if not words:
        return []

    rows_dict = {}
    for word in words:
        y_key = round(word["top"] / 2) * 2
        rows_dict.setdefault(y_key, []).append(word)

    sorted_rows = [
        sorted(words, key=lambda w: w["x0"])
        for _, words in sorted(rows_dict.items())
    ]

    all_x0 = sorted({round(w["x0"] / 5) * 5 for row in sorted_rows for w in row})

    def snap_to_col(x0):
        return min(all_x0, key=lambda col: abs(col - round(x0 / 5) * 5))

    grid = []
    for row_words in sorted_rows:
        row_dict = {}
        for word in row_words:
            col = snap_to_col(word["x0"])
            row_dict[col] = row_dict.get(col, "") + " " + word["text"]
        grid.append({col: _fix_spacing(text.strip()) for col, text in row_dict.items()})

    # Apply stricter filtering
    if len(grid) < MIN_TABLE_ROWS:
        return []

    if not _is_real_table(grid):
        return []

    all_cols   = sorted({col for row in grid for col in row})
    table_data = [[row.get(col, "") for col in all_cols] for row in grid]
    df         = pd.DataFrame(table_data)

    first_row_filled = sum(1 for c in table_data[0] if c.strip())
    avg_filled = sum(
        sum(1 for c in row if c.strip()) for row in table_data[1:]
    ) / max(len(table_data) - 1, 1)

    if first_row_filled >= avg_filled and len(table_data) > 1:
        df.columns = table_data[0]
        df = df[1:].reset_index(drop=True)

    df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all").fillna("")

    if df.empty:
        return []

    try:
        return [df.to_markdown(index=False)]
    except Exception:
        return []


def _clean_page_text(text: str) -> str:
    """
    Remove common PDF noise: headers, footers, page numbers,
    and sideways watermark text from the NBC 2020.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip page number lines e.g. "4-9", "4-10", "B-2"
        if re.fullmatch(r"[A-Z]?-?\d+", stripped):
            continue

        # Skip running headers e.g. "Division B 4.1.5.3." or "4.1.5.3. Division B"
        if re.fullmatch(r"(Division\s+[A-Z]\s+[\d\.]+\.?|[\d\.]+\.?\s+Division\s+[A-Z])", stripped):
            continue

        # Skip footer lines e.g. "National Building Code of Canada 2020 Volume 1"
        if "NationalBuildingCode" in stripped or "National Building Code" in stripped:
            continue

        # Skip the sideways copyright watermark (short fragments of reversed text)
        if re.fullmatch(r"[a-z,\s]{2,15}", stripped) and stripped.islower():
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned)


def _fix_spacing(text: str) -> str:
    """
    Attempt to restore spaces in concatenated PDF text.
    Inserts a space before uppercase letters that follow lowercase,
    and before digits that follow letters and vice versa.
    """
    # "assemblyAreas" → "assembly Areas"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # "areas4.8" → "areas 4.8"
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)

    # "4.8kPa" → "4.8 kPa"
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

    return text

 
def load_split(pdf_path: str, start_page: int = 1, end_page: int = None) -> list:
    """
    Load a PDF and return a list of LangChain Document chunks.
    Text and tables are handled separately:
      - Text is chunked with RecursiveCharacterTextSplitter
      - Tables are stored whole (never split) to preserve structure
 
    :param pdf_path: Absolute or relative path to the PDF file.
    :return: List of Document objects with page_content and metadata.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Loading PDF: {pdf_path.name}")
 
    text_documents  = []
    table_documents = []
 
    with pdfplumber.open(pdf_path) as pdf:

        total_pages = len(pdf.pages)
        end = end_page if end_page else total_pages

        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num < start_page or page_num > end:
                continue
 
            # 1. Extract plain text
            text = page.extract_text() or ""
            text = _clean_page_text(text)
            text = _fix_spacing(text)

            # Extract table title from text
            # Building code tables are labelled "Table X.X.X.X" in the text
            table_titles = re.findall(
                r"Table\s+[\d\.]+[^\n]*", text, re.IGNORECASE
            )
            title_prefix = "\n".join(table_titles) if table_titles else ""
 
            # 2. Try default table extractor first (for bordered tables)
            bordered_tables = page.extract_tables()
            table_markdowns = []
 
            for table in bordered_tables:
                if not table or not table[0]:
                    continue
                headers = [_fix_spacing(str(c).strip()) if c else "" for c in table[0]]
                rows    = [
                    [_fix_spacing(str(c).strip()) if c else "" for c in row]
                    for row in table[1:]
                ]
                df = pd.DataFrame(rows, columns=headers)
                df = df.replace("", pd.NA).dropna(how="all").fillna("")
                if not df.empty:
                    table_markdowns.append(df.to_markdown(index=False))
 
            # 3. If no bordered tables found, try borderless extraction
            # Skip pages that contain figures — dimension labels get misread as tables
            has_figure = bool(re.search(r"Figure\s+[\d\.]+", text, re.IGNORECASE))
            if not table_markdowns and not has_figure:
                table_markdowns = _extract_borderless_tables(page)
 
            # 4. Store text document
            if text.strip():
                text_documents.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "type": "text",
                    }
                ))
 
            # 5. Store each table as its own document (never split)
            for table_md in table_markdowns:
                label = f"[Page {page_num}]\n{title_prefix}\n\n" if title_prefix else f"[Page {page_num}]\n\n"
                table_documents.append(Document(
                    page_content=label + table_md,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num,
                        "type": "table",
                    }
                ))
 
            if page_num % 100 == 0:
                print(f"  Processed {page_num}/{total_pages} pages...")
 
    print(f"\n  Pages loaded   : {total_pages}")
    print(f"  Text documents : {len(text_documents)}")
    print(f"  Table documents: {len(table_documents)}")
 
    # 6. Chunk text documents only
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    text_chunks = splitter.split_documents(text_documents)
 
    # 7. Combine text chunks + whole table documents
    all_chunks = text_chunks + table_documents
 
    print(f"  Text chunks    : {len(text_chunks)}")
    print(f"  Total chunks   : {len(all_chunks)}")
    print(f"  Avg text chunk : {sum(len(c.page_content) for c in text_chunks) // max(len(text_chunks), 1)} chars")
 
    return all_chunks
 
 
# Run standalone
if __name__ == "__main__":
    import sys

    pdf  = sys.argv[1] if len(sys.argv) > 1 else "data/raw/nbc2020.pdf"
    docs = load_split(pdf, start_page=430, end_page=557)

    # Separate by type
    table_chunks = [d for d in docs if d.metadata.get("type") == "table"]
    text_chunks  = [d for d in docs if d.metadata.get("type") == "text"]

    print(f"\n── Table chunks: {len(table_chunks)} | Text chunks: {len(text_chunks)} ──")

    # Preview table chunks
    print("\n" + "=" * 60)
    print("TABLE CHUNKS")
    print("=" * 60)
    for doc in table_chunks[184:186]:
        print(f"\n[Page {doc.metadata.get('page')}]")
        print(doc.page_content[:700])
        print("─" * 60)

    # Preview text chunks from the same pages as the table chunks above
    target_pages = {d.metadata.get("page") for d in table_chunks[184:186]}
    same_page_text = [d for d in text_chunks if d.metadata.get("page") in target_pages]

    print("\n" + "=" * 60)
    print("TEXT CHUNKS (same pages)")
    print("=" * 60)
    for doc in same_page_text[:2]:
        print(f"\n[Page {doc.metadata.get('page')}]")
        print(doc.page_content[:700])
        print("─" * 60)