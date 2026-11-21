import unittest

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData

from document_chunking import build_text_chunks


class WordTokenizer(BaseTokenizer):
    max_tokens: int = 64

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self):
        return self

    def encode(self, text: str, add_special_tokens: bool = False):
        return text.split()

    def decode(self, tokens, skip_special_tokens: bool = True):
        return " ".join(tokens)


class DocumentChunkingTests(unittest.TestCase):
    def test_headings_lists_and_table_become_markdown_text_chunks(self):
        doc = DoclingDocument(name="sample")
        doc.add_heading("Bab Satu", level=1)
        doc.add_text(DocItemLabel.TEXT, "Paragraf pembuka.")
        group = doc.add_list_group()
        doc.add_list_item("Butir pertama", parent=group)
        table = TableData(
            num_rows=2,
            num_cols=2,
            table_cells=[
                TableCell(start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1, text="Nama", column_header=True),
                TableCell(start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2, text="Nilai", column_header=True),
                TableCell(start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=0, end_col_offset_idx=1, text="Ani"),
                TableCell(start_row_offset_idx=1, end_row_offset_idx=2, start_col_offset_idx=1, end_col_offset_idx=2, text="90"),
            ],
        )
        doc.add_table(table)

        chunks = build_text_chunks(doc, WordTokenizer(max_tokens=64), max_tokens=64, overlap_tokens=8)
        combined = "\n\n".join(chunk["text"] for chunk in chunks)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk["type"] == "text" for chunk in chunks))
        self.assertIn("# Bab Satu", combined)
        self.assertIn("- Butir pertama", combined)
        self.assertIn("| Nama", combined)
        self.assertIn("| Ani", combined)
        self.assertNotIn("image_url", combined)

    def test_neighboring_chunks_repeat_overlap_within_same_section(self):
        doc = DoclingDocument(name="overlap")
        doc.add_heading("Bab", level=1)
        first = " ".join(f"awal{i}" for i in range(24))
        second = " ".join(f"akhir{i}" for i in range(24))
        doc.add_text(DocItemLabel.TEXT, first)
        doc.add_text(DocItemLabel.TEXT, second)

        tokenizer = WordTokenizer(max_tokens=12)
        chunks = build_text_chunks(doc, tokenizer, max_tokens=12, overlap_tokens=3)

        self.assertGreater(len(chunks), 2)
        previous_words = chunks[0]["text"].split()
        current_words = chunks[1]["text"].split()
        self.assertTrue(set(previous_words[-3:]).intersection(current_words))
        self.assertTrue(all(chunk["text"].startswith("# Bab") for chunk in chunks))
        self.assertTrue(all(tokenizer.count_tokens(chunk["text"]) <= 12 for chunk in chunks))

    def test_picture_boundary_prevents_text_peer_merge(self):
        doc = DoclingDocument(name="ordered")
        doc.add_heading("Bab", level=1)
        before = doc.add_text(DocItemLabel.TEXT, "Sebelum gambar.")
        picture = doc.add_picture()
        after = doc.add_text(DocItemLabel.TEXT, "Sesudah gambar.")

        chunks = build_text_chunks(doc, WordTokenizer(max_tokens=64), max_tokens=64, overlap_tokens=8)

        self.assertEqual(len(chunks), 2)
        self.assertLess(chunks[0]["_sequence"], doc.body.children.index(picture.get_ref()))
        self.assertGreater(chunks[1]["_sequence"], doc.body.children.index(picture.get_ref()))
        self.assertIn("Sebelum gambar", chunks[0]["text"])
        self.assertIn("Sesudah gambar", chunks[1]["text"])
        self.assertNotIn("Sebelum gambar", chunks[1]["text"])

    def test_heading_only_and_html_are_preserved_safely(self):
        doc = DoclingDocument(name="safe")
        doc.add_heading("<script>alert(1)</script>", level=1)

        chunks = build_text_chunks(doc, WordTokenizer(max_tokens=64), max_tokens=64, overlap_tokens=8)

        self.assertEqual(len(chunks), 1)
        self.assertIn("# &lt;script&gt;alert(1)&lt;/script&gt;", chunks[0]["text"])
        self.assertNotIn("<script>", chunks[0]["text"])

    def test_invalid_overlap_limits_are_rejected(self):
        doc = DoclingDocument(name="invalid")
        with self.assertRaisesRegex(ValueError, "overlap_tokens"):
            build_text_chunks(doc, WordTokenizer(max_tokens=10), max_tokens=10, overlap_tokens=10)


if __name__ == "__main__":
    unittest.main()
