import asyncio
import base64
import importlib
import io
import os
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from PIL import Image


def load_main_with_fake_registry():
    registry = types.ModuleType("model_registry")
    setattr(registry, "get_dense_model", Mock())
    setattr(registry, "get_sparse_model", Mock())
    setattr(registry, "get_reranker", Mock())
    setattr(registry, "get_docling_converter", Mock())
    setattr(registry, "preload_all", Mock())
    sys.modules["model_registry"] = registry
    if "numpy" not in sys.modules:
        numpy = types.ModuleType("numpy")
        setattr(numpy, "ndarray", type("ndarray", (), {}))
        sys.modules["numpy"] = numpy
    sys.modules.pop("main", None)
    return importlib.import_module("main")


def element(label, page=1, text="", image=None):
    return SimpleNamespace(
        label=label,
        text=text,
        prov=[SimpleNamespace(page_no=page)],
        image=None if image is None else SimpleNamespace(pil_image=image),
    )


class FakeDocument:
    def __init__(self, elements, pages=2, markdown="# Full document\n\nMerged pages."):
        self.pages = {number: object() for number in range(1, pages + 1)}
        self._elements = elements
        self.markdown = markdown

    def iterate_items(self):
        return [(item, None) for item in self._elements]

    def export_to_markdown(self):
        return self.markdown


class FakeConverter:
    def __init__(self, elements, fail=False):
        self.elements = elements
        self.fail = fail
        self.converted_path = None

    def convert(self, path):
        self.converted_path = Path(path)
        if not self.converted_path.exists():
            raise AssertionError("temporary PDF must exist during conversion")
        if self.fail:
            raise RuntimeError("conversion failed")
        return SimpleNamespace(document=FakeDocument(self.elements))


class ConcurrentConverter:
    def __init__(self):
        self._state_lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0

    def convert(self, path):
        with self._state_lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.03)
        with self._state_lock:
            self.active -= 1
        return SimpleNamespace(document=FakeDocument([]))


class FakeStorage:
    def __init__(self, error=None):
        self.calls = []
        self.markdown_calls = []
        self.error = error

    def bucket_path(self, object_key):
        return f"documents/{object_key}"

    def upload_png(self, png_bytes, object_key):
        if self.error:
            raise self.error
        self.calls.append((png_bytes, object_key))
        return f"http://signed.example/{object_key}?signature=redacted"

    def upload_markdown(self, markdown_text, object_key):
        if self.error:
            raise self.error
        self.markdown_calls.append((markdown_text, object_key))
        return f"http://signed.example/{object_key}?signature=redacted"


class DoclingProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_with_fake_registry()

    def setUp(self):
        self.chunk_patch = patch.object(self.main, "build_text_chunks", return_value=[])
        self.chunk_patch.start()

    def tearDown(self):
        self.chunk_patch.stop()

    def test_bucket_mode_uploads_binary_png_without_base64(self):
        converter = FakeConverter(
            [element("picture", page=3, image=Image.new("RGB", (100, 90), "red"))]
        )
        storage = FakeStorage()

        with patch.object(base64, "b64encode", side_effect=AssertionError("no base64")):
            response = self.main.process_docling_upload(
                io.BytesIO(b"%PDF-1.7"), converter, storage
            )

        self.assertEqual(response["total_pages"], 2)
        self.assertNotIn("image_base64", response["elements"][0])
        self.assertIn("image_url", response["elements"][0])
        png_bytes, key = storage.calls[0]
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertRegex(key, r"^docling/[0-9a-f-]+/page-3/picture-1\.png$")
        with Image.open(io.BytesIO(png_bytes)) as uploaded:
            self.assertEqual(uploaded.size, (100, 90))
            self.assertEqual(uploaded.format, "PNG")

    def test_bucket_mode_uploads_merged_markdown_and_returns_root_url(self):
        converter = FakeConverter(
            [element("picture", page=3, image=Image.new("RGB", (100, 90), "red"))]
        )
        storage = FakeStorage()

        response = self.main.process_docling_upload(
            io.BytesIO(b"document"), converter, storage, suffix=".docx"
        )

        markdown_text, markdown_key = storage.markdown_calls[0]
        image_key = storage.calls[0][1]
        self.assertEqual(markdown_text, "# Full document\n\nMerged pages.")
        self.assertRegex(markdown_key, r"^docling/[0-9a-f-]+/document\.md$")
        self.assertEqual(
            markdown_key.split("/")[1],
            image_key.split("/")[1],
        )
        self.assertEqual(
            response["markdown_file"],
            f"http://signed.example/{markdown_key}?signature=redacted",
        )
        self.assertEqual(response["bucket_path"], f"documents/{markdown_key}")
        self.assertEqual(response["file_name"], "document.md")
        self.assertEqual(converter.converted_path.suffix, ".docx")
        self.assertFalse(converter.converted_path.exists())

    def test_base64_mode_does_not_require_storage(self):
        converter = FakeConverter(
            [element("picture", page=4, image=Image.new("RGB", (80, 80), "blue"))]
        )

        response = self.main.process_docling_upload(
            io.BytesIO(b"%PDF-1.7"), converter, storage=None
        )

        image_element = response["elements"][0]
        self.assertIsNone(response["markdown_file"])
        self.assertIsNone(response["bucket_path"])
        self.assertEqual(response["file_name"], "document.md")
        self.assertIn("image_base64", image_element)
        self.assertNotIn("image_url", image_element)
        self.assertTrue(base64.b64decode(image_element["image_base64"]).startswith(b"\x89PNG"))

    def test_text_chunks_and_images_keep_document_order_and_small_images_are_filtered(self):
        elements = [
            element("section_header", page=1, text="Pendahuluan"),
            element("text", page=1, text="Isi dokumen."),
            element("table", page=2, image=Image.new("RGB", (100, 100))),
            element("picture", page=2, image=Image.new("RGB", (79, 100))),
            element("picture", page=3, image=Image.new("RGB", (80, 80))),
        ]
        for index, item in enumerate(elements):
            item.self_ref = f"#/items/{index}"
        chunks = [
            {"type": "text", "page": 1, "text": "# Pendahuluan\n\nIsi dokumen.", "_sequence": 0},
            {"type": "text", "page": 2, "text": "| A | B |\n| - | - |\n| 1 | 2 |", "_sequence": 2},
        ]

        with patch.object(self.main, "build_text_chunks", return_value=chunks):
            response = self.main.process_docling_upload(
                io.BytesIO(b"pdf"), FakeConverter(elements), FakeStorage(), chunk_tokenizer=Mock()
            )

        self.assertEqual([item["type"] for item in response["elements"]], ["text", "text", "image"])
        self.assertEqual([item["page"] for item in response["elements"]], [1, 2, 3])
        self.assertIn("| A | B |", response["elements"][1]["text"])

    def test_elements_only_contain_semantic_text_chunks_and_picture_images(self):
        elements = [
            element("section_header", page=1, text="Pendahuluan"),
            element("text", page=1, text="Isi dokumen."),
            element("list_item", page=1, text="Butir"),
            element("table", page=2, image=Image.new("RGB", (100, 100))),
            element("picture", page=3, image=Image.new("RGB", (100, 100))),
        ]
        for index, item in enumerate(elements):
            item.self_ref = f"#/items/{index}"
        storage = FakeStorage()
        chunks = [
            {"type": "text", "page": 1, "text": "# Pendahuluan\n\nIsi dokumen.\n\n- Butir", "_sequence": 0},
            {"type": "text", "page": 2, "text": "| Nama | Nilai |\n| --- | --- |\n| Ani | 90 |", "_sequence": 3},
        ]

        with patch.object(self.main, "build_text_chunks", return_value=chunks):
            response = self.main.process_docling_upload(
                io.BytesIO(b"pdf"), FakeConverter(elements), storage, chunk_tokenizer=Mock()
            )

        self.assertEqual([item["type"] for item in response["elements"]], ["text", "text", "image"])
        self.assertTrue(all("_sequence" not in item for item in response["elements"]))
        self.assertIn("| Nama | Nilai |", response["elements"][1]["text"])
        self.assertEqual(len(storage.calls), 1)
        self.assertIn("/picture-1.png", storage.calls[0][1])
        self.assertIn("image_url", response["elements"][2])

    def test_object_keys_are_unique_between_requests(self):
        picture = element("picture", image=Image.new("RGB", (80, 80)))
        first_storage = FakeStorage()
        second_storage = FakeStorage()

        self.main.process_docling_upload(
            io.BytesIO(b"pdf"), FakeConverter([picture]), first_storage
        )
        self.main.process_docling_upload(
            io.BytesIO(b"pdf"), FakeConverter([picture]), second_storage
        )

        self.assertNotEqual(first_storage.calls[0][1], second_storage.calls[0][1])

    def test_singleton_docling_converter_is_not_used_concurrently(self):
        converter = ConcurrentConverter()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    self.main.process_docling_upload,
                    io.BytesIO(b"pdf"),
                    converter,
                    None,
                )
                for _ in range(2)
            ]
            for future in futures:
                future.result()

        self.assertEqual(converter.maximum_active, 1)

    def test_temporary_pdf_is_removed_after_success_and_failure(self):
        success_converter = FakeConverter([])
        self.main.process_docling_upload(
            io.BytesIO(b"pdf"), success_converter, storage=None
        )
        self.assertFalse(success_converter.converted_path.exists())

        failing_converter = FakeConverter([], fail=True)
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            self.main.process_docling_upload(
                io.BytesIO(b"pdf"), failing_converter, storage=None
            )
        self.assertFalse(failing_converter.converted_path.exists())

        upload_converter = FakeConverter(
            [element("picture", image=Image.new("RGB", (80, 80)))]
        )
        with self.assertRaises(self.main.StorageUpstreamError):
            self.main.process_docling_upload(
                io.BytesIO(b"pdf"),
                upload_converter,
                FakeStorage(self.main.StorageUpstreamError("upload failed")),
            )
        self.assertFalse(upload_converter.converted_path.exists())


class DoclingEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_with_fake_registry()

    def setUp(self):
        self.chunk_patch = patch.object(self.main, "build_text_chunks", return_value=[])
        self.tokenizer_patch = patch.object(
            self.main, "_create_chunk_tokenizer", return_value=Mock()
        )
        self.chunk_patch.start()
        self.tokenizer_patch.start()

    def tearDown(self):
        self.tokenizer_patch.stop()
        self.chunk_patch.stop()

    def _upload_file(self):
        return SimpleNamespace(file=io.BytesIO(b"document"), filename="sample.docx")

    def test_lifespan_preloads_models(self):
        async def enter_lifespan():
            async with self.main.lifespan(self.main.app):
                pass

        with patch.object(self.main, "preload_all") as preload:
            asyncio.run(enter_lifespan())
        preload.assert_called_once_with()

    def test_async_endpoint_runs_blocking_work_in_threadpool(self):
        expected = {"total_pages": 0, "elements": []}
        threadpool = AsyncMock(return_value=expected)

        with patch.object(self.main, "run_in_threadpool", threadpool):
            response = asyncio.run(self.main.extract_docling(self._upload_file()))

        self.assertEqual(response, expected)
        threadpool.assert_awaited_once_with(
            self.main.extract_docling_sync,
            threadpool.await_args.args[1],
            "sample.docx",
        )

    def test_bucket_mode_endpoint_returns_image_url(self):
        converter = FakeConverter(
            [element("picture", page=2, image=Image.new("RGB", (80, 80)))]
        )
        storage = FakeStorage()
        environment = {
            "USE_VISION_BUCKET": "true",
            "BUCKET_ACCESS_KEY": "access",
            "BUCKET_SECRET_KEY": "secret",
            "BUCKET_URL": "http://minio:9000",
            "BUCKET_NAME": "documents",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(self.main, "get_docling_converter", return_value=converter),
            patch.object(self.main, "get_vision_storage", return_value=storage),
        ):
            response = asyncio.run(self.main.extract_docling(self._upload_file()))

        self.assertIn("image_url", response["elements"][0])
        self.assertNotIn("image_base64", response["elements"][0])
        self.assertIn("/document.md?", response["markdown_file"])
        self.assertEqual(len(storage.calls), 1)
        self.assertEqual(len(storage.markdown_calls), 1)

    def test_false_mode_never_initializes_storage(self):
        converter = FakeConverter([])
        with (
            patch.dict(os.environ, {"USE_VISION_BUCKET": "false"}, clear=True),
            patch.object(self.main, "get_docling_converter", return_value=converter),
            patch.object(
                self.main,
                "get_vision_storage",
                side_effect=AssertionError("storage must not initialize"),
            ),
        ):
            response = self.main.extract_docling_sync(io.BytesIO(b"pdf"))

        self.assertEqual(
            response,
            {
                "total_pages": 2,
                "markdown_file": None,
                "bucket_path": None,
                "file_name": "document.md",
                "elements": [],
            },
        )

    def test_incomplete_bucket_config_is_detected_before_docling_conversion(self):
        with (
            patch.dict(os.environ, {"USE_VISION_BUCKET": "true"}, clear=True),
            patch.object(self.main, "get_docling_converter") as converter,
        ):
            with self.assertRaises(self.main.StorageConfigurationError):
                self.main.extract_docling_sync(io.BytesIO(b"pdf"))
        converter.assert_not_called()

    def test_incomplete_bucket_config_returns_503_without_fallback(self):
        with (
            patch.dict(os.environ, {"USE_VISION_BUCKET": "true"}, clear=True),
            patch.object(self.main, "run_in_threadpool", AsyncMock(side_effect=self.main.StorageConfigurationError("missing bucket config"))),
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(self.main.extract_docling(self._upload_file()))

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("missing bucket config", raised.exception.detail)

    def test_storage_failure_and_timeout_return_502_and_504(self):
        cases = [
            (self.main.StorageUpstreamError("storage failed"), 502),
            (self.main.StorageTimeoutError("storage timeout"), 504),
        ]
        for error, expected_status in cases:
            with self.subTest(expected_status=expected_status):
                with patch.object(
                    self.main, "run_in_threadpool", AsyncMock(side_effect=error)
                ):
                    with self.assertRaises(HTTPException) as raised:
                        asyncio.run(self.main.extract_docling(self._upload_file()))
                self.assertEqual(raised.exception.status_code, expected_status)

    def test_explicit_http_exception_is_not_converted_to_500(self):
        expected = HTTPException(status_code=418, detail="expected")
        with patch.object(
            self.main, "run_in_threadpool", AsyncMock(side_effect=expected)
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(self.main.extract_docling(self._upload_file()))
        self.assertIs(raised.exception, expected)


if __name__ == "__main__":
    unittest.main()
