import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError, ReadTimeoutError


def client_error(code, status=400, message="storage error", operation="HeadBucket"):
    return ClientError(
        {
            "Error": {"Code": code, "Message": message},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )


class VisionStorageConfigTests(unittest.TestCase):
    def test_bucket_mode_defaults_to_enabled(self):
        from vision_storage import vision_bucket_enabled

        self.assertTrue(vision_bucket_enabled({}))

    def test_bucket_mode_accepts_whitespace_case_and_supported_values(self):
        from vision_storage import vision_bucket_enabled

        for value in (" TRUE ", "1", " Yes ", "ON"):
            with self.subTest(value=value):
                self.assertTrue(vision_bucket_enabled({"USE_VISION_BUCKET": value}))
        for value in (" FALSE ", "0", " No ", "off"):
            with self.subTest(value=value):
                self.assertFalse(vision_bucket_enabled({"USE_VISION_BUCKET": value}))

    def test_bucket_mode_rejects_empty_and_invalid_values(self):
        from vision_storage import StorageConfigurationError, vision_bucket_enabled

        for value in ("", "   ", "maybe"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(StorageConfigurationError, "USE_VISION_BUCKET"):
                    vision_bucket_enabled({"USE_VISION_BUCKET": value})

    def test_bucket_config_requires_non_empty_values(self):
        from vision_storage import StorageConfigurationError, VisionStorageConfig

        with self.assertRaisesRegex(StorageConfigurationError, "BUCKET_SECRET_KEY"):
            VisionStorageConfig.from_env(
                {
                    "BUCKET_ACCESS_KEY": "access",
                    "BUCKET_SECRET_KEY": " ",
                    "BUCKET_URL": "http://minio:9000",
                    "BUCKET_NAME": "documents",
                }
            )

    def test_presigned_expiry_must_be_positive_and_within_sigv4_limit(self):
        from vision_storage import StorageConfigurationError, VisionStorageConfig

        base = {
            "BUCKET_ACCESS_KEY": "access",
            "BUCKET_SECRET_KEY": "secret",
            "BUCKET_URL": "http://minio:9000",
            "BUCKET_NAME": "documents",
        }
        for value in ("0", "-1", "604801", "invalid", ""):
            with self.subTest(value=value):
                with self.assertRaises(StorageConfigurationError):
                    VisionStorageConfig.from_env(
                        {**base, "BUCKET_PRESIGNED_URL_EXPIRES": value}
                    )

    def test_bucket_url_parser_errors_are_configuration_errors(self):
        from vision_storage import StorageConfigurationError, VisionStorageConfig

        base = {
            "BUCKET_ACCESS_KEY": "access",
            "BUCKET_SECRET_KEY": "secret",
            "BUCKET_NAME": "documents",
        }
        for value in ("http://[invalid", "http://minio.example:not-a-port", "ftp://minio"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(StorageConfigurationError, "BUCKET_URL"):
                    VisionStorageConfig.from_env({**base, "BUCKET_URL": value})

    def test_bucket_name_syntax_is_validated_as_configuration(self):
        from vision_storage import StorageConfigurationError, VisionStorageConfig

        base = {
            "BUCKET_ACCESS_KEY": "access",
            "BUCKET_SECRET_KEY": "secret",
            "BUCKET_URL": "http://minio:9000",
        }
        invalid_names = (
            "ab",
            "BadName",
            "bad name",
            "bad..name",
            "-bad-name",
            "bad-name-",
            "192.168.1.1",
        )
        for value in invalid_names:
            with self.subTest(value=value):
                with self.assertRaisesRegex(StorageConfigurationError, "BUCKET_NAME"):
                    VisionStorageConfig.from_env({**base, "BUCKET_NAME": value})

    def test_client_uses_sigv4_path_style_region_and_bounded_timeouts(self):
        from vision_storage import VisionStorage, VisionStorageConfig

        config = VisionStorageConfig(
            access_key="access",
            secret_key="secret",
            endpoint_url="https://minio.example:9000",
            bucket_name="documents",
            presigned_url_expires=3600,
        )
        with patch("vision_storage.boto3.client") as create_client:
            VisionStorage(config)

        kwargs = create_client.call_args.kwargs
        self.assertEqual(kwargs["endpoint_url"], config.endpoint_url)
        self.assertEqual(kwargs["aws_access_key_id"], config.access_key)
        self.assertEqual(kwargs["aws_secret_access_key"], config.secret_key)
        self.assertEqual(kwargs["region_name"], "us-east-1")
        self.assertEqual(kwargs["config"].signature_version, "s3v4")
        self.assertEqual(kwargs["config"].s3["addressing_style"], "path")
        self.assertGreater(kwargs["config"].connect_timeout, 0)
        self.assertGreater(kwargs["config"].read_timeout, 0)
        self.assertLessEqual(kwargs["config"].retries["max_attempts"], 2)


class VisionStorageTests(unittest.TestCase):
    def setUp(self):
        from vision_storage import VisionStorage, VisionStorageConfig

        self.client = Mock()
        self.client.generate_presigned_url.return_value = "http://signed.example/object?sig=x"
        self.config = VisionStorageConfig(
            access_key="access",
            secret_key="secret",
            endpoint_url="http://minio:9000",
            bucket_name="documents",
            presigned_url_expires=1234,
        )
        self.storage = VisionStorage(self.config, client=self.client)

    def test_missing_bucket_is_created_then_upload_succeeds(self):
        self.client.head_bucket.side_effect = client_error("NoSuchBucket", 404)

        url = self.storage.upload_png(b"\x89PNG\r\n", "docling/request/page-1/picture-1.png")

        self.client.create_bucket.assert_called_once_with(Bucket="documents")
        self.client.put_object.assert_called_once_with(
            Bucket="documents",
            Key="docling/request/page-1/picture-1.png",
            Body=b"\x89PNG\r\n",
            ContentType="image/png",
        )
        self.assertEqual(url, "http://signed.example/object?sig=x")

    def test_existing_bucket_is_not_created_or_checked_for_every_image(self):
        self.storage.upload_png(b"first", "first.png")
        self.storage.upload_png(b"second", "second.png")

        self.client.head_bucket.assert_called_once_with(Bucket="documents")
        self.client.create_bucket.assert_not_called()
        self.assertEqual(self.client.put_object.call_count, 2)

    def test_bucket_creation_race_is_accepted_after_verification(self):
        self.client.head_bucket.side_effect = [
            client_error("NoSuchBucket", 404),
            None,
        ]
        self.client.create_bucket.side_effect = client_error(
            "BucketAlreadyOwnedByYou", 409, operation="CreateBucket"
        )

        self.storage.upload_png(b"png", "image.png")

        self.assertEqual(self.client.head_bucket.call_count, 2)
        self.client.put_object.assert_called_once()

    def test_access_denied_does_not_trigger_bucket_creation(self):
        from vision_storage import StorageUpstreamError

        self.client.head_bucket.side_effect = client_error("AccessDenied", 403)

        with self.assertRaisesRegex(StorageUpstreamError, "akses"):
            self.storage.upload_png(b"png", "image.png")
        self.client.create_bucket.assert_not_called()

    def test_no_such_bucket_during_upload_reensures_and_retries_once(self):
        self.client.head_bucket.side_effect = [
            None,
            client_error("NoSuchBucket", 404),
        ]
        self.client.put_object.side_effect = [
            client_error("NoSuchBucket", 404, operation="PutObject"),
            None,
        ]

        self.storage.upload_png(b"png", "image.png")

        self.client.create_bucket.assert_called_once_with(Bucket="documents")
        self.assertEqual(self.client.put_object.call_count, 2)

    def test_timeout_is_reported_as_storage_timeout(self):
        from vision_storage import StorageTimeoutError

        self.client.head_bucket.side_effect = ReadTimeoutError(endpoint_url="http://minio:9000")

        with self.assertRaises(StorageTimeoutError):
            self.storage.upload_png(b"png", "image.png")

    def test_timeout_during_bucket_creation_is_reported_as_storage_timeout(self):
        from vision_storage import StorageTimeoutError

        self.client.head_bucket.side_effect = client_error("NoSuchBucket", 404)
        self.client.create_bucket.side_effect = ReadTimeoutError(
            endpoint_url="http://minio:9000"
        )

        with self.assertRaises(StorageTimeoutError):
            self.storage.upload_png(b"png", "image.png")

    def test_presigned_url_uses_uploaded_bucket_key_and_expiry(self):
        key = "docling/request/page-4/table-1.png"

        self.storage.upload_png(b"png", key)

        self.client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "documents", "Key": key},
            ExpiresIn=1234,
        )

    def test_markdown_is_uploaded_as_utf8_and_presigned(self):
        key = "docling/request/document.md"

        url = self.storage.upload_markdown("# Judul\n\nIsi é.", key)

        self.client.put_object.assert_called_once_with(
            Bucket="documents",
            Key=key,
            Body="# Judul\n\nIsi é.".encode("utf-8"),
            ContentType="text/markdown; charset=utf-8",
        )
        self.client.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "documents", "Key": key},
            ExpiresIn=1234,
        )
        self.assertEqual(url, "http://signed.example/object?sig=x")

    def test_console_html_head_response_is_not_treated_as_bucket_success(self):
        from vision_storage import StorageUpstreamError

        self.client.head_bucket.return_value = {
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                    "content-type": "text/html",
                    "server": "MinIO Console",
                },
            }
        }

        with self.assertRaisesRegex(StorageUpstreamError, "S3 API"):
            self.storage.upload_png(b"png", "image.png")
        self.client.put_object.assert_not_called()
        self.client.create_bucket.assert_not_called()

    def test_redirect_or_console_response_mentions_s3_api_endpoint(self):
        from vision_storage import StorageUpstreamError

        self.client.head_bucket.side_effect = client_error(
            "PermanentRedirect", 301, "Moved to MinIO Console"
        )

        with self.assertRaisesRegex(StorageUpstreamError, "S3 API"):
            self.storage.upload_png(b"png", "image.png")


if __name__ == "__main__":
    unittest.main()
