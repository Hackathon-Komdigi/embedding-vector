from __future__ import annotations

import ipaddress
import os
import re
import threading
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    ReadTimeoutError,
)
from botocore.parsers import ResponseParserError


_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}
_MAX_SIGV4_EXPIRY_SECONDS = 604800


class StorageConfigurationError(ValueError):
    """The vision bucket environment is absent or invalid."""


class StorageUpstreamError(RuntimeError):
    """A non-timeout object-storage operation failed."""


class StorageTimeoutError(StorageUpstreamError):
    """An object-storage operation timed out after bounded retries."""


def vision_bucket_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    source = os.environ if environ is None else environ
    raw_value = source.get("USE_VISION_BUCKET")
    if raw_value is None:
        return True

    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise StorageConfigurationError(
        "USE_VISION_BUCKET harus bernilai true/false, 1/0, yes/no, atau on/off; "
        "nilai kosong maupun nilai lain tidak valid."
    )


@dataclass(frozen=True)
class VisionStorageConfig:
    access_key: str
    secret_key: str
    endpoint_url: str
    bucket_name: str
    presigned_url_expires: int = 3600

    @classmethod
    def from_env(
        cls, environ: Optional[Mapping[str, str]] = None
    ) -> "VisionStorageConfig":
        source = os.environ if environ is None else environ
        names = (
            "BUCKET_ACCESS_KEY",
            "BUCKET_SECRET_KEY",
            "BUCKET_URL",
            "BUCKET_NAME",
        )
        values = {}
        missing = []
        for name in names:
            raw_value = source.get(name)
            if raw_value is None or not raw_value.strip():
                missing.append(name)
            else:
                values[name] = raw_value.strip()
        if missing:
            raise StorageConfigurationError(
                "Konfigurasi vision bucket wajib diisi dan tidak boleh kosong: "
                + ", ".join(missing)
            )

        endpoint_url = values["BUCKET_URL"]
        try:
            parsed_endpoint = urlsplit(endpoint_url)
            # Accessing .port performs validation that urlsplit itself defers.
            parsed_endpoint.port
        except ValueError as exc:
            raise StorageConfigurationError(
                "BUCKET_URL harus berupa URL endpoint S3 API yang valid."
            ) from exc
        if (
            parsed_endpoint.scheme not in {"http", "https"}
            or not parsed_endpoint.netloc
            or not parsed_endpoint.hostname
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or parsed_endpoint.query
            or parsed_endpoint.fragment
            or any(character.isspace() for character in endpoint_url)
        ):
            raise StorageConfigurationError(
                "BUCKET_URL harus berupa URL endpoint S3 API yang valid dengan skema http atau https, tanpa kredensial, query, atau fragment."
            )

        bucket_name = values["BUCKET_NAME"]
        bucket_looks_like_ip = False
        try:
            ipaddress.ip_address(bucket_name)
            bucket_looks_like_ip = True
        except ValueError:
            pass
        if (
            re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket_name)
            is None
            or ".." in bucket_name
            or ".-" in bucket_name
            or "-." in bucket_name
            or bucket_looks_like_ip
        ):
            raise StorageConfigurationError(
                "BUCKET_NAME harus berupa nama bucket S3 yang valid: 3-63 karakter "
                "huruf kecil, angka, titik, atau tanda hubung."
            )

        raw_expiry = source.get("BUCKET_PRESIGNED_URL_EXPIRES")
        if raw_expiry is None:
            expiry = 3600
        else:
            try:
                expiry = int(raw_expiry.strip())
            except (AttributeError, ValueError) as exc:
                raise StorageConfigurationError(
                    "BUCKET_PRESIGNED_URL_EXPIRES harus berupa bilangan bulat positif."
                ) from exc
            if expiry <= 0 or expiry > _MAX_SIGV4_EXPIRY_SECONDS:
                raise StorageConfigurationError(
                    "BUCKET_PRESIGNED_URL_EXPIRES harus antara 1 dan 604800 detik."
                )

        return cls(
            access_key=values["BUCKET_ACCESS_KEY"],
            secret_key=values["BUCKET_SECRET_KEY"],
            endpoint_url=endpoint_url,
            bucket_name=values["BUCKET_NAME"],
            presigned_url_expires=expiry,
        )


class VisionStorage:
    def __init__(self, config: VisionStorageConfig, client=None):
        self.config = config
        self._client = client or boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )
        self._bucket_ready = False
        self._bucket_lock = threading.Lock()

    def ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        with self._bucket_lock:
            if self._bucket_ready:
                return
            try:
                response = self._client.head_bucket(Bucket=self.config.bucket_name)
                _reject_non_s3_success(response)
            except ClientError as exc:
                if not _is_missing_bucket(exc):
                    raise _translate_storage_error(exc) from exc
                self._create_missing_bucket()
            except (ConnectTimeoutError, ReadTimeoutError) as exc:
                raise StorageTimeoutError(
                    "Timeout saat memeriksa bucket object storage setelah retry terbatas."
                ) from exc
            except (ResponseParserError, BotoCoreError) as exc:
                raise _translate_storage_error(exc) from exc
            self._bucket_ready = True

    def _create_missing_bucket(self) -> None:
        try:
            response = self._client.create_bucket(Bucket=self.config.bucket_name)
            _reject_non_s3_success(response)
            return
        except (ConnectTimeoutError, ReadTimeoutError) as exc:
            raise StorageTimeoutError(
                "Timeout saat membuat bucket setelah retry terbatas."
            ) from exc
        except ClientError as exc:
            if _error_code(exc) not in {
                "BucketAlreadyExists",
                "BucketAlreadyOwnedByYou",
            }:
                raise _translate_storage_error(exc) from exc
        except (ResponseParserError, BotoCoreError) as exc:
            raise _translate_storage_error(exc) from exc

        # Another request/process may have created it after our HEAD request.
        try:
            response = self._client.head_bucket(Bucket=self.config.bucket_name)
            _reject_non_s3_success(response)
        except (ConnectTimeoutError, ReadTimeoutError) as exc:
            raise StorageTimeoutError(
                "Timeout saat memverifikasi bucket yang dibuat secara bersamaan."
            ) from exc
        except (ClientError, ResponseParserError, BotoCoreError) as exc:
            raise _translate_storage_error(exc) from exc

    def bucket_path(self, object_key: str) -> str:
        return f"{self.config.bucket_name}/{object_key}"

    def upload_png(self, png_bytes: bytes, object_key: str) -> str:
        return self._upload_bytes(png_bytes, object_key, "image/png")

    def upload_markdown(self, markdown_text: str, object_key: str) -> str:
        return self._upload_bytes(
            markdown_text.encode("utf-8"),
            object_key,
            "text/markdown; charset=utf-8",
        )

    def _upload_bytes(self, body: bytes, object_key: str, content_type: str) -> str:
        self.ensure_bucket()
        try:
            self._put_object(body, object_key, content_type)
        except ClientError as exc:
            if not _is_missing_bucket(exc):
                raise _translate_storage_error(exc) from exc
            # The bucket may have been deleted between ensure and upload.
            self._bucket_ready = False
            self.ensure_bucket()
            try:
                self._put_object(body, object_key, content_type)
            except (ConnectTimeoutError, ReadTimeoutError) as retry_exc:
                raise StorageTimeoutError(
                    "Timeout saat mengunggah object setelah retry terbatas."
                ) from retry_exc
            except (ClientError, ResponseParserError, BotoCoreError) as retry_exc:
                raise _translate_storage_error(retry_exc) from retry_exc
        except (ConnectTimeoutError, ReadTimeoutError) as exc:
            raise StorageTimeoutError(
                "Timeout saat mengunggah object setelah retry terbatas."
            ) from exc
        except (ResponseParserError, BotoCoreError) as exc:
            raise _translate_storage_error(exc) from exc

        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket_name, "Key": object_key},
                ExpiresIn=self.config.presigned_url_expires,
            )
        except (ConnectTimeoutError, ReadTimeoutError) as exc:
            raise StorageTimeoutError(
                "Timeout saat membuat URL object bertanda tangan."
            ) from exc
        except (ClientError, ResponseParserError, BotoCoreError) as exc:
            raise _translate_storage_error(exc) from exc

    def _put_object(self, body: bytes, object_key: str, content_type: str) -> None:
        response = self._client.put_object(
            Bucket=self.config.bucket_name,
            Key=object_key,
            Body=body,
            ContentType=content_type,
        )
        _reject_non_s3_success(response)


def _reject_non_s3_success(response) -> None:
    """Reject successful HTTP responses served by a Console/HTML endpoint."""
    if not isinstance(response, Mapping):
        return
    metadata = response.get("ResponseMetadata", {})
    headers = metadata.get("HTTPHeaders", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(headers, Mapping):
        return
    normalized_headers = {
        str(key).lower(): str(value).lower() for key, value in headers.items()
    }
    content_type = normalized_headers.get("content-type", "")
    server = normalized_headers.get("server", "")
    location = normalized_headers.get("location", "")
    if "text/html" in content_type or "console" in server or "console" in location:
        raise StorageUpstreamError(
            "Endpoint object storage bukan endpoint S3 API yang dapat digunakan. "
            "Pastikan BUCKET_URL menunjuk ke port/endpoint S3 API MinIO, bukan Console."
        )


def _error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _http_status(exc: ClientError) -> int:
    return int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)


def _is_missing_bucket(exc: ClientError) -> bool:
    return _error_code(exc) in {"404", "NoSuchBucket", "NotFound"} or _http_status(exc) == 404


def _translate_storage_error(exc: Exception) -> StorageUpstreamError:
    if isinstance(exc, (ConnectTimeoutError, ReadTimeoutError)):
        return StorageTimeoutError(
            "Timeout saat mengakses object storage setelah retry terbatas."
        )

    code = _error_code(exc) if isinstance(exc, ClientError) else ""
    status = _http_status(exc) if isinstance(exc, ClientError) else 0
    safe_text = str(exc).lower()
    if (
        status in {301, 302, 303, 307, 308}
        or code in {"PermanentRedirect", "TemporaryRedirect"}
        or "html" in safe_text
        or "console" in safe_text
        or "api port" in safe_text
    ):
        return StorageUpstreamError(
            "Endpoint object storage bukan endpoint S3 API yang dapat digunakan. "
            "Pastikan BUCKET_URL menunjuk ke port/endpoint S3 API MinIO, bukan Console."
        )
    if status == 403 or code in {
        "AccessDenied",
        "InvalidAccessKeyId",
        "SignatureDoesNotMatch",
        "AuthorizationHeaderMalformed",
    }:
        return StorageUpstreamError(
            "Object storage menolak akses atau kredensial/signature tidak valid; "
            "periksa izin bucket dan konfigurasi BUCKET_URL/access key."
        )
    return StorageUpstreamError(
        "Operasi object storage gagal. Periksa konektivitas BUCKET_URL, endpoint S3 API, "
        "kredensial, region, dan izin bucket."
    )


_storage_instance: Optional[VisionStorage] = None
_storage_instance_lock = threading.Lock()


def get_vision_storage() -> VisionStorage:
    """Build and reuse the S3 client lazily after endpoint use requires it."""
    global _storage_instance
    config = VisionStorageConfig.from_env()
    if _storage_instance is not None and _storage_instance.config == config:
        return _storage_instance
    with _storage_instance_lock:
        if _storage_instance is None or _storage_instance.config != config:
            _storage_instance = VisionStorage(config)
        return _storage_instance
