# Embedding Vector Microservice

Service FastAPI untuk dense embedding, sparse embedding SPLADE, reranking, dan ekstraksi dokumen dengan Docling. Dalam mode bucket, hasil Markdown lengkap seluruh halaman diunggah ke S3-compatible storage/MinIO dan dikembalikan sebagai presigned `markdown_file`. Gambar `picture` dan `table` diunggah sebagai PNG binary dan dikembalikan melalui `image_url`; mode kompatibilitas tetap menggunakan `image_base64`.

## Endpoint

| Endpoint | Method | Fungsi |
|---|---|---|
| `/embed/dense` | POST | Dense vector embedding |
| `/embed/sparse/query` | POST | Sparse query vector SPLADE |
| `/embed/sparse/passages` | POST | Sparse passage vectors |
| `/rerank` | POST | Cross-encoder reranking |
| `/extract/docling` | POST | Ekstraksi struktur, teks, OCR, tabel, gambar, dan Markdown dari format dokumen yang didukung Docling |

Swagger UI tersedia di `http://localhost:8003/docs`.

## Konfigurasi

Salin contoh konfigurasi tanpa menimpa `.env` yang sudah ada:

```bash
cp -n .env.example .env
```

Isi nilai berikut:

```env
# Konfigurasi Model Hugging Face
DENSE_MODEL=BAAI/bge-m3
SPARSE_MODEL=naver/splade-cocondenser-ensembledistil
RERANKER_MODEL=BAAI/bge-reranker-large

# Konfigurasi Vision / Image
USE_VISION_BUCKET=true
BUCKET_ACCESS_KEY=replace-with-access-key
BUCKET_SECRET_KEY=replace-with-secret-key
BUCKET_URL=http://minio.example:9000
BUCKET_NAME=regulatory-documents

# Opsional; default 3600, rentang valid 1-604800 detik
BUCKET_PRESIGNED_URL_EXPIRES=3600
```

Aplikasi memuat `.env` dari direktori modul aplikasi, bukan dari current working directory. Environment proses dari systemd, Docker, atau shell tidak ditimpa oleh `.env` (`override=False`), sehingga environment proses memiliki prioritas lebih tinggi.

### Aturan `USE_VISION_BUCKET`

- Tidak ditetapkan: default `true`.
- Nilai true: `true`, `1`, `yes`, `on`.
- Nilai false: `false`, `0`, `no`, `off`.
- Kapitalisasi dan spasi awal/akhir diabaikan.
- Nilai kosong atau selain daftar tersebut ditolak sebagai kesalahan konfigurasi HTTP 503.
- Ketika `true`, `BUCKET_ACCESS_KEY`, `BUCKET_SECRET_KEY`, `BUCKET_URL`, dan `BUCKET_NAME` wajib diisi. `BUCKET_NAME` harus memenuhi format nama bucket S3 (3-63 karakter lowercase, angka, titik, atau tanda hubung; bukan alamat IP).
- Gunakan `USE_VISION_BUCKET=false` untuk kembali ke respons Base64. Dalam mode ini client S3 tidak dibuat dan bucket tidak dihubungi.

Konfigurasi dan client storage baru divalidasi/dibuat saat endpoint Docling memerlukannya. Masalah storage tidak menggagalkan startup maupun endpoint dense, sparse, dan reranker. Jika mode bucket aktif, kegagalan storage tidak diubah menjadi Base64.

## MinIO/S3

`BUCKET_URL` dipakai apa adanya sebagai endpoint S3 API, termasuk hostname, skema, dan port. Aplikasi tidak mengganti port secara otomatis. Port default MinIO biasanya:

- `9000`: S3 API.
- `9001`: Console web.

Pemetaan deployment dapat berbeda, jadi port `9001` tetap sah apabila memang dipetakan ke S3 API. Jika endpoint mengembalikan redirect, HTML Console, atau pesan bahwa request harus memakai API port, endpoint merespons HTTP 502 dengan petunjuk agar `BUCKET_URL` diarahkan ke S3 API. Redirect ke Console tidak dianggap sebagai upload berhasil.

Client menggunakan Signature Version 4, region `us-east-1`, path-style addressing, verifikasi TLS normal untuk HTTPS, timeout terbatas, dan retry terbatas. Jangan masukkan kredensial ke Dockerfile atau source code.

Sebelum upload pertama, service memeriksa bucket. Bucket yang benar-benar tidak ada dibuat otomatis. Race ketika proses lain membuat bucket ditangani dan upload `NoSuchBucket` dicoba ulang secara terbatas setelah ensure/create. Access denied, kredensial/signature salah, endpoint salah, connection refused, dan timeout tidak dianggap sebagai bucket hilang. Service tidak mengubah policy/ACL dan tidak membuat bucket public.

Kredensial minimal harus memiliki izin yang sesuai untuk:

- memeriksa/list bucket (`HeadBucket`/izin ekuivalen);
- membuat bucket bila belum ada;
- `PutObject` untuk upload PNG dan Markdown;
- `GetObject` untuk akses melalui presigned URL.

Jika bucket disiapkan sebelumnya, izin membuat bucket dapat dihilangkan selama `HeadBucket`, `PutObject`, dan signing/`GetObject` tetap diizinkan.

Server Linux harus dapat menjangkau `BUCKET_URL` untuk pemeriksaan dan upload. Hostname/port yang tertulis di presigned URL juga harus dapat dijangkau oleh konsumen `image_url`. Signing yang sukses tidak membuktikan konektivitas konsumen.

## Kontrak `/extract/docling`

Elemen teks, section header, list item, formula, urutan, nomor halaman, dan `total_pages` tetap sama. Gambar tanpa data dilewati; gambar dengan lebar atau tinggi kurang dari 80 piksel tetap difilter.

### Mode bucket (default)

Gambar diserialisasi ke PNG lossless dalam buffer memori lalu diunggah sebagai binary dengan `Content-Type: image/png`. Jalur ini tidak membuat Base64 atau data URI. Object key unik berbentuk `docling/{request_uuid}/page-{page_no}/{element_type}-{image_index}.png`.

Docling mengekspor seluruh isi dokumen—semua halaman dalam urutan dokumen—menjadi satu string Markdown melalui `export_to_markdown()`. Markdown UTF-8 tersebut diunggah sebagai `text/markdown; charset=utf-8` ke `docling/{request_uuid}/document.md`. Root response `markdown_file` berisi presigned GET URL untuk object ini. Suffix aman dari nama upload diteruskan ke file sementara sehingga endpoint dapat menerima format input lain yang didukung instalasi Docling, tanpa menggunakan nama file klien sebagai path server.

```json
{
  "total_pages": 11,
  "markdown_file": "http://storage.example/regulatory-documents/docling/.../document.md?...",
  "elements": [
    {
      "type": "section_header",
      "page": 1,
      "text": "Pendahuluan"
    },
    {
      "type": "text",
      "page": 1,
      "text": "Contoh isi dokumen."
    },
    {
      "type": "table",
      "page": 4,
      "image_url": "http://storage.example/regulatory-documents/docling/...?..."
    }
  ]
}
```

`image_base64` tidak disertakan dalam mode bucket, termasuk sebagai `null`. `image_url` dan `markdown_file` ditandatangani untuk operasi `get_object`, berlaku sesuai `BUCKET_PRESIGNED_URL_EXPIRES`, dan bukan tautan permanen. Service ini tidak menyediakan proxy atau refresh URL.

### Mode Base64

Atur `USE_VISION_BUCKET=false`. Mode ini tidak menghubungi object storage, sehingga `markdown_file` bernilai `null`; elemen gambar tetap memakai Base64:

```json
{
  "total_pages": 11,
  "markdown_file": null,
  "elements": [
    {
      "type": "table",
      "page": 4,
      "image_base64": "iVBORw0KGgoAAA..."
    }
  ]
}
```

Jika satu upload atau signing gagal, request tidak mengembalikan sukses parsial dan elemen gambar tidak dihilangkan diam-diam. Status storage:

- `503`: konfigurasi bucket tidak lengkap/tidak valid;
- `502`: kegagalan upstream storage selain timeout;
- `504`: timeout storage setelah retry terbatas.

## Upload PDF dengan curl

Jangan menambahkan header `Content-Type: application/json`. Opsi `--form` membuat multipart boundary yang benar.

Klien Linux:

```bash
curl --location 'https://embedding.aideaspark.com/extract/docling' \
  --form 'file=@"/home/anjas/Downloads/674-Article Text-2557-1-10-20230220.pdf"'
```

Klien Windows:

```bash
curl --location 'https://embedding.aideaspark.com/extract/docling' \
  --form 'file=@"C:/Users/anjas/Downloads/674-Article Text-2557-1-10-20230220.pdf"'
```

Path file mengikuti komputer yang menjalankan curl. Server Linux menerima isi file melalui multipart `UploadFile`; server tidak mencoba membuka path Linux atau Windows milik klien.

## Menjalankan tanpa Docker

Ubuntu/Debian memerlukan library OCR/gambar yang sama dengan Dockerfile, antara lain Tesseract dan library OpenCV:

```bash
sudo apt update
sudo apt install -y build-essential git tesseract-ocr libgl1 libglib2.0-0 \
  libxcb1 libx11-xcb1 libxrender1 libxext6 libxfixes3 libxkbcommon0 \
  libxdamage1 libxrandr2 libxi6

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Development:

```bash
uvicorn main:app --host 0.0.0.0 --port 8003
```

Production foreground (satu worker karena model menggunakan RAM/VRAM besar):

```bash
/path/absolut/embedding-vector/.venv/bin/gunicorn main:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8003 \
  --timeout 600
```

Docling, serialisasi gambar, dan boto3 bersifat blocking dan dijalankan melalui threadpool endpoint. File upload disalin ke direktori sementara sistem menggunakan `tempfile`, file ditutup sebelum Docling membukanya, lalu selalu dihapus pada jalur sukses maupun gagal. Hanya suffix aman dari nama file klien yang dipertahankan agar Docling dapat mengenali format; nama file klien tidak digunakan sebagai path server. `TMPDIR` dihormati. Direktori tersebut harus tersedia, cukup ruang, dan writable oleh user service.

## systemd

Contoh `/etc/systemd/system/embedding-vector.service` (ganti seluruh placeholder dan gunakan path absolut):

```ini
[Unit]
Description=Embedding Vector API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<service-user>
Group=<service-group>
WorkingDirectory=/opt/embedding-vector
EnvironmentFile=/etc/embedding-vector/embedding-vector.env
RuntimeDirectory=embedding-vector
RuntimeDirectoryMode=0750
Environment=TMPDIR=/run/embedding-vector
ExecStart=/opt/embedding-vector/.venv/bin/gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8003 --timeout 600
Restart=on-failure
RestartSec=5
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

`RuntimeDirectory=embedding-vector` membuat `/run/embedding-vector` dan memberi kepemilikan kepada user service, sehingga `TMPDIR` tetap writable walaupun `PrivateTmp=true`. File environment harus hanya dapat dibaca oleh administrator dan user service karena berisi secret. Contoh pemasangan unit (tidak dijalankan otomatis oleh repository ini):

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now embedding-vector.service
sudo systemctl status embedding-vector.service
journalctl -u embedding-vector.service -f
```

## Docker

Buat `.env` dari `.env.example`, isi kredensial, lalu:

```bash
docker compose up -d --build
docker compose logs -f model_api
```

`docker-compose.yml` meneruskan seluruh `.env` sebagai environment container dan menyediakan tmpfs writable melalui `TMPDIR=/tmp/embedding-vector`. Kredensial tidak disalin ke Dockerfile/image. Untuk Docker CLI:

```bash
docker build -t embedding-vector .
docker run -d --name model_api --env-file .env -p 8003:8003 \
  --tmpfs /tmp/embedding-vector:rw,nosuid,nodev,noexec,size=2g \
  -e TMPDIR=/tmp/embedding-vector embedding-vector
```

## Pengujian

Suite menggunakan mock Docling dan mock client S3. Pengujian tidak memerlukan GPU, checkpoint model, atau koneksi ke bucket:

```bash
python -m unittest discover -v
```

Pengujian integrasi storage harus dilakukan terpisah di environment staging, bukan terhadap bucket pengguna saat development:

1. Isi `BUCKET_*` dengan kredensial staging dan endpoint S3 API yang dapat dijangkau.
2. Jalankan service dengan `USE_VISION_BUCKET=true`.
3. Upload dokumen yang didukung Docling dan berisi gambar memakai curl.
4. Pastikan root `markdown_file` dapat di-GET sebelum expiry, berisi Markdown UTF-8 lengkap, dan object bertipe `text/markdown; charset=utf-8`.
5. Pastikan elemen gambar memiliki `image_url`, tidak memiliki `image_base64`, URL dapat di-GET sebelum expiry, object bertipe `image/png`, dan bucket dibuat hanya bila belum ada.
6. Jalankan ulang dengan `USE_VISION_BUCKET=false`; pastikan `markdown_file` bernilai `null`, elemen gambar memiliki `image_base64`, dan tidak ada request ke storage.

## Keamanan deployment

Endpoint API tidak menambahkan skema autentikasi aplikasi baru atau kuota bucket. Pada deployment yang dapat diakses jaringan publik, tempatkan service di belakang reverse proxy/API gateway yang menerapkan autentikasi, batas ukuran multipart, rate limit, timeout, dan batas concurrency. Terapkan quota/lifecycle bucket di MinIO/S3 sesuai kebijakan operasional agar klien tidak dapat menghabiskan disk sementara, CPU OCR, atau kapasitas bucket. Batas tersebut sengaja dikelola pada lapisan deployment dan storage, bukan dengan hardcode di service.

## Batasan dan performa

Mode bucket mengurangi ukuran JSON dan transfer respons API, terutama untuk gambar beresolusi tinggi. Perubahan ini tidak menghilangkan waktu konversi/OCR Docling, serialisasi PNG, upload storage, atau download gambar oleh konsumen. Presigned URL kedaluwarsa; konsumen harus mengambil gambar sebelum expiry. Tidak ada endpoint refresh URL. Upload yang sudah berhasil sebelum kegagalan elemen berikutnya dapat tetap menjadi object privat yang tidak direferensikan oleh respons gagal.
