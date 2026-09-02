# 🚀 UB SR RAG Embedding Microservice (`ub-sr-rag-embedding`)

**UB SR RAG Embedding Microservice** adalah layanan REST API berkinerja tinggi berbasis **FastAPI** dan **PyTorch** yang dirancang untuk mendukung ekosistem **Retrieval-Augmented Generation (RAG)**. Service ini menyediakan endpoint untuk **Dense Embeddings**, **Sparse Embeddings (SPLADE)**, **Reranking (Cross-Encoder)**, serta **Pengekstrak Dokumen & OCR (Docling)**.

---

## 📑 Daftar Isi
1. [Fitur Utama](#-fitur-utama)
2. [Struktur Proyek](#-struktur-proyek)
3. [Prasyarat & Variabel Lingkungan (.env)](#-prasyarat--variabel-lingkungan-env)
4. [Cara Menjalankan: Tanpa Docker (Non-Docker)](#-cara-menjalankan-tanpa-docker-non-docker)
   - [Persiapan Dependensi Sistem & Virtual Environment](#1-persiapan-dependensi-sistem--virtual-environment)
   - [Instalasi PyTorch (CPU vs GPU)](#2-instalasi-pytorch-cpu-vs-gpu)
   - [Menjalankan Server Foreground (Interaktif)](#3-menjalankan-server-foreground-interaktif)
   - [Menjalankan Background dengan `nohup`](#4-menjalankan-background-menggunakan-nohup)
   - [Menjalankan Background dengan `setsid`](#5-menjalankan-background-menggunakan-setsid)
5. [Cara Menjalankan: Menggunakan Docker](#-cara-menjalankan-menggunakan-docker)
   - [Menggunakan Docker Compose (Direkomendasikan)](#1-menggunakan-docker-compose-direkomendasikan)
   - [Menggunakan Docker CLI Langsung](#2-menggunakan-docker-cli-langsung)
6. [Dokumentasi API Endpoints](#-dokumentasi-api-endpoints)
7. [Troubleshooting & Tips Performa](#-troubleshooting--tips-performa)

---

## ✨ Fitur Utama

- **Dense Vector Embedding**: Memproses teks menjadi dense vector menggunakan library `SentenceTransformers` (misal: model `BAAI/bge-m3`).
- **Sparse Vector Embedding (SPLADE)**: Menghasilkan sparse vectors (kombinasi `indices` dan `values`) menggunakan arsitektur SPLADE (`AutoModelForMaskedLM` & `AutoTokenizer`).
- **Rerank Scoring**: Menghitung skor relevansi pasangan query-dokumen menggunakan `CrossEncoder`.
- **Document Parser & OCR (Docling)**: Mengekstrak teks, struktur header, daftar, rumus matematika, dan gambar (format Base64) dari file PDF secara otomatis.
- **Auto Model Preloading**: Mengunduh dan memuat seluruh model ke memori GPU/RAM pada saat server startup guna menghindari latency pada request pertama.
- **Proxy Mode Support**: Opsi untuk meneruskan request ke server model terpisah via HTTP (`MODEL_API_URL`).

---

## 📁 Struktur Proyek

```text
ub-sr-rag-embedding/
├── main.py              # Entrypoint server FastAPI & pemroses HTTP REST endpoints
├── model_registry.py    # Singleton registry untuk Dense, Sparse, Reranker, & Docling
├── requirements.txt     # Dependensi Python
├── Dockerfile           # Konfigurasi Docker image dengan Tesseract OCR & OpenCV libs
├── docker-compose.yml   # Konfigurasi containerization service
└── README.md            # Dokumentasi lengkap proyek
```

---

## ⚙️ Prasyarat & Variabel Lingkungan (`.env`)

Buat file `.env` pada direktori akar proyek. Anda dapat mengonfigurasi model Hugging Face yang ingin digunakan:

```env
# Konfigurasi Model Hugging Face
DENSE_MODEL=BAAI/bge-m3
SPARSE_MODEL=naver/splade-cocondenser-ensembledistil
RERANKER_MODEL=BAAI/bge-reranker-large

# (Opsional) Mode Proxy: Hanya diisi jika service ini berlaku sebagai proxy ke server embedding terpisah
# MODEL_API_URL=http://127.0.0.1:8003
```

### 💡 Cara Load File `.env` Saat Menjalankan Tanpa Docker (Non-Docker)

Terdapat 3 cara bagaimana file `.env` dimuat saat Anda menjalankan aplikasi secara Non-Docker:

1. **Otomatis via Library `python-dotenv` (Direkomendasikan & Default)**
   - Di dalam file `main.py` dan `model_registry.py`, telah ditambahkan fungsi `load_dotenv()`.
   - Saat Anda menjalankan `python main.py` atau via `gunicorn`/`uvicorn`, aplikasi akan **otomatis membaca file `.env`** yang berada di direktori akar proyek secara langsung tanpa perlu perintah tambahan.

2. **Eksplisit via Flag `--env-file` (Jika Menggunakan Uvicorn CLI)**
   - Anda dapat membacakan file `.env` secara langsung lewat opsi terminal CLI uvicorn:
     ```bash
     uvicorn main:app --env-file .env --host 0.0.0.0 --port 8003
     ```

3. **Eksplisit via Linux Shell (`export` / `env`)**
   - Jika ingin me-load seluruh baris `.env` ke environment variabel Linux sebelum menjalankan gunicorn / nohup / setsid:
     ```bash
     # Load .env ke sesi terminal Linux
     export $(grep -v '^#' .env | xargs)

     # Lalu jalankan dengan nohup / setsid
     nohup gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8003 --timeout 600 > app.log 2>&1 &
     ```

---

## 💻 Cara Menjalankan: Tanpa Docker (Non-Docker)

### 1. Persiapan Dependensi Sistem & Virtual Environment

Di sistem operasi Linux (Ubuntu/Debian), beberapa library sistem diperlukan untuk pengolahan gambar (OpenCV/Docling) dan OCR (Tesseract):

```bash
# Update repository dan install dependensi sistem
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-xcb1 \
    libxrender1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxdamage1 \
    libxrandr2 \
    libxi6

# Buat dan aktifkan virtual environment Python
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Instalasi PyTorch (CPU vs GPU)

Pilihlah salah satu sesuai dengan ketersediaan perangkat keras di server lokal Anda:

#### 🟢 Opsi A: CPU Only
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 🚀 Opsi B: Dengan GPU (NVIDIA CUDA Support)
Jika server Anda memiliki GPU NVIDIA dan NVIDIA CUDA Driver sudah terinstall:

```bash
pip install --upgrade pip

# Install PyTorch versi CUDA (contoh: CUDA 12.1 atau CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install sisa dependensi proyek tanpa menimpa versi torch CUDA
pip install fastapi uvicorn gunicorn pydantic python-dotenv requests sentence-transformers rank_bm25 huggingface_hub langgraph langchain_core langchain_aws langchain_openai langchain_community boto3 jinja2 docling pypdf python-multipart Pillow qdrant-client transformers numpy cachetools
```

---

### 3. Menjalankan Server Foreground (Interaktif)

Untuk keperluan uji coba/testing, Anda dapat menjalankan server secara langsung di terminal:

- **Menggunakan Uvicorn (Development Mode)**:
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 8003 --reload
  ```
- **Menggunakan Gunicorn (Production Mode)**:
  ```bash
  gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8003 --timeout 600
  ```
- **Langsung via Python Script**:
  ```bash
  python main.py
  ```

---

### 4. Menjalankan Background menggunakan `nohup`

`nohup` (No Hang Up) memungkinkan proses tetap berjalan di background meskipun sesi SSH atau terminal ditutup.

#### 🚀 Menjalankan Service:
```bash
nohup gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8003 --timeout 600 > app.log 2>&1 &
```

#### 📊 Memantau & Mengelola Proses:
- **Melihat Log secara Real-time**:
  ```bash
  tail -f app.log
  ```
- **Cek Process ID (PID)**:
  ```bash
  ps aux | grep main:app
  ```
- **Menghentikan Service**:
  ```bash
  pkill -f "gunicorn main:app"
  # Atau matikan spesifik PID:
  kill <PID>
  ```

---

### 5. Menjalankan Background menggunakan `setsid`

`setsid` membuat sesi/process group baru sehingga proses terlepas penuh dari terminal pembuka (detaching).

#### 🚀 Menjalankan Service:
```bash
setsid gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8003 --timeout 600 > app.log 2>&1 &
```

#### 📊 Memantau & Mengelola Proses:
- **Melihat Log**:
  ```bash
  tail -f app.log
  ```
- **Mengecek Status Port / Service**:
  ```bash
  ss -tulpn | grep 8003
  ```
- **Menghentikan Service**:
  ```bash
  pkill -f "gunicorn main:app"
  ```

---

## 🐳 Cara Menjalankan: Menggunakan Docker

### 1. Menggunakan Docker Compose (Direkomendasikan)

#### 🟢 Opsi CPU Only:
Secara default, file `docker-compose.yml` dikonfigurasi untuk CPU:

```bash
# Build dan jalankan di background
docker-compose up -d --build

# Cek log aplikasi
docker-compose logs -f model_api
```

#### 🚀 Opsi GPU Supported:
Jika ingin menggunakan GPU NVIDIA di Docker, pastikan `nvidia-container-toolkit` telah terpasang di host server, lalu tambahkan blok `reservations` pada `docker-compose.yml`:

```yaml
services:
  model_api:
    build: .
    container_name: model_api
    command: >
      gunicorn main:app
      -w 1
      -k uvicorn.workers.UvicornWorker
      --bind 0.0.0.0:8003
      --timeout 600
    ports:
      - "8003:8003"
    volumes:
      - ./.env:/app/.env
    env_file:
      - ./.env
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```
Jalankan dengan:
```bash
docker-compose up -d --build
```

---

### 2. Menggunakan Docker CLI Langsung

#### 🟢 Build Docker Image:
```bash
docker build -t ub-sr-rag-embedding .
```

#### 🟢 Jalankan Container (CPU Only):
```bash
docker run -d \
  --name model_api \
  --env-file .env \
  -p 8003:8003 \
  ub-sr-rag-embedding
```

#### 🚀 Jalankan Container (GPU Enabled):
```bash
docker run -d \
  --name model_api \
  --gpus all \
  --env-file .env \
  -p 8003:8003 \
  ub-sr-rag-embedding
```

---

## 📡 Dokumentasi API Endpoints

Setelah server berjalan, Anda dapat mengakses dokumentasi **Interactive Swagger UI** di browser:
👉 **`http://localhost:8003/docs`**

### Ringkasan Endpoints:

| Endpoint | Method | Deskripsi | Example Payload |
|---|---|---|---|
| `/embed/dense` | `POST` | Menghasilkan Dense Vector Embeddings | `{"texts": ["Halo dunia"], "normalize_embeddings": true}` |
| `/embed/sparse/query` | `POST` | Menghasilkan Sparse Query Vector (SPLADE) | `{"text": "kata kunci pencarian"}` |
| `/embed/sparse/passages` | `POST` | Menghasilkan Sparse Passage Vectors Batch | `{"texts": ["Paragraf 1", "Paragraf 2"]}` |
| `/rerank` | `POST` | Menghitung skor relevansi query & passages | `{"query": "Siapa CEO Google?", "texts": ["Sundar Pichai", "Kucing hitam"]}` |
| `/extract/docling` | `POST` | Parser Dokumen PDF + OCR gambar & tabel | `multipart/form-data` dengan parameter `file` |

---

### 📝 Contoh Request & Response

#### 1. Dense Embedding (`POST /embed/dense`)
```bash
curl -X POST "http://localhost:8003/embed/dense" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": ["Kecerdasan Buatan dalam RAG"],
       "normalize_embeddings": true
     }'
```
**Response Sample:**
```json
{
  "vectors": [
    [0.0123, -0.0456, 0.0789, ...]
  ]
}
```

#### 2. Sparse Query Vector (`POST /embed/sparse/query`)
```bash
curl -X POST "http://localhost:8003/embed/sparse/query" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "dokumen RAG"
     }'
```
**Response Sample:**
```json
{
  "indices": [101, 2054, 4832],
  "values": [0.421, 1.892, 0.954]
}
```

#### 3. Reranker (`POST /rerank`)
```bash
curl -X POST "http://localhost:8003/rerank" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Apa itu RAG?",
       "texts": [
         "RAG adalah Retrieval-Augmented Generation.",
         "Resep membuat nasi goreng jawa."
       ]
     }'
```
**Response Sample:**
```json
{
  "scores": [8.4215, -4.1203]
}
```

---

## 💡 Troubleshooting & Tips Performa

1. **Konsumsi Memori GPU/RAM (Out of Memory / OOM)**:
   - Penggunaan worker Gunicorn diset ke `-w 1` secara default karena model ML (Dense + SPLADE + Reranker + Docling) memuat checkpoint berukuran gigabyte ke memori. Menambah jumlah worker akan menduplikasi pemuatan model di setiap worker.
2. **Timeout pada Pengolahan Dokumen PDF (Docling)**:
   - Parameter `--timeout 600` di Gunicorn disiapkan untuk pemrosesan file PDF berhalaman banyak yang membutuhkan OCR tingkat tinggi.
3. **Deteksi Perangkat Keras (CUDA vs CPU)**:
   - Script `model_registry.py` secara otomatis mendeteksi ketersediaan CUDA (`torch.cuda.is_available()`). Jika CUDA terdeteksi, model otomatis dijalankan di GPU (`cuda`).

