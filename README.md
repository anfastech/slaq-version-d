# 🚀 SLAQ AI Speech Diagnosis — Full System Setup Guide

A clean, simplified, and production-ready README for running the full system: Redis, Celery, Django, and AI model downloads.

---

## ⚡ Initial Setup

### Python Environment

> Python version 3.10.11 ⚠️

```sh
python --version
# Python 3.10.11

python -m venv venv
venv\Scripts\activate
```

```sh
# if not, version py 10
# Create a new environment with Python 3.10.11
conda create -n py310 python=3.10.11

# Activate the environment
conda activate py310

# Verify
python --version
```

### Database Setup (development)

```shell
pip install psycopg2-binary
python .\setup_database.py
```

---

## ⚡ Download AI Models (One-time Only)

This step downloads ~3GB of audio AI models.

**Run this command in Terminal 1:**

```sh
# Do not activate virtual environment (❌venv)
pip install transformers torch
python download_model.py
```

Models are stored in:

```
C:\Users\<YOUR_USERNAME>\.cache\huggingface\hub
```

---

## ⚡ Install Required Libraries

```sh
venv\Scripts\activate
pip install -r requirements.txt
```

**Note on PyTorch Installation:**

The `requirements.txt` file may specify `torch==2.2.2+cpu`, but PyPI does not distribute PyTorch with the `+cpu` suffix. To resolve this, remove `torch`, `torchvision`, `torchaudio`, and `torchcodec` from `requirements.txt` and install them separately:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Or for CUDA support (if available):

```sh
pip install torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check installed versions:

```sh
pip show torch torchvision torchaudio
```

If versions are incorrect, uninstall and reinstall:

```sh
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ⚡ Start System Services

### 2.1 🚚 Redis Server

Redis is required for Celery background task processing.

#### Ensure Redis is Installed

Install using the official MSI installer (recommended for Windows):
👉 [https://github.com/microsoftarchive/redis/releases](https://github.com/microsoftarchive/redis/releases)
Download:

```
Redis-x64-3.2.100.msi
```

During installation:

* ✔ Add Redis to PATH
* ✔ Install as a Windows Service

#### Start Redis

```sh
redis-server
```

Or start Windows service:

```sh
net start Redis
```

---

### 2.2 🔁 Celery Worker

Celery executes the AI analysis jobs (audio processing, ML pipeline).

#### Start Celery (Terminal 2)

1. Activate Virtual Environment

```sh
venv\Scripts\activate
```

2. Install Dependencies

```sh
pip install -r requirements.txt
```

3. Run Celery

**⚠️ Windows Limitation:** The `prefork` pool doesn't work on Windows because it requires Unix `fork()` which Windows doesn't support.

**For Windows:**
```sh
# Development (single-threaded)
celery -A slaq_project worker --pool=solo -l info

# Production (multi-threaded with concurrency)
celery -A slaq_project worker --pool=threads --concurrency=4 -l info
```

**For Linux/Unix (Production Recommended):**
```sh
# Production (multi-process with concurrency)
celery -A slaq_project worker --pool=prefork --concurrency=4 -l info
```

> **💡 Production Tip:** Adjust `--concurrency` based on your CPU cores (typically 2-4x CPU cores). For heavy ML tasks, start with 2-4 workers.

---

## ⚡ Start Django Server

The main backend web server.

#### Steps (Terminal 3)

1. Activate Environment

```sh
venv\Scripts\activate
```

2. Install Dependencies

```sh
pip install -r requirements.txt
```

3. Run Migrations

```powershell
# Apply database migrations
python manage.py makemigrations
python manage.py makemigrations core
python manage.py makemigrations diagnosis
python manage.py migrate
```

4. Run Django

```sh
python manage.py runserver
```

Your app is now available at:

```
http://127.0.0.1:8000/
```

---

## 📝 Notes

* Always keep **Redis running** before starting Celery or Django.
* Celery and Django must run in **separate terminals**.
* If models aren't found, delete HuggingFace cache and re-run the download script.
* Keep `requirements.txt` updated.

### Database Cleanup (if needed)

If you encounter migration issues:

```bash
# Delete the problematic migration file
python manage.py migrate diagnosis zero

# Delete the migration file that was just created
del diagnosis\migrations\0002_analysisresult_stutter_frequency_and_more.py

# Create new clean migrations
python manage.py makemigrations diagnosis

# Apply the migrations
python manage.py migrate diagnosis
```

Check migrations and schema:

```shell
# See what migrations are applied
python manage.py showmigrations diagnosis

# Check database schema
python manage.py dbshell
# Then run: \d diagnosis_analysisresult;
```

---

## 🎉 System Overview

| Component              | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| **Redis**              | Message broker for Celery                |
| **Celery Worker**      | Runs background AI analysis              |
| **Django Server**      | Backend API and business logic           |
| **HuggingFace Models** | Speech processing + stuttering detection |

---

## Author ❤️ SLAQ Research AI team

Clean & production-ready setup guide generated with the help of AI.

Ready to deploy. 🚀
