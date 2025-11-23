### Anfas Noted : 23/11/2025

- The pip install failure occurs because the `requirements.txt` file specifies `torch==2.2.2+cpu`, but PyPI (the Python Package Index that pip uses) does not distribute PyTorch with the `+cpu` suffix. This suffix is typically used in conda environments or other package managers for CPU-only builds, but pip's available torch versions (as listed in the error) are standard releases without that variant. To resolve this, update `requirements.txt` to use `torch==2.2.2` (which will install the CPU version by default on systems without CUDA) or follow PyTorch's official installation guide for pip, which often recommends commands like `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` for CPU-only installations.
> Reminder : remove torch torchvision torchaudio torchcodec from requirement.txt becuase it wanna install from another source. run : `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` Or `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`

---

### Start for enviornment

database 
```shell
pip install psycopg2-binary
python .\setup_database.py
```

```shell
 python --version
#  Python 3.10.11
```

```shell
python -m venv venv
.\venv\Scripts\activate
```

```shell
pip install -r requirements.txt
```

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

so here the whole package will install with out conflict.

check torch torchvision torchaudio versions

```shell
pip show torch torchvision torchaudio
# Name: torch
# Version: 2.5.1+cpu
# Name: torchvision
# Version: 0.20.1+cpu
# Name: torchaudio
# Version: 2.5.1+cpu
```

if no then 

```shell
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu121
```

--- 


clean data base 

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


```shell
# See what migrations are applied
python manage.py showmigrations diagnosis

# Check database schema
python manage.py dbshell
# Then run: \d diagnosis_analysisresult;
```