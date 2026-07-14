from pathlib import Path
import re
import shutil
import sys
import py_compile

PROJECT_DIR = Path(__file__).resolve().parent
APP_FILE = PROJECT_DIR / "app.py"
BACKUP_FILE = PROJECT_DIR / "app_backup_sebelum_fix.py"

if not APP_FILE.exists():
    print(f"ERROR: app.py tidak ditemukan di: {APP_FILE}")
    sys.exit(1)

source = APP_FILE.read_text(encoding="utf-8")

replacement = """def numeric_series(series):
    \"\"\"
    Mengubah kolom angka menjadi numerik.
    Mendukung angka biasa, persen, dan format Indonesia.
    \"\"\"
    values = series.astype(str).str.strip()
    values = values.str.replace("%", "", regex=False)

    def parse_value(value):
        value = str(value).strip()

        if value == "" or value.lower() in {"nan", "none"}:
            return np.nan

        if "," in value and "." in value:
            value = value.replace(".", "")
            value = value.replace(",", ".")
        elif "," in value:
            value = value.replace(",", ".")

        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    return values.apply(parse_value)
"""

pattern = (
    r"def numeric_series\(series\):"
    r"[\s\S]*?"
    r"(?=\n\ndef find_existing_file)"
)

updated, count = re.subn(
    pattern,
    replacement.rstrip(),
    source,
    count=1
)

if count != 1:
    print("ERROR: Fungsi numeric_series() tidak ditemukan.")
    print("Tidak ada perubahan yang dilakukan.")
    sys.exit(1)

shutil.copy2(APP_FILE, BACKUP_FILE)
APP_FILE.write_text(updated, encoding="utf-8")

try:
    py_compile.compile(str(APP_FILE), doraise=True)
except Exception as error:
    shutil.copy2(BACKUP_FILE, APP_FILE)
    print("ERROR: app.py masih gagal diperiksa.")
    print(error)
    print("File asli telah dikembalikan dari backup.")
    sys.exit(1)

print("=" * 60)
print("BERHASIL!")
print(f"File diperbaiki : {APP_FILE}")
print(f"File backup     : {BACKUP_FILE}")
print("Syntax app.py   : OK")
print("=" * 60)
print("Lanjutkan dengan:")
print("git add app.py")
print('git commit -m "fix indentation app py"')
print("git push origin master")