from pathlib import Path
import csv
import gc
import io
import os
import re

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix
from statsmodels.tsa.arima.model import ARIMA


app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY",
    "skripsi_apriori_timeseries"
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024



BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_HASIL_APRIORI = OUTPUT_DIR / "hasil_apriori.csv"
FILE_PREDIKSI_HARIAN = OUTPUT_DIR / "prediksi_arima_harian.csv"
FILE_PREDIKSI_RINGKASAN = OUTPUT_DIR / "prediksi_arima_90_hari.csv"
FILE_REKOMENDASI = OUTPUT_DIR / "rekomendasi_stok.csv"

ALLOWED_EXTENSIONS = {"csv"}

MIN_SUPPORT_DEFAULT = 0.01
MIN_CONFIDENCE_DEFAULT = 0.10
MIN_LIFT = 1.0

TOP_PRODUCTS = 50
FORECAST_HORIZON = 90
ARIMA_ORDER = (2, 1, 2)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def read_csv_flexible(source):
    """
    Membaca CSV dengan pemisah koma, titik koma, atau tab secara otomatis.
    """
    return pd.read_csv(
        source,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        on_bad_lines="skip"
    )


def read_apriori_csv(file_storage):
    """
    Membaca CSV khusus untuk proses Apriori.

    Semua kolom dibaca sebagai teks terlebih dahulu agar nama produk,
    merek, atau nilai teks seperti "Bustong" tidak dipaksa menjadi angka.
    Fungsi ini hanya mengambil Transaction_ID dan Product_Name sehingga
    proses Apriori lebih ringan dan tidak memengaruhi proses Time Series.
    """
    if file_storage is None:
        raise ValueError("File CSV Apriori belum dipilih.")

    if not getattr(file_storage, "filename", ""):
        raise ValueError("Nama file CSV Apriori tidak ditemukan.")

    file_storage.stream.seek(0)
    raw_data = file_storage.stream.read()

    if not raw_data:
        raise ValueError("File CSV Apriori kosong.")

    decoded_text = None
    selected_encoding = None

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            decoded_text = raw_data.decode(encoding)
            selected_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        raise ValueError(
            "Encoding file tidak didukung. "
            "Simpan ulang file sebagai CSV UTF-8."
        )

   
    sample = decoded_text[:10000]

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters=",;\t|"
        )
        separator = dialect.delimiter
    except csv.Error:
        first_line = next(
            (
                line
                for line in decoded_text.splitlines()
                if line.strip()
            ),
            ""
        )
        candidates = [",", ";", "\t", "|"]
        separator = max(
            candidates,
            key=lambda item: first_line.count(item)
        )

        if first_line.count(separator) == 0:
            separator = ","

    read_options = {
        "sep": separator,
        "dtype": str,
        "keep_default_na": False,
        "on_bad_lines": "skip"
    }

    try:
        data = pd.read_csv(
            io.StringIO(decoded_text),
            engine="c",
            **read_options
        )
    except Exception:
        # Fallback untuk CSV dengan struktur yang tidak didukung parser C.
        data = pd.read_csv(
            io.StringIO(decoded_text),
            engine="python",
            **read_options
        )

    if data.empty:
        raise ValueError("File CSV Apriori tidak memiliki baris data.")

    data = normalize_columns(data)

    transaction_col = find_column(
        data,
        [
            "Transaction_ID",
            "Transaction ID",
            "Kode Transaksi",
            "ID Transaksi",
            "TransactionID"
        ]
    )
    product_col = find_column(
        data,
        [
            "Product_Name",
            "Product Name",
            "Nama Produk",
            "Produk",
            "ProductName"
        ]
    )

    missing_columns = []

    if not transaction_col:
        missing_columns.append("Transaction_ID")

    if not product_col:
        missing_columns.append("Product_Name")

    if missing_columns:
        available_columns = ", ".join(
            data.columns.astype(str).tolist()
        )
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(missing_columns)
            + ". Kolom yang terbaca: "
            + available_columns
        )

    clean_data = data[
        [transaction_col, product_col]
    ].copy()
    clean_data.columns = [
        "Transaction_ID",
        "Product_Name"
    ]

    clean_data["Transaction_ID"] = (
        clean_data["Transaction_ID"]
        .astype(str)
        .str.strip()
    )
    clean_data["Product_Name"] = (
        clean_data["Product_Name"]
        .astype(str)
        .str.strip()
    )

    clean_data = clean_data[
        (clean_data["Transaction_ID"] != "")
        & (clean_data["Product_Name"] != "")
    ].copy()


    clean_data = clean_data.drop_duplicates(
        subset=[
            "Transaction_ID",
            "Product_Name"
        ]
    ).reset_index(drop=True)

    if clean_data.empty:
        raise ValueError(
            "Tidak ada transaksi valid yang dapat dianalisis."
        )

    app.logger.info(
        "CSV Apriori dibaca | file=%s | encoding=%s | "
        "separator=%r | baris=%s | transaksi=%s | produk=%s",
        file_storage.filename,
        selected_encoding,
        separator,
        len(clean_data),
        clean_data["Transaction_ID"].nunique(),
        clean_data["Product_Name"].nunique()
    )

    return clean_data


def normalize_columns(data):
    data = data.copy()

    data.columns = [
        re.sub(r"\s+", " ", str(column).strip())
        for column in data.columns
    ]

    return data


def normalize_key(value):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower()
    ).strip("_")


def find_column(data, candidates):
    normalized_columns = {
        normalize_key(column): column
        for column in data.columns
    }

    for candidate in candidates:
        key = normalize_key(candidate)

        if key in normalized_columns:
            return normalized_columns[key]

    return None


def numeric_series(series):
    """
    Mengubah kolom angka menjadi numerik.
    Cocok untuk Qty, support, confidence, dan hasil prediksi.
    """
    values = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    
    contains_comma = values.str.contains(",", regex=False, na=False)
    contains_dot = values.str.contains(".", regex=False, na=False)

    indonesia_mask = contains_comma & contains_dot

    values.loc[indonesia_mask] = (
        values.loc[indonesia_mask]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    values.loc[contains_comma & ~contains_dot] = (
        values.loc[contains_comma & ~contains_dot]
        .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(values, errors="coerce")


def find_existing_file(filename):
    """
    Mencari file pada folder output, folder utama, dan folder data.
    """
    locations = [
        OUTPUT_DIR / filename,
        BASE_DIR / filename,
        DATA_DIR / filename
    ]

    for path in locations:
        if path.exists() and path.is_file():
            return path

    return None



def get_evaluation_data():
    """
    Nilai evaluasi berdasarkan hasil pengolahan offline penelitian.
    """
    return [
        {
            "rank": 1,
            "model": "ARIMA",
            "total_products": 50,
            "mae": 42.46,
            "rmse": 76.86,
            "mape": 102.14,
            "status": "Model Terbaik"
        },
        {
            "rank": 2,
            "model": "SARIMA",
            "total_products": 50,
            "mae": 53.98,
            "rmse": 84.78,
            "mape": 173.76,
            "status": "Model Pembanding"
        },
        {
            "rank": 3,
            "model": "Prophet",
            "total_products": 50,
            "mae": 110.42,
            "rmse": 129.56,
            "mape": 497.01,
            "status": "Model Pembanding"
        }
    ]


def validate_timeseries_dataset(data):
    data = normalize_columns(data)

    date_col = find_column(
        data,
        ["Date", "Tanggal", "Transaction_Date"]
    )
    product_col = find_column(
        data,
        [
            "Product_Name",
            "Product Name",
            "Produk",
            "Nama Produk"
        ]
    )
    qty_col = find_column(
        data,
        [
            "Qty",
            "Quantity",
            "Jumlah",
            "Jumlah Terjual"
        ]
    )

    missing_columns = []

    if not date_col:
        missing_columns.append("Date")

    if not product_col:
        missing_columns.append("Product_Name")

    if not qty_col:
        missing_columns.append("Qty")

    if missing_columns:
        raise ValueError(
            "Dataset harus memiliki kolom: "
            + ", ".join(missing_columns)
            + "."
        )

    clean_data = data[
        [date_col, product_col, qty_col]
    ].copy()

    clean_data.columns = [
        "Date",
        "Product_Name",
        "Qty"
    ]

    clean_data["Date"] = pd.to_datetime(
        clean_data["Date"],
        errors="coerce",
        dayfirst=True
    )

    clean_data["Product_Name"] = (
        clean_data["Product_Name"]
        .astype(str)
        .str.strip()
    )

    clean_data["Qty"] = numeric_series(
        clean_data["Qty"]
    )

    clean_data = clean_data.dropna(
        subset=[
            "Date",
            "Product_Name",
            "Qty"
        ]
    )

    clean_data = clean_data[
        clean_data["Product_Name"] != ""
    ].copy()

    clean_data = clean_data[
        clean_data["Qty"] >= 0
    ].copy()

    if clean_data.empty:
        raise ValueError(
            "Dataset tidak memiliki data valid yang dapat diproses."
        )

    return clean_data


def forecast_one_product(product_series):
    """
    Menjalankan ARIMA (2,1,2) untuk satu produk.
    Jika model gagal, sistem menggunakan rata-rata historis agar produk
    tetap memiliki hasil prediksi.
    """
    if product_series.empty:
        return np.zeros(FORECAST_HORIZON)

    if len(product_series) < 10:
        mean_value = max(
            float(product_series.mean()),
            0
        )

        return np.repeat(
            mean_value,
            FORECAST_HORIZON
        )

    try:
        model = ARIMA(
            product_series,
            order=ARIMA_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit()

        prediction = fitted_model.forecast(
            steps=FORECAST_HORIZON
        )

        prediction = np.asarray(
            prediction,
            dtype=float
        )

    except Exception:
        mean_value = max(
            float(product_series.mean()),
            0
        )

        prediction = np.repeat(
            mean_value,
            FORECAST_HORIZON
        )

    prediction = np.nan_to_num(
        prediction,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    prediction = np.clip(
        prediction,
        0,
        None
    )

    return prediction


def process_timeseries(data):
    """
    Memproses seluruh produk atau maksimal 50 produk terlaris.
    Hasil berisi prediksi harian 90 hari untuk setiap produk dan
    ringkasan total 90 hari.
    """
    clean_data = validate_timeseries_dataset(data)

    product_totals = (
        clean_data.groupby("Product_Name")["Qty"]
        .sum()
        .sort_values(ascending=False)
    )

    product_list = (
        product_totals
        .head(TOP_PRODUCTS)
        .index
        .tolist()
    )

    if not product_list:
        raise ValueError(
            "Tidak ada produk yang dapat diprediksi."
        )

    filtered_data = clean_data[
        clean_data["Product_Name"].isin(product_list)
    ].copy()

    daily_rows = []
    summary_rows = []
    failed_products = []

    for product_number, product_name in enumerate(
        product_list,
        start=1
    ):
        try:
            product_history = filtered_data[
                filtered_data["Product_Name"]
                == product_name
            ].copy()

            product_daily = (
                product_history
                .groupby("Date")["Qty"]
                .sum()
                .sort_index()
            )

            product_dates = pd.date_range(
                start=product_daily.index.min(),
                end=product_daily.index.max(),
                freq="D"
            )

            product_series = (
                product_daily
                .reindex(
                    product_dates,
                    fill_value=0
                )
                .astype(float)
            )

            prediction = forecast_one_product(
                product_series
            )

            future_dates = pd.date_range(
                start=product_series.index.max()
                + pd.Timedelta(days=1),
                periods=FORECAST_HORIZON,
                freq="D"
            )

            
            rounded_prediction = np.round(
                prediction,
                2
            )

            for forecast_date, forecast_value in zip(
                future_dates,
                rounded_prediction
            ):
                daily_rows.append({
                    "Tanggal": forecast_date,
                    "Product_Name": product_name,
                    "Prediksi_ARIMA": float(
                        forecast_value
                    )
                })

            summary_rows.append({
                "No": product_number,
                "Product_Name": product_name,
                "Prediksi_90_Hari": round(
                    float(
                        rounded_prediction.sum()
                    ),
                    2
                )
            })

        except Exception as error:
            failed_products.append({
                "product_name": product_name,
                "error": str(error)
            })

    if not summary_rows:
        raise ValueError(
            "Tidak ada produk yang berhasil diprediksi."
        )

    daily_result = pd.DataFrame(daily_rows)
    summary_result = pd.DataFrame(summary_rows)

    daily_result = daily_result.sort_values(
        ["Product_Name", "Tanggal"]
    ).reset_index(drop=True)

    summary_result = summary_result.reset_index(
        drop=True
    )

    summary_result["No"] = range(
        1,
        len(summary_result) + 1
    )

    return {
        "harian": daily_result,
        "ringkasan": summary_result,
        "produk_gagal": failed_products
    }


def build_timeseries_context(
    daily_data,
    summary_data,
    selected_product=None,
    failed_products=None,
    success_message=None
):
    """
    Menyiapkan seluruh prediksi semua produk untuk halaman Time Series.
    Pergantian produk dilakukan langsung di browser tanpa upload ulang.
    """
    daily_data = normalize_columns(daily_data)
    summary_data = normalize_columns(summary_data)

    daily_date_col = find_column(
        daily_data,
        ["Tanggal", "Date"]
    )
    daily_product_col = find_column(
        daily_data,
        ["Product_Name", "Product Name"]
    )
    daily_prediction_col = find_column(
        daily_data,
        ["Prediksi_ARIMA", "Prediksi ARIMA"]
    )

    summary_no_col = find_column(
        summary_data,
        ["No"]
    )
    summary_product_col = find_column(
        summary_data,
        ["Product_Name", "Product Name"]
    )
    summary_prediction_col = find_column(
        summary_data,
        ["Prediksi_90_Hari", "Prediksi 90 Hari"]
    )

    if (
        not daily_date_col
        or not daily_product_col
        or not daily_prediction_col
    ):
        raise ValueError(
            "File prediksi harian tidak memiliki struktur yang benar."
        )

    if (
        not summary_product_col
        or not summary_prediction_col
    ):
        raise ValueError(
            "File ringkasan prediksi tidak memiliki struktur yang benar."
        )

    daily_data[daily_date_col] = pd.to_datetime(
        daily_data[daily_date_col],
        errors="coerce",
        dayfirst=True
    )

    daily_data[daily_product_col] = (
        daily_data[daily_product_col]
        .astype(str)
        .str.strip()
    )

    daily_data[daily_prediction_col] = numeric_series(
        daily_data[daily_prediction_col]
    )

    daily_data = daily_data.dropna(
        subset=[
            daily_date_col,
            daily_product_col,
            daily_prediction_col
        ]
    )

    summary_data[summary_product_col] = (
        summary_data[summary_product_col]
        .astype(str)
        .str.strip()
    )

    summary_data[summary_prediction_col] = numeric_series(
        summary_data[summary_prediction_col]
    )

    summary_data = summary_data.dropna(
        subset=[
            summary_product_col,
            summary_prediction_col
        ]
    ).reset_index(drop=True)

    if summary_data.empty:
        raise ValueError(
            "Ringkasan prediksi tidak memiliki data valid."
        )

    product_list = (
        summary_data[summary_product_col]
        .drop_duplicates()
        .tolist()
    )

    requested_product = (
        str(selected_product).strip()
        if selected_product is not None
        else ""
    )

    if requested_product not in product_list:
        requested_product = product_list[0]

    selected_product = requested_product

   
    all_forecast_records = []

    sorted_daily_data = daily_data.sort_values(
        [daily_product_col, daily_date_col]
    )

    for _, row in sorted_daily_data.iterrows():
        all_forecast_records.append({
            "date": row[daily_date_col].strftime("%d-%m-%Y"),
            "product_name": str(row[daily_product_col]),
            "arima": round(
                float(row[daily_prediction_col]),
                2
            )
        })

    summary_records = []

    for index, row in summary_data.iterrows():
        if (
            summary_no_col
            and pd.notna(row[summary_no_col])
        ):
            try:
                row_number = int(
                    float(row[summary_no_col])
                )
            except (TypeError, ValueError):
                row_number = index + 1
        else:
            row_number = index + 1

        summary_records.append({
            "no": row_number,
            "product_name": str(
                row[summary_product_col]
            ),
            "prediksi_90_hari": round(
                float(row[summary_prediction_col]),
                2
            )
        })

    selected_summary = next(
        (
            item
            for item in summary_records
            if item["product_name"] == selected_product
        ),
        None
    )

    selected_daily = [
        item
        for item in all_forecast_records
        if item["product_name"] == selected_product
    ]

    selected_total = (
        selected_summary["prediksi_90_hari"]
        if selected_summary
        else 0
    )

    selected_daily_total = round(
        sum(item["arima"] for item in selected_daily),
        2
    )

    return {
        "forecast_data": all_forecast_records,
        "summary_data": summary_records,
        "product_list": product_list,
        "selected_product": selected_product,
        "total_products": len(product_list),
        "prediksi_total_terpilih": selected_total,
        "jumlah_harian": selected_daily_total,
        "selisih_total": round(
            selected_total - selected_daily_total,
            2
        ),
        "produk_gagal": failed_products or [],
        "evaluation_data": get_evaluation_data(),
        "error_message": None,
        "success_message": success_message
    }


def empty_timeseries_context(error_message=None):
    return {
        "forecast_data": [],
        "summary_data": [],
        "product_list": [],
        "selected_product": "",
        "total_products": 0,
        "prediksi_total_terpilih": 0,
        "jumlah_harian": 0,
        "selisih_total": 0,
        "produk_gagal": [],
        "evaluation_data": get_evaluation_data(),
        "error_message": error_message,
        "success_message": None
    }


def load_saved_timeseries_context(
    selected_product=None,
    success_message=None
):
    daily_file = find_existing_file(
        "prediksi_arima_harian.csv"
    )
    summary_file = find_existing_file(
        "prediksi_arima_90_hari.csv"
    )

    if not daily_file or not summary_file:
        return None

    daily_data = read_csv_flexible(
        daily_file
    )
    summary_data = read_csv_flexible(
        summary_file
    )

    return build_timeseries_context(
        daily_data,
        summary_data,
        selected_product=selected_product,
        success_message=success_message
    )


def process_apriori(
    data,
    min_support,
    min_confidence
):
    """
    Menjalankan Apriori dengan market basket sparse.

    Versi ini tidak memakai groupby().agg(list), sehingga lebih cepat
    dan lebih hemat memori untuk dataset transaksi berukuran besar.
    Karena hasil akhir aplikasi hanya menggunakan aturan satu produk
    menuju satu produk, pencarian itemset dibatasi sampai panjang 2.
    """
    data = normalize_columns(data)

    transaction_col = find_column(
        data,
        [
            "Transaction_ID",
            "Transaction ID",
            "Kode Transaksi"
        ]
    )
    product_col = find_column(
        data,
        [
            "Product_Name",
            "Product Name",
            "Nama Produk"
        ]
    )

    if not transaction_col or not product_col:
        raise ValueError(
            "Dataset Apriori harus memiliki kolom "
            "Transaction_ID dan Product_Name."
        )

    # Hanya ambil dua kolom yang benar-benar dibutuhkan Apriori.
    clean_data = data[
        [transaction_col, product_col]
    ].copy()
    clean_data.columns = [
        "Transaction_ID",
        "Product_Name"
    ]

    clean_data = clean_data.dropna(
        subset=[
            "Transaction_ID",
            "Product_Name"
        ]
    )

    clean_data["Transaction_ID"] = (
        clean_data["Transaction_ID"]
        .astype(str)
        .str.strip()
    )
    clean_data["Product_Name"] = (
        clean_data["Product_Name"]
        .astype(str)
        .str.strip()
    )

    clean_data = clean_data[
        (clean_data["Transaction_ID"] != "")
        & (clean_data["Product_Name"] != "")
    ].drop_duplicates(
        subset=[
            "Transaction_ID",
            "Product_Name"
        ]
    ).reset_index(drop=True)

    if clean_data.empty:
        raise ValueError(
            "Dataset Apriori tidak memiliki transaksi valid."
        )

    transaction_codes, transaction_names = pd.factorize(
        clean_data["Transaction_ID"],
        sort=False
    )
    product_codes, product_names = pd.factorize(
        clean_data["Product_Name"],
        sort=False
    )

    total_transactions = len(transaction_names)
    total_products = len(product_names)

    if total_transactions == 0 or total_products == 0:
        raise ValueError(
            "Keranjang transaksi tidak berhasil dibentuk."
        )

    # Bentuk matriks transaksi-produk langsung dalam format CSR sparse.
    encoded_sparse = csr_matrix(
        (
            np.ones(
                len(clean_data),
                dtype=np.uint8
            ),
            (transaction_codes, product_codes)
        ),
        shape=(
            total_transactions,
            total_products
        ),
        dtype=np.uint8
    )

    encoded_sparse.data[:] = 1
    encoded_sparse = encoded_sparse.astype(bool)

    basket_sets = pd.DataFrame.sparse.from_spmatrix(
        encoded_sparse,
        columns=product_names.astype(str)
    )

    app.logger.info(
        "Market basket sparse | transaksi=%s | produk=%s | bentuk=%s",
        total_transactions,
        total_products,
        basket_sets.shape
    )

    frequent_itemsets = apriori(
        basket_sets,
        min_support=min_support,
        use_colnames=True,
        max_len=2,
        low_memory=False
    )

    if frequent_itemsets.empty:
        pd.DataFrame(
            columns=[
                "Antecedent",
                "Consequent",
                "Support (%)",
                "Confidence (%)",
                "Lift"
            ]
        ).to_csv(
            FILE_HASIL_APRIORI,
            sep=";",
            index=False,
            encoding="utf-8-sig"
        )

        del basket_sets
        del encoded_sparse
        del clean_data
        gc.collect()

        return {
            "rules_data": [],
            "total_transactions": total_transactions,
            "total_products": total_products,
            "total_rules": 0
        }

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if not rules.empty:
        rules = rules[
            (rules["support"] >= min_support)
            & (rules["confidence"] >= min_confidence)
            & (rules["lift"] > MIN_LIFT)
            & (rules["antecedents"].apply(len) == 1)
            & (rules["consequents"].apply(len) == 1)
        ].copy()

    if not rules.empty:
        rules = rules.sort_values(
            by=[
                "confidence",
                "lift",
                "support"
            ],
            ascending=[
                False,
                False,
                False
            ]
        ).reset_index(drop=True)

    rules_formatted = []
    output_rows = []

    for _, row in rules.iterrows():
        antecedent = next(
            iter(row["antecedents"])
        )
        consequent = next(
            iter(row["consequents"])
        )

        rules_formatted.append({
            "antecedent": antecedent,
            "consequent": consequent,
            "support": round(
                float(row["support"]),
                4
            ),
            "confidence": round(
                float(row["confidence"]),
                4
            ),
            "lift": round(
                float(row["lift"]),
                2
            )
        })

        output_rows.append({
            "Antecedent": antecedent,
            "Consequent": consequent,
            "Support (%)": round(
                float(row["support"]) * 100,
                2
            ),
            "Confidence (%)": round(
                float(row["confidence"]) * 100,
                2
            ),
            "Lift": round(
                float(row["lift"]),
                2
            )
        })

    pd.DataFrame(
        output_rows,
        columns=[
            "Antecedent",
            "Consequent",
            "Support (%)",
            "Confidence (%)",
            "Lift"
        ]
    ).to_csv(
        FILE_HASIL_APRIORI,
        sep=";",
        index=False,
        encoding="utf-8-sig"
    )

    app.logger.info(
        "Apriori selesai | frequent_itemsets=%s | aturan=%s",
        len(frequent_itemsets),
        len(rules_formatted)
    )

    del basket_sets
    del encoded_sparse
    del clean_data
    del frequent_itemsets
    del rules
    gc.collect()

    return {
        "rules_data": rules_formatted,
        "total_transactions": total_transactions,
        "total_products": total_products,
        "total_rules": len(rules_formatted)
    }


def prepare_recommendation_data(data):
    data = normalize_columns(data)

    no_col = find_column(data, ["No"])
    product_col = find_column(
        data,
        [
            "Product Name",
            "Product_Name",
            "Produk"
        ]
    )
    prediction_col = find_column(
        data,
        [
            "Prediksi 90 Hari",
            "Prediksi_90_Hari"
        ]
    )
    category_col = find_column(
        data,
        [
            "Kategori Prediksi",
            "Kategori_Prediksi"
        ]
    )
    pair_col = find_column(
        data,
        [
            "Produk_Pasangan",
            "Produk Pasangan"
        ]
    )
    support_col = find_column(
        data,
        ["Support (%)", "Support"]
    )
    confidence_col = find_column(
        data,
        [
            "Confidence (%)",
            "Confidence"
        ]
    )
    lift_col = find_column(
        data,
        ["Lift"]
    )
    relation_col = find_column(
        data,
        [
            "Hubungan Apriori",
            "Hubungan_Apriori"
        ]
    )
    recommendation_col = find_column(
        data,
        ["Rekomendasi"]
    )

    required_columns = {
        "Product Name": product_col,
        "Prediksi 90 Hari": prediction_col,
        "Kategori Prediksi": category_col,
        "Produk Pasangan": pair_col,
        "Hubungan Apriori": relation_col,
        "Rekomendasi": recommendation_col
    }

    missing_columns = [
        name
        for name, column
        in required_columns.items()
        if not column
    ]

    if missing_columns:
        raise ValueError(
            "Kolom rekomendasi tidak lengkap: "
            + ", ".join(missing_columns)
        )

    recommendation_data = []

    for index, row in data.iterrows():
        def clean_optional(column):
            if (
                not column
                or pd.isna(row[column])
            ):
                return None

            value = str(row[column]).strip()

            if (
                value == ""
                or value.lower() == "nan"
            ):
                return None

            return value

        if (
            no_col
            and pd.notna(row[no_col])
        ):
            try:
                row_number = int(
                    float(row[no_col])
                )
            except (TypeError, ValueError):
                row_number = index + 1
        else:
            row_number = index + 1

        recommendation_data.append({
            "no": row_number,
            "product_name": str(
                row[product_col]
            ).strip(),
            "prediksi_90_hari": clean_optional(
                prediction_col
            ),
            "kategori_prediksi": str(
                row[category_col]
            ).strip(),
            "produk_pasangan": (
                clean_optional(pair_col)
                or "-"
            ),
            "support": clean_optional(
                support_col
            ),
            "confidence": clean_optional(
                confidence_col
            ),
            "lift": clean_optional(
                lift_col
            ),
            "hubungan_apriori": str(
                row[relation_col]
            ).strip(),
            "rekomendasi": str(
                row[recommendation_col]
            ).strip()
        })

    return {
        "recommendation_data": recommendation_data,
        "total_items": len(
            recommendation_data
        ),
        "count_high": sum(
            item["kategori_prediksi"]
            == "Tinggi"
            for item in recommendation_data
        ),
        "count_medium": sum(
            item["kategori_prediksi"]
            == "Sedang"
            for item in recommendation_data
        ),
        "count_low": sum(
            item["kategori_prediksi"]
            == "Rendah"
            for item in recommendation_data
        ),
        "count_strong": sum(
            item["hubungan_apriori"]
            == "Kuat"
            for item in recommendation_data
        ),
        "count_not_strong": sum(
            item["hubungan_apriori"]
            == "Tidak kuat"
            for item in recommendation_data
        )
    }


@app.route("/")
def dashboard():
    return render_template(
        "index.html"
    )

@app.route(
    "/apriori",
    methods=["GET", "POST"]
)
def apriori_route():
    """
    Menampilkan halaman Apriori dan memproses dataset CSV
    yang diunggah oleh pengguna.
    """

    min_support = MIN_SUPPORT_DEFAULT
    min_confidence = MIN_CONFIDENCE_DEFAULT

    if request.method == "GET":
        return render_template(
            "apriori.html",
            rules_data=None,
            total_transactions=0,
            total_products=0,
            total_rules=0,
            min_support=min_support,
            min_confidence=min_confidence,
            error_message=None
        )

    try:

        uploaded_file = request.files.get("file")

        if uploaded_file is None:
            raise ValueError(
                "File dataset Apriori tidak ditemukan."
            )

        if uploaded_file.filename == "":
            raise ValueError(
                "Pilih file dataset Apriori terlebih dahulu."
            )


        if not allowed_file(uploaded_file.filename):
            raise ValueError(
                "Dataset Apriori harus berformat CSV."
            )

        min_support_input = request.form.get(
            "min_support",
            MIN_SUPPORT_DEFAULT
        )

        min_confidence_input = request.form.get(
            "min_confidence",
            MIN_CONFIDENCE_DEFAULT
        )

        try:
            min_support = float(min_support_input)
        except (TypeError, ValueError):
            raise ValueError(
                "Minimum support harus berupa angka."
            )

        try:
            min_confidence = float(min_confidence_input)
        except (TypeError, ValueError):
            raise ValueError(
                "Minimum confidence harus berupa angka."
            )


        if not 0 < min_support <= 1:
            raise ValueError(
                "Minimum support harus berada pada "
                "rentang lebih dari 0 sampai 1."
            )

        if not 0 < min_confidence <= 1:
            raise ValueError(
                "Minimum confidence harus berada pada "
                "rentang lebih dari 0 sampai 1."
            )

        data = read_apriori_csv(
            uploaded_file
        )

        if data.empty:
            raise ValueError(
                "Dataset tidak memiliki data yang dapat diproses."
            )

        app.logger.info(
            "Memulai Apriori | file=%s | baris=%s | "
            "transaksi=%s | produk=%s | support=%s | "
            "confidence=%s",
            uploaded_file.filename,
            len(data),
            data["Transaction_ID"].nunique(),
            data["Product_Name"].nunique(),
            min_support,
            min_confidence
        )


        context = process_apriori(
            data=data,
            min_support=min_support,
            min_confidence=min_confidence
        )
        
        if not isinstance(context, dict):
            raise ValueError(
                "Hasil proses Apriori tidak memiliki "
                "format yang benar."
            )


        context.setdefault(
            "rules_data",
            []
        )

        context.setdefault(
            "total_transactions",
            data["Transaction_ID"].nunique()
        )

        context.setdefault(
            "total_products",
            data["Product_Name"].nunique()
        )

        context.setdefault(
            "total_rules",
            len(context["rules_data"])
        )


        app.logger.info(
            "Apriori berhasil | transaksi=%s | "
            "produk=%s | aturan=%s",
            context["total_transactions"],
            context["total_products"],
            context["total_rules"]
        )

        return render_template(
            "apriori.html",
            **context,
            min_support=min_support,
            min_confidence=min_confidence,
            error_message=None
        )



    except ValueError as error:
        app.logger.warning(
            "Validasi Apriori gagal: %s",
            error
        )

        return render_template(
            "apriori.html",
            rules_data=None,
            total_transactions=0,
            total_products=0,
            total_rules=0,
            min_support=min_support,
            min_confidence=min_confidence,
            error_message=str(error)
        )


    except Exception as error:
        app.logger.exception(
            "Terjadi kesalahan saat menjalankan Apriori."
        )

        return render_template(
            "apriori.html",
            rules_data=None,
            total_transactions=0,
            total_products=0,
            total_rules=0,
            min_support=min_support,
            min_confidence=min_confidence,
            error_message=(
                "Terjadi kesalahan saat memproses dataset. "
                "Silakan periksa struktur file CSV dan coba kembali. "
                f"Detail: {error}"
            )
        )


@app.route(
    "/timeseries",
    methods=["GET", "POST"]
)
def timeseries():
    if request.method == "POST":
        try:
            uploaded_file = request.files.get(
                "file"
            )

            if (
                not uploaded_file
                or uploaded_file.filename == ""
            ):
                raise ValueError(
                    "Pilih file dataset Time Series terlebih dahulu."
                )

            if not allowed_file(
                uploaded_file.filename
            ):
                raise ValueError(
                    "Dataset Time Series harus berformat CSV."
                )

            raw_data = read_csv_flexible(
                uploaded_file
            )

            result = process_timeseries(
                raw_data
            )

            daily_result = result["harian"]
            summary_result = result["ringkasan"]

            daily_output = daily_result.copy()

            daily_output["Tanggal"] = (
                pd.to_datetime(
                    daily_output["Tanggal"]
                )
                .dt.strftime("%d-%m-%Y")
            )

            daily_output.to_csv(
                FILE_PREDIKSI_HARIAN,
                index=False,
                encoding="utf-8-sig"
            )

            summary_result.to_csv(
                FILE_PREDIKSI_RINGKASAN,
                index=False,
                encoding="utf-8-sig"
            )

            first_product = str(
                summary_result.iloc[0][
                    "Product_Name"
                ]
            )

            return redirect(
                url_for(
                    "timeseries",
                    product_name=first_product,
                    uploaded="1"
                )
            )

        except Exception as error:
            return render_template(
                "timeseries.html",
                **empty_timeseries_context(
                    error_message=str(error)
                )
            )

    selected_product = request.args.get(
        "product_name",
        default=None,
        type=str
    )

    success_message = None

    if request.args.get("uploaded") == "1":
        success_message = (
            "Prediksi seluruh produk berhasil diproses. "
            "Pilih produk melalui dropdown untuk mengubah grafik."
        )

    try:
        saved_context = load_saved_timeseries_context(
            selected_product=selected_product,
            success_message=success_message
        )

        if saved_context:
            return render_template(
                "timeseries.html",
                **saved_context
            )

    except Exception as error:
        return render_template(
            "timeseries.html",
            **empty_timeseries_context(
                error_message=str(error)
            )
        )

    return render_template(
        "timeseries.html",
        **empty_timeseries_context()
    )


@app.route(
    "/recommendation",
    methods=["GET", "POST"]
)
def recommendation():
    try:
        if request.method == "POST":
            uploaded_file = request.files.get(
                "file"
            )

            if (
                not uploaded_file
                or uploaded_file.filename == ""
            ):
                raise ValueError(
                    "Pilih file rekomendasi terlebih dahulu."
                )

            if not allowed_file(
                uploaded_file.filename
            ):
                raise ValueError(
                    "File rekomendasi harus berformat CSV."
                )

            data = read_csv_flexible(
                uploaded_file
            )

            context = (
                prepare_recommendation_data(
                    data
                )
            )

            return render_template(
                "recommendation.html",
                **context
            )

        recommendation_file = (
            find_existing_file(
                "rekomendasi_stok.csv"
            )
        )

        if recommendation_file:
            data = read_csv_flexible(
                recommendation_file
            )

            context = (
                prepare_recommendation_data(
                    data
                )
            )

            return render_template(
                "recommendation.html",
                **context
            )

        return render_template(
            "recommendation.html",
            recommendation_data=None
        )

    except Exception as error:
        return render_template(
            "recommendation.html",
            recommendation_data=None,
            error_message=str(error)
        )


@app.route("/about")
def about():
    return render_template(
        "about.html"
    )


@app.route("/health")
def health():
    return {
        "status": "ok"
    }


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    debug_mode = (
        os.environ.get(
            "FLASK_DEBUG",
            "0"
        )
        == "1"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode
    )
