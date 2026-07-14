import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


warnings.filterwarnings("ignore")

ARIMA_ORDER = (1, 1, 1)
HORIZON = 90
JUMLAH_PRODUK = 50


def validasi_dataset(data):
    data.columns = data.columns.str.strip()

    kolom_wajib = {
        "Date",
        "Product_Name",
        "Qty"
    }

    kolom_kurang = kolom_wajib - set(data.columns)

    if kolom_kurang:
        raise ValueError(
            "Dataset harus memiliki kolom Date, "
            "Product_Name, dan Qty."
        )

    data = data[
        [
            "Date",
            "Product_Name",
            "Qty"
        ]
    ].copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
        dayfirst=True
    )

    data["Product_Name"] = (
        data["Product_Name"]
        .astype(str)
        .str.strip()
    )

    data["Qty"] = pd.to_numeric(
        data["Qty"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "Date",
            "Product_Name",
            "Qty"
        ]
    )

    data = data[
        data["Product_Name"] != ""
    ].copy()

    data["Qty"] = data["Qty"].clip(lower=0)

    if data.empty:
        raise ValueError(
            "Dataset tidak memiliki data yang dapat diproses."
        )

    return data


def pilih_produk_terlaris(data):
    total_produk = (
        data.groupby(
            "Product_Name",
            as_index=False
        )["Qty"]
        .sum()
        .sort_values(
            "Qty",
            ascending=False
        )
        .head(JUMLAH_PRODUK)
    )

    daftar_produk = total_produk[
        "Product_Name"
    ].tolist()

    return daftar_produk


def bentuk_data_harian(data, produk):
    data_produk = data[
        data["Product_Name"] == produk
    ].copy()

    data_harian = (
        data_produk.groupby("Date")["Qty"]
        .sum()
        .sort_index()
    )

    tanggal_lengkap = pd.date_range(
        start=data_harian.index.min(),
        end=data_harian.index.max(),
        freq="D"
    )

    data_harian = data_harian.reindex(
        tanggal_lengkap,
        fill_value=0
    )

    data_harian.index.name = "Date"

    return data_harian.astype(float)


def prediksi_satu_produk(data_harian):
    if len(data_harian) < 10:
        raise ValueError(
            "Data historis produk terlalu sedikit."
        )

    model = ARIMA(
        data_harian,
        order=ARIMA_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    hasil_model = model.fit()

    prediksi = hasil_model.forecast(
        steps=HORIZON
    )

    prediksi = np.asarray(
        prediksi,
        dtype=float
    )

    prediksi = np.clip(
        prediksi,
        0,
        None
    )

    return prediksi


def proses_timeseries(data):
    data = validasi_dataset(data)

    daftar_produk = pilih_produk_terlaris(
        data
    )

    hasil_harian = []
    hasil_ringkasan = []
    produk_gagal = []

    for produk in daftar_produk:
        try:
            data_harian = bentuk_data_harian(
                data,
                produk
            )

            prediksi = prediksi_satu_produk(
                data_harian
            )

            tanggal_awal = (
                data_harian.index.max()
                + pd.Timedelta(days=1)
            )

            tanggal_prediksi = pd.date_range(
                start=tanggal_awal,
                periods=HORIZON,
                freq="D"
            )

            for tanggal, nilai in zip(
                tanggal_prediksi,
                prediksi
            ):
                hasil_harian.append({
                    "Tanggal": tanggal,
                    "Product_Name": produk,
                    "Prediksi_ARIMA": round(
                        float(nilai),
                        2
                    )
                })

            hasil_ringkasan.append({
                "Product_Name": produk,
                "Prediksi_90_Hari": round(
                    float(prediksi.sum()),
                    2
                )
            })

        except Exception as error:
            produk_gagal.append({
                "Product_Name": produk,
                "Error": str(error)
            })

    if not hasil_ringkasan:
        raise ValueError(
            "Tidak ada produk yang berhasil diprediksi."
        )

    harian = pd.DataFrame(
        hasil_harian
    )

    ringkasan = pd.DataFrame(
        hasil_ringkasan
    )

    ringkasan.insert(
        0,
        "No",
        range(1, len(ringkasan) + 1)
    )

    return {
        "harian": harian,
        "ringkasan": ringkasan,
        "produk_gagal": produk_gagal
    }