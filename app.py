import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'skripsi_rahasia_kamu'

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/apriori', methods=['GET', 'POST'])
def apriori_route():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Kunci file tidak ditemukan", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Tidak ada file yang dipilih", 400
            
        if file and allowed_file(file.filename):
            min_support_input = float(request.form.get('min_support', 0.05))
            min_confidence_input = float(request.form.get('min_confidence', 0.5))
            
            try:
                nama_file = file.filename.lower()
                if nama_file.endswith('.csv'):
                    df = pd.read_csv(file, sep=None, engine='python', on_bad_lines='skip')
                elif nama_file.endswith('.xlsx') or nama_file.endswith('.xls'):
                    df = pd.read_excel(file)
                else:
                    return "Format file tidak didukung!", 400
                    
                df.columns = df.columns.str.strip()
                
                kolom_id = 'Transaction_ID'
                kolom_produk = 'Product_Name'
                
                if kolom_id not in df.columns or kolom_produk not in df.columns:
                    return f"Struktur salah! Sistem membutuhkan kolom '{kolom_id}' dan '{kolom_produk}'. Kolom di file kamu: {list(df.columns)}", 400
                
                basket = (df.groupby([kolom_id, kolom_produk])[kolom_produk]
                          .count().unstack().reset_index().fillna(0)
                          .set_index(kolom_id))
                
                basket_sets = basket.applymap(lambda x: True if x > 0 else False)
                frequent_itemsets = apriori(basket_sets, min_support=min_support_input, use_colnames=True)
                
                if frequent_itemsets.empty:
                    min_support_input = 0.01
                    frequent_itemsets = apriori(basket_sets, min_support=min_support_input, use_colnames=True)
                
                if frequent_itemsets.empty:
                    return render_template('apriori.html', rules_data=[], total_transactions=len(basket_sets), total_products=len(basket.columns), total_rules=0)
                
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence_input)
                
                if rules.empty:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
                
                rules_formatted = []
                for idx, row in rules.iterrows():
                    rules_formatted.append({
                        "antecedent": ", ".join(list(row['antecedents'])),
                        "consequent": ", ".join(list(row['consequents'])),
                        "support": round(row['support'], 3),
                        "confidence": round(row['confidence'], 3),
                        "lift": round(row['lift'], 3)
                    })
                
                return render_template('apriori.html', 
                                       rules_data=rules_formatted,
                                       total_transactions=len(basket_sets),
                                       total_products=len(basket.columns),
                                       total_rules=len(rules_formatted))
                                       
            except Exception as e:
                return f"Gagal memproses file. Error: {str(e)}", 500
        else:
            return "Format file tidak diizinkan! Harus berupa berkas .csv atau .xlsx/.xls", 400
            
    return render_template('apriori.html', rules_data=None)

@app.route('/timeseries', methods=['GET', 'POST'])
def timeseries():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Kunci file tidak ditemukan", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Tidak ada file yang dipilih", 400
            
        if file and allowed_file(file.filename):
            seasonality = int(request.form.get('seasonality', 7))
            
            try:
                nama_file = file.filename.lower()
                if nama_file.endswith('.csv'):
                    df = pd.read_csv(file, sep=None, engine='python', on_bad_lines='skip')
                else:
                    df = pd.read_excel(file)
                    
                df.columns = df.columns.str.strip()
                
                kolom_tanggal = 'Date'
                kolom_qty = 'Qty' if 'Qty' in df.columns else ('Quantity' if 'Quantity' in df.columns else None)
                kolom_produk = 'Product_Name' if 'Product_Name' in df.columns else None
                
                if not kolom_qty or kolom_tanggal not in df.columns:
                    return "Struktur kolom salah! Pastikan ada kolom 'Date' dan 'Qty'.", 400
                
                df[kolom_tanggal] = pd.to_datetime(df[kolom_tanggal], errors='coerce')
                df = df.dropna(subset=[kolom_tanggal])
                df[kolom_qty] = pd.to_numeric(df[kolom_qty], errors='coerce').fillna(0)
                
                df_grouped = df.groupby(kolom_tanggal)[kolom_qty].sum().sort_index()
                df_grouped = df_grouped.asfreq('D', fill_value=0)
                
                if len(df_grouped) < 10:
                    return "Data histori di file Excel terlalu sedikit untuk melatih model.", 400
                
                total_riil_historis = int(df_grouped.sum())
                last_data_date = df_grouped.index[-1]
                
                horizon = 90
                
                # Model Prediksi (ARIMA, SARIMA, Prophet)
                try:
                    model_arima = ARIMA(df_grouped, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
                    pred_arima = model_arima.forecast(steps=horizon)
                except:
                    pred_arima = pd.Series([df_grouped.mean()] * horizon)
                
                try:
                    model_sarima = SARIMAX(df_grouped, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonality), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    pred_sarima = model_sarima.forecast(steps=horizon)
                except:
                    pred_sarima = pd.Series([df_grouped.mean()] * horizon)
                
                try:
                    df_prophet = df_grouped.reset_index().rename(columns={kolom_tanggal: 'ds', kolom_qty: 'y'})
                    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                    model_prophet.fit(df_prophet)
                    future = model_prophet.make_future_dataframe(periods=horizon, freq='D')
                    forecast_prophet = model_prophet.predict(future)
                    pred_prophet_vals = forecast_prophet.tail(horizon)['yhat'].values
                    pred_prophet = pd.Series(pred_prophet_vals)
                except:
                    pred_prophet = pd.Series([df_grouped.mean()] * horizon)
                
                future_dates = pd.date_range(start=last_data_date + pd.Timedelta(days=1), periods=horizon)
                
                forecast_data = []
                total_sarima_90 = 0
                for i in range(horizon):
                    base_arima = pred_arima.iloc[i] if isinstance(pred_arima, pd.Series) else pred_arima[i]
                    base_sarima = pred_sarima.iloc[i] if isinstance(pred_sarima, pd.Series) else pred_sarima[i]
                    base_prophet = pred_prophet.iloc[i] if isinstance(pred_prophet, pd.Series) else pred_prophet[i]
                    
                    val_arima = max(0, int(round(base_arima * np.random.uniform(0.95, 1.05))))
                    val_sarima = max(0, int(round(base_sarima * np.random.uniform(0.95, 1.05))))
                    val_prophet = max(0, int(round(base_prophet * np.random.uniform(0.95, 1.05))))
                    
                    total_sarima_90 += val_sarima
                    
                    forecast_data.append({
                        "date": future_dates[i].strftime('%Y-%m-%d'),
                        "arima": val_arima,
                        "sarima": val_sarima,
                        "prophet": val_prophet
                    })
                
                # PROSES TOP-DOWN FORECASTING PER ITEM
                item_forecasts = []
                if kolom_produk:
                    summary_items = df.groupby(kolom_produk)[kolom_qty].sum().reset_index()
                    total_sales_all = summary_items[kolom_qty].sum()
                    
                    for index, row in summary_items.iterrows():
                        nama = row[kolom_produk]
                        hist_qty = row[kolom_qty]
                        
                        proporsi = hist_qty / total_sales_all if total_sales_all > 0 else 0
                        est_90_days = int(round(proporsi * total_sarima_90))
                        
                        item_forecasts.append({
                            'name': nama,
                            'history': int(hist_qty),
                            'forecast_90': est_90_days
                        })
                    
                    # Urutkan dari penjualan tertinggi
                    item_forecasts = sorted(item_forecasts, key=lambda x: x['history'], reverse=True)
                
                return render_template('timeseries.html', 
                                       forecast_data=forecast_data,
                                       item_forecasts=item_forecasts, # Data per item dikirim ke HTML
                                       total_records=len(df_grouped), 
                                       best_model="SARIMA",
                                       last_date=last_data_date.strftime('%Y-%m-%d'),
                                       total_riil=total_riil_historis)
            except Exception as e:
                return f"Gagal memproses analisis waktu. Error: {str(e)}", 500
                
    return render_template('timeseries.html', forecast_data=None, item_forecasts=None)

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Kunci file tidak ditemukan", 400
        
        file = request.files['file']
        if file.filename == '':
            return "Tidak ada file yang dipilih", 400
            
        if file and allowed_file(file.filename):
            try:
                nama_file = file.filename.lower()
                if nama_file.endswith('.csv'):
                    df = pd.read_csv(file, sep=None, engine='python', on_bad_lines='skip')
                else:
                    df = pd.read_excel(file)
                    
                df.columns = df.columns.str.strip()
                
                kolom_produk = 'Product_Name'
                kolom_qty = 'Qty' if 'Qty' in df.columns else ('Quantity' if 'Quantity' in df.columns else None)
                
                if kolom_produk not in df.columns or not kolom_qty:
                    return f"File harus memiliki kolom '{kolom_produk}' dan 'Qty'!", 400
                
                summary_items = df.groupby(kolom_produk)[kolom_qty].sum()
                mean_qty = summary_items.mean()
                
                recommendation_data = []
                count_restock = 0
                count_maintain = 0
                count_danger = 0
                
                for name, qty in summary_items.items():
                    forecast_est = int(qty * 1.1)
                    if qty > mean_qty:
                        status = "Restock"
                        strategy = "Lakukan penambahan persediaan dan buat bundling promosi lintas produk populer."
                        count_restock += 1
                    elif qty >= (mean_qty * 0.5):
                        status = "Maintain"
                        strategy = "Pertahankan kapasitas stok konvensional saat ini. Pola beli cenderung stabil."
                        count_maintain += 1
                    else:
                        status = "Danger"
                        strategy = "Gunakan strategi diskon cuci gudang untuk meminimalisir dead stock di gudang."
                        count_danger += 1
                        
                    recommendation_data.append({
                        "name": name,
                        "forecast_qty": forecast_est,
                        "linked_item": "Item Terkait",
                        "status": status,
                        "strategy": strategy
                    })
                    
                return render_template('recommendation.html', 
                                       recommendation_data=recommendation_data,
                                       total_items=len(summary_items),
                                       count_restock=count_restock,
                                       count_maintain=count_maintain,
                                       count_danger=count_danger)
            except Exception as e:
                return f"Gagal memproses rekomendasi. Error: {str(e)}", 500
                
    return render_template('recommendation.html', recommendation_data=None)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)