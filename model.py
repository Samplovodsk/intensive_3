import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt
import io
from matplotlib.dates import YearLocator, DateFormatter

MODEL_PATH = 'armature_price_model.joblib'

def load_data(train_path='train.xlsx', test_path='test.xlsx'):
    """Загрузка и подготовка данных."""
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)
    
    train = train.rename(columns={'dt': 'date', 'Цена на арматуру': 'price'})
    test = test.rename(columns={'dt': 'date', 'Цена на арматуру': 'price'})
    
    for df in [train, test]:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.dayofweek
    
    return train, test

def train_model(train):
    """Обучение модели с кэшированием."""
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        train['prediction'] = model.predict(train[['year', 'month', 'week', 'day_of_week']])
        return model, train
    
    X = train[['year', 'month', 'week', 'day_of_week']]
    y = train['price']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    
    train['prediction'] = model.predict(X)
    
    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    print(f"MAE на валидации: {mae:.2f} руб.")
    
    joblib.dump(model, MODEL_PATH)
    return model, train

def predict_future(model, last_date, periods=4):
    """Прогнозирование на будущие периоды."""
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq='W-MON')[1:]
    future_df = pd.DataFrame({
        'date': future_dates,
        'year': future_dates.year,
        'month': future_dates.month,
        'week': future_dates.isocalendar().week,
        'day_of_week': future_dates.dayofweek
    })
    predictions = model.predict(future_df[['year', 'month', 'week', 'day_of_week']])
    return future_df.assign(predicted_price=predictions)

def plot_predictions(train_data, test_data, predictions):
    """Создание графика прогноза."""
    plt.figure(figsize=(14, 7))
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.plot(train_data['date'], train_data['price'], 
             label='Реальные данные (обучение)', color='blue', linewidth=2)
    
    plt.plot(train_data['date'], train_data['prediction'], 
             label='Предсказания (обучение)', color='green', linestyle='--', linewidth=1.5)
    
    if test_data is not None:
        plt.plot(test_data['date'], test_data['price'], 
                 color='blue', linewidth=2)
    
    plt.plot(predictions['date'], predictions['predicted_price'], 
             label='Прогноз', color='orange', linestyle='-', linewidth=3, marker='o')
    
    plt.title('Прогнозирование цен на арматуру', fontsize=16, pad=20)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Цена (руб)', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf