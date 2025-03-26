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
    """Создание графиков прогноза: основного и детализированного."""
    # Создаем фигуру с двумя графиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Основной график (полный обзор)
    ax1.plot(train_data['date'], train_data['price'], 
             label='Реальные данные (обучение)', color='blue', linewidth=2)
    ax1.plot(train_data['date'], train_data['prediction'], 
             label='Предсказания (обучение)', color='green', linestyle='--', linewidth=1.5)
    
    if test_data is not None:
        ax1.plot(test_data['date'], test_data['price'], 
                 color='blue', linewidth=2)
    
    ax1.plot(predictions['date'], predictions['predicted_price'], 
             label='Прогноз', color='orange', linestyle='-', linewidth=3, marker='o')
    
    ax1.set_title('Полный обзор: прогнозирование цен на арматуру', fontsize=14, pad=15)
    ax1.set_ylabel('Цена (руб)', fontsize=10)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_locator(YearLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Детализированный график (последние данные + прогноз)
    combined_dates = pd.concat([test_data['date'] if test_data is not None else train_data['date'].iloc[-12:], 
                              predictions['date']])
    start_date = combined_dates.min() - pd.Timedelta(days=30)
    end_date = combined_dates.max() + pd.Timedelta(days=30)
    
    mask_train = (train_data['date'] >= start_date) & (train_data['date'] <= end_date)
    if test_data is not None:
        mask_test = (test_data['date'] >= start_date) & (test_data['date'] <= end_date)
    
    ax2.plot(train_data.loc[mask_train, 'date'], train_data.loc[mask_train, 'price'], 
             label='Реальные данные (обучение)', color='blue', linewidth=2)
    ax2.plot(train_data.loc[mask_train, 'date'], train_data.loc[mask_train, 'prediction'], 
             label='Предсказания (обучение)', color='green', linestyle='--', linewidth=1.5)
    
    if test_data is not None:
        ax2.plot(test_data.loc[mask_test, 'date'], test_data.loc[mask_test, 'price'], 
                 color='blue', linewidth=2, label='Реальные данные (тест)')
    
    ax2.plot(predictions['date'], predictions['predicted_price'], 
             label='Прогноз', color='orange', linestyle='-', linewidth=3, marker='o')
    
    ax2.set_title('Детализированный вид: последние данные и прогноз', fontsize=14, pad=15)
    ax2.set_xlabel('Дата', fontsize=10)
    ax2.set_ylabel('Цена (руб)', fontsize=10)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_formatter(DateFormatter('%d.%m.%Y'))
    
    # Автоматический поворот дат для лучшей читаемости
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Улучшаем layout
    plt.tight_layout()
    
    # Сохраняем в буфер
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf