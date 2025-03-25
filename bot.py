# bot.py
import telebot
from telebot import types
import pandas as pd
from model import load_data, train_model, predict_future, plot_predictions
import datetime
import time
import requests
import sys

TOKEN = "8130631136:AAHDZRDYaCXSGfB_OtHpzzI-GeGdDvC-D-Q"
bot = telebot.TeleBot(TOKEN, threaded=True)

# Состояния для конечного автомата
class States:
    WAITING_DATE = 1
    WAITING_PERIODS = 2

user_state = {}
predictions_data = {}

# Инициализация модели
def init_model():
    try:
        print("Инициализация модели...")
        train, test = load_data()
        model, train_with_predictions = train_model(train)  # Теперь ожидаем два значения
        last_date = test['date'].iloc[-1] if not test.empty else train['date'].iloc[-1]
        print("Модель готова к работе!")
        return model, train_with_predictions, test, last_date
    except Exception as e:
        print(f"Ошибка инициализации модели: {e}")
        sys.exit(1)

model, train_data, test_data, last_date = init_model()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Сделать прогноз")
    btn2 = types.KeyboardButton("Автопрогноз на 6 недель")
    markup.add(btn1, btn2)
    
    bot.send_message(
        message.chat.id,
        "Бот для прогнозирования цен на арматуру\n\n"
        "Выберите действие:\n"
        "- 'Сделать прогноз' - ввести свою дату и период предсказания\n"
        "- 'Автопрогноз' - прогноз на 6 недель от последней известной даты",
        reply_markup=markup
    )

@bot.message_handler(func=lambda m: m.text == "Автопрогноз на 6 недель")
def auto_predict(message):
    try:
        bot.send_chat_action(message.chat.id, 'typing')
        predictions = predict_future(model, last_date, periods=6)
        
        # Текстовый прогноз
        response = "📊 Автоматический прогноз на 6 недель:\n\n"
        for _, row in predictions.iterrows():
            response += f"📅 {row['date'].strftime('%d.%m.%Y')}: {int(row['predicted_price']):,} руб.\n"
        
        # График прогноза
        plot_buf = plot_predictions(train_data, test_data, predictions)
        
        bot.send_message(message.chat.id, response)
        bot.send_photo(message.chat.id, photo=plot_buf, caption="График прогноза цен на арматуру")
    except Exception as e:
        bot.send_message(message.chat.id, "⚠️ Ошибка прогноза. Попробуйте позже.")
        print(f"Ошибка автопрогноза: {e}")

@bot.message_handler(func=lambda m: m.text == "Сделать прогноз")
def start_custom_predict(message):
    msg = bot.send_message(
        message.chat.id,
        "📅 Введите начальную дату для прогноза в формате ДД.ММ.ГГГГ (например, 01.01.2023):"
    )
    user_state[message.chat.id] = States.WAITING_DATE
    bot.register_next_step_handler(msg, process_date)

def process_date(message):
    try:
        input_date = datetime.datetime.strptime(message.text, "%d.%m.%Y")
        predictions_data[message.chat.id] = {'date': input_date}
        
        msg = bot.send_message(
            message.chat.id,
            "⏳ Введите количество недель для прогноза (от 1 до 12):"
        )
        user_state[message.chat.id] = States.WAITING_PERIODS
        bot.register_next_step_handler(msg, process_periods)
    except ValueError:
        bot.send_message(
            message.chat.id,
            "❌ Неверный формат даты. Пожалуйста, введите дату в формате ДД.ММ.ГГГГ:"
        )
        bot.register_next_step_handler(message, process_date)

def process_periods(message):
    try:
        periods = int(message.text)
        if not 1 <= periods <= 12:
            raise ValueError
        
        user_data = predictions_data[message.chat.id]
        bot.send_chat_action(message.chat.id, 'typing')
        predictions = predict_future(model, user_data['date'], periods)
        
        # Текстовый прогноз
        response = f"📊 Прогноз на {periods} недель с {user_data['date'].strftime('%d.%m.%Y')}:\n\n"
        for _, row in predictions.iterrows():
            response += f"📅 {row['date'].strftime('%d.%m.%Y')}: {int(row['predicted_price']):,} руб.\n"
        
        # График прогноза
        plot_buf = plot_predictions(train_data, test_data, predictions)
        
        bot.send_message(message.chat.id, response)
        bot.send_photo(message.chat.id, photo=plot_buf, caption="График прогноза цен на арматуру")
        user_state.pop(message.chat.id, None)
    except ValueError:
        bot.send_message(
            message.chat.id,
            "❌ Неверное количество недель. Введите число от 1 до 12:"
        )
        bot.register_next_step_handler(message, process_periods)
    except Exception as e:
        bot.send_message(message.chat.id, "⚠️ Ошибка при расчете прогноза.")
        print(f"Ошибка прогноза: {e}")
        user_state.pop(message.chat.id, None)

def start_bot():
    while True:
        try:
            print("Запуск бота...")
            bot.polling(none_stop=True, interval=3, timeout=30)
        except requests.exceptions.ReadTimeout:
            print("Таймаут соединения, переподключение через 5 секунд...")
            time.sleep(5)
        except requests.exceptions.ConnectionError:
            print("Ошибка подключения, переподключение через 10 секунд...")
            time.sleep(10)
        except Exception as e:
            print(f"Неизвестная ошибка: {e}, перезапуск через 15 секунд...")
            time.sleep(15)

if __name__ == "__main__":
    start_bot()