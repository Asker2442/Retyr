"""ИИ-агент поддержки и продаж в Telegram на базе Claude.

Агент отвечает клиентам по базе знаний (knowledge.md), консультирует,
собирает заявки (имя, телефон, запрос) и зовёт живого менеджера,
когда сам не справляется.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
log = logging.getLogger("agent")

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
# Chat ID владельца/менеджера: сюда приходят заявки и вызовы менеджера.
OWNER_CHAT_ID = os.environ.get("OWNER_CHAT_ID")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
BASE_DIR = Path(__file__).parent
KNOWLEDGE_FILE = BASE_DIR / os.environ.get("KNOWLEDGE_FILE", "knowledge.md")
LEADS_FILE = BASE_DIR / "leads.jsonl"
MAX_HISTORY_MESSAGES = 40  # чтобы диалог не разрастался бесконечно
MAX_TOOL_ROUNDS = 5  # защита от зацикливания

client = anthropic.AsyncAnthropic()  # ключ берётся из ANTHROPIC_API_KEY

SYSTEM_PROMPT = f"""Ты — вежливый и живой консультант компании в Telegram.
Твои задачи:
1. Отвечать на вопросы клиентов, опираясь ТОЛЬКО на базу знаний ниже.
   Если ответа в базе нет — честно скажи, что уточнишь, и вызови call_manager.
   Не выдумывай цены, сроки и условия.
2. Помогать клиенту выбрать подходящую услугу/товар и мягко подводить к заявке.
3. Когда клиент готов (хочет записаться, купить, получить расчёт) — узнай имя
   и телефон и вызови save_lead. Не проси данные повторно, если они уже есть.
4. Если клиент злится, просит человека или вопрос нестандартный — вызови call_manager.

Стиль: коротко (2–5 предложений), дружелюбно, на языке клиента, без канцелярита.
Формат: обычный текст без Markdown-разметки (звёздочек, решёток).

<knowledge_base>
{KNOWLEDGE_FILE.read_text(encoding="utf-8")}
</knowledge_base>
"""

TOOLS = [
    {
        "name": "save_lead",
        "description": (
            "Сохранить заявку клиента и отправить её менеджеру. Вызывай, когда "
            "клиент готов купить/записаться и назвал имя и телефон."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Имя клиента"},
                "phone": {"type": "string", "description": "Телефон клиента"},
                "request": {
                    "type": "string",
                    "description": "Что хочет клиент: услуга, детали, удобное время",
                },
            },
            "required": ["name", "phone", "request"],
            "additionalProperties": False,
        },
    },
    {
        "name": "call_manager",
        "description": (
            "Позвать живого менеджера в диалог. Вызывай, если ответа нет в базе "
            "знаний, клиент недоволен или просит человека."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Кратко: почему нужен менеджер и в чём вопрос",
                }
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]

# История диалогов в памяти: chat_id -> список сообщений.
# Для продакшена замените на БД (SQLite/Redis), иначе история сбросится при перезапуске.
histories: dict[int, list] = {}


async def notify_owner(context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    if not OWNER_CHAT_ID:
        log.warning("OWNER_CHAT_ID не задан, уведомление не отправлено: %s", text)
        return
    await context.bot.send_message(chat_id=OWNER_CHAT_ID, text=text)


def client_label(update: Update) -> str:
    user = update.effective_user
    username = f"@{user.username}" if user and user.username else "без username"
    return f"{user.full_name if user else 'клиент'} ({username}, chat {update.effective_chat.id})"


async def run_tool(
    name: str, args: dict, update: Update, context: ContextTypes.DEFAULT_TYPE
) -> str:
    if name == "save_lead":
        lead = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "chat_id": update.effective_chat.id,
            "telegram": client_label(update),
            **args,
        }
        with LEADS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(lead, ensure_ascii=False) + "\n")
        await notify_owner(
            context,
            "🔥 Новая заявка\n"
            f"Имя: {args['name']}\nТелефон: {args['phone']}\n"
            f"Запрос: {args['request']}\nTelegram: {client_label(update)}",
        )
        return "Заявка сохранена и передана менеджеру."
    if name == "call_manager":
        await notify_owner(
            context,
            f"🙋 Нужен менеджер\nПричина: {args['reason']}\nКлиент: {client_label(update)}",
        )
        return "Менеджер уведомлён и скоро подключится."
    return f"Неизвестный инструмент: {name}"


def trim_history(history: list) -> list:
    """Обрезает старые сообщения, начиная историю с обычной реплики клиента."""
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    while trimmed and not (
        trimmed[0]["role"] == "user" and isinstance(trimmed[0]["content"], str)
    ):
        trimmed.pop(0)
    return trimmed


async def ask_claude(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> str:
    chat_id = update.effective_chat.id
    history = trim_history(histories.get(chat_id, []))
    history.append({"role": "user", "content": text})

    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # База знаний одинакова для всех — кешируем её, это дешевле и быстрее.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOLS,
            messages=history,
            # Для чата хватает низкого уровня размышлений: ответы быстрее и дешевле.
            output_config={"effort": "low"},
            # Если модель откажется отвечать, API сам переадресует запрос резервной модели.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        # Сохраняем ответ целиком (включая служебные блоки), не только текст.
        history.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            histories[chat_id] = history
            return "Извините, с этим вопросом я помочь не могу. Позвать менеджера?"

        if response.stop_reason == "pause_turn":
            continue

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            histories[chat_id] = history
            reply = "".join(b.text for b in response.content if b.type == "text").strip()
            return reply or "Уточните, пожалуйста, ваш вопрос 🙂"

        results = []
        for tool in tool_uses:
            try:
                output = await run_tool(tool.name, tool.input, update, context)
                results.append(
                    {"type": "tool_result", "tool_use_id": tool.id, "content": output}
                )
            except Exception as e:  # noqa: BLE001 — ошибку инструмента отдаём модели
                log.exception("Ошибка инструмента %s", tool.name)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool.id,
                        "content": f"Ошибка: {e}",
                        "is_error": True,
                    }
                )
        history.append({"role": "user", "content": results})

    histories[chat_id] = history
    return "Секунду, передаю ваш вопрос менеджеру."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    histories.pop(update.effective_chat.id, None)
    await update.message.reply_text(
        "Здравствуйте! 👋 Я онлайн-консультант. Спрашивайте — подскажу по услугам, "
        "ценам и запишу вас."
    )


async def chat_id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Ваш chat ID: {update.effective_chat.id}")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        reply = await ask_claude(update, context, update.message.text)
    except anthropic.RateLimitError:
        reply = "Сейчас много обращений, напишите, пожалуйста, через минуту."
    except anthropic.APIConnectionError:
        reply = "Проблема со связью, попробуйте ещё раз чуть позже."
    except anthropic.APIStatusError as e:
        log.error("Claude API error %s: %s", e.status_code, e.message)
        reply = "Произошла техническая ошибка. Менеджер скоро ответит вам лично."
        await notify_owner(context, f"⚠️ Ошибка бота ({e.status_code}) у {client_label(update)}")
    await update.message.reply_text(reply)


def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chatid", chat_id_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    log.info("Бот запущен, модель %s", MODEL)
    app.run_polling()


if __name__ == "__main__":
    main()
