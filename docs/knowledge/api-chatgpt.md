# ChatGPT API — заметки

Первая модель тестового стенда. Вкладка «ChatGPT» реализована (дашборд).

## Что реализовано (2026-09-07)
- Эндпоинт: `POST https://api.openai.com/v1/images/edits`, модель `gpt-image-1` (по умолчанию), multipart: `image[]` (до 16), `prompt`, `n=1`, `size` (auto/1024x1024/1536x1024/1024x1536).
- Ответ: `data[].b64_json` → декодируем → сохраняем как outgoing.
- Таймаут запроса: 300 с (генерация бывает долгой).
- Ключ: `OPENAI_API_KEY` из `.env`; без ключа `/api/chatgpt/generate` → 503, в UI индикатор «ключ не задан».

## API стенда
- `GET /api/chatgpt/status` — задан ли ключ
- `POST /api/chatgpt/generate` — files[], prompt, size, model
- `GET /api/chatgpt/requests` — последние запросы (группировка по request_id)

## Грабли
(пополнять по ходу тестов Льва: лимиты, форматы, ошибки)
- Примечание: content_type `image/jpg` клиент может прислать нестандартно — разрешён наравне с jpeg.
