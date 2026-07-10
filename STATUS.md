# STATUS — MeetingTasksAgent (Python / Chainlit / Notion MCP)

Cập nhật: 2026-07-10

## 1. Đã làm được gì

Đã dựng xong toàn bộ khung ứng dụng theo README (extract action items từ transcript → review → tạo task trong Notion), dùng stack: Chainlit (chat UI) + Notion MCP server (qua MCP client tích hợp sẵn của Chainlit) + LLM endpoint OpenAI-compatible của FPT Cloud.

### Cấu trúc file đã tạo

```
app.py                          # Entry point Chainlit: toàn bộ luồng hội thoại
meeting_agent/
  config.py                     # đọc FPT_API_KEY, FPT_BASE_URL, LLM_MODEL từ .env
  state.py                      # SessionState (stage, transcript, tasks, mcp_clients, ...)
  models.py                     # Task / TaskList (pydantic)
  llm.py                        # wrapper OpenAI SDK trỏ vào FPT Cloud, structured-output helper
  extraction.py                 # extract_tasks() + revise_tasks() — gọi LLM lấy JSON có schema
  mcp_tools.py                  # phần dùng chung cho mọi MCP server: cache tool, dispatch, vòng lặp agentic tool-calling
  notion_mapping.py             # riêng cho Notion: lấy schema data source, fuzzy-match field -> property, tạo page
samples/
  transcript_en.txt             # transcript mẫu tiếng Anh
  transcript_vi.txt             # transcript mẫu tiếng Việt
scripts/
  smoke_extract.py              # test riêng phần extraction, không cần chạy Chainlit
.chainlit/config.toml           # đã bật features.mcp (sse/streamable-http/stdio, allowlist npx/uvx)
.env / .env.example             # cấu hình khoá API (file .env đã bị gitignore, không commit)
```

### Luồng hoạt động (`app.py`)
1. `on_chat_start`: chào, xin transcript (paste hoặc đính kèm file `.txt`/`.md`).
2. Nhận transcript → gọi LLM (`extraction.extract_tasks`) → trả JSON structured output (title, owner, due_date, dependencies, source_excerpt) → hiển thị danh sách task dạng Markdown.
3. Người dùng nhắn tự do để chỉnh sửa (vd: "gộp task 2 và 3") → `extraction.revise_tasks` gọi lại LLM để sửa → lặp lại tới khi bấm nút **"Looks good, proceed"**.
4. Sau khi xác nhận: kiểm tra đã kết nối Notion MCP chưa (qua icon 🔌 trong khung chat) → dùng vòng lặp agentic tool-calling để LLM tự tìm database Notion phù hợp → lấy schema database, tự động map field (Owner/Due date, có cả alias tiếng Việt "phụ trách", "hạn") → hiện bảng mapping + xin xác nhận bằng nút bấm (`cl.AskActionMessage`: ✅ Create / ✏️ Edit / ❌ Cancel) — **không bao giờ tự động ghi vào Notion nếu chưa bấm Confirm**.
5. Sau khi bấm Confirm: chạy vòng lặp Python thuần (không qua LLM) để tạo từng page trong Notion, báo kết quả từng task (✅/❌) về chat.

### Đã kiểm chứng
- Toàn bộ file compile sạch (`py_compile`), `chainlit run app.py` khởi động và trả HTTP 200.
- Đã đọc trực tiếp source code của `chainlit`/`mcp` package đang cài (không đoán theo doc/blog) để xác nhận đúng API: `cl.Action(name, payload, label)`, `cl.AskActionMessage`, chữ ký `on_mcp_connect`/`on_mcp_disconnect`, cấu trúc `CallToolResult`/`Tool` của MCP.
- Endpoint FPT Cloud (`https://mkp-api.fptcloud.com/v1/models`) gọi được, trả về 18 model, trong đó nhiều model hỗ trợ `tools`/`response_format` (structured output + function calling) — đang chọn mặc định `DeepSeek-V4-Flash` (200k ctx), có thể đổi qua biến môi trường `LLM_MODEL`.

## 2. Đang bị stuck ở đâu

**API key FPT Cloud bạn cung cấp không gọi được chat completions thật.**

- `GET /v1/models` → hoạt động bình thường (endpoint này thực ra public, không cần key hợp lệ).
- `POST /v1/chat/completions` (dùng để extract task) → trả về `401 Invalid API Key`, đã kiểm chứng bằng cả `curl` thô lẫn qua OpenAI SDK, thử với nhiều model khác nhau (DeepSeek-V4-Flash, gpt-oss-120b) — không phải lỗi do code, mà do bản thân key.
- Đã hỏi lại và bạn chọn "cứ tiếp tục build, xử lý key sau" — nên phần LLM hiện **chưa test end-to-end được** với dữ liệu thật. `scripts/smoke_extract.py` sẽ tái hiện đúng lỗi 401 này khi chạy.

→ Việc cần làm: lấy lại key hợp lệ từ FPT AI Marketplace (key hiện tại có thể đã hết hạn/bị thu hồi/copy thiếu ký tự), rồi cập nhật vào file `.env` (biến `FPT_API_KEY`).

**Chưa test thật với Notion MCP server** (do chưa có transcript thật chạy qua được LLM để tới bước này) — phần này về code đã viết đủ logic (discover tool, map field, tạo page), nhưng chưa có lần chạy thực tế nào tạo page thật trong Notion.

## 3. Các bước để chạy thử app

1. **Cập nhật API key hợp lệ:**
   ```
   # sửa file .env ở root repo
   FPT_API_KEY=<key-hop-le-tu-FPT-AI-Marketplace>
   FPT_BASE_URL=https://mkp-api.fptcloud.com/v1
   LLM_MODEL=DeepSeek-V4-Flash
   ```

2. **Test nhanh phần extraction (không cần mở UI):**
   ```
   uv run python scripts/smoke_extract.py samples/transcript_en.txt
   uv run python scripts/smoke_extract.py samples/transcript_vi.txt
   ```
   Kỳ vọng: in ra JSON danh sách task (title/owner/due_date/dependencies) thay vì lỗi 401.

3. **Chạy app Chainlit:**
   ```
   uv run chainlit run app.py -w
   ```
   Mở trình duyệt tại link được in ra (mặc định `http://localhost:8000`).

4. **Chuẩn bị Notion MCP server** (chọn 1 trong 2 cách, làm qua icon 🔌 trong khung chat của Chainlit):
   - **Cách A — server local qua npx** (cần Node.js): thêm MCP server kiểu `stdio`, command: `npx -y @notionhq/notion-mcp-server`, cần có Notion internal integration token (tạo tại notion.so/my-integrations, và share database đích với integration đó).
   - **Cách B — Notion hosted MCP** (không cần Node.js): thêm MCP server kiểu `streamable-http`, URL: `https://mcp.notion.com/mcp`, đăng nhập OAuth ngay trong dialog.

5. **Đi hết luồng demo:** paste/đính kèm transcript → xem danh sách task được trích xuất → gõ chỉnh sửa nếu cần → bấm "Looks good, proceed" → xác nhận database Notion + bảng mapping field → bấm "✅ Create tasks" → kiểm tra page thật đã được tạo trong Notion đúng field.

## 4. Việc cần làm tiếp theo (sau khi có key)

- Chạy lại `smoke_extract.py` để xác nhận structured output từ FPT model đúng schema.
- Chạy full flow với Notion MCP thật, kiểm tra format `tool_calls[].function.arguments` từ model FPT có đúng chuẩn JSON string không (một số model open-weight trả sai định dạng).
- Nếu ổn, có thể xem thêm phần "Future Improvements" trong README (Jira connector, dependency dạng relation trong Notion, v.v.) — chưa cần thiết cho bản demo.
