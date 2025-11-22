# Grok2API

Grok2API refactored based on **FastAPI**, fully adapted to the latest Web calling format, supporting streaming conversation, image generation, image editing, web search, deep thinking, account pool concurrency, and automatic load balancing integration.


<br>

## Instructions

### Call Count and Quota

- **Basic Account**: Free usage **80 times / 20 hours**
- **Super Account**: Quota TBD (Not tested by author)
- The system automatically load balances the call count of each account. You can view usage and status in real-time on the **Management Page**.

### Image Generation Function

- Automatically trigger image generation by inputting content like "Draw me a moon" in the conversation.
- Returns two images in **Markdown format** each time, consuming 4 quota units.
- **Note: Grok's direct image links are restricted by 403. The system automatically caches images locally. `Base Url` must be set correctly to ensure images display properly!**

### Video Generation Function
- Select `grok-imagine-0.9` model, pass in image and prompt (same format as OpenAI image analysis call).
- Return format is `<video src="{full_video_url}" controls="controls"></video>`
- **Note: Grok's direct video links are restricted by 403. The system automatically caches video locally. `Base Url` must be set correctly to ensure videos display properly!**

```
curl https://your-server-address/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GROK2API_API_KEY" \
  -d '{
    "model": "grok-imagine-0.9",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Make the sun rise"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://your-image.jpg"
            }
          }
        ]
      }
    ]
  }'
```

### About `x_statsig_id`

- `x_statsig_id` is Grok's anti-bot Token, reverse engineering materials are available for reference.
- **Beginners are advised not to modify the configuration and keep the default value.**
- Tried to use Camoufox to bypass 403 and automatically get id, but grok now restricts non-logged-in `x_statsig_id`, so it was abandoned and a fixed value is used to be compatible with all requests.

<br>

## Deployment

### docker-compose

```yaml
services:
  grok2api:
    image: ghcr.io/chenyme/grok2api:latest
    ports:
      - "8000:8000"
    volumes:
      - grok_data:/app/data
      - ./logs:/app/logs
    environment:
      # =====Storage Mode: file, mysql or redis=====
      - STORAGE_MODE=file
      # =====Database Connection URL (Required only when STORAGE_MODE=mysql or redis)=====
      # - DATABASE_URL=mysql://user:password@host:3306/grok2api

      ## MySQL format: mysql://user:password@host:port/database
      ## Redis format: redis://host:port/db or redis://user:password@host:port/db

volumes:
  grok_data:
```

### Environment Variables

| Variable      | Required | Description                                    | Example |
|---------------|----------|------------------------------------------------|---------|
| STORAGE_MODE  | No       | Storage mode: file/mysql/redis                 | file |
| DATABASE_URL  | No       | Database connection URL (Required for MySQL/Redis mode) | mysql://user:pass@host:3306/db |

**Storage Modes:**
- `file`: Local file storage (Default)
- `mysql`: MySQL database storage, DATABASE_URL required
- `redis`: Redis cache storage, DATABASE_URL required

<br>

## Interface Description

> Fully compatible with OpenAI official interface, API requests need authentication via **Authorization header**

| Method | Endpoint                     | Description                        | Auth Required |
|--------|------------------------------|------------------------------------|---------------|
| POST   | `/v1/chat/completions`       | Create chat completion (stream/non-stream) | ✅            |
| GET    | `/v1/models`                 | Get all supported models           | ✅            |
| GET    | `/images/{img_path}`         | Get generated image file           | ❌            |

<br>

<details>
<summary>Management & Statistics Interfaces (Expand to view more)</summary>

| Method | Endpoint                | Description        | Auth |
|--------|-------------------------|--------------------|------|
| GET    | /login                  | Admin login page   | ❌   |
| GET    | /manage                 | Admin console page | ❌   |
| POST   | /api/login              | Admin login auth   | ❌   |
| POST   | /api/logout             | Admin logout       | ✅   |
| GET    | /api/tokens             | Get Token list     | ✅   |
| POST   | /api/tokens/add         | Batch add Tokens   | ✅   |
| POST   | /api/tokens/delete      | Batch delete Tokens| ✅   |
| GET    | /api/settings           | Get system config  | ✅   |
| POST   | /api/settings           | Update system config| ✅   |
| GET    | /api/cache/size         | Get cache size     | ✅   |
| POST   | /api/cache/clear        | Clear all cache    | ✅   |
| POST   | /api/cache/clear/images | Clear image cache  | ✅   |
| POST   | /api/cache/clear/videos | Clear video cache  | ✅   |
| GET    | /api/stats              | Get statistics     | ✅   |

</details>

<br>

## Available Models

| Model Name             | Count  | Account Type | Image Gen/Edit | Deep Thinking | Web Search | Video Gen |
|------------------------|--------|--------------|----------------|---------------|------------|-----------|
| `grok-3-fast`          | 1      | Basic/Super  | ✅             | ❌            | ✅         | ❌        |
| `grok-4-fast`          | 1      | Basic/Super  | ✅             | ✅            | ✅         | ❌        |
| `grok-4-fast-expert`   | 4      | Basic/Super  | ✅             | ✅            | ✅         | ❌        |
| `grok-4-expert`        | 4      | Basic/Super  | ✅             | ✅            | ✅         | ❌        |
| `grok-4-heavy`         | 1      | Super        | ✅             | ✅            | ✅         | ❌        |
| `grok-imagine-0.9`     | -      | Basic/Super  | ✅             | ❌            | ❌         | ✅        |

<br>

## Configuration Parameters

> After service startup, log in to `/login` admin panel to configure parameters

| Parameter                  | Scope   | Required | Description                                    | Default |
|----------------------------|---------|----------|------------------------------------------------|---------|
| admin_username             | global  | No       | Admin panel login username                     | "admin"|
| admin_password             | global  | No       | Admin panel login password                     | "admin"|
| log_level                  | global  | No       | Log level: DEBUG/INFO/...                      | "INFO" |
| image_mode                 | global  | No       | Image return mode: url/base64                  | "url"  |
| image_cache_max_size_mb    | global  | No       | Max image cache size (MB)                      | 512    |
| video_cache_max_size_mb    | global  | No       | Max video cache size (MB)                      | 1024   |
| base_url                   | global  | No       | Service Base URL/Image access base             | ""     |
| api_key                    | grok    | No       | API Key (Optional for enhanced security)       | ""     |
| proxy_url                  | grok    | No       | HTTP Proxy Server Address                      | ""     |
| stream_chunk_timeout       | grok    | No       | Stream chunk timeout (seconds)                 | 120    |
| stream_first_response_timeout | grok | No       | Stream first response timeout (seconds)        | 30     |
| stream_total_timeout       | grok    | No       | Stream total timeout (seconds)                 | 600    |
| cf_clearance               | grok    | No       | Cloudflare Security Token                      | ""     |
| x_statsig_id               | grok    | Yes      | Anti-bot Unique Identifier                     | "ZTpUeXBlRXJyb3I6IENhbm5vdCByZWFkIHByb3BlcnRpZXMgb2YgdW5kZWZpbmVkIChyZWFkaW5nICdjaGlsZE5vZGVzJyk=" |
| filtered_tags              | grok    | No       | Filter response tags (comma separated)         | "xaiartifact,xai:tool_usage_card,grok:render" |
| temporary                  | grok    | No       | Session mode true(temporary)/false             | true   |

<br>

## ⚠️ Notes

This project is for learning and research purposes only, please comply with relevant terms of use!

<br>

> This project is refactored based on the following projects for learning, special thanks to: [LINUX DO](https://linux.do), [VeroFess/grok2api](https://github.com/VeroFess/grok2api), [xLmiler/grok2api_python](https://github.com/xLmiler/grok2api_python)