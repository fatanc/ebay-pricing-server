# eBay Pricing Server — Salvage Vehicle Damage Analysis

FastAPI proxy server for eBay Browse API. Provides vehicle parts pricing for salvage damage cost estimation.

## Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/fatanc/ebay-pricing-server)

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Server health & API status |
| `GET /api/search?q=...` | Search eBay for parts with price/condition filters |
| `GET /api/item/{id}` | Full item details by eBay item ID |
| `GET /api/damage-estimate?vehicle=...&parts=...` | Multi-part cost estimate |
| `GET /docs` | Interactive Swagger API docs |

## Example

```bash
# Search for a specific part
curl "https://your-app.onrender.com/api/search?q=2019+BMW+330i+headlight+assembly&limit=5"

# Get damage estimate for multiple parts
curl "https://your-app.onrender.com/api/damage-estimate?vehicle=2020+Tesla+Model+3&parts=front+bumper,headlight,hood,fender"
```

## Environment Variables

| Variable | Description |
|---|---|
| `EBAY_CLIENT_ID` | eBay production App ID |
| `EBAY_CLIENT_SECRET` | eBay production Cert ID |
| `ANTHROPIC_API_KEY` | Anthropic API key (for AI analysis) |
| `EBAY_MARKETPLACE` | Default: `EBAY_US` |

## Local Development

```bash
pip install -r requirements.txt
python ebay_server.py
# Server runs on http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```
# Deploy trigger: Thu Mar 12 11:04:14 UTC 2026
