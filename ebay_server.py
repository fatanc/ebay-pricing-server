"""
eBay Proxy Server for Salvage Vehicle Damage Analysis Platform
FastAPI server that proxies eBay Browse API requests for vehicle parts pricing.
"""

import os
import time
import base64
import logging
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Configuration ─────────────────────────────────────────────────────────────

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID", "")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
EBAY_API_BASE = "https://api.ebay.com"
EBAY_MARKETPLACE = os.getenv("EBAY_MARKETPLACE", "EBAY_US")
SERVER_PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ebay_server")

# ── Token Cache ───────────────────────────────────────────────────────────────

class TokenManager:
    """Manages eBay OAuth2 client credentials token with automatic refresh."""

    def __init__(self):
        self.access_token: Optional[str] = None
        self.expires_at: float = 0
        self.http_client: Optional[httpx.AsyncClient] = None

    async def init_client(self):
        self.http_client = httpx.AsyncClient(timeout=30)

    async def close_client(self):
        if self.http_client:
            await self.http_client.aclose()

    async def get_token(self) -> str:
        if self.access_token and time.time() < self.expires_at - 60:
            return self.access_token

        logger.info("Refreshing eBay OAuth token...")
        credentials = base64.b64encode(
            f"{EBAY_CLIENT_ID}:{EBAY_CLIENT_SECRET}".encode()
        ).decode()

        resp = await self.http_client.post(
            f"{EBAY_API_BASE}/identity/v1/oauth2/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {credentials}",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
        )

        if resp.status_code != 200:
            logger.error(f"Token refresh failed: {resp.status_code} {resp.text}")
            raise HTTPException(502, "Failed to authenticate with eBay")

        data = resp.json()
        self.access_token = data["access_token"]
        self.expires_at = time.time() + data.get("expires_in", 7200)
        logger.info(f"Token refreshed, expires in {data.get('expires_in')}s")
        return self.access_token


token_mgr = TokenManager()

# ── App Lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await token_mgr.init_client()
    # Pre-warm the token
    try:
        await token_mgr.get_token()
        logger.info("eBay token pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Token pre-warm failed (will retry on first request): {e}")
    yield
    await token_mgr.close_client()


app = FastAPI(
    title="Salvage Vehicle eBay Pricing API",
    description="Proxy server for eBay Browse API — provides vehicle parts pricing for salvage damage analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Response Models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ebay_token: str
    anthropic_key: str
    marketplace: str

class PriceResult(BaseModel):
    title: str
    price: float
    currency: str
    condition: Optional[str] = None
    item_id: str
    image_url: Optional[str] = None
    item_url: str
    seller_feedback: Optional[float] = None
    shipping_cost: Optional[float] = None

class SearchResponse(BaseModel):
    query: str
    marketplace: str
    total: int
    results: list[PriceResult]
    avg_price: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and API connectivity."""
    token_status = "valid" if token_mgr.access_token and time.time() < token_mgr.expires_at else "needs_refresh"
    anthropic_status = "configured" if ANTHROPIC_API_KEY else "missing"
    return HealthResponse(
        status="ok",
        ebay_token=token_status,
        anthropic_key=anthropic_status,
        marketplace=EBAY_MARKETPLACE,
    )


@app.get("/api/search", response_model=SearchResponse)
async def search_parts(
    q: str = Query(..., description="Search query, e.g. '2018 BMW 3 Series front bumper'"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
    sort: str = Query("BEST_MATCH", description="Sort: BEST_MATCH, price, -price, newlyListed"),
    min_price: Optional[float] = Query(None, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, description="Maximum price filter"),
    condition: Optional[str] = Query(None, description="Filter: NEW, USED, PARTS_ONLY, etc."),
    marketplace: Optional[str] = Query(None, description="Override marketplace, e.g. EBAY_DE"),
):
    """
    Search eBay for vehicle parts/components and return pricing data.
    Designed for salvage vehicle damage cost estimation.
    """
    token = await token_mgr.get_token()
    mktplace = marketplace or EBAY_MARKETPLACE

    params = {"q": q, "limit": str(limit)}

    # Sort mapping — BEST_MATCH is eBay's default (no param needed)
    if sort and sort != "BEST_MATCH":
        sort_map = {"price": "price", "-price": "-price", "newlyListed": "newlyListed"}
        if sort in sort_map:
            params["sort"] = sort_map[sort]

    # Price filter
    filters = []
    if min_price is not None:
        filters.append(f"price:[{min_price}..{'*' if max_price is None else ''}]")
    if max_price is not None:
        if min_price is None:
            filters.append(f"price:[*..{max_price}]")
        else:
            filters.append(f"price:[{min_price}..{max_price}]")
    if condition:
        filters.append(f"conditions:{{{condition}}}")
    if filters:
        params["filter"] = ",".join(filters)

    resp = await token_mgr.http_client.get(
        f"{EBAY_API_BASE}/buy/browse/v1/item_summary/search",
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": mktplace,
        },
        params=params,
    )

    if resp.status_code == 401:
        # Token expired mid-flight — force refresh and retry once
        token_mgr.access_token = None
        token = await token_mgr.get_token()
        resp = await token_mgr.http_client.get(
            f"{EBAY_API_BASE}/buy/browse/v1/item_summary/search",
            headers={
                "Authorization": f"Bearer {token}",
                "X-EBAY-C-MARKETPLACE-ID": mktplace,
            },
            params=params,
        )

    if resp.status_code != 200:
        logger.error(f"eBay search failed: {resp.status_code} {resp.text[:300]}")
        raise HTTPException(resp.status_code, f"eBay API error: {resp.status_code}")

    data = resp.json()
    total = data.get("total", 0)
    items = data.get("itemSummaries", [])

    results = []
    for item in items:
        price_val = float(item.get("price", {}).get("value", 0))
        currency = item.get("price", {}).get("currency", "USD")

        shipping = None
        shipping_opts = item.get("shippingOptions", [])
        if shipping_opts:
            ship_cost = shipping_opts[0].get("shippingCost", {})
            if ship_cost.get("value"):
                shipping = float(ship_cost["value"])

        image = None
        if item.get("image", {}).get("imageUrl"):
            image = item["image"]["imageUrl"]
        elif item.get("thumbnailImages"):
            image = item["thumbnailImages"][0].get("imageUrl")

        results.append(PriceResult(
            title=item.get("title", ""),
            price=price_val,
            currency=currency,
            condition=item.get("condition", None),
            item_id=item.get("itemId", ""),
            image_url=image,
            item_url=item.get("itemWebUrl", ""),
            seller_feedback=item.get("seller", {}).get("feedbackPercentage"),
            shipping_cost=shipping,
        ))

    prices = [r.price for r in results if r.price > 0]
    return SearchResponse(
        query=q,
        marketplace=mktplace,
        total=total,
        results=results,
        avg_price=round(sum(prices) / len(prices), 2) if prices else None,
        min_price=min(prices) if prices else None,
        max_price=max(prices) if prices else None,
    )


@app.get("/api/item/{item_id}")
async def get_item_details(item_id: str):
    """Get detailed info for a specific eBay item by ID."""
    token = await token_mgr.get_token()

    resp = await token_mgr.http_client.get(
        f"{EBAY_API_BASE}/buy/browse/v1/item/{item_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": EBAY_MARKETPLACE,
        },
    )

    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"eBay item lookup failed: {resp.status_code}")

    return resp.json()


@app.get("/api/damage-estimate")
async def damage_price_estimate(
    vehicle: str = Query(..., description="Vehicle, e.g. '2019 Toyota Camry'"),
    parts: str = Query(..., description="Comma-separated damaged parts, e.g. 'front bumper,headlight,hood'"),
    marketplace: Optional[str] = Query(None),
):
    """
    Multi-part price lookup: searches eBay for each damaged part on a vehicle
    and returns aggregated cost estimate.
    """
    part_list = [p.strip() for p in parts.split(",") if p.strip()]
    if not part_list:
        raise HTTPException(400, "No parts specified")

    mktplace = marketplace or EBAY_MARKETPLACE
    estimates = []
    total_min = 0.0
    total_avg = 0.0
    total_max = 0.0

    for part in part_list:
        query = f"{vehicle} {part}"
        try:
            result = await search_parts(q=query, limit=10, sort="BEST_MATCH", min_price=15.0, max_price=None, condition=None, marketplace=mktplace)
            estimates.append({
                "part": part,
                "query": query,
                "results_found": result.total,
                "avg_price": result.avg_price,
                "min_price": result.min_price,
                "max_price": result.max_price,
                "sample_results": [r.model_dump() for r in result.results[:3]],
            })
            total_min += result.min_price or 0
            total_avg += result.avg_price or 0
            total_max += result.max_price or 0
        except Exception as e:
            estimates.append({
                "part": part,
                "query": query,
                "error": str(e),
                "results_found": 0,
            })

    return {
        "vehicle": vehicle,
        "marketplace": mktplace,
        "parts_analyzed": len(part_list),
        "total_estimate": {
            "min": round(total_min, 2),
            "avg": round(total_avg, 2),
            "max": round(total_max, 2),
        },
        "per_part": estimates,
    }


# ── Auction URL Fetch Endpoint ────────────────────────────────────────────────

@app.get("/api/fetch-lot")
async def fetch_lot(url: str = Query(..., description="Auction listing URL")):
    """
    Fetch and extract vehicle data from any auction listing URL.
    Tier 1: Native JSON extraction for AutoBidMaster (richest data).
    Tier 2: AI-powered HTML parsing for any other auction site.
    """
    import json as json_mod
    import re

    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url

    headers_fetch = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Cookie": "screenSize=1920x1080",
    }

    # ── Copart/IAAI → route through AutoBidMaster ──
    # These sites are JS-rendered and block server scraping.
    # AutoBidMaster mirrors Copart/IAAI lots with SSR content.
    copart_lot = None
    if "copart.com" in url:
        import re as _re
        m = _re.search(r'/lot/(\d+)', url)
        if m:
            copart_lot = m.group(1)
        else:
            # Try extracting lot number from the end of URL
            m = _re.search(r'(\d{7,9})', url)
            if m:
                copart_lot = m.group(1)
        if copart_lot:
            url = f"https://www.autobidmaster.com/en/search/lot/{copart_lot}/"
            logger.info(f"Redirecting Copart lot {copart_lot} through AutoBidMaster")
        else:
            raise HTTPException(400, "Could not extract lot number from Copart URL. Please use the full lot URL (e.g. copart.com/lot/12345678)")

    if "iaai.com" in url:
        import re as _re
        m = _re.search(r'/VehicleDetail/(\d+)', url) or _re.search(r'(\d{7,9})', url)
        if m:
            iaai_lot = m.group(1)
            url = f"https://www.autobidmaster.com/en/search/lot/{iaai_lot}/"
            logger.info(f"Redirecting IAAI lot {iaai_lot} through AutoBidMaster")

    # ── Tier 1: AutoBidMaster native extraction ──
    if "autobidmaster.com" in url:
        # Ensure fallback=true for SSR content
        fetch_url = url if "fallback=true" in url else url.rstrip("/") + "/?fallback=true"
        try:
            resp = await token_mgr.http_client.get(fetch_url, headers=headers_fetch, follow_redirects=True, timeout=20)
            if resp.status_code != 200:
                raise HTTPException(502, f"Could not fetch page: HTTP {resp.status_code}")

            text = resp.text
            # Extract __REACT_QUERY_STATE__
            scripts = re.findall(r'<script[^>]*>(.*?)</script>', text, re.DOTALL)
            lot = None
            for script in scripts:
                m = re.search(r'window\.__REACT_QUERY_STATE__\s*=\s*(\{)', script)
                if not m:
                    continue
                start = m.start(1)
                depth = 0
                end = start
                for j in range(start, len(script)):
                    if script[j] == '{': depth += 1
                    elif script[j] == '}': depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
                try:
                    state = json_mod.loads(script[start:end])
                    for q in state.get("queries", []):
                        qdata = q.get("state", {}).get("data", {})
                        if isinstance(qdata, dict) and "lot" in qdata:
                            lot = qdata["lot"]
                            break
                except json_mod.JSONDecodeError:
                    continue
                if lot:
                    break

            if not lot:
                raise HTTPException(422, "Could not extract lot data from AutoBidMaster page")

            # Extract images (full resolution)
            images = []
            for img in lot.get("images", []):
                img_url = img.get("hdr") or img.get("full") or img.get("thumbnail")
                if img_url:
                    images.append({"url": img_url, "label": img.get("label")})

            # Extract title info
            title_info = lot.get("title", {})
            title_name = title_info.get("name", "") if isinstance(title_info, dict) else str(title_info)

            return {
                "source": "autobidmaster",
                "source_url": url,
                "extraction_method": "native_json",
                "vehicle": {
                    "year": lot.get("year"),
                    "make": lot.get("make", ""),
                    "model": lot.get("model", ""),
                    "description": lot.get("description", ""),
                    "vin": lot.get("vin", ""),
                    "color": lot.get("color", ""),
                    "engine": lot.get("engineSize", ""),
                    "cylinders": lot.get("cylinders"),
                    "drive": lot.get("drive", ""),
                    "transmission": lot.get("transmission", ""),
                    "fuel": lot.get("fuel", ""),
                    "body_style": lot.get("bodyStyle", ""),
                    "odometer": lot.get("odometer"),
                    "odometer_type": lot.get("odometerType", "mi"),
                    "odometer_brand": lot.get("odometerBrand", ""),
                },
                "auction": {
                    "platform": lot.get("inventoryAuction", ""),
                    "lot_number": lot.get("lotNumber"),
                    "current_bid": lot.get("currentBid") or lot.get("highBid"),
                    "buy_now": lot.get("buyItNow"),
                    "suggested_bid": lot.get("suggestedBid"),
                    "starting_bid": lot.get("startingBid"),
                    "currency": lot.get("currency", "USD"),
                    "sale_date": lot.get("saleDate"),
                    "sale_status": lot.get("saleStatusString", ""),
                    "sold": lot.get("sold", False),
                    "location": lot.get("locationName", ""),
                    "location_state": lot.get("locationState", ""),
                },
                "condition": {
                    "primary_damage": lot.get("primaryDamage", ""),
                    "secondary_damage": lot.get("secondaryDamage", ""),
                    "title_type": title_name,
                    "title_state": lot.get("titleState", ""),
                    "run_drives": lot.get("runDrives", ""),
                    "driveable": lot.get("drivable"),
                    "keys": lot.get("keysStatus", ""),
                    "airbag_status": lot.get("airbagStatus", ""),
                    "engine_missing": lot.get("engineMissing", False),
                    "transmission_missing": lot.get("transmissionMissing", False),
                },
                "valuation": {
                    "acv": float(lot["acv"]) if lot.get("acv") else None,
                    "repair_cost": float(lot["repairCost"]) if lot.get("repairCost") else None,
                },
                "images": images,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"AutoBidMaster extraction failed: {e}")
            raise HTTPException(500, f"Failed to parse AutoBidMaster page: {str(e)}")

    # ── Tier 2: AI-powered extraction for any other URL ──
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "Anthropic API key not configured — needed for non-AutoBidMaster URLs")

    try:
        resp = await token_mgr.http_client.get(url, headers=headers_fetch, follow_redirects=True, timeout=20)
        if resp.status_code != 200:
            raise HTTPException(502, f"Could not fetch page: HTTP {resp.status_code}")

        html = resp.text
        if len(html) < 500:
            raise HTTPException(422, "Page returned too little content — may require JavaScript or login")

        # Truncate HTML to fit in context (keep first 30k chars, should include all vehicle data)
        html_truncated = html[:30000]

        # Use Claude to extract structured data
        extract_prompt = """Extract vehicle auction listing data from this HTML page. 
Return ONLY valid JSON (no markdown, no backticks):
{
  "vehicle": {"year": 2020, "make": "Toyota", "model": "Camry", "description": "full title/description", "vin": "if found", "color": "", "engine": "", "drive": "", "transmission": "", "fuel": "", "odometer": null},
  "auction": {"platform": "site name", "lot_number": null, "current_bid": null, "buy_now": null, "sale_date": "", "sale_status": "", "location": ""},
  "condition": {"primary_damage": "", "secondary_damage": "", "title_type": "", "run_drives": "", "driveable": null, "keys": ""},
  "valuation": {"acv": null, "repair_cost": null},
  "images": [{"url": "full image URL", "label": null}]
}
Extract ALL image URLs that show the vehicle (full/high resolution preferred over thumbnails).
If a field is not found, use null or empty string. Extract prices as numbers without $ or commas."""

        ai_resp = await token_mgr.http_client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": f"{extract_prompt}\n\nHTML:\n{html_truncated}"}],
            },
            timeout=60,
        )

        if ai_resp.status_code != 200:
            err = ai_resp.json().get("error", {}).get("message", "AI extraction failed")
            raise HTTPException(502, err)

        ai_text = "".join(c.get("text", "") for c in ai_resp.json().get("content", []))
        cleaned = ai_text.replace("```json", "").replace("```", "").strip()
        # Fix trailing commas
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

        try:
            result = json_mod.loads(cleaned)
        except json_mod.JSONDecodeError:
            raise HTTPException(422, "Could not parse extracted data")

        result["source"] = re.sub(r'https?://(?:www\.)?([^/]+).*', r'\1', url)
        result["source_url"] = url
        result["extraction_method"] = "ai_parsed"
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI lot extraction failed: {e}")
        raise HTTPException(500, f"Failed to extract data: {str(e)}")


# ── VIN Decoder Endpoint ──────────────────────────────────────────────────────

@app.get("/api/vin/{vin}")
async def decode_vin(vin: str):
    """Decode a VIN using the free NHTSA vPIC API."""
    vin = vin.strip().upper()
    if len(vin) != 17:
        raise HTTPException(400, "VIN must be exactly 17 characters")

    resp = await token_mgr.http_client.get(
        f"https://vpic.nhtsa.dot.gov/api/vehicles/decodevin/{vin}?format=json",
        timeout=15,
    )
    if resp.status_code != 200:
        raise HTTPException(502, "NHTSA API error")

    raw = resp.json()
    fields = {}
    for item in raw.get("Results", []):
        if item.get("Value") and item["Value"].strip() and item["Value"].strip() != "Not Applicable":
            fields[item["Variable"]] = item["Value"].strip()

    # Map to clean structure
    make = fields.get("Make", "").title()
    model = fields.get("Model", "")
    year = fields.get("Model Year", "")
    trim = fields.get("Trim", "")
    body = fields.get("Body Class", "")
    drive = fields.get("Drive Type", "")
    fuel = fields.get("Fuel Type - Primary", "")
    engine = fields.get("Displacement (L)", "")
    cylinders = fields.get("Engine Number of Cylinders", "")
    plant_country = fields.get("Plant Country", "")
    vehicle_type = fields.get("Vehicle Type", "")
    gvwr = fields.get("Gross Vehicle Weight Rating From", "")

    vehicle_str = f"{year} {make} {model}".strip()
    if trim:
        vehicle_str += f" {trim}"

    return {
        "vin": vin,
        "vehicle": vehicle_str,
        "year": year,
        "make": make,
        "model": model,
        "trim": trim,
        "body_class": body,
        "drive_type": drive,
        "fuel_type": fuel,
        "engine": f"{engine}L {cylinders}cyl".strip("L cyl") if engine or cylinders else "",
        "plant_country": plant_country,
        "vehicle_type": vehicle_type,
        "all_fields": fields,
    }


# ── Market Value Endpoint ─────────────────────────────────────────────────────

@app.get("/api/market-value")
async def get_market_value(
    vehicle: str = Query(..., description="e.g. '2020 Tesla Model 3'"),
    marketplace: Optional[str] = Query(None),
):
    """
    Estimate market value by searching eBay for comparable clean-title vehicle listings.
    Returns clean-title range + estimated salvage value.
    """
    token = await token_mgr.get_token()
    mktplace = marketplace or EBAY_MARKETPLACE
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": mktplace}

    # Search for clean title vehicles
    clean_resp = await token_mgr.http_client.get(
        f"{EBAY_API_BASE}/buy/browse/v1/item_summary/search",
        headers=headers,
        params={
            "q": vehicle,
            "limit": "20",
            "category_ids": "6001",
            "filter": "price:[3000..],conditions:{USED}",
            "sort": "price",
        },
    )

    clean_prices = []
    clean_listings = []
    if clean_resp.status_code == 200:
        data = clean_resp.json()
        for item in data.get("itemSummaries", []):
            price = float(item.get("price", {}).get("value", 0))
            if price >= 3000:
                clean_prices.append(price)
                clean_listings.append({
                    "title": item.get("title", ""),
                    "price": price,
                    "url": item.get("itemWebUrl", ""),
                    "image": item.get("image", {}).get("imageUrl"),
                    "condition": item.get("condition"),
                })

    # Calculate market value ranges
    if clean_prices:
        clean_prices.sort()
        # Remove outliers (top/bottom 10%)
        trim_count = max(1, len(clean_prices) // 10)
        trimmed = clean_prices[trim_count:-trim_count] if len(clean_prices) > 4 else clean_prices
        avg_clean = sum(trimmed) / len(trimmed)
        low_clean = trimmed[0]
        high_clean = trimmed[-1]
    else:
        avg_clean = low_clean = high_clean = 0

    # Salvage value is typically 20-40% of clean retail
    salvage_low = round(avg_clean * 0.15, -2) if avg_clean else 0
    salvage_avg = round(avg_clean * 0.25, -2) if avg_clean else 0
    salvage_high = round(avg_clean * 0.40, -2) if avg_clean else 0

    return {
        "vehicle": vehicle,
        "clean_title": {
            "avg": round(avg_clean, 0),
            "low": round(low_clean, 0),
            "high": round(high_clean, 0),
            "sample_count": len(clean_prices),
            "listings": clean_listings[:5],
        },
        "salvage_estimate": {
            "low": salvage_low,
            "avg": salvage_avg,
            "high": salvage_high,
            "note": "Estimated at 15-40% of clean retail depending on damage severity",
        },
    }


# ── Analyze Endpoint (proxies Anthropic vision API) ──────────────────────────

class AnalyzeRequest(BaseModel):
    images: list[dict]  # [{media_type: str, data: str (base64)}]
    vehicle: str
    mileage: str = ""
    notes: str = ""

@app.post("/api/analyze")
async def analyze_damage(req: AnalyzeRequest):
    """
    Analyze vehicle damage from images using Claude vision.
    Keeps the Anthropic API key server-side.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "Anthropic API key not configured")
    if not req.images:
        raise HTTPException(400, "No images provided")

    # Build message content with images
    content = []
    for img in req.images[:8]:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.get("media_type", "image/jpeg"),
                "data": img["data"],
            }
        })

    prompt = f"""You are an expert vehicle damage assessor and insurance adjuster with 20 years of experience. Analyze these images of a {req.vehicle}{(' with ' + req.mileage + ' miles') if req.mileage else ''}.
{('Additional context: ' + req.notes) if req.notes else ''}

Perform a thorough professional assessment. For each damaged component, provide detailed analysis.

Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "vehicle": "{req.vehicle}",
  "overall_assessment": "2-3 sentence professional summary of the damage",
  "estimated_severity": "total_loss | heavy | moderate | light",
  "structural_integrity": "compromised | likely_safe | safe",
  "driveable": true or false,
  "safety_concerns": ["List any safety issues, e.g. airbag deployed, structural damage, lighting failure"],
  "recommended_next_steps": ["e.g. Get structural frame check", "Replace all lighting before driving"],
  "damaged_parts": [
    {{
      "part_name": "Front bumper cover",
      "category": "body | lighting | glass | mechanical | structural | interior | electrical",
      "severity": "critical | major | moderate | minor",
      "confidence": 85,
      "damage_type": "crack | dent | scratch | shatter | deformation | puncture | discoloration | misalignment",
      "description": "Detailed description of the visible damage",
      "action": "Replace | Repair | Repair or Replace | Monitor",
      "repair_possible": false,
      "repair_difficulty": "easy | moderate | hard | specialist",
      "labor_hours_estimate": 1.5,
      "paint_required": true,
      "oem_vs_aftermarket": "OEM recommended | Aftermarket OK | Either",
      "hidden_damage_risk": "low | medium | high",
      "hidden_damage_note": "Brief note about what hidden damage to check for",
      "ebay_query": "{req.vehicle} front bumper cover"
    }}
  ]
}}

Important guidelines:
- confidence is 0-100, how confident you are in the assessment from the photos
- labor_hours_estimate is for a professional body shop
- hidden_damage_risk flags parts where damage behind visible panels is likely
- Be specific about damage_type, not generic
- Always flag safety concerns prominently
- For structural parts, always recommend professional inspection"""

    content.append({"type": "text", "text": prompt})

    try:
        resp = await token_mgr.http_client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=90,
        )

        if resp.status_code != 200:
            error_data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            error_msg = error_data.get("error", {}).get("message", f"Anthropic API error: {resp.status_code}")
            logger.error(f"Anthropic API error: {resp.status_code} - {error_msg}")
            raise HTTPException(resp.status_code, error_msg)

        data = resp.json()
        text = "".join(c.get("text", "") for c in data.get("content", []))
        cleaned = text.replace("```json", "").replace("```", "").strip()

        import json, re

        # Try parsing directly first
        try:
            result = json.loads(cleaned)
            return result
        except json.JSONDecodeError:
            pass

        # JSON repair: common issues from LLM output
        repaired = cleaned

        # Fix trailing commas before } or ]
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # Fix unescaped quotes inside string values
        # This handles cases like "description": "The "bumper" is cracked"
        lines = repaired.split('\n')
        fixed_lines = []
        for line in lines:
            # If line has a key:value pattern with string value
            m = re.match(r'^(\s*"[^"]+"\s*:\s*")(.*)(",?\s*)$', line)
            if m:
                prefix, val, suffix = m.groups()
                # Escape unescaped quotes in the value
                val = val.replace('\\"', '\x00').replace('"', '\\"').replace('\x00', '\\"')
                line = prefix + val + suffix
            fixed_lines.append(line)
        repaired = '\n'.join(fixed_lines)

        try:
            result = json.loads(repaired)
            logger.info("JSON repaired successfully")
            return result
        except json.JSONDecodeError:
            pass

        # Last resort: try to extract just the damaged_parts array and build minimal response
        try:
            # Find the outermost JSON object
            start = repaired.index('{')
            depth = 0
            end = start
            for i in range(start, len(repaired)):
                if repaired[i] == '{': depth += 1
                elif repaired[i] == '}': depth -= 1
                if depth == 0:
                    end = i + 1
                    break
            # Truncate at last complete object/array
            truncated = repaired[start:end]
            result = json.loads(truncated)
            logger.info("JSON recovered via truncation")
            return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Final fallback: extract what we can
        logger.warning(f"JSON parse failed, attempting field extraction. Raw length: {len(cleaned)}")
        fallback = {
            "vehicle": req.vehicle,
            "overall_assessment": "Analysis completed but response format was unexpected. Please try again.",
            "estimated_severity": "moderate",
            "structural_integrity": "unknown",
            "driveable": True,
            "safety_concerns": [],
            "recommended_next_steps": ["Re-run analysis or consult a professional"],
            "damaged_parts": []
        }

        # Try to extract individual parts from the malformed JSON
        part_pattern = r'\{[^{}]*"part_name"\s*:\s*"([^"]+)"[^{}]*\}'
        for match in re.finditer(part_pattern, cleaned):
            try:
                part_json = match.group(0)
                # Fix trailing commas
                part_json = re.sub(r',\s*}', '}', part_json)
                part = json.loads(part_json)
                fallback["damaged_parts"].append(part)
            except json.JSONDecodeError:
                continue

        if fallback["damaged_parts"]:
            logger.info(f"Extracted {len(fallback['damaged_parts'])} parts from malformed JSON")
        return fallback

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# ── Frontend HTML ─────────────────────────────────────────────────────────────

FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Salvage Analyst — Vehicle Damage Assessment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#F8FAFC;--surface:#FFF;--border:#E2E8F0;--border-hover:#CBD5E1;--text-primary:#0F172A;--text-secondary:#475569;--text-tertiary:#94A3B8;--accent:#2563EB;--accent-hover:#1D4ED8;--accent-light:#EFF6FF;--accent-subtle:#DBEAFE;--danger:#EF4444;--success:#10B981;--warning:#F59E0B;--shadow-sm:0 1px 2px rgba(0,0,0,.04);--shadow-md:0 4px 12px rgba(0,0,0,.06);--radius-sm:8px;--radius-md:12px;--radius-lg:16px}
body{font-family:'DM Sans',-apple-system,sans-serif;background:var(--bg);color:var(--text-primary);-webkit-font-smoothing:antialiased}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideIn{from{opacity:0;transform:translateX(-12px)}to{opacity:1;transform:translateX(0)}}
@keyframes progressGlow{0%,100%{box-shadow:0 0 8px rgba(37,99,235,.3)}50%{box-shadow:0 0 16px rgba(37,99,235,.5)}}
.app{max-width:960px;margin:0 auto;padding:32px 24px 64px;min-height:100vh}
.header{text-align:center;margin-bottom:40px;animation:fadeIn .5s ease}
.logo{display:inline-flex;align-items:center;gap:10px;margin-bottom:8px}
.logo-icon{width:40px;height:40px;background:var(--accent);border-radius:10px;display:flex;align-items:center;justify-content:center}
.logo-icon svg{color:#fff}
.title{font-family:'Space Mono',monospace;font-size:22px;font-weight:700;letter-spacing:-.5px}
.subtitle{font-size:14px;color:var(--text-tertiary);margin-top:4px}
.steps{display:flex;align-items:center;justify-content:center;margin-bottom:36px;animation:fadeIn .5s ease .1s both}
.step{display:flex;align-items:center;gap:8px;padding:8px 16px;font-size:13px;font-weight:500;color:var(--text-tertiary)}
.step.active{color:var(--accent)}
.step.done{color:var(--success)}
.step-n{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;background:var(--border);color:var(--text-tertiary);flex-shrink:0}
.step.active .step-n{background:var(--accent);color:#fff}
.step.done .step-n{background:var(--success);color:#fff}
.step-line{width:32px;height:2px;background:var(--border);flex-shrink:0}
.step-line.done{background:var(--success)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:28px;box-shadow:var(--shadow-sm);animation:fadeIn .4s ease;margin-bottom:20px}
.card-title{font-size:16px;font-weight:600;margin-bottom:20px;display:flex;align-items:center;gap:8px}
.badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;background:var(--accent-light);color:var(--accent)}
.dropzone{border:2px dashed var(--border);border-radius:var(--radius-md);padding:40px 24px;text-align:center;cursor:pointer;transition:all .2s;background:var(--bg)}
.dropzone:hover,.dropzone.dragover{border-color:var(--accent);background:var(--accent-light)}
.dropzone-label{font-size:15px;font-weight:500;color:var(--text-secondary);margin-top:12px}
.dropzone-sub{font-size:13px;color:var(--text-tertiary);margin-top:4px}
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:12px;margin-top:16px}
.img-thumb{position:relative;aspect-ratio:1;border-radius:var(--radius-sm);overflow:hidden;border:1px solid var(--border)}
.img-thumb img{width:100%;height:100%;object-fit:cover}
.img-rm{position:absolute;top:6px;right:6px;width:24px;height:24px;background:rgba(0,0,0,.6);border:none;border-radius:50%;color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .15s;font-size:14px}
.img-thumb:hover .img-rm{opacity:1}
.form-row{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px}
.fg{display:flex;flex-direction:column;gap:6px}
.fl{font-size:12px;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.5px}
.fi{padding:10px 14px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:14px;font-family:'DM Sans',sans-serif;color:var(--text-primary);background:var(--bg);transition:border-color .15s;outline:none}
.fi:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-subtle)}
.fi::placeholder{color:var(--text-tertiary)}
textarea.fi{resize:vertical;min-height:72px}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;padding:12px 28px;border-radius:var(--radius-sm);font-size:14px;font-weight:600;font-family:'DM Sans',sans-serif;cursor:pointer;border:none;transition:all .15s}
.btn-p{background:var(--accent);color:#fff}
.btn-p:hover{background:var(--accent-hover);transform:translateY(-1px);box-shadow:var(--shadow-md)}
.btn-p:disabled{opacity:.5;cursor:not-allowed;transform:none}
.btn-s{background:var(--bg);color:var(--text-secondary);border:1px solid var(--border)}
.btn-s:hover{border-color:var(--border-hover);background:#fff}
.btn-g{background:var(--success);color:#fff}
.btn-g:hover{background:#059669;transform:translateY(-1px)}
.btn-row{display:flex;gap:12px;margin-top:24px;justify-content:flex-end}
.progress{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 24px;animation:fadeIn .4s ease}
.progress-track{width:100%;height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin-top:24px;max-width:320px}
.progress-fill{height:100%;background:var(--accent);border-radius:3px;transition:width .4s;animation:progressGlow 2s infinite}
.progress-label{font-size:18px;font-weight:600;margin-top:20px}
.progress-sub{font-size:14px;color:var(--text-tertiary);margin-top:6px}
.rh{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;animation:fadeIn .4s ease}
.rt{font-family:'Space Mono',monospace;font-size:20px;font-weight:700}
.sev-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid}
.sev-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.overview{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-lg);padding:24px;margin-bottom:20px;animation:fadeIn .4s ease .1s both}
.overview-text{font-size:15px;color:var(--text-secondary);line-height:1.6}
.overview-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:20px}
.stat{background:var(--bg);border-radius:var(--radius-sm);padding:14px;text-align:center}
.stat-v{font-family:'Space Mono',monospace;font-size:22px;font-weight:700}
.stat-l{font-size:11px;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:.5px;margin-top:4px}
.pc{border:1px solid var(--border);border-radius:var(--radius-md);overflow:hidden;margin-bottom:12px;background:var(--surface);transition:box-shadow .2s;animation:slideIn .3s ease both}
.pc:hover{box-shadow:var(--shadow-md)}
.pc-h{display:flex;align-items:center;gap:14px;padding:16px 20px;cursor:pointer;user-select:none}
.pc-sev{width:4px;height:36px;border-radius:2px;flex-shrink:0}
.pc-info{flex:1}
.pc-name{font-size:15px;font-weight:600}
.pc-action{font-size:13px;color:var(--text-secondary);margin-top:2px}
.pc-chev{color:var(--text-tertiary);transition:transform .2s;flex-shrink:0}
.pc-chev.open{transform:rotate(90deg)}
.pc-body{padding:0 20px 20px;border-top:1px solid var(--border);animation:fadeIn .2s ease}
.pc-desc{font-size:14px;color:var(--text-secondary);line-height:1.6;padding-top:16px}
.pr-card{border:1px solid var(--border);border-radius:var(--radius-md);overflow:hidden;margin-bottom:16px;background:var(--surface);animation:fadeIn .4s ease both}
.pr-head{padding:18px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);background:var(--bg)}
.pr-name{font-size:15px;font-weight:600}
.pr-avg{font-family:'Space Mono',monospace;font-size:18px;font-weight:700;color:var(--accent)}
.pr-range{font-size:12px;color:var(--text-tertiary);margin-top:2px}
.lg{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;padding:16px 20px}
.li{display:flex;gap:12px;padding:12px;border:1px solid var(--border);border-radius:var(--radius-sm);transition:all .15s;text-decoration:none;color:inherit}
.li:hover{border-color:var(--accent);background:var(--accent-light);transform:translateY(-1px);box-shadow:var(--shadow-sm)}
.li-img{width:56px;height:56px;border-radius:6px;object-fit:cover;background:var(--bg);flex-shrink:0}
.li-info{flex:1;min-width:0}
.li-title{font-size:13px;font-weight:500;line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.li-meta{display:flex;align-items:center;gap:8px;margin-top:4px}
.li-price{font-family:'Space Mono',monospace;font-size:14px;font-weight:700;color:var(--success)}
.li-cond{font-size:11px;color:var(--text-tertiary);padding:1px 6px;background:var(--bg);border-radius:4px}
.total-bar{background:var(--surface);border:2px solid var(--accent);border-radius:var(--radius-lg);padding:24px 28px;display:flex;align-items:center;justify-content:space-between;margin-top:24px;animation:fadeIn .4s ease .3s both}
.total-v{font-family:'Space Mono',monospace;font-size:28px;font-weight:700;color:var(--accent)}
.total-r{font-size:13px;color:var(--text-tertiary);margin-top:2px;text-align:right}
.error{background:#FEF2F2;border:1px solid #FECACA;color:#991B1B;padding:12px 16px;border-radius:var(--radius-sm);font-size:14px;margin-bottom:16px;display:flex;align-items:center;justify-content:space-between;animation:fadeIn .3s ease}
.spinner{width:24px;height:24px;animation:spin 1s linear infinite}
@media(max-width:640px){.form-row{grid-template-columns:1fr 1fr}.overview-stats{grid-template-columns:1fr 1fr}.lg{grid-template-columns:1fr}.app{padding:20px 16px 48px}.step{padding:6px 8px;font-size:12px}}
.ac-wrap{position:relative}
.ac-list{position:absolute;top:100%;left:0;right:0;max-height:220px;overflow-y:auto;background:var(--surface);border:1px solid var(--accent);border-radius:0 0 var(--radius-sm) var(--radius-sm);box-shadow:var(--shadow-md);z-index:100;margin-top:-1px}
.ac-item{padding:8px 14px;font-size:14px;cursor:pointer;transition:background .1s}
.ac-item:hover,.ac-item.hl{background:var(--accent-light);color:var(--accent)}
.ac-item.selected{font-weight:600;color:var(--accent)}
.ac-empty{padding:10px 14px;font-size:13px;color:var(--text-tertiary);font-style:italic}
.alert-bar{display:flex;align-items:flex-start;gap:12px;padding:16px 20px;border-radius:var(--radius-md);margin-bottom:16px;font-size:14px;line-height:1.5;animation:fadeIn .4s ease}
.alert-bar.warn{background:#FEF3C7;border:1px solid #FDE68A;color:#92400E}
.alert-bar.danger{background:#FEE2E2;border:1px solid #FECACA;color:#991B1B}
.alert-bar.info{background:var(--accent-light);border:1px solid var(--accent-subtle);color:#1E40AF}
.alert-bar.ok{background:#ECFDF5;border:1px solid #A7F3D0;color:#065F46}
.alert-icon{flex-shrink:0;margin-top:1px}
.tag{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:.3px}
.tag-body{background:#DBEAFE;color:#1E40AF}
.tag-lighting{background:#FEF3C7;color:#92400E}
.tag-glass{background:#E0E7FF;color:#3730A3}
.tag-mechanical{background:#F3E8FF;color:#6B21A8}
.tag-structural{background:#FEE2E2;color:#991B1B}
.tag-interior{background:#F0FDF4;color:#166534}
.tag-electrical{background:#FFF7ED;color:#9A3412}
.conf-bar{width:100%;height:4px;background:var(--border);border-radius:2px;overflow:hidden;margin-top:6px}
.conf-fill{height:100%;border-radius:2px;transition:width .6s ease}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px 20px;margin-top:16px}
.detail-item{display:flex;flex-direction:column;gap:2px}
.detail-label{font-size:11px;font-weight:600;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:.5px}
.detail-value{font-size:13px;color:var(--text-primary);font-weight:500}
.hidden-risk{display:flex;align-items:center;gap:8px;padding:10px 14px;border-radius:var(--radius-sm);margin-top:12px;font-size:13px}
.hidden-risk.low{background:#F0FDF4;color:#166534;border:1px solid #BBF7D0}
.hidden-risk.medium{background:#FEF9C3;color:#854D0E;border:1px solid #FDE68A}
.hidden-risk.high{background:#FEE2E2;color:#991B1B;border:1px solid #FECACA}
.next-steps{margin-top:20px;padding:20px;background:var(--bg);border-radius:var(--radius-md);border:1px solid var(--border)}
.next-step-item{display:flex;align-items:flex-start;gap:10px;padding:8px 0;font-size:14px;color:var(--text-secondary)}
.next-step-item+.next-step-item{border-top:1px solid var(--border)}
.next-step-num{width:22px;height:22px;border-radius:50%;background:var(--accent);color:#fff;font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.labor-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px;padding:16px;background:var(--bg);border-radius:var(--radius-md)}
@media(max-width:640px){.detail-grid{grid-template-columns:1fr}.labor-summary{grid-template-columns:1fr}}
</style>
</head>
<body>
<div id="app" class="app"></div>
<script>
const API = '';
const SEV = {critical:{bg:'#FEE2E2',bd:'#F87171',tx:'#991B1B',dot:'#EF4444'},major:{bg:'#FEF3C7',bd:'#FBBF24',tx:'#92400E',dot:'#F59E0B'},moderate:{bg:'#FEF9C3',bd:'#FACC15',tx:'#854D0E',dot:'#EAB308'},minor:{bg:'#ECFDF5',bd:'#6EE7B7',tx:'#065F46',dot:'#10B981'}};
const SLABEL = {critical:'Replace',major:'Repair / Replace',moderate:'Repair',minor:'Inspect'};

const YEARS=[];for(let y=2026;y>=1990;y--)YEARS.push(String(y));

const VEHICLES={"Acura":["ILX","Integra","MDX","NSX","RDX","RLX","TL","TLX","TSX","ZDX"],"Alfa Romeo":["4C","Giulia","Giulietta","Stelvio","Tonale"],"Aston Martin":["DB11","DB12","DBS","DBX","Vantage"],"Audi":["A3","A4","A5","A6","A7","A8","e-tron","e-tron GT","Q3","Q4 e-tron","Q5","Q7","Q8","R8","RS3","RS5","RS6","RS7","S3","S4","S5","S6","S7","S8","TT"],"Bentley":["Bentayga","Continental GT","Flying Spur"],"BMW":["1 Series","2 Series","3 Series","4 Series","5 Series","6 Series","7 Series","8 Series","i3","i4","i5","i7","iX","iX3","M2","M3","M4","M5","M8","X1","X2","X3","X4","X5","X6","X7","Z4"],"Buick":["Enclave","Encore","Encore GX","Envision","LaCrosse","Regal"],"Cadillac":["ATS","CT4","CT5","CT6","CTS","Escalade","Lyriq","XT4","XT5","XT6"],"Chevrolet":["Blazer","Bolt","Camaro","Colorado","Corvette","Cruze","Equinox","Express","Impala","Malibu","Silverado 1500","Silverado 2500","Silverado 3500","Spark","Suburban","Tahoe","Trailblazer","Traverse","Trax"],"Chrysler":["300","Pacifica","Voyager"],"Citroën":["Berlingo","C3","C3 Aircross","C4","C5 Aircross","C5 X","ë-C4"],"Cupra":["Ateca","Born","Formentor","Leon","Tavascan"],"Dacia":["Duster","Jogger","Logan","Sandero","Spring"],"Dodge":["Challenger","Charger","Durango","Grand Caravan","Hornet","Ram 1500","Ram 2500","Ram 3500"],"Ferrari":["296 GTB","488","812","F8","Portofino","Purosangue","Roma","SF90"],"Fiat":["500","500e","500L","500X","Panda","Punto","Tipo"],"Ford":["Bronco","Bronco Sport","EcoSport","Edge","Escape","Expedition","Explorer","F-150","F-250","F-350","Fiesta","Flex","Focus","Fusion","Galaxy","Kuga","Maverick","Mondeo","Mustang","Mustang Mach-E","Puma","Ranger","Transit"],"Genesis":["G70","G80","G90","GV60","GV70","GV80"],"GMC":["Acadia","Canyon","Hummer EV","Sierra 1500","Sierra 2500","Sierra 3500","Terrain","Yukon"],"Honda":["Accord","Civic","CR-V","Fit","HR-V","Insight","Jazz","Odyssey","Passport","Pilot","Prologue","Ridgeline"],"Hyundai":["Accent","Elantra","i10","i20","i30","Ioniq","Ioniq 5","Ioniq 6","Kona","Nexo","Palisade","Santa Cruz","Santa Fe","Sonata","Staria","Tucson","Veloster","Venue"],"Infiniti":["Q50","Q60","QX50","QX55","QX60","QX80"],"Jaguar":["E-Pace","F-Pace","F-Type","I-Pace","XE","XF","XJ"],"Jeep":["Cherokee","Compass","Gladiator","Grand Cherokee","Grand Wagoneer","Renegade","Wagoneer","Wrangler"],"Kia":["Carnival","Ceed","EV6","EV9","Forte","K5","Niro","Optima","Picanto","Rio","Seltos","Sorento","Soul","Sportage","Stinger","Telluride"],"Lamborghini":["Aventador","Huracán","Revuelto","Urus"],"Land Rover":["Defender","Discovery","Discovery Sport","Range Rover","Range Rover Evoque","Range Rover Sport","Range Rover Velar"],"Lexus":["ES","GX","IS","LC","LS","LX","NX","RC","RX","RZ","TX","UX"],"Lincoln":["Aviator","Corsair","MKC","MKZ","Nautilus","Navigator"],"Lotus":["Eletre","Emira","Evija"],"Maserati":["Ghibli","GranTurismo","Grecale","Levante","MC20","Quattroporte"],"Mazda":["CX-3","CX-30","CX-5","CX-50","CX-60","CX-9","CX-90","Mazda2","Mazda3","Mazda6","MX-30","MX-5 Miata"],"McLaren":["570S","600LT","720S","750S","765LT","Artura"],"Mercedes-Benz":["A-Class","AMG GT","B-Class","C-Class","CLA","CLE","CLS","E-Class","EQA","EQB","EQC","EQE","EQS","G-Class","GLA","GLB","GLC","GLE","GLS","S-Class","SL","Sprinter","V-Class","Vito"],"Mini":["Clubman","Convertible","Cooper","Countryman","Hardtop"],"Mitsubishi":["ASX","Eclipse Cross","L200","Mirage","Outlander","Outlander Sport","Pajero","Space Star","Triton"],"Nissan":["370Z","Altima","Ariya","Armada","Frontier","Juke","Kicks","Leaf","Maxima","Murano","Navara","Pathfinder","Qashqai","Rogue","Sentra","Titan","Versa","X-Trail","Z"],"Opel":["Astra","Combo","Corsa","Crossland","Grandland","Mokka","Vivaro","Zafira"],"Peugeot":["2008","208","3008","308","408","5008","508","Rifter","Traveller"],"Polestar":["Polestar 1","Polestar 2","Polestar 3","Polestar 4"],"Porsche":["718 Boxster","718 Cayman","911","Cayenne","Macan","Panamera","Taycan"],"Ram":["1500","2500","3500","ProMaster"],"Renault":["Arkana","Captur","Clio","Espace","Kangoo","Koleos","Megane","Scenic","Trafic","Twingo","Zoe"],"Rivian":["R1S","R1T","R2"],"Rolls-Royce":["Cullinan","Dawn","Ghost","Phantom","Spectre","Wraith"],"Saab":["9-3","9-5"],"SEAT":["Arona","Ateca","Ibiza","Leon","Tarraco"],"Škoda":["Enyaq","Fabia","Kamiq","Karoq","Kodiaq","Octavia","Scala","Superb"],"Smart":["EQ fortwo","#1","#3"],"Subaru":["Ascent","BRZ","Crosstrek","Forester","Impreza","Legacy","Outback","Solterra","WRX"],"Suzuki":["Across","Baleno","Ignis","Jimny","S-Cross","Swift","Vitara"],"Tesla":["Cybertruck","Model 3","Model S","Model X","Model Y"],"Toyota":["4Runner","86","Avalon","bZ4X","C-HR","Camry","Corolla","Corolla Cross","GR86","GR Supra","Highlander","Land Cruiser","Mirai","Prius","RAV4","Sequoia","Sienna","Tacoma","Tundra","Venza","Yaris","Yaris Cross"],"Volkswagen":["Arteon","Atlas","Atlas Cross Sport","Caddy","Golf","Golf GTI","Golf R","ID.3","ID.4","ID.5","ID.7","ID. Buzz","Jetta","Passat","Polo","T-Cross","T-Roc","Taos","Tiguan","Touareg","Touran","Transporter"],"Volvo":["C40 Recharge","EX30","EX90","S60","S90","V60","V90","XC40","XC60","XC90"]};

const MAKES=Object.keys(VEHICLES).sort();

let state = {step:'upload',images:[],previews:[],vehicle:{year:'',make:'',model:'',trim:''},vin:'',vinData:null,mileage:'',notes:'',lotUrl:'',lotData:null,lotImagesLoaded:false,auctionData:null,lotValuation:null,result:null,pricing:null,marketValue:null,pricingLoading:false,progress:0,error:null,expanded:null,progressTimer:null};

function $(sel){return document.querySelector(sel)}
function h(tag,attrs,...children){
  const el=document.createElement(tag);
  if(attrs)Object.entries(attrs).forEach(([k,v])=>{
    if(k==='on'&&typeof v==='object')Object.entries(v).forEach(([ev,fn])=>el.addEventListener(ev,fn));
    else if(k==='style'&&typeof v==='object')Object.assign(el.style,v);
    else if(k==='className')el.className=v;
    else if(k==='innerHTML')el.innerHTML=v;
    else el.setAttribute(k,v);
  });
  children.flat(Infinity).filter(c=>c!=null&&c!==false).forEach(c=>el.appendChild(typeof c==='string'?document.createTextNode(c):c));
  return el;
}

function makeAC(label,key,options,value,onChange){
  const wrap=h('div',{className:'fg'});
  wrap.appendChild(h('label',{className:'fl'},label));
  const acWrap=h('div',{className:'ac-wrap'});
  let hlIdx=-1;
  let isOpen=false;

  const inp=h('input',{className:'fi',placeholder:options.length?'Type to search...':'Select make first',value:value||'',on:{
    focus:()=>{isOpen=true;inp.value=inp.value||'';showList();},
    input:()=>{hlIdx=-1;showList();},
    keydown:e=>{
      const items=acWrap.querySelectorAll('.ac-item');
      if(e.key==='ArrowDown'){e.preventDefault();hlIdx=Math.min(hlIdx+1,items.length-1);highlightItem(items);}
      else if(e.key==='ArrowUp'){e.preventDefault();hlIdx=Math.max(hlIdx-1,0);highlightItem(items);}
      else if(e.key==='Enter'&&hlIdx>=0&&items[hlIdx]){e.preventDefault();selectItem(items[hlIdx].textContent);}
      else if(e.key==='Escape'){closeList();inp.blur();}
    },
    blur:()=>{setTimeout(()=>{closeList();if(!value&&inp.value)inp.value='';if(value)inp.value=value;},180);}
  }});
  inp.setAttribute('autocomplete','off');
  acWrap.appendChild(inp);

  function showList(){
    let list=acWrap.querySelector('.ac-list');
    if(!list){list=h('div',{className:'ac-list'});acWrap.appendChild(list);}
    list.innerHTML='';
    const q=(inp.value||'').toLowerCase();
    const filtered=options.filter(o=>o.toLowerCase().includes(q));
    if(filtered.length===0){list.appendChild(h('div',{className:'ac-empty'},'No matches'));}
    else filtered.slice(0,50).forEach((opt,i)=>{
      const item=h('div',{className:'ac-item'+(opt===value?' selected':''),on:{
        mousedown:e=>{e.preventDefault();selectItem(opt);}
      }},opt);
      list.appendChild(item);
    });
  }

  function highlightItem(items){
    items.forEach((el,i)=>{el.classList.toggle('hl',i===hlIdx);});
    if(items[hlIdx])items[hlIdx].scrollIntoView({block:'nearest'});
  }

  function selectItem(val){
    isOpen=false;hlIdx=-1;
    const list=acWrap.querySelector('.ac-list');
    if(list)list.remove();
    inp.value=val;
    onChange(val);
  }

  function closeList(){
    isOpen=false;hlIdx=-1;
    const list=acWrap.querySelector('.ac-list');
    if(list)list.remove();
  }

  wrap.appendChild(acWrap);
  return wrap;
}

function render(){
  const app=$('#app');
  app.innerHTML='';
  app.appendChild(renderHeader());
  app.appendChild(renderSteps());
  if(state.error&&state.step==='upload')app.appendChild(renderError());
  if(state.step==='upload')renderUpload(app);
  else if(state.step==='analyzing')app.appendChild(renderAnalyzing());
  else if(state.step==='results')renderResults(app);
  else if(state.step==='pricing')renderPricing(app);
}

function renderHeader(){
  return h('div',{className:'header'},
    h('div',{className:'logo'},
      h('div',{className:'logo-icon',innerHTML:'<svg width="22" height="22" fill="none" viewBox="0 0 24 24"><path d="M5 17h1m12 0h1M3 11l2-6h14l2 6M3 11v6h18v-6M3 11h18M7.5 17a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm9 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'}),
      h('div',{className:'title'},'SALVAGE ANALYST')),
    h('div',{className:'subtitle'},'AI-powered vehicle damage assessment & parts pricing'));
}

function renderSteps(){
  const sn=state.step==='upload'?1:state.step==='analyzing'?2:state.step==='results'?3:4;
  const wrap=h('div',{className:'steps'});
  [{n:1,l:'Upload'},{n:2,l:'Analysis'},{n:3,l:'Results'},{n:4,l:'Pricing'}].forEach((s,i)=>{
    if(i>0)wrap.appendChild(h('div',{className:'step-line'+(sn>s.n-1?' done':'')}));
    const cls='step'+(sn===s.n?' active':'')+(sn>s.n?' done':'');
    wrap.appendChild(h('div',{className:cls},h('div',{className:'step-n'},sn>s.n?'✓':String(s.n)),h('span',null,s.l)));
  });
  return wrap;
}

function renderError(){
  return h('div',{className:'error'},
    h('span',null,state.error),
    h('button',{style:{background:'none',border:'none',cursor:'pointer',color:'#991B1B',fontSize:'16px'},on:{click:()=>{state.error=null;render();}}},'✕'));
}

function renderUpload(app){
  // Auction URL card
  const card0=h('div',{className:'card'});
  card0.appendChild(h('div',{className:'card-title'},
    h('span',{innerHTML:'<svg width="18" height="18" fill="none" viewBox="0 0 24 24"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'}),
    ' Import from Auction URL ',
    h('span',{className:'badge',style:{background:'#ECFDF5',color:'#059669'}},'Quick start')));

  const urlRow=h('div',{style:{display:'grid',gridTemplateColumns:'1fr auto',gap:'12px'}});
  const urlGroup=h('div',{className:'fg'});
  const urlInput=h('input',{className:'fi',placeholder:'Paste Copart, IAAI, AutoBidMaster, or any auction URL...',value:state.lotUrl||'',style:{fontSize:'13px'},on:{input:e=>{state.lotUrl=e.target.value;}}});
  urlGroup.appendChild(urlInput);
  urlRow.appendChild(urlGroup);

  const urlBtn=h('button',{className:'btn btn-p',style:{alignSelf:'flex-end',padding:'10px 20px',whiteSpace:'nowrap'},on:{click:async()=>{
    const u=(state.lotUrl||'').trim();
    if(!u){state.error='Please paste an auction URL';render();return;}
    urlBtn.disabled=true;urlBtn.innerHTML='<span style="display:inline-flex;align-items:center;gap:6px"><svg width="16" height="16" viewBox="0 0 24 24" style="animation:spin 1s linear infinite"><circle cx="12" cy="12" r="10" stroke="#fff" stroke-width="3" fill="none" opacity=".3"/><path d="M12 2a10 10 0 019.8 8" stroke="#fff" stroke-width="3" stroke-linecap="round" fill="none"/></svg> Fetching...</span>';
    try{
      const r=await fetch(API+'/api/fetch-lot?'+new URLSearchParams({url:u}));
      if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.detail||'Failed to fetch lot');}
      const d=await r.json();
      state.lotData=d;
      // Auto-fill vehicle fields
      const v=d.vehicle||{};
      if(v.year)state.vehicle.year=String(v.year);
      if(v.make)state.vehicle.make=v.make;
      if(v.model)state.vehicle.model=v.model;
      if(v.description)state.vehicle.trim=v.description.replace(v.year+' '+v.make+' '+v.model,'').trim();
      if(v.odometer)state.mileage=String(v.odometer);
      if(v.vin)state.vin=v.vin;
      // Build notes from condition data
      const c=d.condition||{};
      const noteParts=[];
      if(c.primary_damage)noteParts.push('Primary damage: '+c.primary_damage);
      if(c.secondary_damage)noteParts.push('Secondary: '+c.secondary_damage);
      if(c.run_drives)noteParts.push(c.run_drives);
      if(c.title_type)noteParts.push('Title: '+c.title_type);
      if(c.keys)noteParts.push('Keys: '+c.keys);
      if(noteParts.length)state.notes=noteParts.join(' · ');
      // Store auction data for later
      state.auctionData=d.auction||{};
      state.lotValuation=d.valuation||{};
      render();
    }catch(e){state.error=e.message;render();}
  }}},'Import');
  urlRow.appendChild(urlBtn);
  card0.appendChild(urlRow);

  // Show lot data summary if imported
  if(state.lotData){
    const ld=state.lotData;
    const v=ld.vehicle||{};
    const a=ld.auction||{};
    const c=ld.condition||{};
    const val=ld.valuation||{};
    const summary=h('div',{style:{marginTop:'16px',padding:'16px',background:'var(--bg)',borderRadius:'var(--radius-md)',border:'1px solid var(--border)'}});

    const row1=h('div',{style:{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'8px'}});
    row1.appendChild(h('div',null,
      h('div',{style:{fontWeight:'600',fontSize:'15px'}},v.description||(v.year+' '+v.make+' '+v.model)),
      h('div',{style:{fontSize:'12px',color:'var(--text-tertiary)',marginTop:'2px'}},
        [a.platform,a.lot_number?'Lot #'+a.lot_number:'',a.location].filter(Boolean).join(' · '))));
    row1.appendChild(h('span',{className:'tag '+(ld.extraction_method==='native_json'?'tag-body':'tag-electrical'),style:{fontSize:'11px'}},
      ld.source||'unknown'));
    summary.appendChild(row1);

    const chips=h('div',{style:{display:'flex',flexWrap:'wrap',gap:'6px',marginTop:'8px'}});
    if(a.current_bid)chips.appendChild(h('span',{className:'tag tag-body'},'Bid: $'+Number(a.current_bid).toLocaleString()));
    if(a.buy_now)chips.appendChild(h('span',{className:'tag tag-lighting'},'Buy Now: $'+Number(a.buy_now).toLocaleString()));
    if(val.acv)chips.appendChild(h('span',{className:'tag tag-body',style:{background:'#ECFDF5',color:'#059669'}},'ACV: $'+Number(val.acv).toLocaleString()));
    if(c.primary_damage)chips.appendChild(h('span',{className:'tag tag-structural'},c.primary_damage));
    if(c.run_drives)chips.appendChild(h('span',{className:'tag '+(c.run_drives.toLowerCase().includes('run')?'tag-body':'tag-structural')},c.run_drives));
    if(c.keys)chips.appendChild(h('span',{className:'tag tag-lighting'},'Keys: '+c.keys));
    if(v.odometer)chips.appendChild(h('span',{className:'tag tag-electrical'},Number(v.odometer).toLocaleString()+' mi'));
    summary.appendChild(chips);

    // Show images from lot with option to use them
    const lotImgs=ld.images||[];
    if(lotImgs.length>0){
      const imgSection=h('div',{style:{marginTop:'12px'}});
      imgSection.appendChild(h('div',{style:{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:'8px'}},
        h('span',{style:{fontSize:'12px',fontWeight:'600',color:'var(--text-secondary)'}},lotImgs.length+' auction photos'),
        state.lotImagesLoaded?h('span',{className:'tag tag-body'},'✓ Loaded'):
        h('button',{className:'btn btn-p',style:{padding:'6px 14px',fontSize:'12px'},on:{click:async()=>{
          // Download images and convert to blobs for the analyzer
          imgSection.querySelector('button').textContent='Loading photos...';
          imgSection.querySelector('button').disabled=true;
          const loaded=[];
          const previews=[];
          for(const img of lotImgs.slice(0,8)){
            try{
              const r=await fetch(img.url);
              if(!r.ok)continue;
              const blob=await r.blob();
              const file=new File([blob],'lot_'+(loaded.length+1)+'.jpg',{type:blob.type||'image/jpeg'});
              loaded.push(file);
              previews.push(URL.createObjectURL(blob));
            }catch{}
          }
          state.images=loaded;
          state.previews=previews;
          state.lotImagesLoaded=true;
          render();
        }}},'Use These Photos')));

      const strip=h('div',{style:{display:'flex',gap:'6px',overflowX:'auto',padding:'2px 0'}});
      lotImgs.slice(0,10).forEach(img=>{
        strip.appendChild(h('img',{src:img.url,style:{width:'72px',height:'54px',objectFit:'cover',borderRadius:'6px',border:'1px solid var(--border)',flexShrink:'0'},loading:'lazy'}));
      });
      imgSection.appendChild(strip);
      summary.appendChild(imgSection);
    }
    card0.appendChild(summary);
  }

  card0.appendChild(h('div',{style:{fontSize:'12px',color:'var(--text-tertiary)',marginTop:'12px'}},'Supports AutoBidMaster, Copart (via ABM), IAAI, and most auction sites'));
  app.appendChild(card0);

  // Photos card
  const card1=h('div',{className:'card'});
  card1.appendChild(h('div',{className:'card-title'},
    h('span',{innerHTML:'<svg width="18" height="18" fill="none" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="4" stroke="currentColor" stroke-width="1.5"/><circle cx="8.5" cy="8.5" r="1.5" stroke="currentColor" stroke-width="1.5"/><path d="M21 15l-5-5L5 21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'}),
    ' Damage Photos ',
    h('span',{className:'badge'},state.images.length+'/8')));

  const inp=h('input',{type:'file',accept:'image/*',multiple:true,style:{display:'none'},on:{change:e=>handleFiles(e.target.files)}});
  card1.appendChild(inp);

  const dz=h('div',{className:'dropzone',on:{
    click:()=>inp.click(),
    drop:e=>{e.preventDefault();e.currentTarget.classList.remove('dragover');handleFiles(e.dataTransfer.files);},
    dragover:e=>{e.preventDefault();e.currentTarget.classList.add('dragover');},
    dragleave:e=>{e.preventDefault();e.currentTarget.classList.remove('dragover');}
  }},
    h('div',{innerHTML:'<svg width="48" height="48" fill="none" viewBox="0 0 48 48"><path d="M24 32V16m0 0l-8 8m8-8l8 8" stroke="#94A3B8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><rect x="4" y="4" width="40" height="40" rx="12" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="6 4"/></svg>'}),
    h('div',{className:'dropzone-label'},'Drop photos here or click to browse'),
    h('div',{className:'dropzone-sub'},'Upload up to 8 images · JPG, PNG, WebP'));
  card1.appendChild(dz);

  if(state.previews.length>0){
    const grid=h('div',{className:'img-grid'});
    state.previews.forEach((p,i)=>{
      const thumb=h('div',{className:'img-thumb'},
        h('img',{src:p}),
        h('button',{className:'img-rm',on:{click:e=>{e.stopPropagation();state.images.splice(i,1);state.previews.splice(i,1);render();}}},'✕'));
      grid.appendChild(thumb);
    });
    card1.appendChild(grid);
  }
  app.appendChild(card1);

  // Vehicle card
  const card2=h('div',{className:'card'});
  card2.appendChild(h('div',{className:'card-title'},
    h('span',{innerHTML:'<svg width="22" height="22" fill="none" viewBox="0 0 24 24"><path d="M5 17h1m12 0h1M3 11l2-6h14l2 6M3 11v6h18v-6M3 11h18M7.5 17a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm9 0a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'}),
    ' Vehicle Information'));

  // VIN decode row
  const vinRow=h('div',{style:{display:'grid',gridTemplateColumns:'1fr auto',gap:'12px',marginBottom:'16px'}});
  const vinGroup=h('div',{className:'fg'});
  vinGroup.appendChild(h('label',{className:'fl'},'VIN (auto-fills vehicle details)'));
  const vinInput=h('input',{className:'fi',placeholder:'e.g. WVWZZZ1KZAW123456',value:state.vin||'',style:{fontFamily:"'Space Mono',monospace",letterSpacing:'1px'},on:{input:e=>{state.vin=e.target.value.toUpperCase();}}});
  vinGroup.appendChild(vinInput);
  vinRow.appendChild(vinGroup);
  const vinBtn=h('button',{className:'btn btn-p',style:{alignSelf:'flex-end',padding:'10px 20px'},on:{click:async()=>{
    const v=(state.vin||'').trim();
    if(v.length!==17){state.error='VIN must be exactly 17 characters';render();return;}
    vinBtn.disabled=true;vinBtn.textContent='Decoding...';
    try{
      const r=await fetch(API+'/api/vin/'+v);
      if(!r.ok)throw new Error('VIN decode failed');
      const d=await r.json();
      state.vinData=d;
      if(d.year)state.vehicle.year=d.year;
      if(d.make)state.vehicle.make=d.make;
      if(d.model)state.vehicle.model=d.model;
      if(d.trim)state.vehicle.trim=d.trim;
      render();
    }catch(e){state.error=e.message;render();}
  }}},'Decode');
  vinRow.appendChild(vinBtn);
  card2.appendChild(vinRow);

  // Show VIN decode result if available
  if(state.vinData){
    const vd=state.vinData;
    const chips=h('div',{style:{display:'flex',flexWrap:'wrap',gap:'6px',marginBottom:'16px'}});
    const addChip=(label,val)=>{if(val)chips.appendChild(h('span',{className:'tag tag-body',style:{fontSize:'12px',padding:'4px 10px'}},label+': '+val));};
    addChip('Body',vd.body_class);addChip('Drive',vd.drive_type);addChip('Fuel',vd.fuel_type);addChip('Engine',vd.engine);addChip('Origin',vd.plant_country);
    card2.appendChild(chips);
  }

  // Divider
  card2.appendChild(h('div',{style:{borderTop:'1px solid var(--border)',margin:'0 0 16px',paddingTop:'0'}}));
  card2.appendChild(h('div',{style:{fontSize:'12px',color:'var(--text-tertiary)',marginBottom:'12px'}},'Or enter manually:'));

  const row=h('div',{className:'form-row'});

  // Year dropdown
  row.appendChild(makeAC('Year *','year',YEARS,state.vehicle.year,v=>{state.vehicle.year=v;render();}));

  // Make dropdown (all makes)
  row.appendChild(makeAC('Make *','make',MAKES,state.vehicle.make,v=>{state.vehicle.make=v;state.vehicle.model='';render();}));

  // Model dropdown (filtered by selected make)
  const models=state.vehicle.make&&VEHICLES[state.vehicle.make]?VEHICLES[state.vehicle.make]:[];
  row.appendChild(makeAC('Model *','model',models,state.vehicle.model,v=>{state.vehicle.model=v;render();}));

  // Trim - plain input
  row.appendChild(h('div',{className:'fg'},
    h('label',{className:'fl'},'Trim'),
    h('input',{className:'fi',placeholder:'SE, Sport, Limited...',value:state.vehicle.trim,on:{input:e=>{state.vehicle.trim=e.target.value;}}})));
  card2.appendChild(row);

  const row2=h('div',{style:{display:'grid',gridTemplateColumns:'1fr 2fr',gap:'12px',marginTop:'12px'}});
  row2.appendChild(h('div',{className:'fg'},
    h('label',{className:'fl'},'Mileage'),
    h('input',{className:'fi',placeholder:'45,000',value:state.mileage,on:{input:e=>{state.mileage=e.target.value;}}})));
  row2.appendChild(h('div',{className:'fg'},
    h('label',{className:'fl'},'Additional Notes'),
    h('textarea',{className:'fi',placeholder:'E.g. front-end collision, side impact, hail damage...',value:state.notes,on:{input:e=>{state.notes=e.target.value;}}})));
  card2.appendChild(row2);

  const btnRow=h('div',{className:'btn-row'});
  const btn=h('button',{className:'btn btn-p',on:{click:analyze}},
    h('span',{innerHTML:'<svg width="18" height="18" fill="none" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="2"/><path d="M20 20l-4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'}),
    ' Analyze Damage');
  if(state.images.length===0)btn.disabled=true;
  btnRow.appendChild(btn);
  card2.appendChild(btnRow);
  app.appendChild(card2);
}

function renderAnalyzing(){
  const wrap=h('div',{className:'progress'});
  wrap.appendChild(h('div',{innerHTML:'<svg width="24" height="24" viewBox="0 0 24 24" style="animation:spin 1s linear infinite"><circle cx="12" cy="12" r="10" stroke="#CBD5E1" stroke-width="3" fill="none"/><path d="M12 2a10 10 0 019.8 8" stroke="#3B82F6" stroke-width="3" stroke-linecap="round" fill="none"/></svg>'}));
  const msg=state.progress<30?'Examining photos...':state.progress<60?'Identifying damaged components...':state.progress<85?'Assessing severity levels...':'Finalizing report...';
  wrap.appendChild(h('div',{className:'progress-label'},'Analyzing vehicle damage...'));
  wrap.appendChild(h('div',{className:'progress-sub'},msg));
  const track=h('div',{className:'progress-track'});
  track.appendChild(h('div',{className:'progress-fill',style:{width:state.progress+'%'}}));
  wrap.appendChild(track);
  return wrap;
}

function renderResults(app){
  const r=state.result;if(!r)return;
  const parts=r.damaged_parts||[];
  const counts={critical:0,major:0,moderate:0,minor:0};
  parts.forEach(p=>counts[p.severity]=(counts[p.severity]||0)+1);
  const os=r.estimated_severity||'moderate';
  const oc=SEV[os==='total_loss'||os==='heavy'?'critical':os==='moderate'?'major':'minor'];
  const vName=state.vehicle.year+' '+state.vehicle.make+' '+state.vehicle.model+(state.vehicle.trim?' '+state.vehicle.trim:'');

  // Header
  const hdr=h('div',{className:'rh'});
  hdr.appendChild(h('div',null,
    h('div',{className:'rt'},'Damage Assessment'),
    h('div',{style:{fontSize:'14px',color:'var(--text-tertiary)',marginTop:'4px'}},vName)));
  const badge=h('div',{className:'sev-badge',style:{background:oc.bg,borderColor:oc.bd,color:oc.tx}},
    h('span',{className:'sev-dot',style:{background:oc.dot}}),' ',os.replace('_',' ').toUpperCase());
  hdr.appendChild(badge);
  app.appendChild(hdr);

  // Safety alerts
  if(r.driveable===false){
    app.appendChild(h('div',{className:'alert-bar danger'},
      h('div',{className:'alert-icon',innerHTML:'<svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'}),
      h('div',null,h('strong',null,'Vehicle is NOT driveable'),' — Do not operate this vehicle. Tow to a repair facility for inspection.')));
  }
  if(r.structural_integrity==='compromised'){
    app.appendChild(h('div',{className:'alert-bar danger'},
      h('div',{className:'alert-icon',innerHTML:'<svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'}),
      h('div',null,h('strong',null,'Structural integrity may be compromised'),' — Professional frame/unibody inspection strongly recommended before any repair decisions.')));
  }
  if(r.safety_concerns&&r.safety_concerns.length>0){
    const sc=h('div',{className:'alert-bar warn'},
      h('div',{className:'alert-icon',innerHTML:'<svg width="20" height="20" fill="none" viewBox="0 0 24 24"><path d="M12 8v4m0 4h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/></svg>'}),
      h('div',null));
    sc.lastChild.appendChild(h('strong',null,'Safety concerns identified:'));
    const ul=h('div',{style:{marginTop:'4px'}});
    r.safety_concerns.forEach(c=>ul.appendChild(h('div',{style:{padding:'2px 0'}},'\u26A0 '+c)));
    sc.lastChild.appendChild(ul);
    app.appendChild(sc);
  }

  // Overview card with better stats
  const ov=h('div',{className:'overview'});
  ov.appendChild(h('div',{className:'overview-text'},r.overall_assessment));

  const stats=h('div',{className:'overview-stats'});
  [['critical','#EF4444','Critical'],['major','#F59E0B','Major'],['moderate','#EAB308','Moderate'],['minor','#10B981','Minor']].forEach(([k,c,l])=>{
    stats.appendChild(h('div',{className:'stat'},
      h('div',{className:'stat-v',style:{color:c}},String(counts[k]||0)),
      h('div',{className:'stat-l'},l)));
  });
  ov.appendChild(stats);

  // Labor & repair summary
  const totalHours=parts.reduce((s,p)=>s+(p.labor_hours_estimate||0),0);
  const replaceCount=parts.filter(p=>p.action==='Replace').length;
  const paintCount=parts.filter(p=>p.paint_required).length;
  const ls=h('div',{className:'labor-summary'});
  ls.appendChild(h('div',{className:'stat'},
    h('div',{className:'stat-v',style:{color:'var(--accent)'}},totalHours.toFixed(1)+'h'),
    h('div',{className:'stat-l'},'Est. Labor')));
  ls.appendChild(h('div',{className:'stat'},
    h('div',{className:'stat-v',style:{color:'var(--danger)'}},String(replaceCount)),
    h('div',{className:'stat-l'},'Need Replacement')));
  ls.appendChild(h('div',{className:'stat'},
    h('div',{className:'stat-v',style:{color:'#7C3AED'}},String(paintCount)),
    h('div',{className:'stat-l'},'Need Paint')));
  ov.appendChild(ls);
  app.appendChild(ov);

  // Uploaded photos strip
  if(state.previews.length>0){
    const strip=h('div',{style:{display:'flex',gap:'8px',marginBottom:'16px',overflowX:'auto',padding:'4px 0'}});
    state.previews.forEach(p=>{
      strip.appendChild(h('img',{src:p,style:{width:'80px',height:'60px',objectFit:'cover',borderRadius:'8px',border:'1px solid var(--border)',flexShrink:'0'}}));
    });
    app.appendChild(strip);
  }

  // Part cards — expanded by default, much richer
  const catOrder=['structural','body','lighting','glass','mechanical','electrical','interior'];
  const catLabels={structural:'Structural',body:'Body Panels',lighting:'Lighting',glass:'Glass',mechanical:'Mechanical',electrical:'Electrical',interior:'Interior'};
  const catTag={structural:'tag-structural',body:'tag-body',lighting:'tag-lighting',glass:'tag-glass',mechanical:'tag-mechanical',electrical:'tag-electrical',interior:'tag-interior'};

  const grouped={};
  parts.forEach(p=>{const c=p.category||'body';if(!grouped[c])grouped[c]=[];grouped[c].push(p);});

  let partIdx=0;
  catOrder.forEach(cat=>{
    if(!grouped[cat])return;
    app.appendChild(h('div',{style:{fontSize:'12px',fontWeight:'700',color:'var(--text-tertiary)',textTransform:'uppercase',letterSpacing:'1px',margin:'24px 0 10px',padding:'0 4px'}},catLabels[cat]||cat));

    grouped[cat].forEach(p=>{
      const i=parts.indexOf(p);
      const s=SEV[p.severity]||SEV.minor;
      const open=state.expanded===i;
      const pc=h('div',{className:'pc',style:{animationDelay:partIdx*60+'ms'}});
      partIdx++;

      // Header row
      const hd=h('div',{className:'pc-h',on:{click:()=>{state.expanded=open?null:i;render();}}});
      hd.appendChild(h('div',{className:'pc-sev',style:{background:s.dot}}));

      const info=h('div',{className:'pc-info'});
      const nameRow=h('div',{style:{display:'flex',alignItems:'center',gap:'8px'}});
      nameRow.appendChild(h('span',{className:'pc-name'},p.part_name));
      nameRow.appendChild(h('span',{className:'tag '+(catTag[p.category]||'tag-body')},p.damage_type||p.category||'body'));
      info.appendChild(nameRow);

      // Confidence + action line
      const metaRow=h('div',{style:{display:'flex',alignItems:'center',gap:'12px',marginTop:'4px'}});
      metaRow.appendChild(h('span',{style:{color:s.tx,fontWeight:'500',fontSize:'13px'}},p.action));
      if(p.confidence!=null){
        const confColor=p.confidence>=80?'#10B981':p.confidence>=60?'#F59E0B':'#EF4444';
        metaRow.appendChild(h('span',{style:{fontSize:'12px',color:'var(--text-tertiary)'}},'·'));
        metaRow.appendChild(h('span',{style:{fontSize:'12px',color:confColor,fontWeight:'600'}},p.confidence+'% confidence'));
      }
      if(p.labor_hours_estimate){
        metaRow.appendChild(h('span',{style:{fontSize:'12px',color:'var(--text-tertiary)'}},'·'));
        metaRow.appendChild(h('span',{style:{fontSize:'12px',color:'var(--text-tertiary)'}},p.labor_hours_estimate+'h labor'));
      }
      info.appendChild(metaRow);

      // Confidence mini bar
      if(p.confidence!=null){
        const confColor=p.confidence>=80?'#10B981':p.confidence>=60?'#F59E0B':'#EF4444';
        const cb=h('div',{className:'conf-bar'});
        cb.appendChild(h('div',{className:'conf-fill',style:{width:p.confidence+'%',background:confColor}}));
        info.appendChild(cb);
      }
      hd.appendChild(info);

      hd.appendChild(h('div',{className:'sev-badge',style:{background:s.bg,borderColor:s.bd,color:s.tx,fontSize:'12px',padding:'3px 10px'}},
        h('span',{className:'sev-dot',style:{background:s.dot,width:'6px',height:'6px'}}),' ',p.severity));
      hd.appendChild(h('div',{className:'pc-chev'+(open?' open':''),innerHTML:'<svg width="16" height="16" fill="none" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'}));
      pc.appendChild(hd);

      if(open){
        const body=h('div',{className:'pc-body'});
        body.appendChild(h('div',{className:'pc-desc'},p.description));

        const dg=h('div',{className:'detail-grid'});
        const addDetail=(label,val)=>{if(val!=null&&val!=='')dg.appendChild(h('div',{className:'detail-item'},h('div',{className:'detail-label'},label),h('div',{className:'detail-value'},String(val))));};
        addDetail('Recommended Action',p.action);
        addDetail('Repair Difficulty',p.repair_difficulty?(p.repair_difficulty.charAt(0).toUpperCase()+p.repair_difficulty.slice(1)):null);
        addDetail('Repair Possible',p.repair_possible?'Yes':'No');
        addDetail('Paint Required',p.paint_required?'Yes':'No');
        addDetail('Parts Source',p.oem_vs_aftermarket);
        addDetail('Est. Labor',p.labor_hours_estimate?p.labor_hours_estimate+' hours':null);
        addDetail('Damage Type',p.damage_type?(p.damage_type.charAt(0).toUpperCase()+p.damage_type.slice(1)):null);
        addDetail('Confidence',p.confidence!=null?p.confidence+'%':null);
        body.appendChild(dg);

        if(p.hidden_damage_risk&&p.hidden_damage_risk!=='low'){
          const hrCls='hidden-risk '+(p.hidden_damage_risk||'low');
          const hrIcon=p.hidden_damage_risk==='high'?'\u26A0':'\u2139\uFE0F';
          body.appendChild(h('div',{className:hrCls},
            h('span',null,hrIcon),
            h('div',null,
              h('strong',null,'Hidden damage risk: '+p.hidden_damage_risk.toUpperCase()),
              p.hidden_damage_note?h('div',{style:{marginTop:'2px',fontWeight:'400'}},' '+p.hidden_damage_note):null)));
        }
        pc.appendChild(body);
      }
      app.appendChild(pc);
    });
  });

  // Recommended next steps
  if(r.recommended_next_steps&&r.recommended_next_steps.length>0){
    const ns=h('div',{className:'next-steps'});
    ns.appendChild(h('div',{style:{fontSize:'14px',fontWeight:'600',marginBottom:'12px',color:'var(--text-primary)'}},'Recommended Next Steps'));
    r.recommended_next_steps.forEach((step,i)=>{
      ns.appendChild(h('div',{className:'next-step-item'},
        h('div',{className:'next-step-num'},String(i+1)),
        h('div',null,step)));
    });
    app.appendChild(ns);
  }

  // Disclaimer
  app.appendChild(h('div',{style:{fontSize:'11px',color:'var(--text-tertiary)',marginTop:'20px',padding:'12px 16px',background:'var(--bg)',borderRadius:'var(--radius-sm)',lineHeight:'1.5',border:'1px solid var(--border)'}},'This assessment is generated by AI based on uploaded photos and may not capture all damage — particularly hidden structural, mechanical, or electrical issues. Always have a qualified mechanic or body shop perform an in-person inspection before making repair or purchase decisions.'));

  const btnRow=h('div',{className:'btn-row'});
  btnRow.appendChild(h('button',{className:'btn btn-s',on:{click:resetAll}},'Start Over'));
  btnRow.appendChild(h('button',{className:'btn btn-g',on:{click:fetchPricing}},
    h('span',{innerHTML:'<svg width="16" height="16" fill="none" viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 100 7h5a3.5 3.5 0 110 7H6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'}),
    ' Look Up eBay Prices'));
  app.appendChild(btnRow);
}

function renderPricing(app){
  if(state.pricingLoading){
    const wrap=h('div',{className:'progress'});
    wrap.appendChild(h('div',{innerHTML:'<svg width="24" height="24" viewBox="0 0 24 24" style="animation:spin 1s linear infinite"><circle cx="12" cy="12" r="10" stroke="#CBD5E1" stroke-width="3" fill="none"/><path d="M12 2a10 10 0 019.8 8" stroke="#3B82F6" stroke-width="3" stroke-linecap="round" fill="none"/></svg>'}));
    wrap.appendChild(h('div',{className:'progress-label'},'Fetching prices & market data...'));
    wrap.appendChild(h('div',{className:'progress-sub'},'Searching eBay listings, calculating market value, comparing OEM vs aftermarket'));
    app.appendChild(wrap);return;
  }
  if(!state.pricing)return;
  const pd=state.pricing;
  const vName=state.vehicle.year+' '+state.vehicle.make+' '+state.vehicle.model;

  // Header
  const hdr=h('div',{className:'rh'});
  hdr.appendChild(h('div',null,
    h('div',{className:'rt'},'Repair Cost & Market Analysis'),
    h('div',{style:{fontSize:'14px',color:'var(--text-tertiary)',marginTop:'4px'}},vName+' · '+pd.length+' parts priced')));
  app.appendChild(hdr);

  // ── Bid Decision Panel ──
  const tPartsAvg=pd.reduce((s,p)=>s+(p.avg_price||0),0);
  const tPartsMin=pd.reduce((s,p)=>s+(p.min_price||0),0);
  const tPartsMax=pd.reduce((s,p)=>s+(p.max_price||0),0);
  const tLabor=pd.reduce((s,p)=>s+(p.labor_hours||0),0);
  const laborRate=75;
  const laborCost=tLabor*laborRate;
  const paintParts=pd.filter(p=>p.paint).length;
  const paintCost=paintParts*350;
  const totalRepairLow=tPartsMin+laborCost*0.8+paintCost*0.7;
  const totalRepairAvg=tPartsAvg+laborCost+paintCost;
  const totalRepairHigh=tPartsMax+laborCost*1.3+paintCost*1.3;

  const mv=state.marketValue;
  const acvFromLot=state.lotValuation?.acv;
  const cleanAvg=acvFromLot||mv?.clean_title?.avg||0;
  const cleanSource=acvFromLot?'Auction ACV':'eBay comps';
  const salvageAvg=mv?.salvage_estimate?.avg||0;
  const auc=state.auctionData||{};

  // Decision card
  const decision=h('div',{className:'card',style:{borderWidth:'2px',borderColor:cleanAvg&&totalRepairAvg>cleanAvg*0.75?'var(--danger)':cleanAvg&&totalRepairAvg>cleanAvg*0.5?'var(--warning)':'var(--success)'}});
  const decTitle=h('div',{style:{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:'20px'}});
  decTitle.appendChild(h('div',{className:'card-title',style:{margin:0}},'Bid Decision Summary'));

  if(cleanAvg>0){
    const ratio=totalRepairAvg/cleanAvg;
    const verdict=ratio>0.75?'LIKELY TOTAL LOSS':ratio>0.5?'MARGINAL — HIGH RISK':'REPAIR VIABLE';
    const vColor=ratio>0.75?'var(--danger)':ratio>0.5?'var(--warning)':'var(--success)';
    decTitle.appendChild(h('div',{className:'sev-badge',style:{background:ratio>0.75?'#FEE2E2':ratio>0.5?'#FEF3C7':'#ECFDF5',borderColor:vColor,color:vColor,fontWeight:'700'}},verdict));
  }
  decision.appendChild(decTitle);

  // Numbers grid
  const numGrid=h('div',{style:{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:'16px'}});

  // Market value column
  const mvCol=h('div',{style:{padding:'16px',background:'var(--bg)',borderRadius:'var(--radius-sm)'}});
  mvCol.appendChild(h('div',{className:'stat-l',style:{marginBottom:'8px'}},'CLEAN TITLE VALUE'));
  if(cleanAvg>0){
    mvCol.appendChild(h('div',{className:'stat-v',style:{color:'var(--accent)',fontSize:'26px'}},'$'+cleanAvg.toLocaleString(undefined,{maximumFractionDigits:0})));
    mvCol.appendChild(h('div',{style:{fontSize:'11px',color:'var(--text-tertiary)',marginTop:'4px'}},'Source: '+cleanSource));
    if(!acvFromLot&&mv?.clean_title)mvCol.appendChild(h('div',{style:{fontSize:'11px',color:'var(--text-tertiary)'}},'$'+mv.clean_title.low.toLocaleString(undefined,{maximumFractionDigits:0})+' – $'+mv.clean_title.high.toLocaleString(undefined,{maximumFractionDigits:0})+' range'));
  }else{
    mvCol.appendChild(h('div',{style:{fontSize:'14px',color:'var(--text-tertiary)'}},'No data found'));
  }
  numGrid.appendChild(mvCol);

  // Repair cost column
  const repCol=h('div',{style:{padding:'16px',background:'var(--bg)',borderRadius:'var(--radius-sm)'}});
  repCol.appendChild(h('div',{className:'stat-l',style:{marginBottom:'8px'}},'ESTIMATED REPAIR COST'));
  repCol.appendChild(h('div',{className:'stat-v',style:{color:'var(--danger)',fontSize:'26px'}},'$'+totalRepairAvg.toLocaleString(undefined,{maximumFractionDigits:0})));
  repCol.appendChild(h('div',{style:{fontSize:'12px',color:'var(--text-tertiary)',marginTop:'4px'}},'$'+totalRepairLow.toLocaleString(undefined,{maximumFractionDigits:0})+' – $'+totalRepairHigh.toLocaleString(undefined,{maximumFractionDigits:0})));
  repCol.appendChild(h('div',{style:{fontSize:'11px',color:'var(--text-tertiary)',marginTop:'2px'}},'Parts + labor + paint'));
  numGrid.appendChild(repCol);

  // Max bid column
  const bidCol=h('div',{style:{padding:'16px',background:'var(--bg)',borderRadius:'var(--radius-sm)'}});
  bidCol.appendChild(h('div',{className:'stat-l',style:{marginBottom:'8px'}},'SUGGESTED MAX BID'));
  if(cleanAvg>0){
    const maxBid=Math.max(0,cleanAvg-totalRepairAvg)*0.85;
    bidCol.appendChild(h('div',{className:'stat-v',style:{color:maxBid>0?'var(--success)':'var(--danger)',fontSize:'26px'}},maxBid>0?'$'+maxBid.toLocaleString(undefined,{maximumFractionDigits:0}):'$0'));
    bidCol.appendChild(h('div',{style:{fontSize:'12px',color:'var(--text-tertiary)',marginTop:'4px'}},'(Market − Repair) × 85%'));
    bidCol.appendChild(h('div',{style:{fontSize:'11px',color:'var(--text-tertiary)',marginTop:'2px'}},'15% profit margin built in'));
  }else{
    bidCol.appendChild(h('div',{style:{fontSize:'14px',color:'var(--text-tertiary)'}},'Need market value'));
  }
  numGrid.appendChild(bidCol);
  decision.appendChild(numGrid);

  // Auction context
  if(auc.current_bid||auc.buy_now){
    const aucBar=h('div',{style:{display:'flex',gap:'12px',marginTop:'16px',flexWrap:'wrap'}});
    if(auc.current_bid)aucBar.appendChild(h('div',{style:{padding:'10px 16px',background:'var(--bg)',borderRadius:'var(--radius-sm)',flex:1,minWidth:'140px'}},
      h('div',{className:'stat-l'},'CURRENT BID'),
      h('div',{style:{fontFamily:"'Space Mono',monospace",fontSize:'18px',fontWeight:'700',color:'var(--warning)',marginTop:'4px'}},'$'+Number(auc.current_bid).toLocaleString())));
    if(auc.buy_now)aucBar.appendChild(h('div',{style:{padding:'10px 16px',background:'var(--bg)',borderRadius:'var(--radius-sm)',flex:1,minWidth:'140px'}},
      h('div',{className:'stat-l'},'BUY NOW'),
      h('div',{style:{fontFamily:"'Space Mono',monospace",fontSize:'18px',fontWeight:'700',color:'#7C3AED',marginTop:'4px'}},'$'+Number(auc.buy_now).toLocaleString())));
    if(auc.sale_date)aucBar.appendChild(h('div',{style:{padding:'10px 16px',background:'var(--bg)',borderRadius:'var(--radius-sm)',flex:1,minWidth:'140px'}},
      h('div',{className:'stat-l'},'SALE DATE'),
      h('div',{style:{fontSize:'14px',fontWeight:'600',marginTop:'4px'}},new Date(auc.sale_date).toLocaleDateString())));
    if(auc.sale_status)aucBar.appendChild(h('div',{style:{padding:'10px 16px',background:'var(--bg)',borderRadius:'var(--radius-sm)',flex:1,minWidth:'140px'}},
      h('div',{className:'stat-l'},'STATUS'),
      h('div',{style:{fontSize:'14px',fontWeight:'600',marginTop:'4px'}},auc.sale_status)));
    decision.appendChild(aucBar);
  }

  // Cost breakdown
  const bd=h('div',{style:{marginTop:'16px',padding:'14px 16px',background:'var(--bg)',borderRadius:'var(--radius-sm)',fontSize:'13px',color:'var(--text-secondary)'}});
  bd.appendChild(h('div',{style:{fontWeight:'600',marginBottom:'8px',color:'var(--text-primary)'}},'Cost Breakdown'));
  const addRow=(l,v,note)=>{const row=h('div',{style:{display:'flex',justifyContent:'space-between',padding:'4px 0'}});row.appendChild(h('span',null,l));const rr=h('span',{style:{fontFamily:"'Space Mono',monospace",fontWeight:'600'}},v);if(note){row.appendChild(h('span',null,rr,' ',h('span',{style:{fontWeight:'400',color:'var(--text-tertiary)',fontSize:'11px'}},note)));}else{row.appendChild(rr);}bd.appendChild(row);};
  addRow('Parts (avg)',  '$'+tPartsAvg.toLocaleString(undefined,{maximumFractionDigits:0}),'('+pd.length+' parts)');
  addRow('Labor',        '$'+laborCost.toLocaleString(undefined,{maximumFractionDigits:0}),'('+tLabor.toFixed(1)+'h × $'+laborRate+'/hr)');
  if(paintParts>0)addRow('Paint & blend','$'+paintCost.toLocaleString(undefined,{maximumFractionDigits:0}),'('+paintParts+' panels)');
  bd.appendChild(h('div',{style:{borderTop:'1px solid var(--border)',margin:'6px 0'}}));
  addRow('Total estimate','$'+totalRepairAvg.toLocaleString(undefined,{maximumFractionDigits:0}));
  decision.appendChild(bd);

  app.appendChild(decision);

  // ── Parts pricing cards ──
  app.appendChild(h('div',{style:{fontSize:'12px',fontWeight:'700',color:'var(--text-tertiary)',textTransform:'uppercase',letterSpacing:'1px',margin:'28px 0 12px',padding:'0 4px'}},'Parts Pricing'));

  pd.forEach((d,idx)=>{
    const card=h('div',{className:'pr-card',style:{animationDelay:idx*80+'ms'}});
    const head=h('div',{className:'pr-head'});
    const headLeft=h('div',null);
    const nameRow=h('div',{style:{display:'flex',alignItems:'center',gap:'8px'}});
    nameRow.appendChild(h('span',{className:'pr-name'},d.part));
    if(d.oem_rec)nameRow.appendChild(h('span',{className:'tag '+(d.oem_rec.includes('OEM')?'tag-structural':'tag-body'),style:{fontSize:'10px'}},d.oem_rec));
    headLeft.appendChild(nameRow);
    headLeft.appendChild(h('div',{style:{fontSize:'12px',color:'var(--text-tertiary)',marginTop:'2px'}},(d.total||0).toLocaleString()+' listings'));
    head.appendChild(headLeft);

    // Price comparison: aftermarket vs OEM
    const headRight=h('div',{style:{textAlign:'right'}});
    headRight.appendChild(h('div',{className:'pr-avg'},d.avg_price?'$'+d.avg_price.toFixed(0):'N/A'));
    if(d.min_price)headRight.appendChild(h('div',{className:'pr-range'},'$'+d.min_price.toFixed(0)+' – $'+d.max_price.toFixed(0)));
    if(d.oem_avg)headRight.appendChild(h('div',{style:{fontSize:'11px',color:'#7C3AED',marginTop:'2px'}},'OEM avg: $'+d.oem_avg.toFixed(0)));
    head.appendChild(headRight);
    card.appendChild(head);

    if(d.results&&d.results.length>0){
      const grid=h('div',{className:'lg'});
      d.results.slice(0,6).forEach(li=>{
        const a=h('a',{className:'li',href:li.item_url,target:'_blank',rel:'noopener noreferrer'});
        if(li.image_url)a.appendChild(h('img',{className:'li-img',src:li.image_url,alt:''}));
        else a.appendChild(h('div',{className:'li-img',style:{display:'flex',alignItems:'center',justifyContent:'center',color:'var(--text-tertiary)'},innerHTML:'<svg width="22" height="22" fill="none" viewBox="0 0 24 24"><path d="M5 17h1m12 0h1M3 11l2-6h14l2 6M3 11v6h18v-6M3 11h18" stroke="currentColor" stroke-width="1.5"/></svg>'}));
        const info=h('div',{className:'li-info'});
        info.appendChild(h('div',{className:'li-title'},li.title));
        const meta=h('div',{className:'li-meta'});
        meta.appendChild(h('span',{className:'li-price'},'$'+(li.price||0).toFixed(2)));
        if(li.condition)meta.appendChild(h('span',{className:'li-cond'},li.condition));
        if(li.shipping_cost!=null&&li.shipping_cost>0)meta.appendChild(h('span',{style:{fontSize:'11px',color:'var(--text-tertiary)'}},'+ $'+li.shipping_cost.toFixed(0)+' ship'));
        info.appendChild(meta);
        a.appendChild(info);
        a.appendChild(h('div',{style:{color:'var(--text-tertiary)',flexShrink:0,alignSelf:'center'},innerHTML:'<svg width="14" height="14" fill="none" viewBox="0 0 24 24"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6m4-3h6v6m-11 5L21 3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'}));
        grid.appendChild(a);
      });
      card.appendChild(grid);
    }
    // External search links
    const extLinks=h('div',{style:{padding:'8px 20px 14px',display:'flex',gap:'10px',flexWrap:'wrap'}});
    const partQuery=d.part;
    const rrrGoogleQuery=encodeURIComponent('site:rrr.lt '+vName+' '+partQuery);
    extLinks.appendChild(h('a',{href:'https://www.google.com/search?q='+rrrGoogleQuery,target:'_blank',rel:'noopener noreferrer',
      style:{fontSize:'12px',color:'#059669',textDecoration:'none',display:'inline-flex',alignItems:'center',gap:'4px',padding:'4px 10px',border:'1px solid #A7F3D0',borderRadius:'20px',background:'#ECFDF5',fontWeight:'500'}},
      h('span',{innerHTML:'<svg width="12" height="12" fill="none" viewBox="0 0 24 24"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6m4-3h6v6m-11 5L21 3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'}),' Find on RRR.lt'));
    const googleQuery=encodeURIComponent(vName+' '+partQuery+' used OEM part buy');
    extLinks.appendChild(h('a',{href:'https://www.google.com/search?q='+googleQuery,target:'_blank',rel:'noopener noreferrer',
      style:{fontSize:'12px',color:'var(--text-tertiary)',textDecoration:'none',display:'inline-flex',alignItems:'center',gap:'4px',padding:'4px 10px',border:'1px solid var(--border)',borderRadius:'20px',fontWeight:'500'}},
      h('span',{innerHTML:'<svg width="12" height="12" fill="none" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7" stroke="currentColor" stroke-width="2"/><path d="M20 20l-4-4" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'}),' Google'));
    card.appendChild(extLinks);
    app.appendChild(card);
  });

  // Comparable vehicles section
  if(mv&&mv.clean_title&&mv.clean_title.listings&&mv.clean_title.listings.length>0){
    app.appendChild(h('div',{style:{fontSize:'12px',fontWeight:'700',color:'var(--text-tertiary)',textTransform:'uppercase',letterSpacing:'1px',margin:'28px 0 12px',padding:'0 4px'}},'Comparable Vehicle Listings (Clean Title)'));
    const compCard=h('div',{className:'pr-card'});
    const compGrid=h('div',{className:'lg'});
    mv.clean_title.listings.forEach(li=>{
      const a=h('a',{className:'li',href:li.url,target:'_blank',rel:'noopener noreferrer'});
      if(li.image)a.appendChild(h('img',{className:'li-img',src:li.image,alt:''}));
      const info=h('div',{className:'li-info'});
      info.appendChild(h('div',{className:'li-title'},li.title));
      info.appendChild(h('div',{className:'li-meta'},h('span',{className:'li-price'},'$'+li.price.toLocaleString(undefined,{maximumFractionDigits:0})),li.condition?h('span',{className:'li-cond'},li.condition):null));
      a.appendChild(info);
      a.appendChild(h('div',{style:{color:'var(--text-tertiary)',flexShrink:0,alignSelf:'center'},innerHTML:'<svg width="14" height="14" fill="none" viewBox="0 0 24 24"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6m4-3h6v6m-11 5L21 3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'}));
      compGrid.appendChild(a);
    });
    compCard.appendChild(compGrid);
    app.appendChild(compCard);
  }

  const btnRow=h('div',{className:'btn-row'});
  btnRow.appendChild(h('button',{className:'btn btn-s',on:{click:()=>{state.step='results';render();}}},'← Back to Assessment'));
  btnRow.appendChild(h('button',{className:'btn btn-s',on:{click:resetAll}},'New Analysis'));
  app.appendChild(btnRow);
}

function handleFiles(files){
  const newFiles=Array.from(files).filter(f=>f.type.startsWith('image/')).slice(0,8-state.images.length);
  newFiles.forEach(f=>{
    state.images.push(f);
    const r=new FileReader();
    r.onload=e=>{state.previews.push(e.target.result);render();};
    r.readAsDataURL(f);
  });
}

async function analyze(){
  if(state.images.length===0){state.error='Please upload at least one image';render();return;}
  if(!state.vehicle.year||!state.vehicle.make||!state.vehicle.model){state.error='Please fill in vehicle year, make, and model';render();return;}
  state.error=null;state.step='analyzing';state.progress=0;render();

  state.progressTimer=setInterval(()=>{state.progress=Math.min(state.progress+Math.random()*12,90);render();},800);

  try{
    const imgs=await Promise.all(state.images.map(f=>new Promise(res=>{
      const r=new FileReader();
      r.onload=e=>{const b64=e.target.result.split(',')[1];res({media_type:f.type||'image/jpeg',data:b64});};
      r.readAsDataURL(f);
    })));

    const vehicleStr=state.vehicle.year+' '+state.vehicle.make+' '+state.vehicle.model+(state.vehicle.trim?' '+state.vehicle.trim:'');
    const resp=await fetch(API+'/api/analyze',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({images:imgs,vehicle:vehicleStr,mileage:state.mileage,notes:state.notes})
    });

    clearInterval(state.progressTimer);

    if(!resp.ok){
      const err=await resp.json().catch(()=>({}));
      throw new Error(err.detail||'Analysis failed: '+resp.status);
    }
    state.result=await resp.json();
    state.progress=100;render();
    setTimeout(()=>{state.step='results';render();},400);
  }catch(e){
    clearInterval(state.progressTimer);
    state.error=e.message||'Analysis failed';state.step='upload';render();
  }
}

async function fetchPricing(){
  if(!state.result)return;
  state.pricingLoading=true;state.step='pricing';render();
  try{
    const vehicleStr=state.vehicle.year+' '+state.vehicle.make+' '+state.vehicle.model;
    const parts=state.result.damaged_parts.filter(p=>p.severity==='critical'||p.severity==='major'||p.action==='Replace');

    // Fetch market value in parallel with parts pricing
    const mvPromise=fetch(API+'/api/market-value?'+new URLSearchParams({vehicle:vehicleStr})).then(r=>r.ok?r.json():null).catch(()=>null);

    const results=await Promise.all(parts.map(async p=>{
      try{
        const q=p.ebay_query||vehicleStr+' '+p.part_name;
        // Search for used/aftermarket (best value)
        const r=await fetch(API+'/api/search?'+new URLSearchParams({q,limit:'6'}));
        if(!r.ok)throw new Error();
        const d=await r.json();
        // Also search for OEM/new
        const oemQ=q+' OEM genuine';
        const r2=await fetch(API+'/api/search?'+new URLSearchParams({q:oemQ,limit:'3',min_price:'20'}));
        const oemData=r2.ok?await r2.json():{results:[],avg_price:null};
        return{part:p.part_name,severity:p.severity,action:p.action,oem_rec:p.oem_vs_aftermarket,labor_hours:p.labor_hours_estimate,paint:p.paint_required,query:q,...d,oem_results:oemData.results||[],oem_avg:oemData.avg_price};
      }catch{return{part:p.part_name,severity:p.severity,total:0,results:[],avg_price:null,min_price:null,max_price:null,oem_results:[],oem_avg:null};}
    }));
    state.pricing=results;
    state.marketValue=await mvPromise;
  }catch{state.error='Failed to fetch pricing';}
  finally{state.pricingLoading=false;render();}
}

function resetAll(){
  clearInterval(state.progressTimer);
  state={step:'upload',images:[],previews:[],vehicle:{year:'',make:'',model:'',trim:''},vin:'',vinData:null,mileage:'',notes:'',lotUrl:'',lotData:null,lotImagesLoaded:false,auctionData:null,lotValuation:null,result:null,pricing:null,marketValue:null,pricingLoading:false,progress:0,error:null,expanded:null,progressTimer:null};
  render();
}

render();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the Salvage Analyst frontend."""
    return FRONTEND_HTML


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting eBay Pricing Server on port {SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
