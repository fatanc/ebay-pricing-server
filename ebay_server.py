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
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting eBay Pricing Server on port {SERVER_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
