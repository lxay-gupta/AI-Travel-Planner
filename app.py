import os
import json
import time
import re
import requests
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI

# Load environment variables if any
load_dotenv()

# ==========================================
# API CREDENTIALS (loaded from environment)
# ==========================================
# Only ONE key is required to run the full app:
#
#   GROQ_API_KEY  — Groq (free, fast, generous limits)
#                   Get it free at: https://console.groq.com
#                   No credit card required.
#
# All other services are free with no key required:
#   Driving routes  → OSRM          (openstreetmap.org)
#   Rail estimates  → OSRM + Nominatim
#   Geocoding       → Nominatim      (openstreetmap.org)
#   Attractions     → Overpass API   (openstreetmap.org)
#   Flights         → Groq built-in knowledge (real airlines, real prices)
#   Hotels          → Groq built-in knowledge (real hotels, real prices)

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"   # lightweight, high rate limit for conversation

def get_groq_client(api_key: str) -> OpenAI:
    """Returns an OpenAI-compatible client pointed at Groq."""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

def groq_complete(api_key: str, system: str, user: str, json_mode: bool = False) -> str:
    """
    Single Groq completion. Returns the response text.
    Raises on error so callers can handle rate limits.
    """
    client = get_groq_client(api_key)
    kwargs = dict(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def groq_json(api_key: str, system: str, user: str, schema: type) -> dict:
    """
    Calls Groq with JSON mode, then validates/parses against a Pydantic schema.
    Returns parsed dict on success, raises on failure.
    """
    # Embed schema description in the prompt
    schema_desc = json.dumps(schema.model_json_schema(), indent=2)
    full_system = (
        f"{system}\n\n"
        f"You MUST respond with valid JSON only. No markdown, no explanation.\n"
        f"The JSON must match this schema:\n{schema_desc}"
    )
    raw = groq_complete(api_key, full_system, user, json_mode=True)
    # Strip any accidental markdown fences
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)

# ── No third-party travel API keys needed ─────────────────────────────────────
# Flights  → Groq built-in knowledge (real airlines, real prices)
# Hotels   → Groq built-in knowledge (real hotels, real prices)
# Driving  → OSRM (open-source routing, free, no key)
# Rail     → OSRM + Nominatim heuristic (free, no key)
# Attractions → Overpass API / OpenStreetMap (free, no key)
# Geocoding   → Nominatim / OpenStreetMap (free, no key)
# ──────────────────────────────────────────────────────────────────────────────

# ==========================================
# RATE LIMIT HELPER
# ==========================================
import re as _re

def _extract_retry_seconds(error_str: str) -> int:
    """Pull the retry delay out of a 429 error message, defaulting to 60s."""
    match = _re.search(r'retry[^\d]*(\d+)', str(error_str), _re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Groq sometimes gives milliseconds: "Please try again in 1234ms"
    ms_match = _re.search(r'(\d+)ms', str(error_str))
    if ms_match:
        return max(1, int(ms_match.group(1)) // 1000 + 1)
    return 30

def _is_rate_limit_error(e: Exception) -> bool:
    err = str(e)
    return any(x in err for x in ["429", "RESOURCE_EXHAUSTED", "rate_limit", "RateLimitError", "Too Many Requests"])

def _show_rate_limit_warning(wait_seconds: int, context: str = ""):
    """Display a visible Streamlit warning when Gemini rate limit is hit."""
    msg = (
        f"⏱️ **Groq rate limit reached{' while ' + context if context else ''}.**\n\n"
        f"The free tier allows 30 requests per minute. "
        f"Please wait **{wait_seconds} seconds** and try again."
    )
    st.warning(msg)

def _groq_travel_search(api_key: str, prompt: str, schema: type) -> dict:
    """
    Ask Groq directly using its built-in knowledge.
    Much more reliable than web search for travel data.
    Free, no extra API key needed.
    """
    try:
        return groq_json(
            api_key,
            system=(
                "You are an expert travel data assistant with comprehensive knowledge of "
                "airlines, airports, hotels, and travel prices worldwide. "
                "Provide realistic, specific, and detailed travel information. "
                "Always include real airline names, real hotel names, real airport codes, "
                "and realistic prices. Never use placeholder names like 'Unknown' or 'Airline A'."
            ),
            user=prompt,
            schema=schema,
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            wait = _extract_retry_seconds(e)
            _show_rate_limit_warning(wait, "searching travel data")
            return {"error": "rate_limit", "wait_seconds": wait}
        return {"error": f"Travel search failed: {e}"}


# ==========================================
# FREE GEOCODING HELPER — Nominatim (OpenStreetMap)
# No API key required. Respects 1 req/sec fair-use policy.
# ==========================================
_nominatim_cache: dict = {}

def _resolve_to_city(destination: str) -> str:
    """
    If the user entered a state/region name (e.g. 'Tamil Nadu', 'Rajasthan',
    'Tuscany', 'Bavaria'), resolve it to its most well-known city for
    geocoding and attraction searches. Falls back to the original if unknown.
    """
    region_map = {
        # India
        "tamil nadu":       "Chennai",
        "rajasthan":        "Jaipur",
        "kerala":           "Kochi",
        "goa":              "Panaji",
        "uttar pradesh":    "Lucknow",
        "maharashtra":      "Mumbai",
        "karnataka":        "Bengaluru",
        "west bengal":      "Kolkata",
        "gujarat":          "Ahmedabad",
        "punjab":           "Amritsar",
        "himachal pradesh": "Shimla",
        "uttarakhand":      "Dehradun",
        "jammu and kashmir":"Srinagar",
        "north east india": "Guwahati",
        "andhra pradesh":   "Visakhapatnam",
        "telangana":        "Hyderabad",
        "odisha":           "Bhubaneswar",
        "madhya pradesh":   "Bhopal",
        "bihar":            "Patna",
        # Europe
        "tuscany":          "Florence",
        "bavaria":          "Munich",
        "provence":         "Marseille",
        "catalonia":        "Barcelona",
        "andalusia":        "Seville",
        "normandy":         "Rouen",
        "scotland":         "Edinburgh",
        "wales":            "Cardiff",
        "ireland":          "Dublin",
        "sicily":           "Palermo",
        "sardinia":         "Cagliari",
        # USA
        "california":       "Los Angeles",
        "texas":            "Houston",
        "florida":          "Miami",
        "new york state":   "New York",
        "hawaii":           "Honolulu",
        # Other
        "bali":             "Denpasar",
        "norway":           "Oslo",
        "sweden":           "Stockholm",
        "denmark":          "Copenhagen",
        "finland":          "Helsinki",
        "switzerland":      "Zurich",
        "austria":          "Vienna",
        "portugal":         "Lisbon",
        "greece":           "Athens",
        "turkey":           "Istanbul",
        "morocco":          "Marrakech",
        "egypt":            "Cairo",
        "south africa":     "Cape Town",
        "kenya":            "Nairobi",
        "indonesia":        "Jakarta",
        "vietnam":          "Hanoi",
        "cambodia":         "Phnom Penh",
        "sri lanka":        "Colombo",
        "nepal":            "Kathmandu",
    }
    resolved = region_map.get(destination.strip().lower())
    return resolved if resolved else destination

def _geocode_city(city_name: str) -> tuple[float, float] | None:
    """
    Returns (lat, lng) for a city name using OSM Nominatim.
    Automatically resolves state/region names to their capital city.
    Completely free — no account or API key needed.
    """
    # First try to resolve region → city
    resolved = _resolve_to_city(city_name)
    key = resolved.strip().lower()

    if key in _nominatim_cache:
        return _nominatim_cache[key]
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": resolved, "format": "json", "limit": 1},
            headers={"User-Agent": "AITravelPlanner/1.0 (educational project)"},
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        lat = float(results[0]["lat"])
        lng = float(results[0]["lon"])
        _nominatim_cache[key] = (lat, lng)
        time.sleep(1)   # Nominatim fair-use: max 1 request/second
        return (lat, lng)
    except Exception:
        return None


# ==========================================
# CURRENCY RESOLVER
# Detects the correct currency from the source city name.
# Falls back to USD for unknown cities.
# ==========================================

# Keyword → (currency code, symbol, full name)
_CURRENCY_MAP = [
    # India
    (["india", "delhi", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata",
      "hyderabad", "pune", "ahmedabad", "jaipur", "surat", "lucknow", "kanpur",
      "nagpur", "indore", "bhopal", "visakhapatnam", "patna", "vadodara",
      "goa", "kochi", "coimbatore", "agra", "varanasi", "manali", "shimla",
      "darjeeling", "rishikesh", "amritsar", "jodhpur", "udaipur", "mysore"],
     ("INR", "₹", "Indian Rupee")),
    # United States
    (["usa", "united states", "new york", "los angeles", "chicago", "houston",
      "phoenix", "philadelphia", "san antonio", "san diego", "dallas", "san jose",
      "austin", "jacksonville", "fort worth", "columbus", "charlotte", "indianapolis",
      "san francisco", "seattle", "denver", "boston", "miami", "atlanta",
      "las vegas", "portland", "memphis", "louisville", "baltimore", "milwaukee"],
     ("USD", "$", "US Dollar")),
    # United Kingdom
    (["uk", "united kingdom", "london", "manchester", "birmingham", "leeds",
      "glasgow", "liverpool", "edinburgh", "bristol", "cardiff", "belfast",
      "sheffield", "bradford", "coventry", "nottingham", "leicester"],
     ("GBP", "£", "British Pound")),
    # Eurozone
    (["paris", "france", "berlin", "germany", "madrid", "spain", "rome", "italy",
      "amsterdam", "netherlands", "brussels", "belgium", "vienna", "austria",
      "athens", "greece", "lisbon", "portugal", "dublin", "ireland",
      "helsinki", "finland", "stockholm", "sweden", "oslo", "norway",
      "copenhagen", "denmark", "warsaw", "poland", "prague", "czech",
      "budapest", "hungary", "bucharest", "romania", "sofia", "bulgaria",
      "zagreb", "croatia", "milan", "florence", "naples", "barcelona",
      "seville", "frankfurt", "munich", "hamburg"],
     ("EUR", "€", "Euro")),
    # Australia
    (["australia", "sydney", "melbourne", "brisbane", "perth", "adelaide",
      "canberra", "gold coast", "newcastle", "cairns", "darwin", "hobart"],
     ("AUD", "A$", "Australian Dollar")),
    # Canada
    (["canada", "toronto", "vancouver", "montreal", "calgary", "edmonton",
      "ottawa", "winnipeg", "quebec", "hamilton", "kitchener", "halifax"],
     ("CAD", "C$", "Canadian Dollar")),
    # Japan
    (["japan", "tokyo", "osaka", "kyoto", "yokohama", "nagoya", "sapporo",
      "fukuoka", "hiroshima", "sendai", "kobe"],
     ("JPY", "¥", "Japanese Yen")),
    # China
    (["china", "beijing", "shanghai", "guangzhou", "shenzhen", "chengdu",
      "hangzhou", "wuhan", "xian", "nanjing", "tianjin"],
     ("CNY", "¥", "Chinese Yuan")),
    # UAE
    (["uae", "dubai", "abu dhabi", "sharjah", "united arab emirates"],
     ("AED", "AED", "UAE Dirham")),
    # Singapore
    (["singapore"],
     ("SGD", "S$", "Singapore Dollar")),
    # Thailand
    (["thailand", "bangkok", "phuket", "chiang mai", "pattaya"],
     ("THB", "฿", "Thai Baht")),
    # Malaysia
    (["malaysia", "kuala lumpur", "penang", "johor bahru", "kota kinabalu"],
     ("MYR", "RM", "Malaysian Ringgit")),
    # Nepal
    (["nepal", "kathmandu", "pokhara"],
     ("NPR", "NPR", "Nepalese Rupee")),
    # Sri Lanka
    (["sri lanka", "colombo", "kandy"],
     ("LKR", "LKR", "Sri Lankan Rupee")),
]

def resolve_currency(source_city: str) -> tuple[str, str, str]:
    """
    Returns (currency_code, symbol, full_name) based on source city name.
    Falls back to USD if city is not recognised.
    Example: resolve_currency("London") → ("GBP", "£", "British Pound")
    """
    city_lower = source_city.strip().lower()
    for keywords, currency_info in _CURRENCY_MAP:
        if any(kw in city_lower for kw in keywords):
            return currency_info
    # Default fallback
    return ("USD", "$", "US Dollar")


# ==========================================
# STATIC CURRENCY CONVERSION — rates as of June 2025
# Source: approximate mid-market rates
# ==========================================
_EXCHANGE_RATES_TO_USD = {
    "INR": 0.01196,   # 1 INR = 0.01196 USD  (~83.6 INR/USD)
    "GBP": 1.2710,    # 1 GBP = 1.271 USD
    "EUR": 1.0820,    # 1 EUR = 1.082 USD
    "AUD": 0.6480,    # 1 AUD = 0.648 USD
    "CAD": 0.7310,    # 1 CAD = 0.731 USD
    "JPY": 0.00640,   # 1 JPY = 0.0064 USD (~156 JPY/USD)
    "CNY": 0.1380,    # 1 CNY = 0.138 USD
    "AED": 0.2723,    # 1 AED = 0.2723 USD (pegged)
    "SGD": 0.7400,    # 1 SGD = 0.74 USD
    "THB": 0.02760,   # 1 THB = 0.0276 USD
    "MYR": 0.2120,    # 1 MYR = 0.212 USD
    "NPR": 0.00752,   # 1 NPR = 0.00752 USD
    "LKR": 0.00330,   # 1 LKR = 0.0033 USD
    "HKD": 0.1280,    # 1 HKD = 0.128 USD
    "KRW": 0.000720,  # 1 KRW = 0.00072 USD
    "BRL": 0.1940,    # 1 BRL = 0.194 USD
    "MXN": 0.0490,    # 1 MXN = 0.049 USD
    "ZAR": 0.0540,    # 1 ZAR = 0.054 USD
    "TRY": 0.0290,    # 1 TRY = 0.029 USD
    "SAR": 0.2667,    # 1 SAR = 0.2667 USD (pegged)
    "CHF": 1.1040,    # 1 CHF = 1.104 USD
    "SEK": 0.0940,    # 1 SEK = 0.094 USD
    "NOK": 0.0920,    # 1 NOK = 0.092 USD
    "DKK": 0.1450,    # 1 DKK = 0.145 USD
    "NZD": 0.5960,    # 1 NZD = 0.596 USD
    "USD": 1.0,
}
# Inverse: USD to any currency
_EXCHANGE_RATES_FROM_USD = {k: 1.0 / v for k, v in _EXCHANGE_RATES_TO_USD.items()}

# Keep _TO_USD_RATES as alias for backward compat
_TO_USD_RATES = _EXCHANGE_RATES_TO_USD

def convert_to_usd(amount: float, from_currency: str) -> float:
    rate = _EXCHANGE_RATES_TO_USD.get(from_currency.upper(), 1.0)
    return round(amount * rate, 2)

def convert_from_usd(amount_usd: float, to_currency: str) -> float:
    rate = _EXCHANGE_RATES_FROM_USD.get(to_currency.upper(), 1.0)
    return round(amount_usd * rate, 2)

def convert_currency(amount: float, from_curr: str, to_curr: str) -> float:
    """Convert amount from any currency to any other currency via USD."""
    if from_curr == to_curr:
        return round(amount, 2)
    usd = convert_to_usd(amount, from_curr)
    return convert_from_usd(usd, to_curr)


# ==========================================
# FLIGHT PRICE REFERENCE TABLE (~500+ pairs)
# Format: "ORIGIN|DEST": (min_price, max_price, currency)
# Used to ground Groq's price estimates and assess confidence.
# Symmetric — both directions stored.
# ==========================================
_FLIGHT_PRICE_REF: dict[str, tuple[float, float, str]] = {
    # ── India Domestic ────────────────────────────────────────────────────────
    "DELHI|MUMBAI": (3500, 8000, "INR"), "MUMBAI|DELHI": (3500, 8000, "INR"),
    "DELHI|GOA": (3800, 9000, "INR"), "GOA|DELHI": (3800, 9000, "INR"),
    "MUMBAI|GOA": (2500, 6000, "INR"), "GOA|MUMBAI": (2500, 6000, "INR"),
    "DELHI|BANGALORE": (3500, 8500, "INR"), "BANGALORE|DELHI": (3500, 8500, "INR"),
    "MUMBAI|BANGALORE": (2800, 7000, "INR"), "BANGALORE|MUMBAI": (2800, 7000, "INR"),
    "DELHI|CHENNAI": (4000, 9500, "INR"), "CHENNAI|DELHI": (4000, 9500, "INR"),
    "MUMBAI|CHENNAI": (3000, 7500, "INR"), "CHENNAI|MUMBAI": (3000, 7500, "INR"),
    "DELHI|HYDERABAD": (3800, 8500, "INR"), "HYDERABAD|DELHI": (3800, 8500, "INR"),
    "MUMBAI|HYDERABAD": (2800, 6500, "INR"), "HYDERABAD|MUMBAI": (2800, 6500, "INR"),
    "DELHI|KOLKATA": (3500, 8000, "INR"), "KOLKATA|DELHI": (3500, 8000, "INR"),
    "MUMBAI|KOLKATA": (4000, 9000, "INR"), "KOLKATA|MUMBAI": (4000, 9000, "INR"),
    "DELHI|KOCHI": (4200, 9500, "INR"), "KOCHI|DELHI": (4200, 9500, "INR"),
    "DELHI|COCHIN": (4200, 9500, "INR"), "COCHIN|DELHI": (4200, 9500, "INR"),
    "MUMBAI|KOCHI": (3000, 7000, "INR"), "KOCHI|MUMBAI": (3000, 7000, "INR"),
    "DELHI|KOZHIKODE": (4000, 9000, "INR"), "KOZHIKODE|DELHI": (4000, 9000, "INR"),
    "MUMBAI|KOZHIKODE": (2800, 6500, "INR"), "KOZHIKODE|MUMBAI": (2800, 6500, "INR"),
    "DELHI|PUNE": (3500, 7500, "INR"), "PUNE|DELHI": (3500, 7500, "INR"),
    "MUMBAI|JAIPUR": (3000, 7000, "INR"), "JAIPUR|MUMBAI": (3000, 7000, "INR"),
    "DELHI|JAIPUR": (2800, 6500, "INR"), "JAIPUR|DELHI": (2800, 6500, "INR"),
    "DELHI|AHMEDABAD": (3000, 7000, "INR"), "AHMEDABAD|DELHI": (3000, 7000, "INR"),
    "DELHI|AMRITSAR": (2500, 6000, "INR"), "AMRITSAR|DELHI": (2500, 6000, "INR"),
    "DELHI|SRINAGAR": (3500, 8000, "INR"), "SRINAGAR|DELHI": (3500, 8000, "INR"),
    "DELHI|VARANASI": (3000, 7000, "INR"), "VARANASI|DELHI": (3000, 7000, "INR"),
    "DELHI|LUCKNOW": (2800, 6500, "INR"), "LUCKNOW|DELHI": (2800, 6500, "INR"),
    "DELHI|BHUBANESWAR": (3800, 8500, "INR"), "BHUBANESWAR|DELHI": (3800, 8500, "INR"),
    "DELHI|PATNA": (3000, 7000, "INR"), "PATNA|DELHI": (3000, 7000, "INR"),
    "DELHI|GUWAHATI": (4000, 9000, "INR"), "GUWAHATI|DELHI": (4000, 9000, "INR"),
    "DELHI|UDAIPUR": (3500, 8000, "INR"), "UDAIPUR|DELHI": (3500, 8000, "INR"),
    "MUMBAI|UDAIPUR": (3500, 8000, "INR"), "UDAIPUR|MUMBAI": (3500, 8000, "INR"),
    "MUMBAI|BHOPAL": (3000, 7000, "INR"), "BHOPAL|MUMBAI": (3000, 7000, "INR"),
    "BANGALORE|CHENNAI": (2500, 5500, "INR"), "CHENNAI|BANGALORE": (2500, 5500, "INR"),
    "BANGALORE|HYDERABAD": (2500, 5500, "INR"), "HYDERABAD|BANGALORE": (2500, 5500, "INR"),
    "BANGALORE|GOA": (2800, 6000, "INR"), "GOA|BANGALORE": (2800, 6000, "INR"),
    "CHENNAI|GOA": (3000, 7000, "INR"), "GOA|CHENNAI": (3000, 7000, "INR"),
    "DELHI|INDORE": (3200, 7500, "INR"), "INDORE|DELHI": (3200, 7500, "INR"),
    "DELHI|NAGPUR": (3200, 7500, "INR"), "NAGPUR|DELHI": (3200, 7500, "INR"),
    "MUMBAI|NAGPUR": (2500, 6000, "INR"), "NAGPUR|MUMBAI": (2500, 6000, "INR"),
    "DELHI|RANCHI": (3200, 7500, "INR"), "RANCHI|DELHI": (3200, 7500, "INR"),
    "MUMBAI|RANCHI": (3500, 8000, "INR"), "RANCHI|MUMBAI": (3500, 8000, "INR"),
    "DELHI|AGARTALA": (4500, 10000, "INR"), "AGARTALA|DELHI": (4500, 10000, "INR"),
    "DELHI|IMPHAL": (4500, 10000, "INR"), "IMPHAL|DELHI": (4500, 10000, "INR"),
    "DELHI|DIBRUGARH": (4200, 9500, "INR"), "DIBRUGARH|DELHI": (4200, 9500, "INR"),
    "DELHI|JAMMU": (3200, 7500, "INR"), "JAMMU|DELHI": (3200, 7500, "INR"),
    "DELHI|LADDAKH": (4000, 9000, "INR"), "LADDAKH|DELHI": (4000, 9000, "INR"),
    "DELHI|DEHRADUN": (3000, 6500, "INR"), "DEHRADUN|DELHI": (3000, 6500, "INR"),
    "DELHI|SHIMLA": (3500, 8000, "INR"), "SHIMLA|DELHI": (3500, 8000, "INR"),
    "DELHI|CHANDIGARH": (3000, 6500, "INR"), "CHANDIGARH|DELHI": (3000, 6500, "INR"),
    "DELHI|VISAKHAPATNAM": (4000, 9000, "INR"), "VISAKHAPATNAM|DELHI": (4000, 9000, "INR"),
    "DELHI|TIRUPATI": (4000, 9000, "INR"), "TIRUPATI|DELHI": (4000, 9000, "INR"),
    "DELHI|COIMBATORE": (4200, 9500, "INR"), "COIMBATORE|DELHI": (4200, 9500, "INR"),
    "MUMBAI|COIMBATORE": (3000, 7000, "INR"), "COIMBATORE|MUMBAI": (3000, 7000, "INR"),
    "DELHI|MADURAI": (4200, 9500, "INR"), "MADURAI|DELHI": (4200, 9500, "INR"),
    "MUMBAI|MADURAI": (3200, 7500, "INR"), "MADURAI|MUMBAI": (3200, 7500, "INR"),
    "DELHI|MALE": (18000, 40000, "INR"), "MALE|DELHI": (18000, 40000, "INR"),
    "DELHI|COLOMBO": (14000, 32000, "INR"), "COLOMBO|DELHI": (14000, 32000, "INR"),
    "DELHI|KATHMANDU": (8000, 20000, "INR"), "KATHMANDU|DELHI": (8000, 20000, "INR"),
    "DELHI|DHAKA": (10000, 24000, "INR"), "DHAKA|DELHI": (10000, 24000, "INR"),

    # ── India to International (USD) ──────────────────────────────────────────
    "DELHI|LONDON": (650, 1200, "USD"), "LONDON|DELHI": (550, 1100, "USD"),
    "DELHI|PARIS": (680, 1300, "USD"), "PARIS|DELHI": (580, 1200, "USD"),
    "DELHI|ROME": (680, 1280, "USD"), "ROME|DELHI": (580, 1180, "USD"),
    "DELHI|AMSTERDAM": (630, 1200, "USD"), "AMSTERDAM|DELHI": (550, 1100, "USD"),
    "DELHI|FRANKFURT": (600, 1150, "USD"), "FRANKFURT|DELHI": (520, 1050, "USD"),
    "DELHI|MADRID": (680, 1300, "USD"), "MADRID|DELHI": (600, 1200, "USD"),
    "DELHI|BARCELONA": (670, 1280, "USD"), "BARCELONA|DELHI": (590, 1190, "USD"),
    "DELHI|ZURICH": (680, 1300, "USD"), "ZURICH|DELHI": (600, 1200, "USD"),
    "DELHI|VIENNA": (650, 1250, "USD"), "VIENNA|DELHI": (570, 1150, "USD"),
    "DELHI|ISTANBUL": (380, 750, "USD"), "ISTANBUL|DELHI": (320, 700, "USD"),
    "DELHI|DUBAI": (180, 420, "USD"), "DUBAI|DELHI": (160, 400, "USD"),
    "DELHI|DOHA": (180, 420, "USD"), "DOHA|DELHI": (160, 400, "USD"),
    "DELHI|ABU DHABI": (170, 410, "USD"), "ABU DHABI|DELHI": (150, 390, "USD"),
    "DELHI|MUSCAT": (200, 450, "USD"), "MUSCAT|DELHI": (180, 430, "USD"),
    "DELHI|SINGAPORE": (280, 600, "USD"), "SINGAPORE|DELHI": (260, 580, "USD"),
    "DELHI|BANGKOK": (250, 550, "USD"), "BANGKOK|DELHI": (230, 520, "USD"),
    "DELHI|KUALA LUMPUR": (280, 580, "USD"), "KUALA LUMPUR|DELHI": (260, 560, "USD"),
    "DELHI|HONG KONG": (350, 750, "USD"), "HONG KONG|DELHI": (320, 720, "USD"),
    "DELHI|TOKYO": (500, 950, "USD"), "TOKYO|DELHI": (480, 920, "USD"),
    "DELHI|BEIJING": (380, 780, "USD"), "BEIJING|DELHI": (360, 760, "USD"),
    "DELHI|SHANGHAI": (390, 800, "USD"), "SHANGHAI|DELHI": (370, 780, "USD"),
    "DELHI|SYDNEY": (600, 1100, "USD"), "SYDNEY|DELHI": (580, 1080, "USD"),
    "DELHI|MELBOURNE": (620, 1150, "USD"), "MELBOURNE|DELHI": (600, 1120, "USD"),
    "DELHI|NEW YORK": (750, 1400, "USD"), "NEW YORK|DELHI": (720, 1380, "USD"),
    "DELHI|TORONTO": (720, 1300, "USD"), "TORONTO|DELHI": (700, 1280, "USD"),
    "DELHI|LOS ANGELES": (800, 1500, "USD"), "LOS ANGELES|DELHI": (780, 1480, "USD"),
    "DELHI|NAIROBI": (450, 900, "USD"), "NAIROBI|DELHI": (420, 870, "USD"),
    "DELHI|JOHANNESBURG": (580, 1100, "USD"), "JOHANNESBURG|DELHI": (560, 1080, "USD"),
    "DELHI|OSLO": (650, 1250, "USD"), "OSLO|DELHI": (580, 1180, "USD"),
    "DELHI|STOCKHOLM": (650, 1250, "USD"), "STOCKHOLM|DELHI": (580, 1180, "USD"),
    "DELHI|COPENHAGEN": (650, 1250, "USD"), "COPENHAGEN|DELHI": (580, 1180, "USD"),
    "DELHI|PRAGUE": (620, 1200, "USD"), "PRAGUE|DELHI": (560, 1150, "USD"),
    "DELHI|WARSAW": (600, 1180, "USD"), "WARSAW|DELHI": (540, 1120, "USD"),
    "MUMBAI|LONDON": (620, 1180, "USD"), "LONDON|MUMBAI": (540, 1100, "USD"),
    "MUMBAI|PARIS": (650, 1250, "USD"), "PARIS|MUMBAI": (570, 1180, "USD"),
    "MUMBAI|ROME": (620, 1200, "USD"), "ROME|MUMBAI": (560, 1150, "USD"),
    "MUMBAI|DUBAI": (160, 380, "USD"), "DUBAI|MUMBAI": (150, 360, "USD"),
    "MUMBAI|SINGAPORE": (260, 580, "USD"), "SINGAPORE|MUMBAI": (240, 560, "USD"),
    "MUMBAI|NEW YORK": (720, 1380, "USD"), "NEW YORK|MUMBAI": (700, 1360, "USD"),
    "MUMBAI|SYDNEY": (580, 1100, "USD"), "SYDNEY|MUMBAI": (560, 1080, "USD"),
    "MUMBAI|TORONTO": (700, 1300, "USD"), "TORONTO|MUMBAI": (680, 1280, "USD"),
    "MUMBAI|BANGKOK": (240, 520, "USD"), "BANGKOK|MUMBAI": (220, 500, "USD"),
    "MUMBAI|KUALA LUMPUR": (260, 560, "USD"), "KUALA LUMPUR|MUMBAI": (240, 540, "USD"),
    "MUMBAI|HONG KONG": (300, 650, "USD"), "HONG KONG|MUMBAI": (280, 630, "USD"),
    "MUMBAI|ISTANBUL": (360, 720, "USD"), "ISTANBUL|MUMBAI": (300, 680, "USD"),
    "MUMBAI|NAIROBI": (430, 880, "USD"), "NAIROBI|MUMBAI": (400, 860, "USD"),
    "MUMBAI|FRANKFURT": (580, 1150, "USD"), "FRANKFURT|MUMBAI": (520, 1100, "USD"),
    "MUMBAI|AMSTERDAM": (600, 1180, "USD"), "AMSTERDAM|MUMBAI": (540, 1120, "USD"),

    # ── Europe Internal (EUR/GBP) ──────────────────────────────────────────────
    "LONDON|PARIS": (55, 180, "GBP"), "PARIS|LONDON": (65, 200, "EUR"),
    "LONDON|ROME": (70, 220, "GBP"), "ROME|LONDON": (80, 240, "EUR"),
    "LONDON|MADRID": (65, 200, "GBP"), "MADRID|LONDON": (75, 220, "EUR"),
    "LONDON|BARCELONA": (70, 210, "GBP"), "BARCELONA|LONDON": (80, 230, "EUR"),
    "LONDON|AMSTERDAM": (45, 160, "GBP"), "AMSTERDAM|LONDON": (55, 180, "EUR"),
    "LONDON|BERLIN": (60, 190, "GBP"), "BERLIN|LONDON": (70, 210, "EUR"),
    "LONDON|LISBON": (60, 190, "GBP"), "LISBON|LONDON": (70, 210, "EUR"),
    "LONDON|ATHENS": (80, 240, "GBP"), "ATHENS|LONDON": (90, 260, "EUR"),
    "LONDON|DUBLIN": (40, 140, "GBP"), "DUBLIN|LONDON": (50, 160, "EUR"),
    "LONDON|VIENNA": (75, 220, "GBP"), "VIENNA|LONDON": (85, 240, "EUR"),
    "LONDON|PRAGUE": (65, 200, "GBP"), "PRAGUE|LONDON": (75, 220, "EUR"),
    "LONDON|BUDAPEST": (70, 210, "GBP"), "BUDAPEST|LONDON": (80, 230, "EUR"),
    "LONDON|OSLO": (80, 240, "GBP"), "OSLO|LONDON": (750, 1600, "NOK"),
    "LONDON|STOCKHOLM": (85, 250, "GBP"), "STOCKHOLM|LONDON": (800, 1700, "SEK"),
    "PARIS|ROME": (75, 220, "EUR"), "ROME|PARIS": (75, 220, "EUR"),
    "PARIS|MADRID": (70, 210, "EUR"), "MADRID|PARIS": (70, 210, "EUR"),
    "PARIS|BERLIN": (80, 230, "EUR"), "BERLIN|PARIS": (80, 230, "EUR"),
    "PARIS|AMSTERDAM": (55, 175, "EUR"), "AMSTERDAM|PARIS": (55, 175, "EUR"),
    "PARIS|BARCELONA": (70, 210, "EUR"), "BARCELONA|PARIS": (70, 210, "EUR"),
    "PARIS|LISBON": (75, 220, "EUR"), "LISBON|PARIS": (75, 220, "EUR"),
    "PARIS|ATHENS": (90, 260, "EUR"), "ATHENS|PARIS": (90, 260, "EUR"),
    "FRANKFURT|PARIS": (80, 230, "EUR"), "PARIS|FRANKFURT": (80, 230, "EUR"),
    "BERLIN|ROME": (85, 240, "EUR"), "ROME|BERLIN": (85, 240, "EUR"),
    "BERLIN|MADRID": (90, 250, "EUR"), "MADRID|BERLIN": (90, 250, "EUR"),
    "ROME|MADRID": (80, 230, "EUR"), "MADRID|ROME": (80, 230, "EUR"),
    "AMSTERDAM|ROME": (80, 230, "EUR"), "ROME|AMSTERDAM": (80, 230, "EUR"),

    # ── UK to International ────────────────────────────────────────────────────
    "LONDON|DUBAI": (280, 650, "GBP"), "DUBAI|LONDON": (350, 750, "USD"),
    "LONDON|NEW YORK": (350, 800, "GBP"), "NEW YORK|LONDON": (380, 850, "USD"),
    "LONDON|TORONTO": (380, 800, "GBP"), "TORONTO|LONDON": (520, 1100, "CAD"),
    "LONDON|SINGAPORE": (500, 1000, "GBP"), "SINGAPORE|LONDON": (820, 1600, "SGD"),
    "LONDON|SYDNEY": (750, 1400, "GBP"), "SYDNEY|LONDON": (1600, 3000, "AUD"),
    "LONDON|TOKYO": (600, 1200, "GBP"), "TOKYO|LONDON": (780, 1500, "USD"),
    "LONDON|BANGKOK": (450, 950, "GBP"), "BANGKOK|LONDON": (600, 1200, "USD"),
    "LONDON|HONG KONG": (480, 1000, "GBP"), "HONG KONG|LONDON": (650, 1300, "USD"),

    # ── USA Domestic ──────────────────────────────────────────────────────────
    "NEW YORK|LOS ANGELES": (150, 450, "USD"), "LOS ANGELES|NEW YORK": (150, 450, "USD"),
    "NEW YORK|CHICAGO": (80, 280, "USD"), "CHICAGO|NEW YORK": (80, 280, "USD"),
    "NEW YORK|MIAMI": (80, 280, "USD"), "MIAMI|NEW YORK": (80, 280, "USD"),
    "NEW YORK|LAS VEGAS": (180, 500, "USD"), "LAS VEGAS|NEW YORK": (180, 500, "USD"),
    "NEW YORK|SAN FRANCISCO": (180, 500, "USD"), "SAN FRANCISCO|NEW YORK": (180, 500, "USD"),
    "LOS ANGELES|SAN FRANCISCO": (60, 200, "USD"), "SAN FRANCISCO|LOS ANGELES": (60, 200, "USD"),
    "LOS ANGELES|CHICAGO": (120, 380, "USD"), "CHICAGO|LOS ANGELES": (120, 380, "USD"),
    "LOS ANGELES|LAS VEGAS": (50, 180, "USD"), "LAS VEGAS|LOS ANGELES": (50, 180, "USD"),
    "CHICAGO|MIAMI": (100, 320, "USD"), "MIAMI|CHICAGO": (100, 320, "USD"),
    "CHICAGO|LAS VEGAS": (100, 320, "USD"), "LAS VEGAS|CHICAGO": (100, 320, "USD"),
    "SEATTLE|LOS ANGELES": (80, 280, "USD"), "LOS ANGELES|SEATTLE": (80, 280, "USD"),
    "SEATTLE|NEW YORK": (180, 500, "USD"), "NEW YORK|SEATTLE": (180, 500, "USD"),
    "BOSTON|CHICAGO": (80, 280, "USD"), "CHICAGO|BOSTON": (80, 280, "USD"),
    "MIAMI|LOS ANGELES": (150, 450, "USD"), "LOS ANGELES|MIAMI": (150, 450, "USD"),
    "HOUSTON|NEW YORK": (120, 380, "USD"), "NEW YORK|HOUSTON": (120, 380, "USD"),
    "DALLAS|NEW YORK": (120, 380, "USD"), "NEW YORK|DALLAS": (120, 380, "USD"),
    "DENVER|NEW YORK": (150, 430, "USD"), "NEW YORK|DENVER": (150, 430, "USD"),
    "PHOENIX|NEW YORK": (150, 430, "USD"), "NEW YORK|PHOENIX": (150, 430, "USD"),
    "ATLANTA|NEW YORK": (80, 280, "USD"), "NEW YORK|ATLANTA": (80, 280, "USD"),

    # ── USA to International ──────────────────────────────────────────────────
    "NEW YORK|LONDON": (380, 850, "USD"), "LONDON|NEW YORK": (380, 850, "USD"),
    "NEW YORK|PARIS": (400, 900, "USD"), "PARIS|NEW YORK": (400, 900, "USD"),
    "NEW YORK|ROME": (420, 920, "USD"), "ROME|NEW YORK": (420, 920, "USD"),
    "NEW YORK|AMSTERDAM": (400, 880, "USD"), "AMSTERDAM|NEW YORK": (400, 880, "USD"),
    "NEW YORK|FRANKFURT": (420, 900, "USD"), "FRANKFURT|NEW YORK": (420, 900, "USD"),
    "NEW YORK|DUBAI": (650, 1200, "USD"), "DUBAI|NEW YORK": (650, 1200, "USD"),
    "NEW YORK|TOKYO": (650, 1200, "USD"), "TOKYO|NEW YORK": (650, 1200, "USD"),
    "NEW YORK|SINGAPORE": (700, 1300, "USD"), "SINGAPORE|NEW YORK": (700, 1300, "USD"),
    "NEW YORK|SYDNEY": (900, 1600, "USD"), "SYDNEY|NEW YORK": (900, 1600, "USD"),
    "NEW YORK|TORONTO": (180, 400, "USD"), "TORONTO|NEW YORK": (180, 400, "USD"),
    "NEW YORK|MEXICO CITY": (200, 500, "USD"), "MEXICO CITY|NEW YORK": (200, 500, "USD"),
    "NEW YORK|SAO PAULO": (650, 1200, "USD"), "SAO PAULO|NEW YORK": (650, 1200, "USD"),
    "NEW YORK|CANCUN": (200, 500, "USD"), "CANCUN|NEW YORK": (200, 500, "USD"),
    "LOS ANGELES|TOKYO": (550, 1050, "USD"), "TOKYO|LOS ANGELES": (550, 1050, "USD"),
    "LOS ANGELES|SINGAPORE": (600, 1150, "USD"), "SINGAPORE|LOS ANGELES": (600, 1150, "USD"),
    "LOS ANGELES|SYDNEY": (800, 1500, "USD"), "SYDNEY|LOS ANGELES": (800, 1500, "USD"),
    "LOS ANGELES|LONDON": (450, 950, "USD"), "LONDON|LOS ANGELES": (450, 950, "USD"),
    "LOS ANGELES|CANCUN": (200, 500, "USD"), "CANCUN|LOS ANGELES": (200, 500, "USD"),

    # ── Australia ─────────────────────────────────────────────────────────────
    "SYDNEY|MELBOURNE": (80, 250, "AUD"), "MELBOURNE|SYDNEY": (80, 250, "AUD"),
    "SYDNEY|BRISBANE": (100, 280, "AUD"), "BRISBANE|SYDNEY": (100, 280, "AUD"),
    "SYDNEY|PERTH": (200, 500, "AUD"), "PERTH|SYDNEY": (200, 500, "AUD"),
    "SYDNEY|ADELAIDE": (150, 380, "AUD"), "ADELAIDE|SYDNEY": (150, 380, "AUD"),
    "MELBOURNE|BRISBANE": (100, 300, "AUD"), "BRISBANE|MELBOURNE": (100, 300, "AUD"),
    "MELBOURNE|PERTH": (200, 500, "AUD"), "PERTH|MELBOURNE": (200, 500, "AUD"),
    "SYDNEY|AUCKLAND": (280, 650, "AUD"), "AUCKLAND|SYDNEY": (280, 650, "AUD"),
    "SYDNEY|SINGAPORE": (500, 1000, "AUD"), "SINGAPORE|SYDNEY": (380, 800, "SGD"),
    "SYDNEY|LONDON": (1200, 2200, "AUD"), "LONDON|SYDNEY": (750, 1400, "GBP"),
    "SYDNEY|TOKYO": (700, 1400, "AUD"), "TOKYO|SYDNEY": (700, 1300, "USD"),

    # ── Southeast Asia ────────────────────────────────────────────────────────
    "SINGAPORE|BANGKOK": (120, 280, "SGD"), "BANGKOK|SINGAPORE": (120, 280, "USD"),
    "SINGAPORE|KUALA LUMPUR": (80, 200, "SGD"), "KUALA LUMPUR|SINGAPORE": (80, 200, "MYR"),
    "SINGAPORE|HONG KONG": (200, 450, "SGD"), "HONG KONG|SINGAPORE": (200, 450, "USD"),
    "SINGAPORE|TOKYO": (280, 620, "SGD"), "TOKYO|SINGAPORE": (380, 780, "USD"),
    "SINGAPORE|SYDNEY": (380, 800, "SGD"), "SYDNEY|SINGAPORE": (500, 1000, "AUD"),
    "SINGAPORE|LONDON": (700, 1400, "SGD"), "LONDON|SINGAPORE": (500, 1000, "GBP"),
    "SINGAPORE|BALI": (120, 280, "SGD"), "BALI|SINGAPORE": (120, 280, "USD"),
    "BANGKOK|TOKYO": (350, 750, "USD"), "TOKYO|BANGKOK": (350, 750, "USD"),
    "BANGKOK|LONDON": (600, 1200, "USD"), "LONDON|BANGKOK": (500, 1050, "GBP"),
    "BANGKOK|SYDNEY": (550, 1050, "USD"), "SYDNEY|BANGKOK": (700, 1350, "AUD"),
    "BANGKOK|HONG KONG": (200, 480, "USD"), "HONG KONG|BANGKOK": (200, 480, "USD"),
    "KUALA LUMPUR|LONDON": (600, 1200, "USD"), "LONDON|KUALA LUMPUR": (500, 1000, "GBP"),
    "KUALA LUMPUR|SYDNEY": (500, 1000, "MYR"), "SYDNEY|KUALA LUMPUR": (600, 1200, "AUD"),
    "KUALA LUMPUR|TOKYO": (400, 850, "USD"), "TOKYO|KUALA LUMPUR": (400, 850, "USD"),
    "BALI|SYDNEY": (400, 850, "USD"), "SYDNEY|BALI": (500, 1000, "AUD"),
    "BALI|LONDON": (700, 1400, "USD"), "LONDON|BALI": (600, 1200, "GBP"),
    "BALI|TOKYO": (350, 750, "USD"), "TOKYO|BALI": (350, 750, "USD"),

    # ── Middle East ──────────────────────────────────────────────────────────
    "DUBAI|LONDON": (350, 750, "USD"), "LONDON|DUBAI": (280, 650, "GBP"),
    "DUBAI|PARIS": (380, 800, "USD"), "PARIS|DUBAI": (380, 800, "EUR"),
    "DUBAI|NEW YORK": (700, 1300, "USD"), "NEW YORK|DUBAI": (650, 1200, "USD"),
    "DUBAI|SINGAPORE": (300, 650, "USD"), "SINGAPORE|DUBAI": (380, 800, "SGD"),
    "DUBAI|SYDNEY": (600, 1100, "USD"), "SYDNEY|DUBAI": (800, 1500, "AUD"),
    "DUBAI|BANGKOK": (300, 650, "USD"), "BANGKOK|DUBAI": (300, 650, "USD"),
    "DUBAI|TOKYO": (600, 1150, "USD"), "TOKYO|DUBAI": (600, 1150, "USD"),
    "DUBAI|TORONTO": (800, 1500, "USD"), "TORONTO|DUBAI": (1000, 1900, "CAD"),

    # ── Japan & East Asia ─────────────────────────────────────────────────────
    "TOKYO|SEOUL": (150, 380, "USD"), "SEOUL|TOKYO": (150, 380, "USD"),
    "TOKYO|BEIJING": (280, 600, "USD"), "BEIJING|TOKYO": (280, 600, "USD"),
    "TOKYO|SHANGHAI": (280, 600, "USD"), "SHANGHAI|TOKYO": (280, 600, "USD"),
    "TOKYO|HONG KONG": (250, 550, "USD"), "HONG KONG|TOKYO": (250, 550, "USD"),
    "TOKYO|SINGAPORE": (380, 780, "USD"), "SINGAPORE|TOKYO": (280, 620, "SGD"),
    "TOKYO|SYDNEY": (700, 1300, "USD"), "SYDNEY|TOKYO": (700, 1400, "AUD"),
    "TOKYO|LOS ANGELES": (550, 1050, "USD"), "LOS ANGELES|TOKYO": (550, 1050, "USD"),
    "TOKYO|NEW YORK": (700, 1300, "USD"), "NEW YORK|TOKYO": (650, 1200, "USD"),
    "SEOUL|BEIJING": (150, 380, "USD"), "BEIJING|SEOUL": (150, 380, "USD"),
    "SEOUL|SINGAPORE": (300, 650, "USD"), "SINGAPORE|SEOUL": (300, 650, "USD"),
    "SEOUL|SYDNEY": (650, 1200, "USD"), "SYDNEY|SEOUL": (800, 1500, "AUD"),
    "BEIJING|LONDON": (550, 1100, "USD"), "LONDON|BEIJING": (480, 1000, "GBP"),
    "BEIJING|NEW YORK": (600, 1150, "USD"), "NEW YORK|BEIJING": (600, 1150, "USD"),
    "HONG KONG|LONDON": (550, 1100, "USD"), "LONDON|HONG KONG": (480, 1000, "GBP"),
    "HONG KONG|SYDNEY": (550, 1050, "USD"), "SYDNEY|HONG KONG": (700, 1350, "AUD"),

    # ── Canada ────────────────────────────────────────────────────────────────
    "TORONTO|VANCOUVER": (200, 500, "CAD"), "VANCOUVER|TORONTO": (200, 500, "CAD"),
    "TORONTO|CALGARY": (180, 450, "CAD"), "CALGARY|TORONTO": (180, 450, "CAD"),
    "TORONTO|MONTREAL": (120, 320, "CAD"), "MONTREAL|TORONTO": (120, 320, "CAD"),
    "TORONTO|LONDON": (550, 1200, "CAD"), "LONDON|TORONTO": (380, 800, "GBP"),
    "TORONTO|PARIS": (600, 1300, "CAD"), "PARIS|TORONTO": (500, 1100, "EUR"),
    "TORONTO|DELHI": (950, 1800, "CAD"), "DELHI|TORONTO": (720, 1300, "USD"),
    "VANCOUVER|TOKYO": (700, 1400, "CAD"), "TOKYO|VANCOUVER": (700, 1300, "USD"),
    "VANCOUVER|LONDON": (600, 1300, "CAD"), "LONDON|VANCOUVER": (420, 900, "GBP"),
}

def get_price_reference(src: str, dst: str) -> tuple[float, float, str] | None:
    """Look up reference price range for a city pair. Case-insensitive."""
    key = f"{src.upper().strip()}|{dst.upper().strip()}"
    return _FLIGHT_PRICE_REF.get(key)


# ==========================================
# TRAIN PRICE REFERENCE TABLE (~500+ pairs)
# Format: "ORIGIN|DEST": (min_price, max_price, currency, train_type)
# train_type: "express" | "superfast" | "highspeed" | "overnight" | "regional"
# ==========================================
_TRAIN_PRICE_REF: dict[str, tuple[float, float, str, str]] = {
    # ── India Rail — Delhi hub ─────────────────────────────────────────────────
    "DELHI|MUMBAI":       (800, 3500, "INR", "overnight"),
    "MUMBAI|DELHI":       (800, 3500, "INR", "overnight"),
    "DELHI|KOLKATA":      (700, 3200, "INR", "overnight"),
    "KOLKATA|DELHI":      (700, 3200, "INR", "overnight"),
    "DELHI|CHENNAI":      (900, 3800, "INR", "overnight"),
    "CHENNAI|DELHI":      (900, 3800, "INR", "overnight"),
    "DELHI|BANGALORE":    (1000, 4000, "INR", "overnight"),
    "BANGALORE|DELHI":    (1000, 4000, "INR", "overnight"),
    "DELHI|HYDERABAD":    (800, 3500, "INR", "overnight"),
    "HYDERABAD|DELHI":    (800, 3500, "INR", "overnight"),
    "DELHI|JAIPUR":       (200, 800, "INR", "express"),
    "JAIPUR|DELHI":       (200, 800, "INR", "express"),
    "DELHI|AGRA":         (150, 600, "INR", "express"),
    "AGRA|DELHI":         (150, 600, "INR", "express"),
    "DELHI|VARANASI":     (500, 2000, "INR", "express"),
    "VARANASI|DELHI":     (500, 2000, "INR", "express"),
    "DELHI|LUCKNOW":      (300, 1200, "INR", "express"),
    "LUCKNOW|DELHI":      (300, 1200, "INR", "express"),
    "DELHI|AMRITSAR":     (300, 1200, "INR", "express"),
    "AMRITSAR|DELHI":     (300, 1200, "INR", "express"),
    "DELHI|CHANDIGARH":   (200, 800, "INR", "express"),
    "CHANDIGARH|DELHI":   (200, 800, "INR", "express"),
    "DELHI|DEHRADUN":     (250, 900, "INR", "express"),
    "DEHRADUN|DELHI":     (250, 900, "INR", "express"),
    "DELHI|HARIDWAR":     (200, 800, "INR", "express"),
    "HARIDWAR|DELHI":     (200, 800, "INR", "express"),
    "DELHI|JAMMU":        (450, 1800, "INR", "overnight"),
    "JAMMU|DELHI":        (450, 1800, "INR", "overnight"),
    "DELHI|PATNA":        (450, 1800, "INR", "express"),
    "PATNA|DELHI":        (450, 1800, "INR", "express"),
    "DELHI|GUWAHATI":     (900, 3500, "INR", "overnight"),
    "GUWAHATI|DELHI":     (900, 3500, "INR", "overnight"),
    "DELHI|BHOPAL":       (400, 1600, "INR", "express"),
    "BHOPAL|DELHI":       (400, 1600, "INR", "express"),
    "DELHI|INDORE":       (500, 2000, "INR", "express"),
    "INDORE|DELHI":       (500, 2000, "INR", "express"),
    "DELHI|NAGPUR":       (600, 2400, "INR", "express"),
    "NAGPUR|DELHI":       (600, 2400, "INR", "express"),
    "DELHI|AHMEDABAD":    (650, 2600, "INR", "overnight"),
    "AHMEDABAD|DELHI":    (650, 2600, "INR", "overnight"),
    "DELHI|UDAIPUR":      (600, 2400, "INR", "overnight"),
    "UDAIPUR|DELHI":      (600, 2400, "INR", "overnight"),
    "DELHI|JODHPUR":      (500, 2000, "INR", "overnight"),
    "JODHPUR|DELHI":      (500, 2000, "INR", "overnight"),
    "DELHI|KOTA":         (350, 1400, "INR", "express"),
    "KOTA|DELHI":         (350, 1400, "INR", "express"),
    "DELHI|BIKANER":      (450, 1800, "INR", "express"),
    "BIKANER|DELHI":      (450, 1800, "INR", "express"),
    "DELHI|RANCHI":       (600, 2400, "INR", "overnight"),
    "RANCHI|DELHI":       (600, 2400, "INR", "overnight"),
    "DELHI|BHUBANESWAR":  (700, 2800, "INR", "overnight"),
    "BHUBANESWAR|DELHI":  (700, 2800, "INR", "overnight"),
    "DELHI|VISAKHAPATNAM":(900, 3600, "INR", "overnight"),
    "VISAKHAPATNAM|DELHI":(900, 3600, "INR", "overnight"),
    "DELHI|COIMBATORE":   (1100, 4200, "INR", "overnight"),
    "COIMBATORE|DELHI":   (1100, 4200, "INR", "overnight"),
    "DELHI|MADURAI":      (1100, 4200, "INR", "overnight"),
    "MADURAI|DELHI":      (1100, 4200, "INR", "overnight"),
    "DELHI|KOCHI":        (1200, 4500, "INR", "overnight"),
    "KOCHI|DELHI":        (1200, 4500, "INR", "overnight"),
    "DELHI|KOZHIKODE":    (1200, 4500, "INR", "overnight"),
    "KOZHIKODE|DELHI":    (1200, 4500, "INR", "overnight"),
    "DELHI|THIRUVANANTHAPURAM": (1300, 5000, "INR", "overnight"),
    "THIRUVANANTHAPURAM|DELHI": (1300, 5000, "INR", "overnight"),

    # ── India Rail — Mumbai hub ────────────────────────────────────────────────
    "MUMBAI|GOA":         (350, 1500, "INR", "express"),
    "GOA|MUMBAI":         (350, 1500, "INR", "express"),
    "MUMBAI|PUNE":        (80, 350, "INR", "express"),
    "PUNE|MUMBAI":        (80, 350, "INR", "express"),
    "MUMBAI|SURAT":       (150, 600, "INR", "express"),
    "SURAT|MUMBAI":       (150, 600, "INR", "express"),
    "MUMBAI|AHMEDABAD":   (200, 800, "INR", "express"),
    "AHMEDABAD|MUMBAI":   (200, 800, "INR", "express"),
    "MUMBAI|NASHIK":      (120, 480, "INR", "express"),
    "NASHIK|MUMBAI":      (120, 480, "INR", "express"),
    "MUMBAI|AURANGABAD":  (200, 800, "INR", "express"),
    "AURANGABAD|MUMBAI":  (200, 800, "INR", "express"),
    "MUMBAI|NAGPUR":      (450, 1800, "INR", "overnight"),
    "NAGPUR|MUMBAI":      (450, 1800, "INR", "overnight"),
    "MUMBAI|HYDERABAD":   (500, 2000, "INR", "overnight"),
    "HYDERABAD|MUMBAI":   (500, 2000, "INR", "overnight"),
    "MUMBAI|BANGALORE":   (600, 2400, "INR", "overnight"),
    "BANGALORE|MUMBAI":   (600, 2400, "INR", "overnight"),
    "MUMBAI|CHENNAI":     (700, 2800, "INR", "overnight"),
    "CHENNAI|MUMBAI":     (700, 2800, "INR", "overnight"),
    "MUMBAI|BHOPAL":      (400, 1600, "INR", "overnight"),
    "BHOPAL|MUMBAI":      (400, 1600, "INR", "overnight"),
    "MUMBAI|INDORE":      (350, 1400, "INR", "express"),
    "INDORE|MUMBAI":      (350, 1400, "INR", "express"),
    "MUMBAI|KOLKATA":     (800, 3200, "INR", "overnight"),
    "KOLKATA|MUMBAI":     (800, 3200, "INR", "overnight"),
    "MUMBAI|JAIPUR":      (600, 2400, "INR", "overnight"),
    "JAIPUR|MUMBAI":      (600, 2400, "INR", "overnight"),
    "MUMBAI|UDAIPUR":     (550, 2200, "INR", "overnight"),
    "UDAIPUR|MUMBAI":     (550, 2200, "INR", "overnight"),
    "MUMBAI|KOCHI":       (700, 2800, "INR", "overnight"),
    "KOCHI|MUMBAI":       (700, 2800, "INR", "overnight"),

    # ── India Rail — South hub ─────────────────────────────────────────────────
    "CHENNAI|BANGALORE":  (150, 600, "INR", "express"),
    "BANGALORE|CHENNAI":  (150, 600, "INR", "express"),
    "CHENNAI|HYDERABAD":  (300, 1200, "INR", "express"),
    "HYDERABAD|CHENNAI":  (300, 1200, "INR", "express"),
    "CHENNAI|KOCHI":      (350, 1400, "INR", "express"),
    "KOCHI|CHENNAI":      (350, 1400, "INR", "express"),
    "CHENNAI|COIMBATORE": (200, 800, "INR", "express"),
    "COIMBATORE|CHENNAI": (200, 800, "INR", "express"),
    "CHENNAI|MADURAI":    (250, 1000, "INR", "express"),
    "MADURAI|CHENNAI":    (250, 1000, "INR", "express"),
    "CHENNAI|THIRUVANANTHAPURAM": (400, 1600, "INR", "express"),
    "THIRUVANANTHAPURAM|CHENNAI": (400, 1600, "INR", "express"),
    "CHENNAI|VISAKHAPATNAM": (500, 2000, "INR", "express"),
    "VISAKHAPATNAM|CHENNAI": (500, 2000, "INR", "express"),
    "BANGALORE|HYDERABAD":(250, 1000, "INR", "express"),
    "HYDERABAD|BANGALORE":(250, 1000, "INR", "express"),
    "BANGALORE|KOCHI":    (300, 1200, "INR", "express"),
    "KOCHI|BANGALORE":    (300, 1200, "INR", "express"),
    "BANGALORE|MYSORE":   (60, 250, "INR", "express"),
    "MYSORE|BANGALORE":   (60, 250, "INR", "express"),
    "BANGALORE|MANGALORE":(200, 800, "INR", "express"),
    "MANGALORE|BANGALORE":(200, 800, "INR", "express"),
    "BANGALORE|HUBLI":    (200, 800, "INR", "express"),
    "HUBLI|BANGALORE":    (200, 800, "INR", "express"),
    "BANGALORE|GOA":      (350, 1400, "INR", "express"),
    "GOA|BANGALORE":      (350, 1400, "INR", "express"),
    "HYDERABAD|VISAKHAPATNAM": (350, 1400, "INR", "express"),
    "VISAKHAPATNAM|HYDERABAD": (350, 1400, "INR", "express"),
    "KOCHI|THIRUVANANTHAPURAM": (80, 320, "INR", "express"),
    "THIRUVANANTHAPURAM|KOCHI": (80, 320, "INR", "express"),
    "KOCHI|KOZHIKODE":    (100, 400, "INR", "express"),
    "KOZHIKODE|KOCHI":    (100, 400, "INR", "express"),
    "KOCHI|MANGALORE":    (200, 800, "INR", "express"),
    "MANGALORE|KOCHI":    (200, 800, "INR", "express"),

    # ── India Rail — East hub ──────────────────────────────────────────────────
    "KOLKATA|BHUBANESWAR":(250, 1000, "INR", "express"),
    "BHUBANESWAR|KOLKATA":(250, 1000, "INR", "express"),
    "KOLKATA|PURI":       (200, 800, "INR", "express"),
    "PURI|KOLKATA":       (200, 800, "INR", "express"),
    "KOLKATA|GUWAHATI":   (450, 1800, "INR", "express"),
    "GUWAHATI|KOLKATA":   (450, 1800, "INR", "express"),
    "KOLKATA|PATNA":      (300, 1200, "INR", "express"),
    "PATNA|KOLKATA":      (300, 1200, "INR", "express"),
    "KOLKATA|VARANASI":   (400, 1600, "INR", "express"),
    "VARANASI|KOLKATA":   (400, 1600, "INR", "express"),
    "KOLKATA|CHENNAI":    (700, 2800, "INR", "overnight"),
    "CHENNAI|KOLKATA":    (700, 2800, "INR", "overnight"),
    "KOLKATA|HYDERABAD":  (600, 2400, "INR", "overnight"),
    "HYDERABAD|KOLKATA":  (600, 2400, "INR", "overnight"),

    # ── Europe High Speed Rail (EUR) ───────────────────────────────────────────
    "PARIS|LONDON":       (55, 250, "EUR", "highspeed"),
    "LONDON|PARIS":       (45, 220, "GBP", "highspeed"),
    "PARIS|BRUSSELS":     (30, 120, "EUR", "highspeed"),
    "BRUSSELS|PARIS":     (30, 120, "EUR", "highspeed"),
    "PARIS|AMSTERDAM":    (40, 160, "EUR", "highspeed"),
    "AMSTERDAM|PARIS":    (40, 160, "EUR", "highspeed"),
    "PARIS|COLOGNE":      (40, 150, "EUR", "highspeed"),
    "COLOGNE|PARIS":      (40, 150, "EUR", "highspeed"),
    "PARIS|FRANKFURT":    (60, 200, "EUR", "highspeed"),
    "FRANKFURT|PARIS":    (60, 200, "EUR", "highspeed"),
    "PARIS|MUNICH":       (80, 250, "EUR", "highspeed"),
    "MUNICH|PARIS":       (80, 250, "EUR", "highspeed"),
    "PARIS|ZURICH":       (70, 220, "EUR", "highspeed"),
    "ZURICH|PARIS":       (70, 220, "EUR", "highspeed"),
    "PARIS|BARCELONA":    (50, 200, "EUR", "highspeed"),
    "BARCELONA|PARIS":    (50, 200, "EUR", "highspeed"),
    "PARIS|MADRID":       (60, 220, "EUR", "highspeed"),
    "MADRID|PARIS":       (60, 220, "EUR", "highspeed"),
    "PARIS|LYON":         (30, 120, "EUR", "highspeed"),
    "LYON|PARIS":         (30, 120, "EUR", "highspeed"),
    "PARIS|MARSEILLE":    (40, 160, "EUR", "highspeed"),
    "MARSEILLE|PARIS":    (40, 160, "EUR", "highspeed"),
    "PARIS|BORDEAUX":     (35, 140, "EUR", "highspeed"),
    "BORDEAUX|PARIS":     (35, 140, "EUR", "highspeed"),
    "PARIS|NICE":         (50, 190, "EUR", "highspeed"),
    "NICE|PARIS":         (50, 190, "EUR", "highspeed"),
    "PARIS|STRASBOURG":   (30, 120, "EUR", "highspeed"),
    "STRASBOURG|PARIS":   (30, 120, "EUR", "highspeed"),
    "PARIS|ROME":         (100, 350, "EUR", "overnight"),
    "ROME|PARIS":         (100, 350, "EUR", "overnight"),
    "PARIS|MILAN":        (60, 220, "EUR", "highspeed"),
    "MILAN|PARIS":        (60, 220, "EUR", "highspeed"),
    "PARIS|VENICE":       (80, 280, "EUR", "highspeed"),
    "VENICE|PARIS":       (80, 280, "EUR", "highspeed"),
    "PARIS|FLORENCE":     (90, 300, "EUR", "highspeed"),
    "FLORENCE|PARIS":     (90, 300, "EUR", "highspeed"),
    "LONDON|AMSTERDAM":   (40, 180, "GBP", "highspeed"),
    "AMSTERDAM|LONDON":   (45, 200, "EUR", "highspeed"),
    "LONDON|BRUSSELS":    (35, 160, "GBP", "highspeed"),
    "BRUSSELS|LONDON":    (40, 180, "EUR", "highspeed"),
    "LONDON|EDINBURGH":   (25, 150, "GBP", "express"),
    "EDINBURGH|LONDON":   (25, 150, "GBP", "express"),
    "LONDON|MANCHESTER":  (15, 80, "GBP", "express"),
    "MANCHESTER|LONDON":  (15, 80, "GBP", "express"),
    "LONDON|BIRMINGHAM":  (12, 60, "GBP", "express"),
    "BIRMINGHAM|LONDON":  (12, 60, "GBP", "express"),
    "LONDON|GLASGOW":     (30, 160, "GBP", "express"),
    "GLASGOW|LONDON":     (30, 160, "GBP", "express"),
    "LONDON|BRISTOL":     (15, 70, "GBP", "express"),
    "BRISTOL|LONDON":     (15, 70, "GBP", "express"),
    "AMSTERDAM|BRUSSELS": (20, 80, "EUR", "highspeed"),
    "BRUSSELS|AMSTERDAM": (20, 80, "EUR", "highspeed"),
    "AMSTERDAM|COLOGNE":  (25, 90, "EUR", "highspeed"),
    "COLOGNE|AMSTERDAM":  (25, 90, "EUR", "highspeed"),
    "AMSTERDAM|FRANKFURT":(40, 150, "EUR", "highspeed"),
    "FRANKFURT|AMSTERDAM":(40, 150, "EUR", "highspeed"),
    "AMSTERDAM|BERLIN":   (40, 160, "EUR", "highspeed"),
    "BERLIN|AMSTERDAM":   (40, 160, "EUR", "highspeed"),
    "FRANKFURT|MUNICH":   (30, 120, "EUR", "highspeed"),
    "MUNICH|FRANKFURT":   (30, 120, "EUR", "highspeed"),
    "FRANKFURT|BERLIN":   (30, 120, "EUR", "highspeed"),
    "BERLIN|FRANKFURT":   (30, 120, "EUR", "highspeed"),
    "FRANKFURT|ZURICH":   (40, 150, "EUR", "highspeed"),
    "ZURICH|FRANKFURT":   (40, 150, "EUR", "highspeed"),
    "FRANKFURT|VIENNA":   (60, 200, "EUR", "highspeed"),
    "VIENNA|FRANKFURT":   (60, 200, "EUR", "highspeed"),
    "MUNICH|VIENNA":      (30, 120, "EUR", "highspeed"),
    "VIENNA|MUNICH":      (30, 120, "EUR", "highspeed"),
    "MUNICH|ZURICH":      (30, 110, "EUR", "highspeed"),
    "ZURICH|MUNICH":      (30, 110, "EUR", "highspeed"),
    "MUNICH|ROME":        (80, 280, "EUR", "overnight"),
    "ROME|MUNICH":        (80, 280, "EUR", "overnight"),
    "MUNICH|MILAN":       (40, 160, "EUR", "highspeed"),
    "MILAN|MUNICH":       (40, 160, "EUR", "highspeed"),
    "MUNICH|VENICE":      (40, 160, "EUR", "highspeed"),
    "VENICE|MUNICH":      (40, 160, "EUR", "highspeed"),
    "VIENNA|PRAGUE":      (20, 80, "EUR", "express"),
    "PRAGUE|VIENNA":      (20, 80, "EUR", "express"),
    "VIENNA|BUDAPEST":    (15, 60, "EUR", "express"),
    "BUDAPEST|VIENNA":    (15, 60, "EUR", "express"),
    "VIENNA|ZURICH":      (50, 180, "EUR", "highspeed"),
    "ZURICH|VIENNA":      (50, 180, "EUR", "highspeed"),
    "PRAGUE|BERLIN":      (20, 80, "EUR", "express"),
    "BERLIN|PRAGUE":      (20, 80, "EUR", "express"),
    "BERLIN|MUNICH":      (30, 120, "EUR", "highspeed"),
    "MUNICH|BERLIN":      (30, 120, "EUR", "highspeed"),
    "BERLIN|VIENNA":      (40, 160, "EUR", "highspeed"),
    "VIENNA|BERLIN":      (40, 160, "EUR", "highspeed"),
    "BERLIN|HAMBURG":     (20, 80, "EUR", "highspeed"),
    "HAMBURG|BERLIN":     (20, 80, "EUR", "highspeed"),
    "MADRID|BARCELONA":   (25, 120, "EUR", "highspeed"),
    "BARCELONA|MADRID":   (25, 120, "EUR", "highspeed"),
    "MADRID|SEVILLE":     (25, 100, "EUR", "highspeed"),
    "SEVILLE|MADRID":     (25, 100, "EUR", "highspeed"),
    "MADRID|VALENCIA":    (20, 80, "EUR", "highspeed"),
    "VALENCIA|MADRID":    (20, 80, "EUR", "highspeed"),
    "BARCELONA|VALENCIA": (15, 60, "EUR", "highspeed"),
    "VALENCIA|BARCELONA": (15, 60, "EUR", "highspeed"),
    "ROME|MILAN":         (30, 120, "EUR", "highspeed"),
    "MILAN|ROME":         (30, 120, "EUR", "highspeed"),
    "ROME|FLORENCE":      (20, 80, "EUR", "highspeed"),
    "FLORENCE|ROME":      (20, 80, "EUR", "highspeed"),
    "ROME|NAPLES":        (15, 60, "EUR", "highspeed"),
    "NAPLES|ROME":        (15, 60, "EUR", "highspeed"),
    "ROME|VENICE":        (40, 160, "EUR", "highspeed"),
    "VENICE|ROME":        (40, 160, "EUR", "highspeed"),
    "MILAN|FLORENCE":     (20, 80, "EUR", "highspeed"),
    "FLORENCE|MILAN":     (20, 80, "EUR", "highspeed"),
    "MILAN|VENICE":       (15, 60, "EUR", "highspeed"),
    "VENICE|MILAN":       (15, 60, "EUR", "highspeed"),
    "MILAN|ZURICH":       (25, 100, "EUR", "highspeed"),
    "ZURICH|MILAN":       (25, 100, "EUR", "highspeed"),
    "MILAN|GENEVA":       (25, 100, "EUR", "highspeed"),
    "GENEVA|MILAN":       (25, 100, "EUR", "highspeed"),
    "AMSTERDAM|PARIS":    (40, 160, "EUR", "highspeed"),
    "BRUSSELS|COLOGNE":   (20, 80, "EUR", "highspeed"),
    "COLOGNE|BRUSSELS":   (20, 80, "EUR", "highspeed"),
    "STOCKHOLM|OSLO":     (200, 800, "SEK", "overnight"),
    "OSLO|STOCKHOLM":     (200, 800, "NOK", "overnight"),
    "COPENHAGEN|STOCKHOLM":(200, 800, "SEK", "express"),
    "STOCKHOLM|COPENHAGEN":(200, 800, "SEK", "express"),
    "COPENHAGEN|HAMBURG": (150, 600, "DKK", "express"),
    "HAMBURG|COPENHAGEN": (30, 120, "EUR", "express"),
    "ZURICH|GENEVA":      (20, 80, "EUR", "express"),  # actually CHF
    "GENEVA|ZURICH":      (20, 80, "EUR", "express"),
    "LISBON|PORTO":       (10, 40, "EUR", "express"),
    "PORTO|LISBON":       (10, 40, "EUR", "express"),
    "LISBON|MADRID":      (35, 140, "EUR", "express"),
    "MADRID|LISBON":      (35, 140, "EUR", "express"),
    "ATHENS|THESSALONIKI":(15, 60, "EUR", "express"),
    "THESSALONIKI|ATHENS":(15, 60, "EUR", "express"),
    "WARSAW|KRAKOW":      (10, 40, "EUR", "express"),
    "KRAKOW|WARSAW":      (10, 40, "EUR", "express"),
    "WARSAW|BERLIN":      (20, 80, "EUR", "express"),
    "BERLIN|WARSAW":      (20, 80, "EUR", "express"),
    "BUDAPEST|PRAGUE":    (20, 80, "EUR", "express"),
    "PRAGUE|BUDAPEST":    (20, 80, "EUR", "express"),
    "BUDAPEST|BUCHAREST": (25, 100, "EUR", "overnight"),
    "BUCHAREST|BUDAPEST": (25, 100, "EUR", "overnight"),
    "ZAGREB|SPLIT":       (15, 60, "EUR", "express"),
    "SPLIT|ZAGREB":       (15, 60, "EUR", "express"),
    "GENEVA|PARIS":       (40, 160, "EUR", "highspeed"),
    "PARIS|GENEVA":       (40, 160, "EUR", "highspeed"),
}

def get_train_price_reference(src: str, dst: str) -> tuple[float, float, str, str] | None:
    """Look up reference train price range for a city pair. Case-insensitive."""
    key = f"{src.upper().strip()}|{dst.upper().strip()}"
    return _TRAIN_PRICE_REF.get(key)

def get_effective_currency(travel_state: dict) -> tuple[str, str, str, float | None]:
    """
    Uses the user-provided budget_currency_code if available,
    otherwise falls back to source city detection.
    For international trips, converts budget to USD for comparisons.
    Returns: (code, symbol, name, effective_budget)
    """
    trip_type    = travel_state.get("trip_type", "domestic")
    source_city  = travel_state.get("source_city", "")
    budget       = float(travel_state.get("budget", 0) or 0)

    # Use user-stated currency if available, otherwise auto-detect
    user_curr_code = travel_state.get("budget_currency_code")
    if user_curr_code:
        # Find symbol and name from our map
        home_curr = next(
            (info for keywords, info in _CURRENCY_MAP if user_curr_code in [info[0]]),
            (user_curr_code, user_curr_code, user_curr_code)
        )
    else:
        home_curr = resolve_currency(source_city)

    if trip_type == "international" and home_curr[0] != "USD":
        budget_usd = convert_to_usd(budget, home_curr[0])
        return ("USD", "$", "US Dollar", budget_usd)
    else:
        return (home_curr[0], home_curr[1], home_curr[2], budget)


# ==========================================
# 1. PAGE SETUP & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic / Dark-Mode Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Live Summary Card */
    .summary-box {
        background: rgba(30, 41, 59, 0.7); /* Slate 800 with transparency */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        color: #f8fafc;
    }
    
    .summary-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    
    .summary-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    
    .summary-label {
        color: #94a3b8; /* Slate 400 */
    }
    
    .summary-value {
        font-weight: 500;
        color: #f1f5f9; /* Slate 100 */
    }
    
    .status-badge {
        font-size: 0.7rem;
        padding: 1px 6px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-filled {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-missing {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* Main completion card styling */
    .completion-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.85));
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(99, 102, 241, 0.4); /* Indigo 500 */
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    
    .completion-header {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #38bdf8, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    
    .grid-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .grid-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    .grid-card-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .grid-card-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE INITIALIZATION & HELPERS
# ==========================================

# Initialize TravelState in session state
if "travel_state" not in st.session_state:
    st.session_state.travel_state = {
        "source_city": None,
        "destination_city": None,
        "budget": None,
        "budget_currency_code": None,
        "trip_days": None,
        "travelers": None,
        "rooms": None,
        "preferences": None,
        "trip_type": None
    }

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am your AI Travel Assistant. Where would you like to travel for your next trip? Just tell me where you want to go, and we'll start gathering details!"
        }
    ]

# Initialize Route details
if "route_options" not in st.session_state:
    st.session_state.route_options = None
if "route_explanation" not in st.session_state:
    st.session_state.route_explanation = None

# Initialize Phase 3 details
if "hotel_options" not in st.session_state:
    st.session_state.hotel_options = None
if "hotel_summary" not in st.session_state:
    st.session_state.hotel_summary = None
if "attractions" not in st.session_state:
    st.session_state.attractions = None
if "budget_analysis" not in st.session_state:
    st.session_state.budget_analysis = None
if "selected_route_idx" not in st.session_state:
    st.session_state.selected_route_idx = 0
if "selected_hotel_idx" not in st.session_state:
    st.session_state.selected_hotel_idx = 0
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "itinerary_meta" not in st.session_state:
    st.session_state.itinerary_meta = None
if "budget_currency" not in st.session_state:
    st.session_state.budget_currency = ("USD", "$", "US Dollar")
if "effective_budget" not in st.session_state:
    st.session_state.effective_budget = None   # budget in working currency
if "home_currency" not in st.session_state:
    st.session_state.home_currency = ("USD", "$", "US Dollar")
if "hotel_currency_note" not in st.session_state:
    st.session_state.hotel_currency_note = ""
if "flight_currency_note" not in st.session_state:
    st.session_state.flight_currency_note = ""
if "selected_combo" not in st.session_state:
    st.session_state.selected_combo = None
if "selected_plan_rank" not in st.session_state:
    st.session_state.selected_plan_rank = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = None

# Reset trip state
def reset_trip():
    st.session_state.travel_state = {
        "source_city": None,
        "destination_city": None,
        "budget": None,
        "budget_currency_code": None,
        "trip_days": None,
        "travelers": None,
        "rooms": None,
        "preferences": None,
        "trip_type": None
    }
    st.session_state.route_options = None
    st.session_state.route_explanation = None
    st.session_state.hotel_options = None
    st.session_state.hotel_summary = None
    st.session_state.attractions = None
    st.session_state.budget_analysis = None
    st.session_state.itinerary = None
    st.session_state.itinerary_meta = None
    st.session_state.budget_currency = ("USD", "$", "US Dollar")
    st.session_state.effective_budget = None
    st.session_state.home_currency = ("USD", "$", "US Dollar")
    st.session_state.hotel_currency_note = ""
    st.session_state.flight_currency_note = ""
    st.session_state.selected_combo = None
    st.session_state.selected_plan_rank = None
    st.session_state.active_tab = None
    st.session_state.selected_route_idx = 0
    st.session_state.selected_hotel_idx = 0
    if "last_selected_inputs" in st.session_state:
        del st.session_state.last_selected_inputs
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Trip reset! Where would you like to travel for your next trip?"
        }
    ]

# ==========================================
# 2.1 API TOOLS — all free, one key total (GROQ_API_KEY)
#   • get_driving_route  → OSRM (OpenStreetMap routing, no key)
#   • search_trains      → Nominatim + OSRM heuristic (no key)
#   • search_flights     → Groq built-in knowledge (real airlines + prices)
#   • search_hotels      → Groq built-in knowledge (real hotels + prices)
#   • search_attractions → Overpass API (OpenStreetMap POIs, no key)
# ==========================================

def get_driving_route(source_city: str, destination_city: str) -> str:
    """
    Get driving route details between source and destination cities using the
    free OSRM routing engine (OpenStreetMap). No API key required.

    Args:
        source_city: The departing city name.
        destination_city: The arrival city name.
    """
    try:
        # ── Geocode both cities via Nominatim ─────────────────────────────────
        origin_coords = _geocode_city(source_city)
        dest_coords   = _geocode_city(destination_city)

        if not origin_coords or not dest_coords:
            return json.dumps({
                "error": f"Could not geocode '{source_city}' or '{destination_city}' via OpenStreetMap.",
                "feasible": False,
            })

        olat, olng = origin_coords
        dlat, dlng = dest_coords

        # Quick straight-line distance to gate obviously infeasible routes
        import math
        def haversine_km(lat1, lon1, lat2, lon2):
            R = 6371
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlam = math.radians(lon2 - lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        straight_km = haversine_km(olat, olng, dlat, dlng)

        # OSRM won't route ocean crossings; gate at 5,000 km straight-line
        if straight_km > 5000:
            return json.dumps({
                "feasible": False,
                "distance_km": round(straight_km),
                "duration": "N/A",
                "estimated_cost_usd": 0.0,
                "estimated_cost_local": "N/A",
                "border_crossing_notes": (
                    f"Straight-line distance is ~{straight_km:,.0f} km — driving is not feasible "
                    "across this distance (likely requires an ocean crossing). Consider flying."
                ),
            })

        # ── Call OSRM public demo server ──────────────────────────────────────
        # Format: /route/v1/driving/{lng1},{lat1};{lng2},{lat2}
        osrm_url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{olng},{olat};{dlng},{dlat}"
            f"?overview=false&annotations=false"
        )
        resp = requests.get(osrm_url, timeout=15)
        resp.raise_for_status()
        body = resp.json()

        if body.get("code") != "Ok" or not body.get("routes"):
            return json.dumps({
                "feasible": False,
                "distance_km": None,
                "duration": "N/A",
                "estimated_cost_usd": 0.0,
                "estimated_cost_local": "N/A",
                "border_crossing_notes": (
                    "OSRM could not find a drivable road route between these cities. "
                    "This may be due to a water crossing, border closure, or extreme distance."
                ),
            })

        route    = body["routes"][0]
        dist_m   = route["distance"]
        dur_secs = route["duration"]
        dist_km  = round(dist_m / 1000, 1)

        dur_h   = int(dur_secs // 3600)
        dur_m   = int((dur_secs % 3600) // 60)
        dur_str = f"{dur_h}h {dur_m}m"

        # Fuel cost estimate: ~0.12 USD/km (global mid-range petrol car)
        fuel_usd = round(dist_km * 0.12, 2)
        feasible = dist_km <= 3500

        return json.dumps({
            "feasible": feasible,
            "distance_km": dist_km,
            "duration": dur_str,
            "estimated_cost_usd": fuel_usd if feasible else 0.0,
            "estimated_cost_local": f"≈ {fuel_usd:.0f} USD (fuel estimate)" if feasible else "N/A",
            "border_crossing_notes": (
                "Route data from OpenStreetMap / OSRM. "
                "Check visa and border requirements before travel."
                if feasible else
                "Driving is impractical over this distance. Flights or rail recommended."
            ),
        })

    except requests.exceptions.Timeout:
        return json.dumps({"error": "OSRM routing timed out. Try again in a moment.", "feasible": False})
    except Exception as e:
        return json.dumps({"error": f"get_driving_route failed: {str(e)}", "feasible": False})


def search_trains(source_city: str, destination_city: str) -> str:
    """
    Search train options using static price reference table + OSRM distance.
    Returns options only if confidence is medium or high.
    """
    try:
        import math
        curr_code = st.session_state.budget_currency[0]

        # ── Check static reference table first ───────────────────────────────
        ref = get_train_price_reference(source_city, destination_city)
        # Also try resolved city names
        src_r = _resolve_to_city(source_city)
        dst_r = _resolve_to_city(destination_city)
        if not ref and (src_r != source_city or dst_r != destination_city):
            ref = get_train_price_reference(src_r, dst_r)

        if ref:
            ref_min, ref_max, ref_curr, train_type = ref
            # Convert to working currency
            fare_min = convert_currency(ref_min, ref_curr, curr_code)
            fare_max = convert_currency(ref_max, ref_curr, curr_code)
            fare_mid = round((fare_min + fare_max) / 2)
            price_confidence = "high"
            fare_label = f"{curr_code} {fare_min:,.0f}–{fare_max:,.0f}"
            verify_note = "Verify on IRCTC (India) or Rail Europe / Trainline (Europe) before booking."
        else:
            # Fall back to OSRM distance heuristic
            origin_coords = _geocode_city(source_city)
            dest_coords   = _geocode_city(destination_city)

            if not origin_coords or not dest_coords:
                return json.dumps({
                    "has_rail_connectivity": False,
                    "options": [],
                    "note": f"Could not geocode '{source_city}' or '{destination_city}'.",
                })

            def haversine_km(lat1, lon1, lat2, lon2):
                R = 6371
                phi1, phi2 = math.radians(lat1), math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlam = math.radians(lon2 - lon1)
                a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
                return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            dist_km = haversine_km(*origin_coords, *dest_coords)

            if dist_km > 4000:
                return json.dumps({
                    "has_rail_connectivity": False,
                    "options": [],
                    "note": f"Rail not feasible — straight-line distance is ~{dist_km:,.0f} km.",
                })

            # OSRM road distance
            olng, olat = origin_coords[1], origin_coords[0]
            dlng, dlat = dest_coords[1], dest_coords[0]
            road_km = dist_km * 1.25
            try:
                r = requests.get(
                    f"https://router.project-osrm.org/route/v1/driving/{olng},{olat};{dlng},{dlat}?overview=false",
                    timeout=12
                )
                if r.status_code == 200 and r.json().get("code") == "Ok":
                    road_km = r.json()["routes"][0]["distance"] / 1000
            except Exception:
                pass

            if road_km < 300:
                speed, train_type, transfers = 120, "regional", 0
            elif road_km < 800:
                speed, train_type, transfers = 150, "intercity", 0
            elif road_km < 2000:
                speed, train_type, transfers = 180, "overnight", 1
            else:
                speed, train_type, transfers = 130, "long-distance", 2

            # Per-km fare in local currency (very rough)
            per_km_rates = {
                "INR": 1.2, "USD": 0.08, "GBP": 0.07,
                "EUR": 0.07, "AUD": 0.12, "CAD": 0.10,
            }
            per_km = per_km_rates.get(curr_code, 0.08)
            fare_mid = round(road_km * per_km)
            fare_min = round(fare_mid * 0.7)
            fare_max = round(fare_mid * 1.4)
            fare_label = f"≈ {curr_code} {fare_mid:,.0f} (estimated)"
            price_confidence = "low"  # heuristic only
            verify_note = "No reference data for this route — verify actual fares with local rail operator."
            ref_curr = curr_code

        # ── Only return option if confidence is medium or high ────────────────
        if price_confidence == "low":
            return json.dumps({
                "has_rail_connectivity": True,
                "options": [],
                "note": (
                    f"Train may be available between {source_city} and {destination_city} "
                    "but no reliable price data exists for this route. "
                    "Check IRCTC, Trainline, or Raileurope for actual availability and fares."
                ),
                "price_confidence": "low",
            })

        # ── Build duration estimate ────────────────────────────────────────────
        type_speeds = {"highspeed": 220, "express": 130, "superfast": 110,
                       "overnight": 90, "regional": 80, "intercity": 100,
                       "long-distance": 90}
        speed_for_dur = type_speeds.get(train_type, 110)

        # Use OSRM if we have ref (geocode for duration calc)
        try:
            origin_coords = _geocode_city(source_city)
            dest_coords   = _geocode_city(destination_city)
            if origin_coords and dest_coords:
                olng, olat = origin_coords[1], origin_coords[0]
                dlng, dlat = dest_coords[1], dest_coords[0]
                r2 = requests.get(
                    f"https://router.project-osrm.org/route/v1/driving/{olng},{olat};{dlng},{dlat}?overview=false",
                    timeout=10
                )
                if r2.status_code == 200 and r2.json().get("code") == "Ok":
                    road_km_for_dur = r2.json()["routes"][0]["distance"] / 1000
                else:
                    road_km_for_dur = 500
            else:
                road_km_for_dur = 500
        except Exception:
            road_km_for_dur = 500

        dur_h  = road_km_for_dur / speed_for_dur
        dur_hh = int(dur_h)
        dur_mm = int((dur_h - dur_hh) * 60)
        dur_str = f"{dur_hh}h {dur_mm}m"

        type_labels = {
            "highspeed": "High-Speed Rail", "express": "Express Train",
            "superfast": "Superfast Express", "overnight": "Overnight Express",
            "regional": "Regional Express", "intercity": "Intercity Express",
            "long-distance": "Long-Distance Rail",
        }
        train_label = type_labels.get(train_type, "Express Train")

        options = [{
            "train_name":        f"{train_label} ({source_city} → {destination_city})",
            "duration":          dur_str,
            "fare_usd":          fare_mid,
            "fare_local":        fare_label,
            "fare_min":          fare_min,
            "fare_max":          fare_max,
            "transfers":         0 if train_type in ("highspeed", "express", "superfast") else 1,
            "price_confidence":  price_confidence,
            "verify_note":       verify_note,
            "details": (
                f"{train_label} service. Fare range: {curr_code} {fare_min:,.0f}–{fare_max:,.0f}. "
                f"Duration estimate: {dur_str}. {verify_note}"
            ),
        }]

        return json.dumps({
            "has_rail_connectivity": True,
            "options": options,
            "price_confidence": price_confidence,
        })

    except Exception as e:
        return json.dumps({"error": f"search_trains failed: {str(e)}", "has_rail_connectivity": False, "options": []})


def search_flights(source_city: str, destination_city: str) -> str:
    """Search flight options using Groq + static price reference table."""
    import datetime
    depart_date = (datetime.date.today() + datetime.timedelta(days=30)).strftime("%B %d, %Y")
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return json.dumps({"error": "GROQ_API_KEY is not configured.", "options": []})

    trip_type  = st.session_state.get("travel_state", {}).get("trip_type", "domestic")
    curr_code  = st.session_state.budget_currency[0]
    is_intl    = trip_type == "international"
    price_curr = "USD" if is_intl else curr_code

    # ── Look up reference prices from our static table ────────────────────────
    ref = get_price_reference(source_city, destination_city)
    if ref:
        ref_min, ref_max, ref_curr = ref
        # Convert reference to working currency if needed
        if ref_curr != price_curr:
            ref_min = convert_currency(ref_min, ref_curr, price_curr)
            ref_max = convert_currency(ref_max, ref_curr, price_curr)
        price_anchor = (
            f"REFERENCE PRICES for {source_city}→{destination_city}: "
            f"{price_curr} {ref_min:,.0f} – {ref_max:,.0f} (verified from static table). "
            f"Your prices MUST fall within or very close to this range."
        )
        has_ref = True
    else:
        price_anchor = (
            f"No static reference available for this exact route. "
            f"Use your best knowledge of typical {price_curr} economy fares for this distance/region. "
            f"NEVER use values below {price_curr} 50 for any flight."
        )
        has_ref = False

    class FlightOption(BaseModel):
        airline: str = Field(..., description="Real airline name e.g. 'IndiGo', 'Air India', 'Emirates', 'British Airways'.")
        operator: str = Field(..., description="Same as airline — the operating carrier name, used for display.")
        price: float = Field(..., description=f"One-way economy price in {price_curr}. Must match reference range if provided.")
        price_label: str = Field(..., description=f"Formatted price e.g. '₹4,200' or '$750 USD'.")
        duration: str = Field(..., description="Total duration e.g. '2h 15m (Direct)', '14h 30m (1 stop via Dubai)'.")
        stops: str = Field(..., description="'Direct', '1 stop via Dubai', '2 stops' etc.")
        source_airport: str = Field(..., description="Departure airport name and IATA code e.g. 'Chhatrapati Shivaji Maharaj Intl (BOM)'.")
        dest_airport: str = Field(..., description="Arrival airport name and IATA code e.g. 'Indira Gandhi Intl (DEL)'.")
        details: str = Field(..., description="2 sentences about this option and why someone would choose it.")
        price_confidence: str = Field(..., description="Your confidence in the price accuracy: 'high' (matches known fares), 'medium' (reasonable estimate), or 'low' (uncertain).")
        verify_note: str = Field(..., description="Always: 'Verify on Google Flights or Skyscanner before booking.'")

    class FlightSearchResult(BaseModel):
        options: list[FlightOption] = Field(..., description="Exactly 3 options: budget, mid-range, premium.")
        currency_note: str = Field(..., description=f"State clearly: All prices in {price_curr}.")
        route_note: Optional[str] = Field(None, description="Any important route notes.")

    prompt = f"""Provide exactly 3 realistic one-way economy flight options from {source_city} to {destination_city} departing ~{depart_date}.

{price_anchor}

RULES:
- ALL prices must be in {price_curr}
- Use REAL airline names that serve this route
- Use REAL airport IATA codes
- Option 1: cheapest (budget carrier, may have stops)
- Option 2: mid-range (1 stop or direct)
- Option 3: premium/fastest (direct, best carrier)
- Set price_confidence = 'high' only if very sure, 'medium' if reasonable estimate, 'low' if guessing
- operator field = exact airline name (same as airline field)
- NEVER use round placeholder prices like 100, 200, 300 — use realistic market fares"""

    result = _groq_travel_search(api_key, prompt, FlightSearchResult)
    if "error" in result and "options" not in result:
        return json.dumps({"error": result.get("error"), "options": []})

    options = result.get("options", [])
    if isinstance(options, list):
        options = [o if isinstance(o, dict) else o.model_dump() for o in options]

    # ── Post-process: validate prices against reference ───────────────────────
    if has_ref and options:
        for opt in options:
            p = float(opt.get("price", 0))
            # If price is wildly outside reference (>3x max or <0.3x min), flag it
            if p > ref_max * 3 or (p < ref_min * 0.3 and p > 0):
                opt["price_confidence"] = "low"
                opt["verify_note"] = (
                    f"⚠️ Price seems outside typical range ({price_curr} {ref_min:,.0f}–{ref_max:,.0f}). "
                    "Verify on Google Flights before booking."
                )

    return json.dumps({
        "options": options,
        "currency_note": result.get("currency_note", f"All prices in {price_curr}"),
        "has_reference": has_ref,
        "note": result.get("route_note"),
    })




def search_hotels(destination_city: str) -> str:
    """Search hotel options using Groq's built-in travel knowledge with real prices and sources."""
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return json.dumps({"error": "GROQ_API_KEY is not configured.", "hotels": []})

    # Resolve state/region to main city
    resolved_city = _resolve_to_city(destination_city)
    curr_code = st.session_state.budget_currency[0]
    curr_sym  = st.session_state.budget_currency[1]

    class HotelOption(BaseModel):
        name: str = Field(..., description="Real hotel name e.g. 'The Leela Palace New Delhi', 'Zostel Delhi'.")
        price_per_night: float = Field(..., description=f"Realistic price per night in {curr_code}. Never use 100/150/200 unless genuinely correct.")
        rating: float = Field(..., description="Realistic guest rating out of 5.0.")
        location: str = Field(..., description="Specific neighbourhood and landmark proximity.")
        features: str = Field(..., description="6 specific real amenities.")
        source: str = Field(..., description="Source note e.g. 'Typical pricing based on Llama knowledge — verify on Booking.com before booking.'")

def _fetch_hotels_batch(api_key: str, resolved_city: str, destination_city: str,
                         curr_code: str, curr_sym: str,
                         tier_focus: str, exclude_names: list[str]) -> list[dict]:
    """
    Fetch one batch of 5 hotels for a specific tier focus.
    Returns list of hotel dicts (may be fewer than 5 if model truncates).
    Sleeps 2s before calling to avoid rate limits.
    """
    time.sleep(2)

    class HotelItem(BaseModel):
        name: str = Field(..., description="Real hotel name.")
        price_per_night: float = Field(..., description=f"Realistic nightly rate in {curr_code}.")
        rating: float = Field(..., description="Guest rating out of 5.0.")
        location: str = Field(..., description="Specific neighbourhood and proximity to landmarks.")
        features: str = Field(..., description="5-6 specific amenities.")
        source: str = Field(..., description="Always: 'Verify on Booking.com or MakeMyTrip before booking.'")

    class HotelBatch(BaseModel):
        hotels: list[HotelItem] = Field(..., description=f"Exactly 5 {tier_focus} hotels.")
        currency_note: str = Field(..., description=f"All prices in {curr_code} ({curr_sym}).")

    exclude_str = ""
    if exclude_names:
        exclude_str = f"\nDo NOT include these already-listed hotels: {', '.join(exclude_names[:10])}"

    prompt = f"""List exactly 5 {tier_focus} hotels in {resolved_city} ({destination_city}).
ALL prices in {curr_code} ({curr_sym}). Use REAL hotels that exist in {resolved_city}.{exclude_str}
Return exactly 5 hotels — no more, no fewer."""

    try:
        result = _groq_travel_search(api_key, prompt, HotelBatch)
        hotels = result.get("hotels", [])
        return [h if isinstance(h, dict) else h.model_dump() for h in hotels]
    except Exception:
        return []


def search_hotels(destination_city: str) -> str:
    """Search 10 hotel options using two batched Groq calls (5 each) to avoid token limits."""
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return json.dumps({"error": "GROQ_API_KEY is not configured.", "hotels": []})

    resolved_city = _resolve_to_city(destination_city)
    curr_code = st.session_state.budget_currency[0]
    curr_sym  = st.session_state.budget_currency[1]

    # ── Batch 1: budget + mid-range ────────────────────────────────────────────
    batch1 = _fetch_hotels_batch(
        api_key, resolved_city, destination_city, curr_code, curr_sym,
        tier_focus="budget (2) and mid-range (3)", exclude_names=[]
    )

    # ── Batch 2: mid-range + luxury ────────────────────────────────────────────
    existing_names = [h.get("name", "") for h in batch1]
    batch2 = _fetch_hotels_batch(
        api_key, resolved_city, destination_city, curr_code, curr_sym,
        tier_focus="mid-range (2) and luxury (3)", exclude_names=existing_names
    )

    # ── Merge and deduplicate ──────────────────────────────────────────────────
    seen = set()
    all_hotels = []
    for h in batch1 + batch2:
        name = h.get("name", "").strip().lower()
        if name and name not in seen:
            seen.add(name)
            all_hotels.append(h)

    # ── Minimum count enforcement: if still < 6, fetch more ───────────────────
    if len(all_hotels) < 6:
        time.sleep(2)
        extra_names = [h.get("name", "") for h in all_hotels]
        extra = _fetch_hotels_batch(
            api_key, resolved_city, destination_city, curr_code, curr_sym,
            tier_focus="any tier (budget, mid-range, or luxury)", exclude_names=extra_names
        )
        for h in extra:
            name = h.get("name", "").strip().lower()
            if name and name not in seen:
                seen.add(name)
                all_hotels.append(h)

    currency_note = f"All hotel prices shown in {curr_code} ({curr_sym}). Verify on Booking.com before booking."

    return json.dumps({
        "hotels": all_hotels,
        "currency_note": currency_note,
        "note": f"Hotels in {resolved_city}. Prices are estimates — verify before booking.",
    })


def search_attractions(destination_city: str) -> str:
    """
    Search tourist attractions in the destination city using the Overpass API
    (OpenStreetMap). Completely free — no API key required.

    Args:
        destination_city: The city name where the user wants to sightsee.
    """
    try:
        # Resolve state/region names to their main city first
        resolved_city = _resolve_to_city(destination_city)

        # ── Geocode city center via Nominatim ─────────────────────────────────
        coords = _geocode_city(resolved_city)
        if not coords:
            return json.dumps({
                "error": f"Could not geocode '{destination_city}' (tried '{resolved_city}').",
                "attractions": [],
            })
        lat, lng = coords

        # ── Overpass API query — tourism & amenity POIs within 15 km ─────────
        # Targets nodes/ways tagged tourism=* or amenity=place_of_worship
        # that also have a name tag (filters out unnamed features).
        radius_m = 15000
        overpass_query = f"""
[out:json][timeout:25];
(
  node["tourism"~"attraction|museum|gallery|artwork|viewpoint|theme_park|zoo|aquarium"]["name"](around:{radius_m},{lat},{lng});
  way["tourism"~"attraction|museum|gallery|artwork|viewpoint|theme_park|zoo|aquarium"]["name"](around:{radius_m},{lat},{lng});
  node["historic"~"monument|castle|ruins|memorial|archaeological_site"]["name"](around:{radius_m},{lat},{lng});
  way["historic"~"monument|castle|ruins|memorial|archaeological_site"]["name"](around:{radius_m},{lat},{lng});
  node["leisure"~"park|nature_reserve|garden"]["name"](around:{radius_m},{lat},{lng});
  way["leisure"~"park|nature_reserve|garden"]["name"](around:{radius_m},{lat},{lng});
);
out center 20;
"""
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            timeout=30,
            headers={"User-Agent": "AITravelPlanner/1.0 (educational project)"},
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])

        if not elements:
            # ── Groq fallback when Overpass returns nothing ───────────────
            api_key = st.session_state.get("api_key", "")
            resolved_city = _resolve_to_city(destination_city)
            if api_key:
                class AttrItem(BaseModel):
                    name: str = Field(..., description="Real attraction name.")
                    category: str = Field(..., description="Category e.g. 'History & Culture', 'Nature', 'Art', 'Entertainment', 'Sightseeing'.")
                    rating: float = Field(..., description="Visitor rating out of 5.0.")
                    description: str = Field(..., description="2 sentence description of the attraction.")
                    estimated_cost: float = Field(..., description="Entry fee in local currency. 0.0 if free.")
                class AttrList(BaseModel):
                    attractions: list[AttrItem] = Field(..., description="Exactly 6 real tourist attractions.")

                result = _groq_travel_search(
                    api_key,
                    f"""List exactly 6 real tourist attractions in {resolved_city} ({destination_city}).
For each: real name, category, visitor rating out of 5, 2-sentence description, and entry fee in local currency (0 if free).
Use only attractions that genuinely exist.""",
                    AttrList,
                )
                if result.get("attractions"):
                    return json.dumps({"attractions": [
                        {
                            "name": a.get("name", ""),
                            "category": a.get("category", "Sightseeing"),
                            "rating": a.get("rating", 4.0),
                            "description": a.get("description", ""),
                            "estimated_cost": a.get("estimated_cost", 0.0),
                        }
                        for a in result["attractions"]
                    ], "note": f"Attractions sourced from Groq knowledge for {resolved_city}."})

            return json.dumps({
                "attractions": [],
                "note": f"No attractions found for '{destination_city}'. Try entering a specific city name (e.g. 'Chennai' instead of 'Tamil Nadu').",
            })

        # ── OSM tag → category mapping ────────────────────────────────────────
        TOURISM_MAP = {
            "museum":              ("History & Culture", 12.0),
            "gallery":             ("Art",               8.0),
            "artwork":             ("Art",               0.0),
            "attraction":          ("Sightseeing",       5.0),
            "viewpoint":           ("Sightseeing",       0.0),
            "theme_park":          ("Entertainment",    25.0),
            "zoo":                 ("Entertainment",    15.0),
            "aquarium":            ("Entertainment",    15.0),
        }
        HISTORIC_MAP = {
            "monument":            ("History & Culture", 0.0),
            "castle":              ("History & Culture", 10.0),
            "ruins":               ("History & Culture", 5.0),
            "memorial":            ("History & Culture", 0.0),
            "archaeological_site": ("History & Culture", 8.0),
        }
        LEISURE_MAP = {
            "park":                ("Nature",            0.0),
            "nature_reserve":      ("Nature",            0.0),
            "garden":              ("Nature",            0.0),
        }

        seen_names = set()
        attractions = []

        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name", "").strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            # Determine category and default entry fee
            tourism_val = tags.get("tourism", "")
            historic_val = tags.get("historic", "")
            leisure_val  = tags.get("leisure", "")

            if tourism_val in TOURISM_MAP:
                category, fee = TOURISM_MAP[tourism_val]
            elif historic_val in HISTORIC_MAP:
                category, fee = HISTORIC_MAP[historic_val]
            elif leisure_val in LEISURE_MAP:
                category, fee = LEISURE_MAP[leisure_val]
            else:
                category, fee = "Sightseeing", 5.0

            # OSM sometimes carries a description or wikipedia summary
            description = (
                tags.get("description")
                or tags.get("wikipedia", "").replace("en:", "").replace("_", " ")
                or f"A notable {category.lower()} attraction in {destination_city}."
            )
            if len(description) > 200:
                description = description[:197] + "…"

            # Use OSM rating if present, otherwise default 4.2
            try:
                rating = float(tags.get("stars", tags.get("rating", 4.2)))
                rating = min(5.0, max(1.0, rating))
            except (ValueError, TypeError):
                rating = 4.2

            attractions.append({
                "name":           name,
                "category":       category,
                "rating":         rating,
                "description":    description,
                "estimated_cost": fee,
            })

            if len(attractions) >= 8:
                break

        if not attractions:
            return json.dumps({
                "attractions": [],
                "note": "OpenStreetMap returned features but none had usable names.",
            })

        return json.dumps({"attractions": attractions})

    except requests.exceptions.Timeout:
        return json.dumps({"error": "Overpass API timed out. Try again in a moment.", "attractions": []})
    except Exception as e:
        return json.dumps({"error": f"search_attractions failed: {str(e)}", "attractions": []})



# ==========================================
# 3. PYDANTIC SCHEMAS
# ==========================================
class ExtractedTravelState(BaseModel):
    source_city: Optional[str] = Field(None, description="The city or airport the traveler is departing from. Extract only if explicitly mentioned. e.g., 'Delhi', 'New York'.")
    destination_city: Optional[str] = Field(None, description="The destination city or region the user wants to visit. Extract only if explicitly mentioned. e.g., 'Manali', 'Paris'.")
    budget: Optional[float] = Field(None, description="The total budget for the trip as a number. Extract numeric value only, do not include currency symbols. e.g., 20000, 1500.")
    budget_currency_code: Optional[str] = Field(None, description="The ISO currency code the user stated their budget in. e.g., 'INR', 'USD', 'GBP', 'EUR'. Extract from context: if user says '£500' that's GBP, '₹10000' is INR, '$2000' is USD. If not explicitly stated, set to null.")
    trip_days: Optional[int] = Field(None, description="The duration of the trip in number of days. e.g., 4, 7.")
    travelers: Optional[int] = Field(None, description="The number of people traveling. e.g., 1, 3.")
    rooms: Optional[int] = Field(None, description="The number of hotel rooms required. e.g., 1, 2.")
    preferences: Optional[str] = Field(None, description="Any other specific preferences, activities, food constraints, or notes.")
    trip_type: Optional[str] = Field(None, description="Whether the trip is 'domestic' or 'international' based on the cities. e.g., if traveling within India it's domestic. If source is in USA and destination is France, it's international.")

class AgentResponse(BaseModel):
    extracted_data: ExtractedTravelState = Field(..., description="Extract all travel information from the conversation history, combining it with the current travel state.")
    reply: str = Field(..., description="A friendly, natural language reply. If any of the 6 required fields (source_city, destination_city, budget, trip_days, travelers, rooms) are missing, ask for EXACTLY ONE missing field. Keep the question highly contextual to the current conversation. Do not ask for multiple things at once. If all required fields are present, provide a warm summary response.")

# Route Agent Schemas
class RouteOption(BaseModel):
    mode: str = Field(..., description="The mode of transport: 'drive', 'train', 'flight', or 'mixed'.")
    operator: str = Field(..., description="The specific carrier/operator name. For flights: airline e.g. 'IndiGo', 'Air India', 'Emirates'. For trains: train name e.g. 'Rajdhani Express', 'Eurostar', 'Vande Bharat'. For driving: 'Self Drive'. For mixed: main carrier.")
    cost: float = Field(..., description="Estimated cost of this option (numeric value in the working currency).")
    duration: str = Field(..., description="Estimated travel duration (e.g. '11h 30m', '1.5 hours').")
    source: str = Field(..., description="Departure terminal, city, or airport.")
    destination: str = Field(..., description="Arrival terminal, city, or airport.")
    description: str = Field(..., description="Brief description starting with operator name (e.g. 'IndiGo operates a direct flight', 'Rajdhani Express runs overnight').")
    feasibility_notes: str = Field(..., description="Border crossing notes, visa requirements, or transfer logistics.")
    price_confidence: str = Field(..., description="'high' if price matches known market rates, 'medium' if estimated, 'low' if uncertain.")

class RouteAgentResponse(BaseModel):
    options: list[RouteOption] = Field(..., description="List of evaluated transport options, ranked from best to worst.")
    explanation: str = Field(..., description="Detailed explanation of the tradeoffs, budget considerations, and rationale for the ranking.")

# Hotel Agent Schemas
class HotelOption(BaseModel):
    name: str = Field(..., description="Name of the hotel.")
    price_per_night: float = Field(..., description="Price per night for one room (numeric).")
    total_cost: float = Field(..., description="Total lodging cost computed as price_per_night * trip_days * rooms.")
    rating: float = Field(..., description="Hotel star rating out of 5.")
    location: str = Field(..., description="Proximity description or area.")
    tradeoffs_explanation: str = Field(..., description="Pros, cons, and tradeoffs for this option.")

class HotelAgentResponse(BaseModel):
    hotels: list[HotelOption] = Field(..., description="Lodging options matching the budget profile, ranked.")
    summary: str = Field(..., description=" Lodging landscape summary.")

# Attractions Agent Schemas
class Attraction(BaseModel):
    name: str = Field(..., description="Attraction name.")
    rating: float = Field(..., description="Visitor rating out of 5.")
    description: str = Field(..., description="Short explanation of the sight.")
    relevance_reason: str = Field(..., description="Why this matches user preferences.")
    estimated_entry_fee: float = Field(..., description="Ticket or permit price (numeric).")

class AttractionsResponse(BaseModel):
    attractions: list[Attraction] = Field(..., description="Ranked local sightseeing recommendations.")

# Budget Agent Schemas
class BudgetBreakdown(BaseModel):
    plan_name: str = Field(..., description="'Economy', 'Balanced', or 'Premium'.")
    transport_cost: float = Field(..., description="Estimated transport cost.")
    lodging_cost: float = Field(..., description="Estimated lodging cost.")
    activities_cost: float = Field(..., description="Calculated attractions ticket fees.")
    food_and_other_cost: float = Field(..., description="Food, drink, and incidentals buffer.")
    total_cost: float = Field(..., description="Sum total of all categories.")
    is_within_budget: bool = Field(..., description="Fits in user's total budget.")
    description: str = Field(..., description="Brief description of what this package offers.")

class BudgetAgentResponse(BaseModel):
    is_feasible: bool = Field(..., description="True if at least one package fits the budget.")
    budget_warning: Optional[str] = Field(None, description="Warning if budget is very tight or exceeded.")
    breakdowns: list[BudgetBreakdown] = Field(..., description="Economy, Balanced, and Premium packages.")
    comparison_summary: str = Field(..., description="tradeoff analysis comparing packages.")


# ===========================================================================
# COMBINATORIAL BUDGET ENGINE
# Pure Python — no LLM call. Generates all valid transport × hotel × attraction
# combinations that fit within the effective budget, ranked cheapest→most expensive.
# ===========================================================================

def compute_attraction_tiers(attractions: list, trip_days: int) -> dict:
    """
    Returns 4 attraction subsets keyed by tier name.
    Attractions are distributed across days for the plan label.
    """
    free   = [a for a in attractions if (a.get("estimated_entry_fee") or 0) == 0]
    paid   = [a for a in attractions if (a.get("estimated_entry_fee") or 0) > 0]
    top3   = (attractions[:3] if len(attractions) >= 3 else attractions)
    all_a  = attractions

    return {
        "No Attractions":       [],
        "Free Only":            free,
        "Standard (Top 3-4)":   top3,
        "Full Experience":      all_a,
    }

def attractions_cost(tier: list) -> float:
    return sum(a.get("estimated_entry_fee") or 0 for a in tier)

def attractions_label(tier: list, trip_days: int) -> str:
    if not tier:
        return "No sightseeing"
    # Distribute across days
    per_day = max(1, len(tier) // max(trip_days, 1))
    parts = []
    for i in range(0, len(tier), max(per_day, 1)):
        day_chunk = tier[i:i+per_day]
        day_num = i // max(per_day, 1) + 2  # Day 2 onward
        names = " + ".join(a["name"] for a in day_chunk)
        parts.append(f"Day {day_num}: {names}")
    return " | ".join(parts)

def compute_combinations(
    route_options: list,
    hotel_options: list,
    attractions: list,
    trip_days: int,
    effective_budget: float,
    curr_code: str,
) -> tuple[list, list]:
    """
    Computes all transport × hotel × attraction-tier combinations.
    Returns (fitting_combos, over_budget_combos), each sorted by total_cost asc.

    Each combo dict has:
      route, hotel, attraction_tier_name, attraction_list,
      transport_cost, hotel_cost, attraction_cost, food_cost, total_cost,
      fits_budget, label
    """
    food_daily = {
        "USD": 60, "GBP": 50, "EUR": 55, "AUD": 80, "CAD": 75,
        "INR": 1500, "JPY": 4000, "AED": 150, "SGD": 60,
        "THB": 600, "MYR": 80,
    }
    food_per_day = food_daily.get(curr_code, 50)

    tiers = compute_attraction_tiers(attractions, trip_days)
    fitting, over = [], []

    # Filter out infeasible routes — those with cost=0 and mode is drive/train
    # (cost=0 means OSRM/ref said it's not feasible, e.g. ocean crossing)
    feasible_routes = []
    for route in route_options:
        mode   = route.get("mode", "").lower()
        cost   = float(route.get("cost") or 0)
        desc   = (route.get("description") or "").lower()
        notes  = (route.get("feasibility_notes") or "").lower()
        # A route is infeasible if cost is 0 and it's drive/train
        # OR if the description/notes explicitly say infeasible/not feasible
        is_infeasible = (
            (mode in ("drive", "train") and cost == 0) or
            any(kw in desc + notes for kw in ["not feasible", "infeasible", "not possible",
                                               "ocean crossing", "physically impossible",
                                               "impractical"])
        )
        if not is_infeasible:
            feasible_routes.append(route)

    # If all routes filtered out, use original list as fallback
    if not feasible_routes:
        feasible_routes = route_options

    for route in feasible_routes:
        t_cost = float(route.get("cost") or 0)
        for hotel in hotel_options:
            h_cost = float(hotel.get("total_cost") or 0)
            for tier_name, tier_list in tiers.items():
                a_cost = attractions_cost(tier_list)
                f_cost = food_per_day * trip_days
                total  = t_cost + h_cost + a_cost + f_cost

                combo = {
                    "route":               route,
                    "hotel":               hotel,
                    "attraction_tier_name":tier_name,
                    "attraction_list":     tier_list,
                    "transport_cost":      t_cost,
                    "hotel_cost":          h_cost,
                    "attraction_cost":     a_cost,
                    "food_cost":           f_cost,
                    "total_cost":          total,
                    "fits_budget":         total <= effective_budget,
                    "label": (
                        f"{route.get('mode','?').capitalize()} → "
                        f"{hotel.get('name','?')} → "
                        f"{tier_name}"
                    ),
                }
                if total <= effective_budget:
                    fitting.append(combo)
                else:
                    over.append(combo)

    fitting.sort(key=lambda x: x["total_cost"])
    over.sort(key=lambda x: x["total_cost"])
    return fitting, over

def derive_econ_balanced_premium(fitting: list, over: list) -> tuple[dict|None, dict|None, dict|None]:
    """Pick Economy (cheapest), Balanced (middle), Premium (most expensive) from fitting."""
    pool = fitting if fitting else over
    if not pool:
        return None, None, None
    n = len(pool)
    economy  = pool[0]
    premium  = pool[-1]
    balanced = pool[n // 2]
    return economy, balanced, premium


# ------------------------------------------------------------------
# Planner Agent Schemas
# ------------------------------------------------------------------
class ItineraryActivity(BaseModel):
    time_slot: str = Field(
        ...,
        description="One of: 'Morning', 'Afternoon', 'Evening', or 'Travel Day'."
    )
    title: str = Field(
        ...,
        description="Short activity title, e.g. 'Visit Eiffel Tower' or 'Check-in & rest'."
    )
    location: str = Field(
        ...,
        description="Specific place name or area, e.g. 'Eiffel Tower, Champ de Mars'."
    )
    duration_hours: float = Field(
        ...,
        description="Estimated time in hours for this activity, e.g. 1.5, 2.0."
    )
    description: str = Field(
        ...,
        description="2-3 sentence narrative: what the traveler will do/see and why it's placed here."
    )
    tips: str = Field(
        ...,
        description="One practical tip: best time to go, booking advice, transport connection, nearby food, etc."
    )
    estimated_cost: float = Field(
        ...,
        description="Estimated per-person cost in the trip's currency (0.0 if free)."
    )
    activity_type: str = Field(
        ...,
        description="Category: 'Sightseeing', 'Culture', 'Nature', 'Food & Dining', 'Shopping', 'Leisure', 'Travel', 'Accommodation'."
    )

class ItineraryDay(BaseModel):
    day_number: int = Field(..., description="Day number starting from 1.")
    date_label: str = Field(
        ...,
        description="Human-readable label, e.g. 'Day 1 — Arrival & Orientation' or 'Day 3 — Cultural Deep Dive'."
    )
    theme: str = Field(
        ...,
        description="One-line theme for the day, e.g. 'Arrival & First Impressions', 'Art & History', 'Nature & Relaxation'."
    )
    activities: list[ItineraryActivity] = Field(
        ...,
        description="Ordered list of activities for the day. Typically 3-4 items across morning/afternoon/evening slots."
    )
    day_summary: str = Field(
        ...,
        description="2-3 sentence recap of the day's arc and what the traveler will have experienced."
    )
    estimated_day_cost: float = Field(
        ...,
        description="Sum of all activity costs for the day (excluding transport and lodging)."
    )

class PlannerAgentResponse(BaseModel):
    itinerary_title: str = Field(
        ...,
        description="Evocative title for the full trip, e.g. 'Paris in 5 Days: Art, Culture & Cuisine'."
    )
    days: list[ItineraryDay] = Field(
        ...,
        description="Complete day-by-day itinerary. Must have exactly trip_days entries."
    )
    planner_notes: str = Field(
        ...,
        description="3-5 sentences of overall planning rationale: how attractions were distributed, pacing logic, arrival/departure day handling."
    )
    total_activities_cost: float = Field(
        ...,
        description="Sum of all per-day estimated_day_cost values."
    )
# ==========================================
def run_requirements_agent(api_key: str, current_state: dict, messages: list, new_input: str):
    """Calls Groq to extract structured travel requirements and generate a follow-up."""

    system_instruction = """You are a friendly AI travel planning assistant. Your job is to have a natural conversation to collect travel details.

Required information to collect:
1. source_city - Where they are traveling FROM
2. destination_city - Where they want to GO
3. budget - Total budget as a NUMBER only (no currency symbols)
4. budget_currency_code - The currency of the budget (e.g. INR, USD, GBP, EUR). ALWAYS ask if not provided.
5. trip_days - Number of days for the trip
6. travelers - Number of people traveling
7. rooms - Number of rooms needed
8. preferences - Any special interests (optional, can be "None")
9. trip_type - "domestic" or "international" based on cities

CRITICAL CURRENCY RULE:
- If the user gives a budget number WITHOUT specifying currency, you MUST ask which currency.
- Example: user says "my budget is 15000" → ask "What currency is that in? (e.g. INR, USD, GBP)"
- If user says "₹15000" → budget_currency_code = "INR"
- If user says "$2000" → budget_currency_code = "USD"
- If user says "£500" → budget_currency_code = "GBP"
- If user says "€1000" → budget_currency_code = "EUR"
- Never assume a currency — always confirm it.

Guidelines:
- Be conversational and friendly
- Ask for ONE missing field at a time
- When you have all required fields including budget_currency_code, congratulate them and say you're ready to plan
- Extract information from context even if not explicitly stated

You must ALWAYS respond with a JSON object containing exactly these fields:
{
  "extracted_state": {
    "source_city": null or "city name",
    "destination_city": null or "city name",
    "budget": null or number,
    "budget_currency_code": null or "INR"/"USD"/"GBP"/"EUR" etc,
    "trip_days": null or number,
    "travelers": null or number,
    "rooms": null or number,
    "preferences": null or "text",
    "trip_type": null or "domestic"/"international"
  },
  "response_text": "Your conversational response to the user"
}"""

    # Build conversation history for Groq
    groq_messages = [{"role": "system", "content": system_instruction}]
    for msg in messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        groq_messages.append({"role": role, "content": msg["content"]})

    # Add current state context
    state_context = f"\nCurrent known state: {json.dumps(current_state)}\nUser message: {new_input}"
    groq_messages.append({"role": "user", "content": state_context})

    try:
        client = get_groq_client(api_key)
        # Use fast lightweight model for conversation — much higher rate limits
        for attempt in range(2):  # 1 auto-retry on rate limit
            try:
                resp = client.chat.completions.create(
                    model=GROQ_FAST_MODEL,
                    messages=groq_messages,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
                clean = re.sub(r"```json|```", "", raw).strip()
                data = json.loads(clean)
                extracted = data.get("extracted_state", {})
                response_text = data.get("response_text", "Could you tell me more about your trip?")
                return extracted, response_text
            except Exception as e:
                if _is_rate_limit_error(e) and attempt == 0:
                    wait = min(_extract_retry_seconds(e), 15)
                    time.sleep(wait)
                    continue
                raise
    except Exception as e:
        if _is_rate_limit_error(e):
            wait = _extract_retry_seconds(e)
            _show_rate_limit_warning(wait, "understanding your requirements")
            return None, f"⏱️ Rate limit reached — please wait {wait} seconds and send your message again."
        return None, "I'm having trouble connecting right now. Please try again or check your API key."
    
def run_route_agent_fact_gathering(api_key: str, travel_state: dict) -> str:
    """
    Route Agent — Direct approach (2 Groq calls):
      1. Call all three transport tool functions directly in Python.
      2. Feed the raw results to Groq once to produce a comparison report.
      3. Call Groq once more to parse the report into structured JSON.

    This replaces the previous iterative tool-calling loop (up to 12 calls)
    with a fixed 2-call flow, dramatically reducing quota usage.
    """
    src  = travel_state['source_city']
    dst  = travel_state['destination_city']
    trip_type = travel_state.get('trip_type', 'domestic')

    # ── Step 1: Call tools with pauses between Groq calls ────────────────────
    driving_raw  = get_driving_route(src, dst)   # OSRM only — no Groq
    trains_raw   = search_trains(src, dst)        # OSRM only — no Groq
    time.sleep(3)                                  # pause before flight Groq call
    flights_raw  = search_flights(src, dst)        # calls Groq
    time.sleep(3)                                  # pause before analysis Groq call

    curr_code   = st.session_state.budget_currency[0]
    curr_symbol = st.session_state.budget_currency[1]
    budget_amt  = travel_state.get('budget', 0)

    analysis_prompt = f"""
You are a Route Planning Agent. Below are the raw results from three transport APIs
for a trip from {src} to {dst}.

DRIVING DATA:
{driving_raw}

RAIL / TRANSIT DATA:
{trains_raw}

FLIGHT DATA:
{flights_raw}

TRIP CONTEXT:
- Budget: {curr_symbol}{budget_amt:,.0f} {curr_code}
- All costs must be expressed in {curr_code}
- Duration: {travel_state['trip_days']} days
- Travelers: {travel_state['travelers']}
- Preferences: {travel_state.get('preferences', 'None')}
- Trip type: {trip_type}

CRITICAL RULES:
1. Use ONLY the data above. Do not invent prices or durations.
2. For FLIGHTS: set operator = exact airline name (e.g. "IndiGo", "Air India", "Emirates").
   If multiple airlines are available, pick the most common/reliable for each tier.
3. For TRAINS: set operator = exact train service name (e.g. "Rajdhani Express", "Eurostar", "Vande Bharat").
   If train name is unclear from the data, use the generic type (e.g. "Intercity Express").
   Only include train if rail data shows genuine connectivity.
4. For DRIVING: set operator = "Self Drive".
5. All prices must be in {curr_code}. If prices seem wrong (<{curr_code} 100 for a flight), note it.
6. Set price_confidence = 'high' if price matches known market rates, 'medium' if estimated from data,
   'low' if uncertain or the data showed suspicious values.
7. The description field MUST start with the operator name.

Write a detailed markdown comparison report ranking options best-to-worst.
"""

    try:
        report = groq_complete(
            api_key,
            system="You are a factual travel analyst. Only use data provided to you. Never invent prices, distances, or durations.",
            user=analysis_prompt,
        )
        return report

    except Exception as e:
        if _is_rate_limit_error(e):
            wait = _extract_retry_seconds(e)
            _show_rate_limit_warning(wait, "analysing routes")
            return f"RATE_LIMIT:{wait}"
        return f"Error executing route fact gathering: {e}"


def parse_route_report_to_structured(api_key: str, comparison_report: str) -> dict:
    """
    Decoupled Structured Output Extraction: Converts the Markdown comparison sheet into structured Pydantic schemas.
    """
    user_prompt = f"""
Analyze the following travel comparison report and extract the structured route options and explanation.

Comparison Report:
{comparison_report}

Convert this into the required JSON schema format.
"""
    try:
        return groq_json(
            api_key,
            system="You are a data extraction assistant. Convert travel route analysis reports into structured JSON.",
            user=user_prompt,
            schema=RouteAgentResponse,
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            wait = _extract_retry_seconds(e)
            _show_rate_limit_warning(wait, "parsing route options")
            return {"error": "rate_limit", "wait_seconds": wait}
        st.error(f"Error parsing route report: {e}")
        return {}


def run_hotel_agent(api_key: str, travel_state: dict) -> dict:
    """
    Hotel Agent — Direct approach (1 Groq call).
    Calls search_hotels directly, then ranks and structures results.
    """
    dst        = travel_state['destination_city']
    budget     = travel_state['budget']
    trip_days  = travel_state['trip_days']
    rooms      = travel_state['rooms']
    prefs      = travel_state.get('preferences', 'None')
    curr_code  = st.session_state.budget_currency[0]
    curr_sym   = st.session_state.budget_currency[1]
    resolved_city = _resolve_to_city(dst)  # resolve region/state to main city

    # Step 1: call tool directly — zero LLM calls
    hotels_raw_str = search_hotels(dst)
    hotels_raw = json.loads(hotels_raw_str)
    currency_note = hotels_raw.get("currency_note", f"Prices in {curr_code}")

    prompt = f"""
You are a lodging agent. Below is raw hotel data for {dst} (resolved city: {resolved_city}).

RAW HOTEL DATA:
{hotels_raw_str}

IMPORTANT: All prices in the raw data are in {curr_code} ({curr_sym}).
Do NOT convert prices. Use them as-is.

TRIP CONTEXT:
- Total budget: {budget} {curr_code}
- Duration: {trip_days} days
- Rooms needed: {rooms}
- Preferences: {prefs}

Instructions:
1. For each hotel compute total_cost = price_per_night × {trip_days} × {rooms}.
   Use the EXACT price_per_night values from the raw data — do not modify them.
2. Include ALL hotels — both those within budget and those over budget (mark them clearly).
3. Rank hotels by rating within each tier (budget/mid/luxury).
4. Write a tradeoffs_explanation for each (2 sentences max).
5. Write a summary of the lodging landscape (2-3 sentences).
6. Return ALL hotels in the structured JSON — do not drop any.
"""
    try:
        result = groq_json(api_key,
            system="You are a lodging data extraction assistant. Return only valid JSON. Never modify prices from raw data.",
            user=prompt,
            schema=HotelAgentResponse,
        )
        result["currency_note"] = currency_note
        return result
    except Exception as e:
        if _is_rate_limit_error(e):
            wait = _extract_retry_seconds(e)
            _show_rate_limit_warning(wait, "searching hotels")
            return {"error": "rate_limit", "wait_seconds": wait}
        st.error(f"Hotel agent error: {e}")
        return {}


def run_attractions_agent(api_key: str, travel_state: dict) -> dict:
    """
    Attractions Agent — Direct approach (1 Groq call):
    Calls search_attractions directly in Python, then asks Groq once to
    rank and structure the results by user preferences.
    """
    dst   = travel_state['destination_city']
    prefs = travel_state.get('preferences', 'None')

    # Step 1: call tool directly — zero Gemini calls
    attractions_raw = search_attractions(dst)

    # Step 2: one Groq call to rank and structure
    prompt = f"""
You are a sightseeing guide agent. Below is raw attractions data for {dst}.

RAW ATTRACTIONS DATA:
{attractions_raw}

TRAVELLER PREFERENCES: {prefs}

Instructions:
1. Rank the attractions by how well they match the traveller's preferences.
   - Nature/Adventure preferences → prioritise parks, valleys, outdoor spots.
   - History/Culture preferences → prioritise temples, museums, landmarks.
   - No preference → rank by rating.
2. For each attraction write a custom_reason (1 sentence) explaining why it
   suits this traveller.
3. Return structured JSON matching the schema exactly.
"""
    try:
        return groq_json(api_key,
            system="You are an attractions data extraction assistant. Return only valid JSON.",
            user=prompt,
            schema=AttractionsResponse,
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            wait = _extract_retry_seconds(e)
            _show_rate_limit_warning(wait, "finding attractions")
            return {"error": "rate_limit", "wait_seconds": wait}
        st.error(f"Attractions agent error: {e}")
        return {}


def run_budget_agent(api_key: str, travel_state: dict, selected_route: dict, selected_hotel: dict, attractions: list) -> dict:
    """
    Budget Agent: Computes Economy, Balanced, and Premium packages.
    All amounts in the working currency (USD for international, local for domestic).
    """
    curr_code   = st.session_state.budget_currency[0]
    curr_symbol = st.session_state.budget_currency[1]
    curr_name   = st.session_state.budget_currency[2]
    budget      = travel_state.get('budget', 0)
    trip_days   = travel_state.get('trip_days', 1)
    travelers   = travel_state.get('travelers', 1)
    rooms       = travel_state.get('rooms', 1)
    trip_type   = travel_state.get('trip_type', 'domestic')

    # Realistic daily food estimates by currency
    food_daily = {
        "USD": 60, "GBP": 50, "EUR": 55, "AUD": 80,
        "CAD": 75, "INR": 1500, "JPY": 4000, "AED": 150,
        "SGD": 60, "THB": 600, "MYR": 80,
    }
    food_per_day = food_daily.get(curr_code, 50)

    route_cost  = selected_route.get('cost', 0) or 0
    hotel_night = selected_hotel.get('price_per_night', 0) or 0
    hotel_total = selected_hotel.get('total_cost', 0) or (hotel_night * trip_days * rooms)

    prompt = f"""
You are a travel budget analyst. ALL amounts must be in {curr_code} ({curr_symbol}, {curr_name}).

TRIP:
- Route: {selected_route.get('mode')} from {selected_route.get('source')} → {selected_route.get('destination')}
- Selected transport cost: {curr_symbol}{route_cost:,.0f} {curr_code}
- Selected hotel: {selected_hotel.get('name')} at {curr_symbol}{hotel_night:,.0f}/night × {trip_days} nights × {rooms} room(s) = {curr_symbol}{hotel_total:,.0f} {curr_code}
- Total budget: {curr_symbol}{budget:,.0f} {curr_code}
- Duration: {trip_days} days | Travelers: {travelers} | Rooms: {rooms}
- Trip type: {trip_type}

ATTRACTIONS (entry fees in {curr_code}):
{json.dumps([{"name": a.get("name"), "fee": a.get("estimated_entry_fee", 0)} for a in attractions[:6]], indent=2)}

FOOD BUDGET GUIDANCE: For {curr_name}, a reasonable daily food+incidentals budget is {curr_symbol}{food_per_day}/person/day.
Total food estimate for {trip_days} days × {travelers} person(s) = {curr_symbol}{food_per_day * trip_days * travelers:,.0f} {curr_code}.

RULES:
1. ALL costs must be in {curr_code}. Never mix currencies.
2. Use REALISTIC costs — the flight alone is {curr_symbol}{route_cost:,.0f}, hotel total is {curr_symbol}{hotel_total:,.0f}.
3. Generate exactly 3 packages:
   - Economy: cheapest transport option, budget hotel, only free attractions
   - Balanced: selected transport ({curr_symbol}{route_cost:,.0f}), selected hotel ({curr_symbol}{hotel_total:,.0f}), top 3 attractions
   - Premium: best/fastest transport, luxury hotel, all attractions + extras
4. For each package provide transport_cost, lodging_cost, activities_cost, food_and_other_cost, total_cost — all in {curr_code}.
5. Compare total_cost to budget of {curr_symbol}{budget:,.0f} {curr_code} to determine feasibility.
6. The Balanced package transport_cost MUST equal {route_cost:,.0f} and lodging_cost MUST equal {hotel_total:,.0f}.
"""

    try:
        return groq_json(api_key,
            system=f"You are a precise travel budget analyst. All costs in {curr_code}. Never use placeholder values. Use realistic figures.",
            user=prompt,
            schema=BudgetAgentResponse,
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            wait = _extract_retry_seconds(e)
            _show_rate_limit_warning(wait, "calculating budget")
            return {"error": "rate_limit", "wait_seconds": wait}
        st.error(f"Budget agent error: {e}")
        return {}

# ------------------------------------------------------------------
# Planner Agent
# ------------------------------------------------------------------
def run_planner_agent(
    api_key: str,
    travel_state: dict,
    selected_route: dict,
    selected_hotel: dict,
    attractions: list,
    budget_analysis: dict,
) -> dict:
    """
    Planner Agent: synthesises all upstream outputs into a realistic
    day-by-day itinerary distributed across the full trip duration.

    Inputs
    ------
    travel_state      – full TravelState dict (source, destination, days, travelers, preferences …)
    selected_route    – the RouteOption the user picked (mode, cost, duration, source, destination)
    selected_hotel    – the HotelOption the user picked (name, location, price_per_night, total_cost)
    attractions       – ranked list of Attraction dicts from the attractions agent
    budget_analysis   – BudgetAgentResponse dict (used to set realistic daily cost expectations)

    Returns
    -------
    PlannerAgentResponse as a plain dict (JSON-deserialised).
    On failure returns {} so the caller can handle gracefully.
    """
    trip_days      = travel_state.get("trip_days", 1)
    destination    = travel_state.get("destination_city", "the destination")
    source         = travel_state.get("source_city", "home")
    travelers      = travel_state.get("travelers", 1)
    preferences    = travel_state.get("preferences") or "No specific preferences stated."
    trip_type      = travel_state.get("trip_type", "domestic")
    currency       = st.session_state.budget_currency[0]  # e.g. "USD", "GBP", "INR"
    curr_symbol    = st.session_state.budget_currency[1]  # e.g. "$", "£", "₹"

    # Compact serialisation of upstream data so the prompt stays focused
    route_summary = (
        f"{selected_route.get('mode', 'unknown').upper()} | "
        f"{selected_route.get('source')} → {selected_route.get('destination')} | "
        f"Duration: {selected_route.get('duration')} | "
        f"Cost: {float(selected_route.get('cost') or 0):,.0f} {currency}"
    )
    hotel_summary = (
        f"{selected_hotel.get('name')} | "
        f"{float(selected_hotel.get('price_per_night') or 0):,.0f} {currency}/night | "
        f"Location: {selected_hotel.get('location')}"
    )

    # Serialise attractions with their fees for scheduling
    attractions_text = "\n".join(
        f"  {i+1}. {a.get('name')} — {a.get('description', '')} "
        f"[Entry fee: {a.get('estimated_entry_fee', 0):,.0f} {currency}]"
        for i, a in enumerate(attractions)
    ) or "  No specific attractions data available."

    # Extract balanced plan cost if available for daily food budget reference
    balanced_plan  = next(
        (p for p in budget_analysis.get("breakdowns", []) if p.get("plan_name") == "Balanced"),
        {}
    )
    daily_food_est = (
        balanced_plan.get("food_and_other_cost", 0) / max(trip_days, 1)
        if balanced_plan else 0
    )

    system_instruction = f"""You are an expert travel itinerary planner with deep knowledge of {destination}.
Your job is to create a detailed, realistic, day-by-day travel itinerary that:
- Covers exactly {trip_days} days (no more, no fewer)
- Respects the traveller's preferences and pace
- Treats Day 1 as an arrival/orientation day (lighter schedule)
- Treats Day {trip_days} as a departure day (morning activity + check-out if multi-day)
- Distributes the ranked attractions across the MIDDLE days to avoid overcrowding
- Slots each attraction into the most logical time of day (e.g. museums in the morning, sunset spots in the evening)
- Mixes paid and free activities across each day
- Includes practical meal/food suggestions in the Evening slot or as standalone Food & Dining activities
- Keeps each day to 3-4 activity slots (Morning / Afternoon / Evening) — never more than 4
- Accounts for realistic travel time between locations; does not schedule back-to-back distant venues

All costs must be in {currency}.
Output ONLY valid JSON matching the schema. No markdown, no preamble."""

    prompt = f"""Build a {trip_days}-day itinerary for {travelers} traveller(s) visiting {destination} from {source}.

=== TRIP CONTEXT ===
Traveller Preferences: {preferences}
Trip Type: {trip_type}

=== SELECTED TRANSPORT ===
{route_summary}

=== SELECTED HOTEL ===
{hotel_summary}

=== RANKED ATTRACTIONS (use these — distribute across days 2 to {max(trip_days-1, 2)}) ===
{attractions_text}

=== BUDGET CONTEXT ===
Estimated daily food + incidentals budget per person: {daily_food_est:,.0f} {currency}

=== PLANNING RULES ===
1. Day 1 must start with arrival/transfer activity, followed by hotel check-in, then gentle orientation.
2. Day {trip_days} must end with hotel check-out and departure transfer; schedule only morning activities.
3. For a 1-day trip, combine a light arrival, 1-2 highlights, and departure.
4. Each day must have a distinct theme (e.g. "Art & History", "Nature Escape", "City Exploration").
5. Activity titles must be specific and vivid — not generic ("Visit Eiffel Tower" not "Sightseeing").
6. Include at least one Food & Dining activity per day.
7. The planner_notes field must explain exactly how you distributed attractions and handled pacing.

Return a PlannerAgentResponse JSON object now.
"""

    try:
        return groq_json(api_key,
            system=system_instruction,
            user=prompt,
            schema=PlannerAgentResponse,
        )

    except Exception as e:
        if _is_rate_limit_error(e):
            wait = _extract_retry_seconds(e)
            _show_rate_limit_warning(wait, "building your itinerary")
            return {"error": "rate_limit", "wait_seconds": wait}
        st.error(f"Planner Agent error: {e}")
        return {}


# ==========================================
# 5. UI LAYOUT & RENDERING
# ==========================================

# Main Header
st.title("✈️ AI Travel Planner")
st.markdown("Your intelligent companion for crafting perfect itineraries. Let's start by gathering your travel requirements!")

# Resolve API Key
api_key = os.environ.get("GROQ_API_KEY", "")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    user_key = st.text_input(
        "Groq API Key",
        value=api_key,
        type="password",
        placeholder="Enter your API key...",
        help="Paste your Groq API key. Get one free at console.groq.com"
    )
    
    if user_key:
        api_key = user_key

    # Store in session state so search_flights / search_hotels can access it
    st.session_state["api_key"] = api_key

    if not api_key:
        st.warning("⚠️ API Key is missing. Please add it to start.")
    else:
        st.success("✅ Groq API Key is ready!")

    # ── API status panel ──────────────────────────────────────────────────────
    with st.expander("🔑 API Status", expanded=False):
        st.markdown("✅ **Nominatim (OSM)** — geocoding *(free, no key)*")
        st.markdown("✅ **OSRM** — driving & rail routing *(free, no key)*")
        st.markdown("✅ **Overpass API (OSM)** — attractions *(free, no key)*")
        st.markdown("✅ **Groq** — flights, hotels, all agents *(uses your Groq key)*")
        
    st.markdown("---")
    
    # Render Live Summary Panel
    # -------------------------------------
    required_fields = ["source_city", "destination_city", "budget", "budget_currency_code", "trip_days", "travelers", "rooms"]
    labels = {
        "source_city": "🛫 Departure City",
        "destination_city": "🛬 Destination City",
        "budget": "💰 Total Budget",
        "budget_currency_code": "💱 Budget Currency",
        "trip_days": "📅 Trip Duration",
        "travelers": "👥 Total Travelers",
        "rooms": "🔑 Hotel Rooms Needed",
        "preferences": "⚙️ Special Preferences",
        "trip_type": "🌐 Trip Category"
    }
    
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown('<div class="summary-title">🗺️ Live Trip Summary</div>', unsafe_allow_html=True)
    
    for field in required_fields:
        val = st.session_state.travel_state.get(field)
        
        # Formatting values
        if val is None:
            val_str = "Missing"
            badge_html = '<span class="status-badge status-missing">?</span>'
        else:
            badge_html = '<span class="status-badge status-filled">✓</span>'
            if field == "budget":
                val_str = f"{val:,.0f}"
            elif field == "trip_days":
                val_str = f"{val} Days"
            elif field == "travelers":
                val_str = f"{val} {'People' if val > 1 else 'Person'}"
            elif field == "rooms":
                val_str = f"{val} {'Rooms' if val > 1 else 'Room'}"
            else:
                val_str = str(val)
                
        st.markdown(f"""
        <div class="summary-item">
            <span class="summary-label">{labels[field]}</span>
            <div>
                <span class="summary-value" style="margin-right: 8px;">{val_str}</span>
                {badge_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Optional fields
    pref_val = st.session_state.travel_state.get("preferences")
    if pref_val:
        st.markdown(f"""
        <div class="summary-item" style="border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 8px; margin-top: 8px;">
            <span class="summary-label">{labels['preferences']}</span>
            <span class="summary-value" style="font-size: 0.8rem; text-align: right; max-width: 60%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{pref_val}">{pref_val}</span>
        </div>
        """, unsafe_allow_html=True)
        
    type_val = st.session_state.travel_state.get("trip_type")
    if type_val:
        st.markdown(f"""
        <div class="summary-item">
            <span class="summary-label">{labels['trip_type']}</span>
            <span class="summary-value" style="text-transform: capitalize; color: #818cf8;">{type_val}</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset Button
    st.button("🔄 Reset Travel Plan", on_click=reset_trip, use_container_width=True)

# Main Content Area
# ==========================================

# Check if all required fields are filled
is_complete = all(st.session_state.travel_state.get(f) is not None for f in required_fields)

# If complete, render the Structured Travel Summary Card
if is_complete:
    budget_val = f"{st.session_state.travel_state.get('budget'):,.0f}" if st.session_state.travel_state.get('budget') is not None else "N/A"

    # Always show the budget in the currency the USER stated, not the working currency
    user_curr_code = st.session_state.travel_state.get('budget_currency_code') or st.session_state.budget_currency[0]
    user_home_curr = next(
        (info for keywords, info in _CURRENCY_MAP if info[0] == user_curr_code),
        (user_curr_code, user_curr_code, user_curr_code)
    )
    display_symbol = user_home_curr[1]
    display_code   = user_home_curr[0]

    curr_code, curr_symbol, curr_name = st.session_state.budget_currency
    trip_type = st.session_state.travel_state.get('trip_type', 'N/A')
    pref = st.session_state.travel_state.get('preferences', 'None provided')
    
    st.markdown(f"""
    <div class="completion-card">
        <div class="completion-header">🎉 Travel Requirements Gathered!</div>
        <p style="color: #cbd5e1; margin-bottom: 20px; font-size: 0.95rem;">
            Excellent! We have successfully gathered all the required travel details. Your structured travel state is complete.
        </p>
        <div class="grid-container">
            <div class="grid-card">
                <div class="grid-card-label">Departure</div>
                <div class="grid-card-value">🛫 {st.session_state.travel_state.get('source_city')}</div>
            </div>
            <div class="grid-card">
                <div class="grid-card-label">Destination</div>
                <div class="grid-card-value">🛬 {st.session_state.travel_state.get('destination_city')}</div>
            </div>
            <div class="grid-card">
                <div class="grid-card-label">Total Budget</div>
                <div class="grid-card-value">💰 {display_symbol}{budget_val} <span style="font-size:0.75rem; color:#94a3b8;">({display_code})</span></div>
            </div>
            <div class="grid-card">
                <div class="grid-card-label">Duration</div>
                <div class="grid-card-value">📅 {st.session_state.travel_state.get('trip_days')} Days</div>
            </div>
            <div class="grid-card">
                <div class="grid-card-label">Travelers</div>
                <div class="grid-card-value">👥 {st.session_state.travel_state.get('travelers')}</div>
            </div>
            <div class="grid-card">
                <div class="grid-card-label">Hotel Rooms</div>
                <div class="grid-card-value">🔑 {st.session_state.travel_state.get('rooms')}</div>
            </div>
        </div>
        <div style="margin-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 15px;">
            <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 5px;"><strong>Trip Category:</strong> <span style="text-transform: capitalize; color: #818cf8;">{trip_type}</span></p>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 5px;"><strong>Special Preferences:</strong> {pref}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Currency banner ───────────────────────────────────────────────────────
    trip_type      = st.session_state.travel_state.get('trip_type', 'domestic')
    user_curr_code = st.session_state.travel_state.get('budget_currency_code')
    raw_budget     = st.session_state.travel_state.get("budget", 0) or 0

    # Resolve home currency from user-stated code or source city
    if user_curr_code:
        home_curr = next(
            (info for keywords, info in _CURRENCY_MAP if info[0] == user_curr_code),
            (user_curr_code, user_curr_code, user_curr_code)
        )
    else:
        home_curr = resolve_currency(st.session_state.travel_state.get('source_city', ''))

    eff_budget = st.session_state.get("effective_budget", raw_budget) or 0
    raw_budget = raw_budget or 0

    if trip_type == "international" and home_curr[0] != "USD":
        eff_budget_usd = eff_budget if eff_budget > 0 else convert_to_usd(raw_budget, home_curr[0])
        st.markdown(f"""
        <div style="background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.35);
                    border-radius: 10px; padding: 12px 20px; margin-bottom: 20px;">
            <span style="color:#fbbf24; font-weight:600; font-size:0.95rem;">🌍 International Trip — Currency Conversion Applied</span><br>
            <span style="color:#94a3b8; font-size:0.85rem;">
                Your budget of <strong style="color:#f1f5f9;">{home_curr[1]}{raw_budget:,.0f} {home_curr[0]}</strong>
                has been converted to <strong style="color:#818cf8;">${eff_budget_usd:,.0f} USD</strong>
                for international cost comparisons (approx. rate: 1 {home_curr[0]} = ${_TO_USD_RATES.get(home_curr[0], 1):.4f} USD).
                All prices on this page are in <strong style="color:#818cf8;">USD</strong>.
                Verify exchange rates before booking.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.3);
                    border-radius: 10px; padding: 12px 20px; margin-bottom: 20px;">
            <span style="color:#818cf8; font-weight:600; font-size:0.95rem;">
                {curr_symbol} All costs calculated in {curr_name} ({curr_code})
            </span>
            <span style="color:#64748b; font-size:0.82rem; margin-left:8px;">
                — {f"as stated by you" if user_curr_code else f"detected from your departure city: {st.session_state.travel_state.get('source_city')}"}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------
    # ROUTE & PLANNING RECOMMENDATIONS
    # ---------------------------------------------
    st.markdown("## 🗺️ Your Customized Travel Plan")

    # Determine which stages are already complete
    routes_done      = st.session_state.route_options is not None
    hotels_done      = st.session_state.hotel_options is not None
    attractions_done = st.session_state.attractions   is not None
    all_done         = routes_done and hotels_done and attractions_done

    if not all_done:
        # Build a status summary so the user knows what's already saved
        stage_status = []
        stage_status.append(f"{'✅' if routes_done      else '⏳'} Routes")
        stage_status.append(f"{'✅' if hotels_done      else '⏳'} Hotels")
        stage_status.append(f"{'✅' if attractions_done else '⏳'} Attractions")

        if not any([routes_done, hotels_done, attractions_done]):
            st.info("✅ Requirements gathered! Generating your travel plan now...")
            # Auto-trigger: resolve currency and kick off generation immediately
            src_city = st.session_state.travel_state.get("source_city", "")
            st.session_state.budget_currency = resolve_currency(src_city)
            do_generate = True
        else:
            # Some stages done — show progress and let user resume
            st.info(
                f"Progress saved — {' · '.join(stage_status)}\n\n"
                "Click below to continue from where we left off."
            )
            do_generate = False

        btn_label = "▶️ Continue Plan Generation"

        if do_generate or st.button(btn_label, use_container_width=True):

            # Resolve effective currency (USD for international, local for domestic)
            if not routes_done:
                eff = get_effective_currency(st.session_state.travel_state)
                st.session_state.budget_currency        = (eff[0], eff[1], eff[2])
                st.session_state.effective_budget       = eff[3]   # budget in working currency
                st.session_state.home_currency          = resolve_currency(
                    st.session_state.travel_state.get("source_city", "")
                )

            curr_code, curr_symbol, curr_name = st.session_state.budget_currency
            effective_budget = st.session_state.get("effective_budget",
                st.session_state.travel_state.get("budget", 0))
            if not routes_done:
                with st.spinner("🚗 Researching routes..."):
                    report = run_route_agent_fact_gathering(api_key, st.session_state.travel_state)

                if isinstance(report, str) and report.startswith("RATE_LIMIT:"):
                    wait = int(report.split(":")[1])
                    _show_rate_limit_warning(wait, "researching routes")
                    st.stop()
                elif "Error" in report:
                    st.error(report)
                    st.stop()
                else:
                    try:
                        structured_res = parse_route_report_to_structured(api_key, report)
                        if structured_res.get("error") == "rate_limit":
                            _show_rate_limit_warning(structured_res.get("wait_seconds", 60), "parsing routes")
                            st.stop()
                        # ✅ Save routes immediately
                        st.session_state.route_options     = structured_res.get("options", [])
                        st.session_state.route_explanation = structured_res.get("explanation", "")
                        st.session_state.selected_route_idx = 0
                        # Store flight currency note from raw route data
                        try:
                            flights_raw = json.loads(search_flights.__doc__ or "{}")
                        except Exception:
                            pass
                    except Exception as e:
                        if _is_rate_limit_error(e):
                            _show_rate_limit_warning(_extract_retry_seconds(e), "parsing routes")
                        else:
                            st.error(f"Error parsing routes: {e}")
                        st.stop()

            # ── STAGE 2: Hotels ───────────────────────────────────────────
            if not hotels_done:
                time.sleep(4)   # pace Groq calls
                with st.spinner("🏨 Searching hotels..."):
                    hotel_res = run_hotel_agent(api_key, st.session_state.travel_state)

                if hotel_res.get("error") == "rate_limit":
                    _show_rate_limit_warning(hotel_res.get("wait_seconds", 60), "searching hotels")
                    st.stop()
                # ✅ Save hotels immediately
                st.session_state.hotel_options  = hotel_res.get("hotels", [])
                st.session_state.hotel_summary  = hotel_res.get("summary", "")
                st.session_state.hotel_currency_note = hotel_res.get("currency_note", "")
                st.session_state.selected_hotel_idx = 0

            # ── STAGE 3: Attractions ──────────────────────────────────────
            if not attractions_done:
                time.sleep(4)   # pace Groq calls
                with st.spinner("🏛️ Finding attractions..."):
                    attractions_res = run_attractions_agent(api_key, st.session_state.travel_state)

                if attractions_res.get("error") == "rate_limit":
                    _show_rate_limit_warning(attractions_res.get("wait_seconds", 60), "finding attractions")
                    st.stop()
                # ✅ Save attractions immediately
                st.session_state.attractions = attractions_res.get("attractions", [])

            # ── All stages done — clean up and render ─────────────────────
            st.session_state.budget_analysis = None
            if "last_selected_inputs" in st.session_state:
                del st.session_state.last_selected_inputs
            st.rerun()

    else:
        # ── Selected plan banner ──────────────────────────────────────────────
        if st.session_state.get("selected_combo"):
            sc = st.session_state.selected_combo
            rank = st.session_state.get("selected_plan_rank", "")
            unit = st.session_state.budget_currency[0]
            sym  = st.session_state.budget_currency[1]
            st.success(
                f"✅ **Plan #{rank} selected** — "
                f"{sc['route'].get('mode','').capitalize()} · "
                f"{sc['hotel'].get('name','')} · "
                f"Total: {sym}{sc['total_cost']:,.0f} {unit} — "
                f"**Click the 🗓️ Itinerary tab above to generate your day-by-day plan**"
            )

        # Create Tabs for visual layout
        tab_transport, tab_hotels, tab_attractions, tab_budget, tab_itinerary = st.tabs([
            "🚗 Transportation",
            "🏨 Lodging Options",
            "🏛️ Local Attractions",
            "💰 Budget Planner",
            "🗓️ Itinerary",
        ])
        
        with tab_transport:
            st.markdown("### 🚗 Available Route Options")

            # Currency note for flights
            flight_currency_note = st.session_state.get("flight_currency_note", "")
            if flight_currency_note:
                st.info(f"💱 {flight_currency_note}")
            trip_type = st.session_state.travel_state.get("trip_type", "domestic")
            if trip_type == "international":
                st.warning("🌍 International trip detected — all flight prices are shown in **USD** for easy comparison. Convert to your local currency before booking.")
            mode_icons = {
                "drive": "🚗 Driving Option",
                "train": "🚆 Rail Travel Option",
                "flight": "✈️ Flight Option",
                "mixed": "🔀 Mixed Mode Option"
            }
            
            for idx, opt in enumerate(st.session_state.route_options):
                mode      = opt.get("mode", "").lower()
                mode_lbl  = mode_icons.get(mode, f"Option: {mode}")
                cost      = float(opt.get("cost") or 0)
                cost_str  = f"{cost:,.0f} {st.session_state.budget_currency[0]}"
                desc      = opt.get("description", "")
                operator  = opt.get("operator", "")
                confidence = opt.get("price_confidence", "medium")
                conf_color = {"high": "#10b981", "medium": "#fbbf24", "low": "#ef4444"}.get(confidence, "#fbbf24")
                conf_label = {"high": "✅ Price verified", "medium": "⚠️ Estimated price", "low": "❓ Uncertain price"}.get(confidence, "⚠️ Estimated price")

                operator_badge = f"""<span style="background:rgba(251,191,36,0.12); color:#fbbf24;
                    border:1px solid rgba(251,191,36,0.3); padding:3px 10px; border-radius:6px;
                    font-size:0.82rem; font-weight:600; margin-right:6px;">🏷️ {operator}</span>""" if operator else ""

                confidence_badge = f"""<span style="background:rgba(255,255,255,0.05); color:{conf_color};
                    border:1px solid {conf_color}44; padding:3px 10px; border-radius:6px;
                    font-size:0.78rem; margin-right:6px;">{conf_label}</span>"""

                src_note = opt.get("source_note", "Verify fare on Google Flights or MakeMyTrip before booking.")

                st.markdown(f"""
                <div class="completion-card" style="border: 1px solid rgba(56, 189, 248, 0.3); margin-bottom: 15px; padding: 20px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:10px;
                                margin-bottom:15px; flex-wrap:wrap; gap:8px;">
                        <span style="font-size:1.25rem; font-weight:700; color:#f8fafc;">#{idx+1}. {mode_lbl}</span>
                        <div style="display:flex; gap:6px; flex-wrap:wrap; align-items:center;">
                            {operator_badge}
                            {confidence_badge}
                            <span style="background:rgba(56,189,248,0.1); color:#38bdf8;
                                         border:1px solid rgba(56,189,248,0.2); padding:4px 10px;
                                         border-radius:6px; font-weight:600; font-size:0.85rem;">
                                ⏱️ {opt.get('duration')}
                            </span>
                            <span style="background:rgba(16,185,129,0.1); color:#10b981;
                                         border:1px solid rgba(16,185,129,0.2); padding:4px 10px;
                                         border-radius:6px; font-weight:600; font-size:0.85rem;">
                                💵 {cost_str}
                            </span>
                        </div>
                    </div>
                    <div style="font-size:0.9rem; color:#cbd5e1; margin-bottom:8px;">
                        <strong>Segment:</strong> {opt.get('source')} → {opt.get('destination')}
                    </div>
                    <div style="font-size:0.9rem; color:#cbd5e1; margin-bottom:8px;">
                        <strong>Operator & Details:</strong> {desc}
                    </div>
                    <div style="font-size:0.85rem; color:#94a3b8; background:rgba(255,255,255,0.02);
                                padding:10px; border-radius:6px; border-left:3px solid #6366f1; margin-bottom:8px;">
                        <strong>Notes:</strong> {opt.get('feasibility_notes')}
                    </div>
                    <div style="font-size:0.75rem; color:#475569;">
                        🔗 {src_note}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### 🧠 Route Trade-offs Analysis")
            st.info(st.session_state.route_explanation)
            
        with tab_hotels:
            st.markdown("### 🏨 Lodging Options (Ranked & Filtered)")

            # Currency note banner
            hotel_currency_note = st.session_state.get("hotel_currency_note", "")
            if hotel_currency_note:
                st.info(f"💱 {hotel_currency_note}")

            if not st.session_state.hotel_options:
                st.warning("No hotels matched your budget criteria.")
            else:
                for idx, opt in enumerate(st.session_state.hotel_options):
                    night_cost = float(opt.get('price_per_night') or 0)
                    tot_cost = float(opt.get('total_cost') or 0)
                    unit = st.session_state.budget_currency[0]
                    
                    st.markdown(f"""
                    <div class="completion-card" style="border: 1px solid rgba(168, 85, 247, 0.3); margin-bottom: 15px; padding: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 10px; margin-bottom: 15px;">
                            <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">#{idx+1}. {opt.get('name')}</span>
                            <div>
                                <span style="background: rgba(168, 85, 247, 0.1); color: #a855f7; border: 1px solid rgba(168, 85, 247, 0.2); padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; margin-right: 8px;">
                                    ★ {opt.get('rating')} / 5.0
                                </span>
                                <span style="background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">
                                    💵 {tot_cost:,.0f} {unit} total
                                </span>
                            </div>
                        </div>
                        <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 8px;">
                            <strong>Rate:</strong> {night_cost:,.0f} {unit} per night
                        </div>
                        <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 8px;">
                            <strong>Location:</strong> {opt.get('location')}
                        </div>
                        <div style="font-size: 0.85rem; color: #94a3b8; background: rgba(255, 255, 255, 0.02); padding: 10px; border-radius: 6px; border-left: 3px solid #a855f7;">
                            <strong>Trade-offs & Features:</strong> {opt.get('tradeoffs_explanation')}
                        </div>
                        <div style="font-size: 0.75rem; color: #475569; margin-top: 8px;">
                            🔗 {opt.get('source', 'Verify prices on Booking.com or MakeMyTrip before booking.')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 🧠 Lodging Landscape Analysis")
                st.info(st.session_state.hotel_summary)
                
        with tab_attractions:
            st.markdown("### 🏛️ Recommended Sights (Ranked by Relevance)")
            
            if not st.session_state.attractions:
                st.info("No sights found for this city.")
            else:
                for idx, opt in enumerate(st.session_state.attractions):
                    fee = opt.get('estimated_entry_fee')
                    unit = st.session_state.budget_currency[0]
                    fee_str = "Free" if fee == 0 else f"{fee:,.0f} {unit}"
                    
                    st.markdown(f"""
                    <div class="completion-card" style="border: 1px solid rgba(236, 72, 153, 0.3); margin-bottom: 15px; padding: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 10px; margin-bottom: 15px;">
                            <span style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">#{idx+1}. {opt.get('name')}</span>
                            <div>
                                <span style="background: rgba(236, 72, 153, 0.1); color: #ec4899; border: 1px solid rgba(236, 72, 153, 0.2); padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; margin-right: 8px;">
                                    ★ {opt.get('rating')}
                                </span>
                                <span style="background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2); padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;">
                                    🎟️ {fee_str}
                                </span>
                            </div>
                        </div>
                        <div style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 8px;">
                            <strong>Description:</strong> {opt.get('description')}
                        </div>
                        <div style="font-size: 0.85rem; color: #94a3b8; background: rgba(255, 255, 255, 0.02); padding: 10px; border-radius: 6px; border-left: 3px solid #ec4899;">
                            <strong>Relevance Reason:</strong> {opt.get('relevance_reason')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        with tab_budget:
            st.markdown("### 💰 All Plans Fitting Your Budget")

            unit         = st.session_state.budget_currency[0]
            curr_sym     = st.session_state.budget_currency[1]
            eff_budget   = st.session_state.get("effective_budget",
                              st.session_state.travel_state.get("budget", 0)) or 0
            trip_days    = st.session_state.travel_state.get("trip_days", 1)
            attractions  = st.session_state.attractions or []

            route_options_list = st.session_state.route_options or []
            hotel_options_list = st.session_state.hotel_options or []

            # ── Attraction tier dropdown ──────────────────────────────────
            tiers_available = list(compute_attraction_tiers(attractions, trip_days).keys())
            selected_tier = st.selectbox(
                "🏛️ Attraction Package",
                options=tiers_available,
                index=2,  # Default: Standard (Top 3-4)
                help="Choose how many attractions to include. Combinations update instantly."
            )

            # ── Run combinatorial engine ──────────────────────────────────
            tiers      = compute_attraction_tiers(attractions, trip_days)
            tier_list  = tiers[selected_tier]

            # Build filtered route/hotel options based on selected tier only
            fitting, over_budget = compute_combinations(
                route_options=route_options_list,
                hotel_options=hotel_options_list,
                attractions=tier_list,
                trip_days=trip_days,
                effective_budget=eff_budget,
                curr_code=unit,
            )

            economy, balanced, premium = derive_econ_balanced_premium(fitting, over_budget)

            # ── Summary banner ────────────────────────────────────────────
            total_combos = len(fitting) + len(over_budget)
            if fitting:
                st.success(
                    f"✅ **{len(fitting)} of {total_combos} combinations fit your budget** of "
                    f"{curr_sym}{eff_budget:,.0f} {unit} — ranked cheapest to most expensive below."
                )
            else:
                st.warning(
                    f"⚠️ No combinations fit your budget of {curr_sym}{eff_budget:,.0f} {unit} "
                    f"with the current attraction tier. Try selecting **'No Attractions'** or **'Free Only'**."
                )

            # ── Economy / Balanced / Premium summary table ────────────────
            if economy:
                st.markdown("#### 📊 Package Summary")
                pkg_rows = []
                for label, combo in [("🟢 Economy", economy), ("🔵 Balanced", balanced), ("🟣 Premium", premium)]:
                    if combo:
                        pkg_rows.append({
                            "Package":                label,
                            "Transport":              f"{combo['route'].get('mode','').capitalize()} — {curr_sym}{combo['transport_cost']:,.0f}",
                            "Hotel":                  f"{combo['hotel'].get('name','')} — {curr_sym}{combo['hotel_cost']:,.0f}",
                            "Attractions":            combo["attraction_tier_name"],
                            f"Total ({unit})":        f"{curr_sym}{combo['total_cost']:,.0f}",
                            "Fits Budget?":           "✅ Yes" if combo["fits_budget"] else "❌ Over",
                        })
                st.table(pkg_rows)

            st.markdown("---")

            # ── Full ranked combinations list ─────────────────────────────
            display_list = fitting if fitting else over_budget[:10]
            section_label = (
                f"#### ✅ All {len(fitting)} Plans Within Budget — Ranked Cheapest to Most Expensive"
                if fitting else
                f"#### ❌ No plans fit budget — showing {len(display_list)} closest over-budget options"
            )
            st.markdown(section_label)

            TRANSPORT_ICONS = {"flight": "✈️", "drive": "🚗", "train": "🚆", "mixed": "🔀"}
            TIER_ICONS = {
                "No Attractions": "🚫",
                "Free Only": "🆓",
                "Standard (Top 3-4)": "🏛️",
                "Full Experience": "🌟",
            }

            for rank, combo in enumerate(display_list, 1):
                route    = combo["route"]
                hotel    = combo["hotel"]
                t_icon   = TRANSPORT_ICONS.get(route.get("mode", ""), "🚌")
                tier_ico = TIER_ICONS.get(combo["attraction_tier_name"], "🏛️")
                fits     = combo["fits_budget"]
                border   = "rgba(16,185,129,0.4)" if fits else "rgba(239,68,68,0.3)"
                badge_bg = "rgba(16,185,129,0.1)" if fits else "rgba(239,68,68,0.1)"
                badge_c  = "#10b981" if fits else "#ef4444"
                badge_t  = f"✅ {curr_sym}{combo['total_cost']:,.0f}" if fits else f"❌ {curr_sym}{combo['total_cost']:,.0f} (over by {curr_sym}{combo['total_cost']-eff_budget:,.0f})"

                # Build attraction day breakdown
                attr_names = " + ".join(a["name"] for a in combo["attraction_list"]) if combo["attraction_list"] else "No sightseeing"

                st.markdown(f"""
                <div style="border:1px solid {border}; border-radius:12px; padding:16px 20px;
                            margin-bottom:12px; background:rgba(255,255,255,0.02);">
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                flex-wrap:wrap; gap:8px; margin-bottom:12px;">
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span style="background:rgba(99,102,241,0.15); color:#818cf8; font-weight:700;
                                         padding:3px 10px; border-radius:20px; font-size:0.8rem;">#{rank}</span>
                            <span style="color:#f1f5f9; font-weight:600; font-size:1rem;">
                                {t_icon} {route.get('mode','').capitalize()} → 🏨 {hotel.get('name','')} → {tier_ico} {combo['attraction_tier_name']}
                            </span>
                        </div>
                        <span style="background:{badge_bg}; color:{badge_c}; font-weight:700;
                                     padding:4px 14px; border-radius:20px; font-size:0.9rem;">
                            {badge_t}
                        </span>
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:10px;">
                        <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:8px 12px;">
                            <div style="color:#64748b; font-size:0.72rem; margin-bottom:2px;">Transport</div>
                            <div style="color:#cbd5e1; font-size:0.88rem; font-weight:600;">{curr_sym}{combo['transport_cost']:,.0f}</div>
                            <div style="color:#475569; font-size:0.72rem;">{route.get('duration','')}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:8px 12px;">
                            <div style="color:#64748b; font-size:0.72rem; margin-bottom:2px;">Hotel ({trip_days}n)</div>
                            <div style="color:#cbd5e1; font-size:0.88rem; font-weight:600;">{curr_sym}{combo['hotel_cost']:,.0f}</div>
                            <div style="color:#475569; font-size:0.72rem;">{curr_sym}{hotel.get('price_per_night',0):,.0f}/night</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:8px 12px;">
                            <div style="color:#64748b; font-size:0.72rem; margin-bottom:2px;">Attractions</div>
                            <div style="color:#cbd5e1; font-size:0.88rem; font-weight:600;">{curr_sym}{combo['attraction_cost']:,.0f}</div>
                            <div style="color:#475569; font-size:0.72rem;">{len(combo['attraction_list'])} sights</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:8px 12px;">
                            <div style="color:#64748b; font-size:0.72rem; margin-bottom:2px;">Food & misc</div>
                            <div style="color:#cbd5e1; font-size:0.88rem; font-weight:600;">{curr_sym}{combo['food_cost']:,.0f}</div>
                            <div style="color:#475569; font-size:0.72rem;">{trip_days} days</div>
                        </div>
                    </div>
                    <div style="color:#64748b; font-size:0.8rem; border-top:1px solid rgba(255,255,255,0.05);
                                padding-top:8px;">
                        🏛️ {attr_names}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # "Use this plan" button — sets route + hotel selections
                if fits:
                    r_idx = next((i for i,r in enumerate(route_options_list) if r.get("mode")==route.get("mode") and r.get("cost")==route.get("cost")), 0)
                    h_idx = next((i for i,h in enumerate(hotel_options_list) if h.get("name")==hotel.get("name")), 0)
                    if st.button(f"✅ Use Plan #{rank}", key=f"use_plan_{rank}"):
                        st.session_state.selected_route_idx   = r_idx
                        st.session_state.selected_hotel_idx   = h_idx
                        st.session_state.selected_combo       = combo   # full combo dict
                        st.session_state.selected_plan_rank   = rank
                        st.session_state.budget_analysis      = {"is_feasible": True, "breakdowns": [], "comparison_summary": "", "budget_warning": None}
                        st.session_state.itinerary            = None
                        st.session_state.itinerary_meta       = None
                        st.session_state.active_tab           = "itinerary"
                        st.rerun()

            st.markdown("---")
            if st.button("🔄 Reset & Recalculate Everything", use_container_width=True):
                st.session_state.route_options = None
                st.session_state.route_explanation = None
                st.session_state.hotel_options = None
                st.session_state.hotel_summary = None
                st.session_state.attractions = None
                st.session_state.budget_analysis = None
                st.session_state.itinerary = None
                st.session_state.itinerary_meta = None
                st.rerun()

        # ──────────────────────────────────────────────────────────────────
        # TAB 5 — ITINERARY
        # ──────────────────────────────────────────────────────────────────
        with tab_itinerary:
            st.markdown("### 🗓️ Your Day-by-Day Itinerary")

            # ── Determine the "signature" of the current plan ─────────────
            current_itinerary_meta = (
                st.session_state.get("selected_route_idx", 0),
                st.session_state.get("selected_hotel_idx", 0),
                st.session_state.travel_state.get("trip_days"),
                st.session_state.travel_state.get("destination_city"),
            )

            # ── Gate: need route + hotel selected ────────────────────────
            if not st.session_state.route_options or not st.session_state.hotel_options:
                st.info(
                    "💡 Visit the **💰 Budget Planner** tab first, pick a plan with "
                    "'✅ Use Plan #N', then come back here to generate your itinerary."
                )
            else:
                # ── Selected Plan Cost Summary Card ───────────────────────
                combo = st.session_state.get("selected_combo")
                if combo:
                    unit = st.session_state.budget_currency[0]
                    sym  = st.session_state.budget_currency[1]
                    eff_bud = st.session_state.get("effective_budget",
                        st.session_state.travel_state.get("budget", 0)) or 0
                    remaining = eff_bud - combo["total_cost"]
                    remaining_color = "#10b981" if remaining >= 0 else "#ef4444"
                    remaining_label = f"Budget remaining: {sym}{abs(remaining):,.0f}" if remaining >= 0 else f"Over budget by: {sym}{abs(remaining):,.0f}"

                    st.markdown(f"""
                    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.3);
                                border-radius:12px; padding:18px 22px; margin-bottom:20px;">
                        <div style="font-size:1rem; font-weight:700; color:#818cf8; margin-bottom:14px;">
                            📋 Selected Plan #{st.session_state.get('selected_plan_rank','')} — Cost Breakdown
                        </div>
                        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:14px;">
                            <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:10px 14px;">
                                <div style="color:#64748b; font-size:0.72rem; margin-bottom:4px;">
                                    {combo['route'].get('mode','').upper()}
                                </div>
                                <div style="color:#f1f5f9; font-weight:700; font-size:1rem;">
                                    {sym}{combo['transport_cost']:,.0f}
                                </div>
                                <div style="color:#475569; font-size:0.72rem;">
                                    {combo['route'].get('operator', combo['route'].get('mode',''))}
                                </div>
                            </div>
                            <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:10px 14px;">
                                <div style="color:#64748b; font-size:0.72rem; margin-bottom:4px;">HOTEL</div>
                                <div style="color:#f1f5f9; font-weight:700; font-size:1rem;">
                                    {sym}{combo['hotel_cost']:,.0f}
                                </div>
                                <div style="color:#475569; font-size:0.72rem;">
                                    {combo['hotel'].get('name','')}
                                </div>
                            </div>
                            <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:10px 14px;">
                                <div style="color:#64748b; font-size:0.72rem; margin-bottom:4px;">ATTRACTIONS</div>
                                <div style="color:#f1f5f9; font-weight:700; font-size:1rem;">
                                    {sym}{combo['attraction_cost']:,.0f}
                                </div>
                                <div style="color:#475569; font-size:0.72rem;">
                                    {combo['attraction_tier_name']}
                                </div>
                            </div>
                            <div style="background:rgba(255,255,255,0.04); border-radius:8px; padding:10px 14px;">
                                <div style="color:#64748b; font-size:0.72rem; margin-bottom:4px;">FOOD & MISC</div>
                                <div style="color:#f1f5f9; font-weight:700; font-size:1rem;">
                                    {sym}{combo['food_cost']:,.0f}
                                </div>
                                <div style="color:#475569; font-size:0.72rem;">
                                    {st.session_state.travel_state.get('trip_days',1)} days est.
                                </div>
                            </div>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;
                                    border-top:1px solid rgba(255,255,255,0.08); padding-top:12px;">
                            <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9;">
                                💰 Total: {sym}{combo['total_cost']:,.0f} {unit}
                            </div>
                            <div style="color:{remaining_color}; font-weight:600; font-size:0.9rem;">
                                {remaining_label}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Generate button / cached result ───────────────────────
                needs_generation = (
                    st.session_state.itinerary is None
                    or st.session_state.itinerary_meta != current_itinerary_meta
                )

                if needs_generation:
                    st.info(
                        "Your transport and lodging are locked in. Click below to let the "
                        "Planner Agent craft a personalised day-by-day itinerary."
                    )
                    if st.button("✨ Generate My Itinerary", use_container_width=True, type="primary"):
                        # Resolve the selected route & hotel from session state
                        r_idx = st.session_state.get("selected_route_idx", 0)
                        h_idx = st.session_state.get("selected_hotel_idx", 0)
                        sel_route  = (st.session_state.route_options or [{}])[r_idx] if st.session_state.route_options else {}
                        sel_hotel  = (st.session_state.hotel_options or [{}])[h_idx] if st.session_state.hotel_options else {}

                        with st.spinner("✈️ Planning your perfect trip — this takes about 15 seconds…"):
                            result = run_planner_agent(
                                api_key=api_key,
                                travel_state=st.session_state.travel_state,
                                selected_route=sel_route,
                                selected_hotel=sel_hotel,
                                attractions=st.session_state.attractions or [],
                                budget_analysis=st.session_state.budget_analysis or {},
                            )

                        if result and result.get("days"):
                            st.session_state.itinerary = result
                            st.session_state.itinerary_meta = current_itinerary_meta
                            st.rerun()
                        elif result.get("error") == "rate_limit":
                            _show_rate_limit_warning(result.get("wait_seconds", 60), "building your itinerary")
                        else:
                            st.error(
                                "The Planner Agent could not generate an itinerary. "
                                "Please check your Groq API key and try again."
                            )

                # ── Render cached itinerary ───────────────────────────────
                if st.session_state.itinerary and not needs_generation:
                    plan = st.session_state.itinerary
                    unit = st.session_state.budget_currency[0]

                    # ── Trip title banner ─────────────────────────────────
                    st.markdown(f"""
                    <div class="completion-card" style="border: 1px solid rgba(99,102,241,0.5); padding: 24px; margin-bottom: 24px;">
                        <div class="completion-header" style="font-size:1.4rem; margin-bottom: 10px;">
                            📋 {plan.get('itinerary_title', 'Your Custom Travel Itinerary')}
                        </div>
                        <p style="color:#94a3b8; font-size:0.9rem; line-height:1.6; margin:0;">
                            {plan.get('planner_notes', '')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Activity type → colour accent mapping ─────────────
                    SLOT_COLOURS = {
                        "Morning":   {"bg": "rgba(251,191,36,0.08)",  "border": "#fbbf24", "badge": "#78350f"},
                        "Afternoon": {"bg": "rgba(56,189,248,0.08)",  "border": "#38bdf8", "badge": "#0c4a6e"},
                        "Evening":   {"bg": "rgba(168,85,247,0.08)", "border": "#a855f7", "badge": "#4a044e"},
                        "Travel Day":{"bg": "rgba(100,116,139,0.08)","border": "#64748b", "badge": "#1e293b"},
                    }
                    ACTIVITY_ICONS = {
                        "Sightseeing":   "🏛️",
                        "Culture":       "🎭",
                        "Nature":        "🌿",
                        "Food & Dining": "🍽️",
                        "Shopping":      "🛍️",
                        "Leisure":       "☕",
                        "Travel":        "🚌",
                        "Accommodation": "🏨",
                    }
                    SLOT_ICONS = {
                        "Morning":    "🌅",
                        "Afternoon":  "☀️",
                        "Evening":    "🌙",
                        "Travel Day": "🚀",
                    }

                    # ── Day cards ─────────────────────────────────────────
                    days = plan.get("days", [])
                    for day in days:
                        day_num   = day.get("day_number", "?")
                        date_lbl  = day.get("date_label", f"Day {day_num}")
                        theme     = day.get("theme", "")
                        day_cost  = day.get("estimated_day_cost", 0)
                        summary   = day.get("day_summary", "")
                        activities = day.get("activities", [])

                        # Default expanded for Day 1, collapsed for rest
                        with st.expander(
                            f"📅  {date_lbl}  ·  {theme}  ·  Est. {day_cost:,.0f} {unit}",
                            expanded=(day_num == 1),
                        ):
                            # Day summary strip
                            st.markdown(f"""
                            <div style="background:rgba(255,255,255,0.02); border-radius:8px;
                                        padding:12px 16px; margin-bottom:16px;
                                        border-left: 3px solid #6366f1;">
                                <span style="color:#94a3b8; font-size:0.88rem; line-height:1.6;">{summary}</span>
                            </div>
                            """, unsafe_allow_html=True)

                            # Activity cards inside the day
                            for act in activities:
                                slot        = act.get("time_slot", "Morning")
                                act_type    = act.get("activity_type", "Sightseeing")
                                title       = act.get("title", "Activity")
                                location    = act.get("location", "")
                                duration_h  = act.get("duration_hours", 1.0)
                                description = act.get("description", "")
                                tips        = act.get("tips", "")
                                cost        = act.get("estimated_cost", 0.0)
                                source      = act.get("source", "")

                                colours     = SLOT_COLOURS.get(slot, SLOT_COLOURS["Afternoon"])
                                act_icon    = ACTIVITY_ICONS.get(act_type, "📍")
                                slot_icon   = SLOT_ICONS.get(slot, "⏰")
                                cost_str    = "Free" if cost == 0 else f"{cost:,.0f} {unit}"
                                dur_str     = f"{duration_h:.0f}h" if duration_h == int(duration_h) else f"{duration_h:.1f}h"
                                source_html = f'<div style="color:#475569; font-size:0.75rem; margin-top:6px;">🔗 Source: {source}</div>' if source else ""

                                card_bg     = colours['bg']
                                card_border = colours['border']
                                cost_bg     = 'rgba(16,185,129,0.12)' if cost == 0 else 'rgba(251,191,36,0.1)'
                                cost_color  = '#10b981' if cost == 0 else '#fbbf24'

                                # Use st.container + st.markdown with unsafe_allow_html
                                with st.container():
                                    st.markdown(
                                        f"""<div style="background:{card_bg}; border:1px solid {card_border}33;
                                            border-left:3px solid {card_border}; border-radius:10px;
                                            padding:14px 18px; margin-bottom:10px;">
                                            <div style="display:flex; justify-content:space-between;
                                                        align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
                                                <div style="display:flex; align-items:center; gap:8px;">
                                                    <span style="font-size:1.05rem;">{slot_icon} {act_icon}</span>
                                                    <span style="font-weight:600; color:#f1f5f9; font-size:0.98rem;">{title}</span>
                                                </div>
                                                <div style="display:flex; gap:6px; flex-wrap:wrap;">
                                                    <span style="background:rgba(255,255,255,0.06); color:#cbd5e1;
                                                                 padding:2px 8px; border-radius:12px; font-size:0.75rem;">{slot}</span>
                                                    <span style="background:rgba(255,255,255,0.06); color:#94a3b8;
                                                                 padding:2px 8px; border-radius:12px; font-size:0.75rem;">⏱ {dur_str}</span>
                                                    <span style="background:{cost_bg}; color:{cost_color};
                                                                 padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;">🎟 {cost_str}</span>
                                                </div>
                                            </div>
                                            <div style="color:#64748b; font-size:0.8rem; margin-bottom:8px;">📍 {location}</div>
                                            <div style="color:#cbd5e1; font-size:0.87rem; line-height:1.55; margin-bottom:8px;">{description}</div>
                                            <div style="background:rgba(255,255,255,0.03); border-radius:6px;
                                                        padding:8px 12px; font-size:0.82rem; color:#94a3b8;">
                                                💡 <em>{tips}</em>
                                            </div>
                                            {source_html}
                                        </div>""",
                                        unsafe_allow_html=True
                                    )

                    # ── Trip-total cost footer ────────────────────────────
                    total_act_cost = plan.get("total_activities_cost", 0)
                    st.markdown(f"""
                    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25);
                                border-radius:10px; padding:14px 20px; margin-top:20px;
                                display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#94a3b8; font-size:0.9rem;">
                            Total estimated sightseeing &amp; activity costs
                        </span>
                        <span style="color:#818cf8; font-weight:700; font-size:1.05rem;">
                            {total_act_cost:,.0f} {unit}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Regenerate button ─────────────────────────────────
                    st.markdown("")
                    if st.button("🔄 Regenerate Itinerary", use_container_width=False):
                        st.session_state.itinerary = None
                        st.session_state.itinerary_meta = None
                        st.rerun()
                
        st.markdown("---")

# Display Chat History
st.markdown("### Chat with your Travel Agent")
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input & Processing
if not api_key:
    st.info("💡 Please enter your Groq API key in the sidebar to begin chatting.")
else:
    if prompt := st.chat_input("I want to go to Paris for 5 days..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Rerun to show user message immediately, then generate response
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    extracted, reply = run_requirements_agent(
                        api_key=api_key,
                        current_state=st.session_state.travel_state,
                        messages=st.session_state.messages,
                        new_input=prompt
                    )
                    
                    # Merge extracted fields into session travel state
                    if extracted:
                        for field in ["source_city", "destination_city", "budget", "budget_currency_code", "trip_days", "travelers", "rooms", "preferences", "trip_type"]:
                            val = extracted.get(field) if isinstance(extracted, dict) else getattr(extracted, field, None)
                            if val is not None:
                                st.session_state.travel_state[field] = val
                    
                    # Add reply to messages
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.markdown(reply)
                    
        # Force rerun to update live summary cards and layout changes
        st.rerun()
