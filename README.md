# ✈️ AI Travel Planner

A multi-agent AI travel planning app that turns a simple chat conversation into a complete, budget-aware trip plan — transport comparisons, hotel options, local attractions, a combinatorial budget optimizer, and a generated day-by-day itinerary.

**Only one API key required. Completely free. No credit card needed.**

---

## What it does

1. **Chat** with an AI agent to describe your trip — it asks for departure city, destination, budget (and currency), dates, and travelers, one question at a time
2. **Compare routes** — driving, rail, and flights, with realistic prices grounded against a static reference table of 500+ city pairs
3. **Browse hotels** — 10 real options across budget, mid-range, and luxury tiers
4. **Discover attractions** — top sights near your destination, ranked by your stated preferences
5. **See every plan that fits your budget** — a combinatorial engine computes every transport × hotel × attraction combination, filters by budget, and ranks them cheapest to most expensive
6. **Generate a day-by-day itinerary** — morning, afternoon, and evening activities for every day, built around your selected plan

---

## Setup

### 1. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card required)
3. Go to **API Keys** → **Create API Key**
4. Copy the key — it starts with `gsk_...`

### 2. Download or clone the project

```
travel_planner/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### 3. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

Save and close.

### 4. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Next time you want to run it

```bash
cd travel_planner
source venv/bin/activate
streamlit run app.py
```

---

## API key security

- Your `.env` file stays **only on your machine** — it is excluded from version control by `.gitignore`
- Never share or commit your `.env` file — it contains your real API key
- Anyone who clones this repo needs to create their own `.env` with their own Groq key (copy `.env.example` → `.env`)

---

## How it works

### Architecture

```
Chat (Requirements Agent — collects trip details, currency, budget)
         ↓
   Route Agent  →  Hotels Agent  →  Attractions Agent
   (parallel research using free geocoding + Groq knowledge)
         ↓
   Combinatorial Budget Engine
   (pure Python — every transport × hotel × attraction combo, ranked by cost)
         ↓
      Planner Agent
   (day-by-day itinerary built around your selected plan)
```

### Agents & data sources

| Agent | Responsibility | Data source |
|---|---|---|
| Requirements | Collects trip details via natural conversation | Groq (`llama-3.1-8b-instant`) |
| Route | Compares driving, rail, and flights | OSRM (driving/rail) + Groq + static price reference table |
| Hotels | Finds 10 real hotels across price tiers | Groq (`llama-3.3-70b-versatile`), fetched in 2 batches |
| Attractions | Discovers and ranks top sights | Overpass API (OpenStreetMap), with Groq fallback for sparse regions |
| Budget | Computes every valid plan combination | Pure Python — zero API calls |
| Planner | Generates day-by-day itinerary | Groq (`llama-3.3-70b-versatile`) |

### Free services used (no key needed)

| Service | Used for |
|---|---|
| [OSRM](https://router.project-osrm.org) | Driving distances, durations, and rail distance estimates |
| [Nominatim](https://nominatim.openstreetmap.org) | City geocoding (name → coordinates), with state/region resolution |
| [Overpass API](https://overpass-api.de) | Tourist attractions from OpenStreetMap |

### Price accuracy safeguards

- **Static reference tables**: ~500 flight pairs and ~500 train pairs (India + Continental Europe + major international routes) ground every price estimate against realistic market ranges
- **Confidence scoring**: every flight and train option includes a `high` / `medium` / `low` confidence badge; low-confidence train prices are hidden entirely rather than shown as a guess
- **Currency system**: the user explicitly states their budget currency in conversation; for international trips, all comparisons convert to USD using a static exchange-rate table (not estimated by the AI)
- **Infeasible route filtering**: driving/train options across oceans or with zero-cost data are automatically excluded from budget combinations

---

## UI Tabs

| Tab | Content |
|---|---|
| 🚗 Transportation | Driving, rail, and flight options with operator names and price confidence |
| 🏨 Lodging Options | 10 hotel cards across budget/mid/luxury tiers |
| 🏛️ Local Attractions | Top sights ranked by your stated preferences |
| 💰 Budget Planner | Every fitting plan combination, ranked cheapest → most expensive, plus an Economy/Balanced/Premium summary |
| 🗓️ Itinerary | Day-by-day schedule with a cost breakdown of your selected plan |

---

## Rate limits

The free Groq tier is generous (30+ requests/minute depending on model), but the app still spaces out API calls with short pauses between agent stages to stay well within limits. If a rate limit is ever hit, the app shows exactly how many seconds to wait — your progress is saved, so you can resume instead of starting over.

---

## Notes

- Trip state is held in memory and does not persist across browser refreshes
- Exchange rates and price reference tables are static (set at build time), not live — always verify final prices before booking
- Built for `llama-3.3-70b-versatile` (planning agents) and `llama-3.1-8b-instant` (conversation agent) via Groq

---

## License

This project is provided as-is for personal and educational use.
