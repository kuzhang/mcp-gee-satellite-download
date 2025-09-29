# GEE MCP Server

Python MCP server exposing a tool to download Google Earth Engine imagery as GeoTIFF.

## Prerequisites

- Google Earth Engine account approved
- Python 3.10+
- Local Earth Engine auth completed for the user account

## Credentials

### Server (GEE)
The server supports two auth methods. Use ONE of these:

1) Service account key file (recommended)
- Place `gee-key.json` in the project root (same folder as `server.py`).
- The file should be a standard Google service account JSON with fields like `type`, `project_id`, `private_key_id`, `private_key`, `client_email`, `client_id`, etc.
- Optionally set `GEE_PROJECT` in `.env` (or it will use `project_id` from the key file).

2) Environment variables
- Set the following in your environment (or `.env` if your launcher loads it):
```ini
GEE_PROJECT=your-gcp-project-id
GEE_SERVICE_ACCOUNT=service-account@your-project.iam.gserviceaccount.com
GEE_PRIVATE_KEY_PATH=C:\path\to\service-account.json
# Alternative to GEE_PRIVATE_KEY_PATH:
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
```

Notes:
- The server automatically detects `gee-key.json` if present; otherwise it tries env vars, then default user auth.
- Run `earthengine authenticate` once if you prefer user auth locally.

### Client (LLM)
The client uses an LLM for prompt parsing. Create a `.env` in the project root with:
```ini
OPENAI_API_KEY=sk-...
# Optional, defaults to gpt-4o-mini
OPENAI_MODEL=gpt-4o-mini
```

### Passing env to the server
When the client spawns the server (stdio), it forwards the most relevant GEE env vars if set:
- `GEE_PROJECT`
- `GEE_SERVICE_ACCOUNT`
- `GEE_PRIVATE_KEY_PATH`
- `GOOGLE_APPLICATION_CREDENTIALS`

If you rely on `gee-key.json`, no additional env configuration is required.


## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Authenticate Earth Engine (one-time):

```bash
earthengine authenticate
```

If running on a headless server, set service account credentials via env `GOOGLE_APPLICATION_CREDENTIALS` or use `ee.ServiceAccountCredentials` in code.

## Run Server (MCP over stdio)

```bash
python server.py
```

The server exposes one tool:

- `download_satellite_image(DownloadParams) -> str`: Downloads a GeoTIFF to a temp file and returns its absolute path.

### DownloadParams
- `latitude` (float)
- `longitude` (float)
- `start_date` (YYYY-MM-DD)
- `end_date` (YYYY-MM-DD)
- `scale` (int, meters, default 10)
- `buffer_m` (int, default 1000)
- `bands` (str, comma-separated; default `B4,B3,B2`)
- `dataset` (str, default `COPERNICUS/S2`)

## Run Client (CLI)

The CLI connects to the server over stdio, prompts for parameters, and calls the tool:

```bash
python client.py
```

## MCP Client Config Example (Cursor / Claude Desktop)

Add a server entry similar to:

```json
{
  "mcpServers": {
    "gee-mcp": {
      "command": "python",
      "args": ["server.py"],
      "env": {}
    }
  }
}
```

## Notes
- Ensure your date window and location intersect the dataset footprint.
- For Sentinel-2, the tool prefers the least cloudy image automatically.
- Large regions or small scales may produce big downloads; adjust `buffer_m` and `scale` accordingly.
- Don't give too large region to download at once, the split function in server may fails.


## Server functions and capabilities

### Tools exposed
- `download_satellite_image(DownloadParams) -> { files: string[], message: string }`
  - Downloads one or more GeoTIFFs from Google Earth Engine for a user-defined Area of Interest (AOI).

### Inputs (DownloadParams)
- Geographic input (provide exactly one):
  - `latitude`, `longitude` (float): point center; uses `buffer_m` to create AOI
  - `bounding_box` (number[4]): `[min_lon, min_lat, max_lon, max_lat]` (preferred for rectangles)
  - `region_geojson` (string): full GeoJSON for complex polygons
- Temporal and data options:
  - `start_date`, `end_date` (YYYY-MM-DD)
  - `dataset` (string): default `COPERNICUS/S2_SR_HARMONIZED`
  - `bands` (string): comma-separated band names (default `B4,B3,B2`)
  - `scale` (int, meters): target resolution
  - `buffer_m` (int): only used with point input
  - `max_cloud_cover` (float, %): filter cap for Sentinel-2
  - `image_count` (int): number of least-cloudy images to download

### Processing behavior
- Sorts by cloud cover (Sentinel-2) and selects up to `image_count` images.
- Saves files to `download/` with capture date in filename.
- Preflight size check and automatic tiling:
  - Estimates pixel count/bytes and splits AOI into quadrants when needed.
  - If a tile still exceeds GEE size limits, recursively splits further (limited depth) and/or increases `scale` (coarsens resolution) until the request fits.
- Robust region handling:
  - Converts `bounding_box` to a rectangular polygon.
  - Uses AOI bounds as the download region.
- Logging: detailed execution written to `server.log` (authentication, preflight, tiling, retries).

### Outputs
- Returns `{ files: [absolute_file_paths], message: string }`.
  - May return multiple files when AOI is split or multiple dates are requested.
  - Message indicates if fewer images were available than requested, or if tiling/scale backoff occurred.

### Limits and notes
- Very large AOIs and/or small `scale` produce large requests; server will split tiles and increase `scale` automatically to satisfy GEE limits (~48 MB per request for thumbnails/downloads).
- Cloud-cover filtering currently applies to Sentinel‑2 collections (field `CLOUDY_PIXEL_PERCENTAGE`).
- If you need strict masking to the polygon footprint, consider adding an explicit `clip(AOI)` step before download.

