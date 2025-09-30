import os
import json
import tempfile
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field, model_validator
import ee
import requests
import time
from typing import Optional, List, Dict, Any


class DownloadParams(BaseModel):
	latitude: Optional[float] = Field(None, description="Latitude of the point of interest.")
	longitude: Optional[float] = Field(None, description="Longitude of the point of interest.")
	region_geojson: Optional[str] = Field(None, description="A GeoJSON string defining the region of interest.")
	bounding_box: Optional[List[float]] = Field(None, description="A bounding box as a list of four coordinates: [min_lon, min_lat, max_lon, max_lat].")
	start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
	end_date: str = Field(..., description="End date (YYYY-MM-DD)")
	scale: int = Field(10, description="Pixel resolution in meters")
	buffer_m: int = Field(1000, description="Buffer distance in meters around the point (only used with latitude/longitude).")
	bands: Optional[str] = Field("B4,B3,B2", description="Comma-separated band names to download, e.g., B4,B3,B2")
	dataset: str = Field(
		"COPERNICUS/S2_SR_HARMONIZED",
		description="ImageCollection ID, e.g., COPERNICUS/S2_SR_HARMONIZED or LANDSAT/LC08/C02/T1_L2",
	)
	max_cloud_cover: Optional[float] = Field(20.0, description="Maximum cloud cover percentage (0-100).")
	image_count: Optional[int] = Field(1, description="Number of least cloudy images to download.")

	@model_validator(mode='before')
	@classmethod
	def check_coords_or_geojson(cls, data: Any) -> Any:
		"""Ensure either coordinates, a GeoJSON string, or a bounding box is provided."""
		if not isinstance(data, dict):
			return data
		
		lat, lon = data.get('latitude'), data.get('longitude')
		geojson = data.get('region_geojson')
		bbox = data.get('bounding_box')
		coords_provided = lat is not None and lon is not None
		geojson_provided = geojson is not None
		bbox_provided = bbox is not None

		provided_options = sum([coords_provided, geojson_provided, bbox_provided])

		if provided_options == 0:
			raise ValueError("Either latitude/longitude, a region_geojson string, or a bounding_box must be provided.")
		if provided_options > 1:
			raise ValueError("Provide only one of latitude/longitude, a region_geojson string, or a bounding_box.")
		
		if bbox_provided and len(bbox) != 4:
			raise ValueError("The bounding_box must contain exactly four coordinates: [min_lon, min_lat, max_lon, max_lat].")

		return data


# Define GEE limitations
MAX_PIXELS_PER_DOWNLOAD = 4e7  # Reduced for safety, well under the ~50M byte limit

# Configure logging to write to a file and stderr
log_file_path = os.path.join(os.path.dirname(__file__), "server.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SERVER - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # Overwrite log on each run
        logging.StreamHandler()
    ],
    force=True # Required to reconfigure logging in some environments
)

load_dotenv()
gee_project = os.getenv("GEE_PROJECT")

def setup_gee() -> None:
	logging.info("Start the GEE authentication process")
	# Try authenticating with gee-key.json first
	key_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "gee-key.json"))
	if os.path.exists(key_path):
		try:
			with open(key_path) as f:
				key_data = json.load(f)
			project = gee_project or key_data.get("project_id")
			if not project:
				raise ValueError("GEE project ID not found in key file or GEE_PROJECT env var.")
			credentials = ee.ServiceAccountCredentials(key_data["client_email"], key_path)
			ee.Initialize(credentials, project=project)
			logging.info("Authenticated to GEE using gee-key.json")
			return
		except Exception as e:
			logging.warning(f"Could not authenticate with gee-key.json, falling back. Error: {e}")

	# If key file fails, try service account from env vars as fallback
	if not gee_project:
		raise RuntimeError("GEE_PROJECT is not set and gee-key.json failed or was not found.")

	service_account = os.getenv("GEE_SERVICE_ACCOUNT") or os.getenv("EE_SERVICE_ACCOUNT")
	env_key_path = os.getenv("GEE_PRIVATE_KEY_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if service_account and env_key_path and os.path.exists(env_key_path):
		try:
			credentials = ee.ServiceAccountCredentials(service_account, env_key_path)
			ee.Initialize(credentials, project=gee_project)
			logging.info("Authenticated to GEE using environment variables.")
			return
		except Exception as e:
			logging.warning(f"Could not authenticate with GEE env vars, falling back. Error: {e}")

	# Fallback to default user credentials or interactive auth
	try:
		ee.Initialize(project=gee_project)
		logging.info("Authenticated to GEE using default user credentials.")
	except Exception:
		try:
			logging.info("Attempting interactive GEE authentication...")
			auth_mode = os.getenv("GEE_AUTH_MODE", "gcloud")
			ee.Authenticate(auth_mode=auth_mode)
			ee.Initialize(project=gee_project)
			logging.info("Interactive GEE authentication successful.")
		except Exception as auth_error:
			raise RuntimeError(f"Fatal: All GEE authentication methods failed. Last error: {auth_error}")


# helper function to check if a region is too large to download and split it into smaller tiles if necessary
def _check_and_split_region(region: ee.Geometry, scale: int) -> List[ee.Geometry]:
	"""
	Checks if a region is too large to download by calculating estimated pixel count
	and splits it into smaller tiles if necessary.
	"""
	# Use ee.Image.pixelArea() to get the size of each pixel, then calculate total area
	try:
		# Create a dummy image to calculate the pixel area
		proj = ee.Projection('EPSG:4326').atScale(scale)
		pixel_area_image = ee.Image.pixelArea().reproject(proj)
		
		# Calculate the area of the region
		area = region.area(maxError=1).getInfo()
		
		# Calculate the number of pixels
		# Get the area of a single pixel to estimate total pixels
		# We sample a single point at the centroid for this
		pixel_area_at_centroid = pixel_area_image.reduceRegion(
			reducer=ee.Reducer.first(),
			geometry=region.centroid(maxError=1),
			scale=scale
		).get('area').getInfo()

		if pixel_area_at_centroid is None or pixel_area_at_centroid == 0:
			raise ValueError("Could not determine pixel area.")

		total_pixels = area / pixel_area_at_centroid

	except Exception as e:
		logging.error(f"Could not compute region size: {e}. Assuming it's too large and splitting.")
		total_pixels = MAX_PIXELS_PER_DOWNLOAD + 1 # Force split

	if total_pixels <= MAX_PIXELS_PER_DOWNLOAD:
		logging.info(f"Region size is acceptable ({total_pixels:.0f} pixels).")
		return [region]
	else:
		# Simple 2x2 split
		logging.info(f"Region too large ({total_pixels:.0f} pixels). Splitting into 4 quadrants.")
		bounds = region.bounds().getInfo()['coordinates'][0]
		center = region.centroid(maxError=1).getInfo()['coordinates']
		lon_c, lat_c = center[0], center[1]
		
		min_lon, min_lat = bounds[0][0], bounds[0][1]
		max_lon, max_lat = bounds[2][0], bounds[2][1]

		# Define the four new quadrants
		bottom_left = ee.Geometry.Rectangle([min_lon, min_lat, lon_c, lat_c])
		bottom_right = ee.Geometry.Rectangle([lon_c, min_lat, max_lon, lat_c])
		top_left = ee.Geometry.Rectangle([min_lon, lat_c, lon_c, max_lat])
		top_right = ee.Geometry.Rectangle([lon_c, lat_c, max_lon, max_lat])
		
		# Recursively check and split each new tile
		tiles = []
		tiles.extend(_check_and_split_region(bottom_left, scale))
		tiles.extend(_check_and_split_region(bottom_right, scale))
		tiles.extend(_check_and_split_region(top_left, scale))
		tiles.extend(_check_and_split_region(top_right, scale))
		
		return tiles


def _split_region_quadrants(region: ee.Geometry) -> List[ee.Geometry]:
	"""Split a region into four rectangular quadrants using its bounds and centroid."""
	bounds = region.bounds().getInfo()['coordinates'][0]
	center = region.centroid(maxError=1).getInfo()['coordinates']
	lon_c, lat_c = center[0], center[1]
	min_lon, min_lat = bounds[0][0], bounds[0][1]
	max_lon, max_lat = bounds[2][0], bounds[2][1]
	return [
		ee.Geometry.Rectangle([min_lon, min_lat, lon_c, lat_c]),
		ee.Geometry.Rectangle([lon_c, min_lat, max_lon, lat_c]),
		ee.Geometry.Rectangle([min_lon, lat_c, lon_c, max_lat]),
		ee.Geometry.Rectangle([lon_c, lat_c, max_lon, max_lat]),
	]


def _download_with_fallback(selected_bands: ee.Image, region: ee.Geometry, scale: int, prefix: str, depth: int = 0) -> List[str]:
	"""Try to download a region. On size-limit error, split into quadrants recursively; as a last resort, increase scale."""
	# Prepare output directory
	download_dir = os.path.join(os.path.dirname(__file__), 'download')
	os.makedirs(download_dir, exist_ok=True)

	# Build download parameters
	download_region = region.bounds().getInfo()["coordinates"]
	download_params = {
		"scale": scale,
		"region": json.dumps(download_region),
		"format": "GEO_TIFF",
	}
	try:
		url = selected_bands.getDownloadURL(download_params)
		file_path = os.path.join(download_dir, f"{prefix}.tif")
		with requests.get(url, stream=True) as r:
			r.raise_for_status()
			with open(file_path, "wb") as f:
				for chunk in r.iter_content(chunk_size=8192):
					if chunk:
						f.write(chunk)
		return [os.path.abspath(file_path)]
	except Exception as e:
		err_msg = str(e)
		# GEE size limit error → split or scale backoff
		if "Total request size" in err_msg and "must be less than or equal to" in err_msg:
			logging.info(f"Tile too large at scale={scale}. depth={depth}. Attempting fallback...")
			if depth < 2:
				results: List[str] = []
				for idx, sub in enumerate(_split_region_quadrants(region), start=1):
					results.extend(_download_with_fallback(selected_bands, sub, scale, f"{prefix}_q{idx}", depth + 1))
				return results
			# Last resort: increase scale (coarser resolution)
			if scale < 160 and depth < 5:
				new_scale = min(scale * 2, 160)
				logging.info(f"Increasing scale to {new_scale} and retrying for prefix={prefix}")
				return _download_with_fallback(selected_bands, region, new_scale, f"{prefix}_s{new_scale}", depth + 1)
			# Give up if still failing
			logging.warning(f"Giving up on region after fallbacks for prefix={prefix}: {err_msg}")
			return []
		# Non-size error → re-raise to be handled by caller
		raise


app = FastMCP("gee-mcp")


@app.tool()
def download_satellite_image(args: DownloadParams) -> Dict[str, Any]:
	"""Download satellite images from GEE and return a dictionary with file paths and a status message.
	Arguments include coordinates, date range, dataset, bands, scale, buffer, cloud cover, and image count.
	"""
	
	# Determine the region of interest from either GeoJSON, a bounding box, or a point
	if args.region_geojson:
		try:
			geojson_dict = json.loads(args.region_geojson)
			logging.info(f"GeoJSON: {geojson_dict}")
			region = ee.Geometry(geojson_dict)
		except Exception as e:
			raise ValueError(f"Invalid GeoJSON provided: {e}")
	elif args.bounding_box:
		try:
			min_lon, min_lat, max_lon, max_lat = args.bounding_box
			logging.info(f"Bounding box: {min_lon}, {min_lat}, {max_lon}, {max_lat}")
			# Define the four corners of the rectangle to create a Polygon
			coords = [
				[min_lon, min_lat],
				[max_lon, min_lat],
				[max_lon, max_lat],
				[min_lon, max_lat],
				[min_lon, min_lat] # Close the polygon
			]
			region = ee.Geometry.Polygon(coords)
		except Exception as e:
			raise ValueError(f"Invalid bounding_box provided. Ensure it is a list of four numbers. Error: {e}")
	else:
		point = ee.Geometry.Point([args.longitude, args.latitude])
		logging.info(f"Point: {point}")
		region = point.buffer(args.buffer_m)

	# --- Pre-flight check and AOI Tiling ---
	logging.info("Performing pre-flight checks for region size...")
	tiles = _check_and_split_region(region, args.scale)
	if len(tiles) > 1:
		logging.info(f"Region is too large for a single download. Splitting into {len(tiles)} tiles.")

	downloaded_files = []
	for i, tile_region in enumerate(tiles):
		if len(tiles) > 1:
			logging.info(f"Processing tile {i + 1} of {len(tiles)}...")

		collection = (
			ee.ImageCollection(args.dataset)
			.filterDate(args.start_date, args.end_date)
			.filterBounds(tile_region)
		)

		# Filter by cloud cover if the dataset supports it and the param is provided
		if args.dataset.startswith("COPERNICUS/S2") and args.max_cloud_cover is not None:
			collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', args.max_cloud_cover))

		# First, check if any images meet the cloud cover criteria before proceeding
		collection_size = collection.size().getInfo()
		if collection_size == 0:
			logging.warning(f"No images found for tile {i + 1} that meet the specified cloud cover rate. Skipping.")
			continue

		# Sort by cloud cover and limit to the number of images requested
		collection = collection.sort("CLOUDY_PIXEL_PERCENTAGE").limit(args.image_count)

		image_list = collection.toList(args.image_count)
		num_images_found = image_list.size().getInfo()
		
		for j in range(num_images_found):
			image = ee.Image(image_list.get(j))

			# Get the image's capture date for the filename
			date_string = ee.Date(image.get('system:time_start')).format('YYYYMMdd').getInfo()

			# Select bands for each image
			selected_bands = None
			if args.bands:
				band_list = [b.strip() for b in args.bands.split(",") if b.strip()]
				selected_bands = image.select(band_list)
			else:
				selected_bands = image
			
			# Use the region's bounds for the download area
			download_region = tile_region.bounds().getInfo()["coordinates"]

			download_params = {
				"scale": args.scale,
				"region": json.dumps(download_region),
				"format": "GEO_TIFF",
			}

			# Generate a filename prefix including capture date and tile index
			prefix = f"gee_image_{date_string}_t{i+1}_n{j+1}"

			# Use robust fallback downloader (recursive split/scale backoff)
			file_paths = _download_with_fallback(selected_bands, tile_region, args.scale, prefix)
			if not file_paths:
				logging.warning(f"No files produced for tile {i+1}, image {j+1} after fallbacks.")
			else:
				downloaded_files.extend(file_paths)
	
	message = f"Successfully downloaded {len(downloaded_files)} image(s) across {len(tiles)} tile(s)."
	if len(tiles) > 1:
		message += f" (Note: Some images were skipped due to insufficient cloud cover or no images found in some tiles.)"


	return {"files": downloaded_files, "message": message}

@app.prompt()
def get_llm_prompt(query: str) -> str:
	"""Generates a a prompt for the LLM to use to answer the query"""

	return f"""
	You are an expert GEE assistant responsible for translating user requests into precise tool calls.
	Your task is to answer the user's query about satellite image downloads by using the `download_satellite_image` tool.

	**VERY IMPORTANT:** You must determine the user's region of interest and format it correctly.
	- If the user provides a latitude and longitude, use the `latitude` and `longitude` parameters.
	- If the user provides a bounding box (e.g., [min_lon, min_lat, max_lon, max_lat]), you have two options:
	  1. **(Preferred)** Pass the coordinates as a simple list to the `bounding_box` parameter.
	  2. Convert it into a valid GeoJSON string and pass it to the `region_geojson` parameter.
	- Use the `bounding_box` parameter for simple rectangular areas. Use `region_geojson` for more complex shapes like polygons.
	- Coordinates MUST be in [longitude, latitude] order. Do NOT swap to [latitude, longitude].
	- For bounding boxes, preserve the exact order [min_lon, min_lat, max_lon, max_lat].
	- For GeoJSON, every coordinate pair must be [lon, lat].
	- Provide only one of: coordinates, a bounding box, or GeoJSON.

	Extract all other parameters like dates, cloud cover, and image count from the query.

	Query: {query}
	"""


if __name__ == "__main__":
	# Initialize GEE once before starting the server
	setup_gee()
	# Run MCP server (stdio by default)
	app.run(transport="stdio")


