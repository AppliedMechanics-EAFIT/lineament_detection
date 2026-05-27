import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.preprocessing.rasterized_training_data import (
        rasterize_line_vectors_to_mask,
        tile_prestacked_raster_and_mask,
    )
except ModuleNotFoundError:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from preprocessing.rasterized_training_data import (
        rasterize_line_vectors_to_mask,
        tile_prestacked_raster_and_mask,
    )


EARTH_SEARCH_STAC = "https://earth-search.aws.element84.com/v1/search"
SGC_ARCGIS_ROOT = "https://srvags.sgc.gov.co/arcgis/rest/services"
SGC_MGC2023_FEATURE = (
    "https://srvags.sgc.gov.co/arcgis/rest/services/"
    "Mapa_Geologico_Colombia/Mapa_Geologico_Colombia_V2023/FeatureServer"
)


DEFAULT_AOIS = {
    "california_san_andreas": [-122.6, 37.5, -122.2, 38.0],
    "turkiye_north_anatolian": [30.0, 40.2, 31.2, 41.1],
    "iceland_thingvellir_rift": [-22.0, 63.9, -20.8, 64.4],
}


def _download_file(url, destination, timeout=180):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)


def _stac_search_s2(bbox, datetime_range, limit):
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": datetime_range,
        "limit": int(limit),
    }
    response = requests.post(EARTH_SEARCH_STAC, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data.get("features", [])


def download_non_colombia_sentinel2(
    output_dir,
    datetime_range,
    bands,
    max_scenes_per_aoi=2,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = []

    for aoi_name, bbox in DEFAULT_AOIS.items():
        features = _stac_search_s2(bbox=bbox, datetime_range=datetime_range, limit=30)
        if not features:
            report.append({"aoi": aoi_name, "bbox": bbox, "scenes": []})
            continue

        features = sorted(features, key=lambda item: item.get("properties", {}).get("eo:cloud_cover", 9999.0))
        selected = features[: max_scenes_per_aoi]

        aoi_folder = output_dir / aoi_name
        aoi_folder.mkdir(parents=True, exist_ok=True)

        scene_rows = []
        for feature in selected:
            scene_id = feature.get("id", "unknown_scene")
            cloud = feature.get("properties", {}).get("eo:cloud_cover")
            assets = feature.get("assets", {})
            downloaded = []

            for band in bands:
                if band not in assets:
                    continue
                href = assets[band]["href"]
                destination = aoi_folder / f"{scene_id}_{band}.tif"
                _download_file(href, destination)
                downloaded.append(str(destination))

            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "cloud_cover": cloud,
                    "downloaded_files": downloaded,
                }
            )

        report.append({"aoi": aoi_name, "bbox": bbox, "scenes": scene_rows})

    return report


def _download_sgc_layer_geojson(layer_id, output_path):
    layer_url = f"{SGC_MGC2023_FEATURE}/{layer_id}"

    meta = requests.get(f"{layer_url}?f=pjson", timeout=40)
    meta.raise_for_status()
    meta_json = meta.json()
    max_records = int(meta_json.get("maxRecordCount", 2000))

    count_response = requests.get(
        f"{layer_url}/query",
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=40,
    )
    count_response.raise_for_status()
    count = int(count_response.json().get("count", 0))

    features = []
    offset = 0
    while offset < count:
        query = requests.get(
            f"{layer_url}/query",
            params={
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "outSR": "4326",
                "resultOffset": offset,
                "resultRecordCount": max_records,
            },
            timeout=120,
        )
        query.raise_for_status()
        payload = query.json()
        batch = payload.get("features", [])
        if not batch:
            break
        features.extend(batch)
        offset += len(batch)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump({"type": "FeatureCollection", "features": features}, file_obj)

    return {
        "layer_id": layer_id,
        "count": len(features),
        "path": str(output_path),
    }


def download_sgc_colombia_vectors(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_mapping = {
        703: "sgc_mgc2023_pliegues.geojson",
        704: "sgc_mgc2023_fallas.geojson",
        705: "sgc_mgc2023_tectonica.geojson",
    }

    rows = []
    for layer_id, filename in layer_mapping.items():
        row = _download_sgc_layer_geojson(layer_id=layer_id, output_path=output_dir / filename)
        rows.append(row)

    return rows


def write_download_notes(notes_path, non_colombia_report, colombia_report):
    notes_path = Path(notes_path)
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Open Data Notes",
        "",
        f"Generated at UTC: {timestamp}",
        "",
        "## Non-Colombia Open Data",
        "Source catalog: https://earth-search.aws.element84.com/v1/search",
        "Collection: sentinel-2-l2a",
        "Asset provider: AWS open-data COG assets exposed through Earth Search STAC",
        "",
    ]

    for row in non_colombia_report:
        lines.append(f"### AOI: {row['aoi']}")
        lines.append(f"BBox: {row['bbox']}")
        if not row.get("scenes"):
            lines.append("No scenes downloaded")
            lines.append("")
            continue
        for scene in row["scenes"]:
            lines.append(f"- Scene: {scene['scene_id']}")
            lines.append(f"  Cloud cover: {scene.get('cloud_cover')}")
            lines.append("  Files:")
            for file_path in scene.get("downloaded_files", []):
                lines.append(f"  - {file_path}")
        lines.append("")

    lines.extend(
        [
            "## Colombia (SGC)",
            "Source server: https://srvags.sgc.gov.co/arcgis/rest/services",
            "Service: Mapa_Geologico_Colombia/Mapa_Geologico_Colombia_V2023/FeatureServer",
            "",
        ]
    )
    for row in colombia_report:
        lines.append(f"- Layer {row['layer_id']}: {row['count']} features")
        lines.append(f"  File: {row['path']}")

    with notes_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download open non-Colombia Sentinel-2 data, download SGC Colombia structural vectors, "
            "and optionally prepare training tiles from a Colombia stack raster."
        )
    )
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument(
        "--datetime-range",
        default="2024-01-01T00:00:00Z/2024-12-31T23:59:59Z",
        help="STAC datetime interval for Sentinel-2 search",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=["red", "green", "blue", "nir"],
        help="Sentinel-2 asset keys to download",
    )
    parser.add_argument("--max-scenes-per-aoi", type=int, default=2)
    parser.add_argument("--skip-non-colombia", action="store_true")
    parser.add_argument("--skip-colombia", action="store_true")

    parser.add_argument(
        "--stack-raster-colombia",
        default=None,
        help="Optional stack raster path to connect downloaded Colombia vectors to training tile generation",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min-positive-fraction", type=float, default=0.001)
    parser.add_argument("--max-nan-fraction", type=float, default=0.25)
    parser.add_argument("--drop-all-zero-tiles", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    non_colombia_dir = data_root / "OpenData" / "non_colombia" / "sentinel2"
    colombia_dir = data_root / "Colombia"
    colombia_vectors_dir = colombia_dir / "labels" / "sgc_mgc2023"
    notes_path = data_root / "NOTES_OPEN_DATA.md"

    non_colombia_report = []
    colombia_report = []

    if not args.skip_non_colombia:
        non_colombia_report = download_non_colombia_sentinel2(
            output_dir=non_colombia_dir,
            datetime_range=args.datetime_range,
            bands=args.bands,
            max_scenes_per_aoi=args.max_scenes_per_aoi,
        )

    if not args.skip_colombia:
        colombia_report = download_sgc_colombia_vectors(colombia_vectors_dir)

    write_download_notes(
        notes_path=notes_path,
        non_colombia_report=non_colombia_report,
        colombia_report=colombia_report,
    )

    prep_report = None
    if args.stack_raster_colombia:
        colombia_training_dir = colombia_dir / "training"
        mask_raster_path = colombia_training_dir / "sgc_faults_mask.tif"
        stack_out_dir = colombia_training_dir / "stacks"
        mask_out_dir = colombia_training_dir / "masks"
        tile_report_path = colombia_training_dir / "tile_report.json"

        vector_paths = [colombia_vectors_dir / "sgc_mgc2023_fallas.geojson"]

        rasterize_line_vectors_to_mask(
            reference_raster_path=args.stack_raster_colombia,
            vector_paths=vector_paths,
            output_mask_raster_path=mask_raster_path,
        )

        prep_report = tile_prestacked_raster_and_mask(
            stack_raster_path=args.stack_raster_colombia,
            mask_raster_path=mask_raster_path,
            output_stack_dir=stack_out_dir,
            output_mask_dir=mask_out_dir,
            tile_size=args.tile_size,
            stride=args.stride,
            min_positive_fraction=args.min_positive_fraction,
            max_nan_fraction=args.max_nan_fraction,
            drop_all_zero_tiles=args.drop_all_zero_tiles,
            output_report_path=tile_report_path,
        )

    summary = {
        "data_root": str(data_root),
        "non_colombia_dir": str(non_colombia_dir),
        "colombia_vectors_dir": str(colombia_vectors_dir),
        "notes": str(notes_path),
        "non_colombia": non_colombia_report,
        "colombia": colombia_report,
        "pipeline_prep": prep_report,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()