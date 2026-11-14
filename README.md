# EUDR coffee traceability pipeline (Uganda)

Plot validation and deforestation-risk screening for coffee plots, built to run on **any Ugandan district**:
edit the district names in `config.yaml`, run the notebooks in order, open the results in QGIS.

> Status: notebook 01 (study-area boundaries) is complete. Plot data used for demos is **synthetic** and will be labelled as such.

## Setup
1. Open this folder in VS Code.
2. Select your conda env as the notebook kernel (the one with geopandas), then once, in a terminal:
   `pip install -e .`
3. Edit `config.yaml` (districts, boundary file path).
4. Run `notebooks/01_boundaries_study_area.ipynb` top to bottom.

## Folders
| Folder | Purpose |
|---|---|
| `notebooks/` | Numbered steps, run in order |
| `src/eudr/` | Reusable code the notebooks call |
| `data/input/` | Real plot files supplied by users |
| `data/synthetic/` | Generated demo data |
| `outputs/<run_name>/` | GeoPackage, reports, maps, run manifests (open `gis/*.gpkg` in QGIS) |
| `qgis/` | QGIS project and styles |

Every run writes a manifest (`outputs/<run_name>/manifests/`) recording inputs (with file hashes),
parameters and package versions, so results can be reproduced and audited.
