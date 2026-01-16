import pandas as pd
import numpy as np
import pickle
import os
import re
import traceback  # Import traceback module to print detailed errors

# --- Dependencies ---
# Ensure you have installed: pip install rasterio geopandas pandas
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point

# --- [!! 1. File Path Definitions !!] ---
# (We assume all files are in the same folder)

# A. Your Input Files
STATION_CSV_PATH = 'result_API_modified.csv'
STATION_MAP_CSV = 'suzhou_subway_station.csv'
RASTER_POP_PATH = 'suzhou_population_clipped.tif'

# B. Census Data Files
PATH_AGE = '02-02.csv'
PATH_EDU = 'educated_level.csv'
PATH_HOUSE = '02-06.csv'
PATH_EMP = '02-09.csv'  # Using your newly modified file

# C. Your Output File
OUTPUT_PICKLE_FILE = 'station_static_features_REAL.pkl'

# --- [!! 2. Name Corrections !!] ---
# Correct the names in the 'Suzhou Subway' CSV to match the "Region/District" names in the census data CSVs
NAME_CORRECTIONS = {
    '苏州工业园区': '工业园区',
    # '虎丘区' in Subway CSV -> '虎丘区、高新区' in Census CSV
    '虎丘区': '虎丘区、高新区',
    # Names in 'educated_level.csv' also need to match
    '高新区、虎丘区': '虎丘区、高新区'
}


# --- 3. Helper Functions (Unchanged Logic) ---

def clean_numeric(value):
    """
    Removes commas from numeric strings and converts them to floats.
    Returns NaN if the value is empty or '-'.
    """
    if pd.isna(value) or str(value).strip() == '-':
        return np.nan
    cleaned = re.sub(r"[^0-9.-]", "", str(value))
    if cleaned == "":
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def load_automated_map(map_csv_path, corrections):
    """
    [!! AUTOMATED !!]
    Loads the Station -> District mapping from 'suzhou_subway_station.csv'.
    """
    print(f"Automatically loading station-district mapping from {map_csv_path}...")
    try:
        df_map = pd.read_csv(map_csv_path, header=1, encoding='gbk')
        if '中文站名' not in df_map.columns or '所在地' not in df_map.columns:
            print(f"!! ERROR: '{map_csv_path}' must contain columns '中文站名' and '所在地'.")
            return None
        df_map = df_map.dropna(subset=['中文站名', '所在地'])
        df_map = df_map.drop_duplicates(subset=['中文站名'])
        df_map['所在地'] = df_map['所在地'].str.strip()
        df_map['所在地'] = df_map['所在地'].replace(corrections)
        station_map = pd.Series(df_map['所在地'].values, index=df_map['中文站名']).to_dict()
        print(f"Successfully loaded {len(station_map)} Station -> District mappings.")
        return station_map
    except Exception as e:
        print(f"!! ERROR: Error reading station map CSV '{map_csv_path}': {e}")
        return None


def load_station_data_from_csv(csv_path):
    """
    Loads station coordinates and IDs from the 'result_API_modified.csv' file.
    """
    print(f"Loading station coordinates from {csv_path}...")
    try:
        df_stations = pd.read_csv(csv_path, encoding='utf-8-sig')
        if 'station_name' not in df_stations.columns or 'lng' not in df_stations.columns or 'lat' not in df_stations.columns:
            print("!! ERROR: CSV file must contain 'station_name', 'lng', and 'lat' columns.")
            return {}, []
        coords_db = {}
        station_ids_list = []
        for _, row in df_stations.iterrows():
            station_id = row['station_name']
            lon, lat = row['lng'], row['lat']
            if pd.isna(station_id) or pd.isna(lon) or pd.isna(lat): continue
            coords_db[station_id] = (float(lon), float(lat))
            station_ids_list.append(station_id)
        print(f"Successfully loaded {len(station_ids_list)} station coordinates.")
        return coords_db, station_ids_list
    except Exception as e:
        print(f"!! ERROR: Error reading station coordinates CSV '{csv_path}': {e}")
        return {}, []


def get_population_from_raster(station_id, station_coords_db, raster_src, radius_m=800):
    """
    [!! REAL IMPLEMENTATION - v2 FIX !!]
    Queries the sum of population within a station buffer from the opened WorldPop raster data.
    Fixed the issue where 'EPSG:102025' was unrecognized.
    """
    if station_id not in station_coords_db:
        print(f"WARNING: Station {station_id} has no coordinates, cannot calculate population. Returning 0.")
        return 0.0
    lon, lat = station_coords_db[station_id]
    try:
        # 1. Create WGS84 (lon/lat) point
        gdf_point = gpd.GeoDataFrame([{'geometry': Point(lon, lat), 'id': station_id}], crs='EPSG:4326')

        # 2. Project to the Raster file's CRS
        gdf_point_proj = gdf_point.to_crs(raster_src.crs)

        # 3. [!! FIX !!] Convert to a common metric CRS (EPSG:3857) to create the buffer.
        #    The original 'EPSG:102025' is an ESRI code and may not be recognized by the proj library.
        gdf_point_metric = gdf_point_proj.to_crs("EPSG:3857")

        # 4. Create buffer in meters
        gdf_buffer_metric = gdf_point_metric.buffer(radius_m)

        # 5. Convert buffer back to the Raster file's CRS for masking
        gdf_buffer_proj = gdf_buffer_metric.to_crs(raster_src.crs)

        # 6. Execute mask
        shapes = gdf_buffer_proj.geometry
        out_image, out_transform = mask(raster_src, shapes, crop=True, nodata=0)

        population_sum = out_image[0].sum()
        return max(0.0, float(population_sum))

    except Exception as e:
        # Print detailed error
        print(f"!! ERROR: Error extracting population for station {station_id} (lon={lon}, lat={lat}): {e}")
        # traceback.print_exc() # Uncomment if full stack trace is needed
        return 0.0

def generate_station_level_socio_demographics(
        station_to_district_map, all_station_ids,
        path_age, path_edu, path_house, path_emp,  # [!! Added path_emp !!]
        raster_src, station_coords_db
):
    """
    [!! v6 FIX !!]
    Merges macro census data and (real) raster population data to generate station-level socio-demographic features.
    Fixed the loading logic for 02-09.csv to match your new file (header=0).
    Kept the v5 fix for educated_level.csv (quotechar='"').
    Added the district_emp_to_pop_ratio feature.
    """
    print("Starting to load macro census data...")

    try:
        # --- 1. Load Age Data (02-02.csv) ---
        print(f"  - Loading: {path_age} (Age)")
        df_age = pd.read_csv(path_age, header=2, encoding='utf-8-sig')
        df_age.columns = df_age.columns.str.strip()
        print(f"    - (DEBUG) {path_age} Original columns: {df_age.columns.to_list()}")
        df_age = df_age.rename(
            columns={
                "地    区": "district",
                "Aged 0-14": "age_0_14",
                "Aged 15-64": "age_15_64",
                "Aged 65 and Over": "age_65_plus",
                "Resident Population": "total_pop_age"
            })
        df_age = df_age[df_age['district'] != '全    市'].set_index("district")
        df_age.index = df_age.index.str.strip()
        for col in ["age_0_14", "age_15_64", "age_65_plus", "total_pop_age"]:
            df_age[col] = df_age[col].apply(clean_numeric)
        print(f"    - (DEBUG) {path_age} Loaded successfully.")

        # --- 2. Load Education Data (educated_level.csv) ---
        print(f"  - Loading: {path_edu} (Education)")
        # [!! v5 FIX !!] Keeping quotechar='"' fix, specific to this file format
        df_edu = pd.read_csv(path_edu, header=0, sep='\t', encoding='gbk', quotechar='"')
        df_edu.columns = df_edu.columns.str.strip()
        print(f"    - (DEBUG) {path_edu} Original columns: {df_edu.columns.to_list()}")

        if '地区' not in df_edu.columns:
            print(f"!! CRITICAL ERROR: '地区' column not found in {path_edu}.")
            return pd.DataFrame()

        df_edu['地区'] = df_edu['地区'].str.strip()
        df_edu = df_edu.rename(
            columns={
                "地区": "district",
                "大学": "edu_tertiary_uni",
                "高中": "edu_senHigh",
                "初中": "edu_junMid",
                "小学": "edu_primary"
            })
        df_edu = df_edu[df_edu['district'] != '苏州市'].set_index("district")
        df_edu.index = df_edu.index.str.strip()
        df_edu = df_edu.rename(index={'高新区、虎丘区': '虎丘区、高新区'})
        edu_cols = ['edu_tertiary_uni', 'edu_senHigh', 'edu_junMid', 'edu_primary']
        for col in edu_cols:
            df_edu[col] = df_edu[col].apply(clean_numeric)
        df_edu['edu_tertiary_college'] = 0
        df_edu['edu_tertiary_postgrad'] = 0
        df_edu['edu_none'] = 0
        df_edu['total_pop_edu'] = df_edu[edu_cols].sum(axis=1)
        print(f"    - (DEBUG) {path_edu} Loaded successfully.")

        # --- 3. Load Housing Data (02-06.csv) ---
        print(f"  - Loading: {path_house} (Housing)")
        df_house = pd.read_csv(path_house, header=0, encoding='utf-8-sig')
        df_house.columns = df_house.columns.str.strip()
        print(f"    - (DEBUG) {path_house} Original columns: {df_house.columns.to_list()}")
        df_house = df_house.rename(
            columns={
                "地    区": "district",
                "年末总户数\n(户)  \nTotal ouseholds\nat Year-end\n(household)": "households"
            })
        df_house = df_house[df_house['district'] != '全    市'].set_index("district")
        df_house.index = df_house.index.str.strip()
        df_house['households'] = df_house['households'].apply(clean_numeric)
        print(f"    - (DEBUG) {path_house} Loaded successfully.")

        # --- 4. [!! v6 FIX !!] Load Employment Data (02-09.csv) ---
        print(f"  - Loading: {path_emp} (Employment)")
        # [!! v6 FIX !!] Your new file has header=0 and is comma separated
        df_emp = pd.read_csv(path_emp, header=0, encoding='utf-8-sig')
        df_emp.columns = df_emp.columns.str.strip()
        print(f"    - (DEBUG) {path_emp} Original columns: {df_emp.columns.to_list()}")

        # [!! v6 FIX !!] Using column names from the new file (header=0)
        df_emp = df_emp.rename(
            columns={
                "地    区": "district",
                "Employment Population (10000 persons)": "emp_total",
                "第一产业        Primary \nIndustry": "emp_primary",
                "第二产业           Secondary Industry": "emp_secondary",
                "第三产业          Tertiary \nIndustry": "emp_tertiary"
            })

        if 'district' not in df_emp.columns:
            print(f"!! CRITICAL ERROR: 'district' column not found in {path_emp} after renaming.")
            return pd.DataFrame()

        df_emp = df_emp[df_emp['district'] != '全    市'].set_index("district")
        df_emp.index = df_emp.index.str.strip()

        emp_cols_to_check = ["emp_total", "emp_primary", "emp_secondary", "emp_tertiary"]
        for col in emp_cols_to_check:
            if col not in df_emp.columns:
                print(f"!! CRITICAL ERROR: Expected column '{col}' not found in {path_emp}")
                return pd.DataFrame()
            df_emp[col] = df_emp[col].apply(clean_numeric)
        print(f"    - (DEBUG) {path_emp} Loaded successfully.")

    except FileNotFoundError as e:
        print(f"!! CRITICAL ERROR: Census data file not found: {e}")
        traceback.print_exc()
        return pd.DataFrame()
    except KeyError as e:
        print(f"!! CRITICAL ERROR: Expected column not found in census CSV: {e}")
        print("  (Please check the (DEBUG) Original columns output above and ensure the keys in the rename dictionary match exactly)")
        traceback.print_exc()
        return pd.DataFrame()
    except Exception as e:
        print(f"!! CRITICAL ERROR: Unexpected error while loading census data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    print("Macro data loaded and cleaned.")
    print(f"Starting feature generation for {len(all_station_ids)} stations...")
    station_features = []
    missing_districts = set()

    for station_id in all_station_ids:
        if station_id not in station_to_district_map:
            print(f"WARNING: Station '{station_id}' (from '{STATION_CSV_PATH}') not found in '{STATION_MAP_CSV}', skipping.")
            continue

        district = station_to_district_map[station_id]

        if district not in df_age.index or \
                district not in df_edu.index or \
                district not in df_house.index or \
                district not in df_emp.index:
            if district not in missing_districts:
                print(f"WARNING: District '{district}' not found in all census data (age, edu, house, emp).")
                if district not in df_age.index: print(f"  - Missing in: {path_age}")
                if district not in df_edu.index: print(f"  - Missing in: {path_edu}")
                if district not in df_house.index: print(f"  - Missing in: {path_house}")
                if district not in df_emp.index: print(f"  - Missing in: {path_emp}")
                print(f"  (Please check 'NAME_CORRECTIONS' dictionary)")
                missing_districts.add(district)
            continue

        # 1. PopAct_i: From Raster query
        pop_act_i = get_population_from_raster(
            station_id, station_coords_db, raster_src, radius_m=800
        )

        # 2. Inherit socio-economic structure from the district
        stats_age = df_age.loc[district]
        stats_edu = df_edu.loc[district]
        stats_house = df_house.loc[district]
        stats_emp = df_emp.loc[district]

        # 3. Calculate Features
        total_pop_age = stats_age['total_pop_age'] + 1e-6  # Resident population
        age_0_14_pct = stats_age['age_0_14'] / total_pop_age
        age_15_64_pct = stats_age['age_15_64'] / total_pop_age
        age_65_plus_pct = stats_age['age_65_plus'] / total_pop_age

        total_pop_edu = stats_edu['total_pop_edu'] + 1e-6
        edu_primary_pct = (stats_edu['edu_none'] + stats_edu['edu_primary']) / total_pop_edu
        edu_junMid_pct = stats_edu['edu_junMid'] / total_pop_edu
        edu_senHigh_pct = stats_edu['edu_senHigh'] / total_pop_edu
        edu_tertiary_pct = (stats_edu['edu_tertiary_college'] + stats_edu['edu_tertiary_uni'] + stats_edu[
            'edu_tertiary_postgrad']) / total_pop_edu

        households = stats_house['households']
        avg_household_size_i = total_pop_age / (households + 1e-6)

        emp_total = stats_emp['emp_total'] * 10000 + 1e-6  # Employed population (Unit: 10k -> persons)
        emp_primary_pct = (stats_emp['emp_primary'] * 10000) / emp_total
        emp_secondary_pct = (stats_emp['emp_secondary'] * 10000) / emp_total
        emp_tertiary_pct = (stats_emp['emp_tertiary'] * 10000) / emp_total

        # [!! v6 New Feature !!] District Employed Pop / District Resident Pop Ratio
        district_emp_to_pop_ratio = emp_total / total_pop_age

        station_features.append({
            'station_id': station_id, 'district': district, 'PopAct_i': pop_act_i,
            'age_0_14_pct': age_0_14_pct, 'age_15_64_pct': age_15_64_pct, 'age_65_plus_pct': age_65_plus_pct,
            'edu_primary_pct': edu_primary_pct, 'edu_junMid_pct': edu_junMid_pct,
            'edu_senHigh_pct': edu_senHigh_pct, 'edu_tertiary_pct': edu_tertiary_pct,
            'avg_household_size_i': avg_household_size_i,
            'emp_primary_pct': emp_primary_pct,
            'emp_secondary_pct': emp_secondary_pct,
            'emp_tertiary_pct': emp_tertiary_pct,
            'district_emp_to_pop_ratio': district_emp_to_pop_ratio  # [!! v6 New !!]
        })

    print("All station features generated.")
    if not station_features:
        return pd.DataFrame()
    return pd.DataFrame(station_features).set_index('station_id')


# --- 4. Main Execution ---
if __name__ == "__main__":

    print("--- Starting Fully Automated Process ---")

    # Step 1: Automatically load Station -> District Mapping
    print("\n[Step 1/4] Automatically loading Station -> District mapping...")
    station_to_district_map = load_automated_map(STATION_MAP_CSV, NAME_CORRECTIONS)
    if not station_to_district_map:
        print("!! ERROR: Failed to load station-district mapping. Aborting.")
        exit()

    # Step 2: Load Station Coordinates
    print("\n[Step 2/4] Loading station coordinates...")
    station_coords_db, all_station_ids = load_station_data_from_csv(STATION_CSV_PATH)
    if not all_station_ids:
        print("!! ERROR: Failed to load station coordinate data. Aborting.")
        exit()

    # Step 3: Execute Calculations
    print("\n[Step 3/4] Executing population and feature calculations...")
    df_features = pd.DataFrame()
    try:
        with rasterio.open(RASTER_POP_PATH) as raster_src:
            print(f"  - Successfully opened clipped raster file: {RASTER_POP_PATH}")
            df_features = generate_station_level_socio_demographics(
                station_to_district_map=station_to_district_map,
                all_station_ids=all_station_ids,
                path_age=PATH_AGE,
                path_edu=PATH_EDU,
                path_house=PATH_HOUSE,
                path_emp=PATH_EMP,  # [!! New !!]
                raster_src=raster_src,
                station_coords_db=station_coords_db
            )
    except FileNotFoundError:
        print(f"!! ERROR: Raster file '{RASTER_POP_PATH}' not found.")
        print("  Please ensure you have run the clipping script.")
    except Exception as e:
        print(f"!! An unexpected error occurred: {e}")
        traceback.print_exc()  # Print full stack

    # Step 4: Save Results
    print("\n[Step 4/4] Saving results...")
    if not df_features.empty:
        with open(OUTPUT_PICKLE_FILE, 'wb') as f:
            pickle.dump(df_features, f)
        print("\n--- Success! Generated Features (First 5 rows) ---")
        print(df_features.head())
        print(f"\nComplete features saved to: {OUTPUT_PICKLE_FILE}")
    else:
        print("\n!! WARNING: No features were generated.")
        print("  Please check above for 'Region not found' warnings or 'Critical Errors'.")

    print("\n--- Fully Automated Process Finished ---")