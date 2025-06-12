# utils/preprocessing_utils.py

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.spatial import KDTree
from sklearn.preprocessing import LabelEncoder

class LakeDataPreprocessor:
    def __init__(self, sentinel_nc_path, carra_nc_path, geojson_path, output_dir):
        self.sentinel_path = sentinel_nc_path
        self.carra_path = carra_nc_path
        self.geojson_path = geojson_path
        self.output_dir = output_dir
        self.time_range = pd.date_range("2019-01-01", "2019-12-31")  # Full year
        os.makedirs(self.output_dir, exist_ok=True)


    def load_data(self):
        # Load Sentinel, CARRA, and GeoJSON datasets
        print("Loading Sentinel, CARRA, and GeoJSON datasets...")
        self.sentinel_ds = xr.open_dataset(self.sentinel_path)
        self.carra_ds = xr.open_dataset(self.carra_path)
        self.gdf = gpd.read_file(self.geojson_path)

        self.lat_grid = self.carra_ds["latitude"].values
        self.lon_grid = self.carra_ds["longitude"].values

        coords = np.vstack([self.lat_grid.ravel(), self.lon_grid.ravel()]).T
        self.kdtree = KDTree(coords)
        print("Datasets loaded successfully.")


    def get_nearest_grid_idx(self, point):
        # Find nearest CARRA grid (y,x) to lake centroid
        dist, idx = self.kdtree.query([point.y, point.x])
        y, x = np.unravel_index(idx, self.lat_grid.shape)
        return y, x


    def find_valid_nearest_value(self, varname, center_y, center_x, max_search_radius=5):
        """
        Attempts to find a non-NaN spatial location near (center_y, center_x) for a given variable.
        Searches first valid_time slice. Expands search radius if necessary.
        """
        # Use the first time slice to check validity
        first_timestep = self.carra_ds[varname].isel(valid_time=0).values  # Shape: (y, x)

        # print(f"\n➔ Finding valid '{varname}' near (y={center_y}, x={center_x})...")

        for radius in range(max_search_radius + 1):
            y_min = max(center_y - radius, 0)
            y_max = min(center_y + radius + 1, first_timestep.shape[0])
            x_min = max(center_x - radius, 0)
            x_max = min(center_x + radius + 1, first_timestep.shape[1])

            window = first_timestep[y_min:y_max, x_min:x_max]
            valid_mask = ~np.isnan(window)

            if np.any(valid_mask):
                dy, dx = np.argwhere(valid_mask)[0]
                new_y = y_min + dy
                new_x = x_min + dx
                # print(f"    ✔ Found valid point at (y={new_y}, x={new_x}) within radius {radius} pixels.")
                return new_y, new_x

            # print(f"    ✖ No valid value at radius={radius}, expanding search...")

        # print(f"    ⚠ No valid point found after {max_search_radius} pixels. Keeping original (y={center_y}, x={center_x}).")
        return center_y, center_x


    def extract_carra_features(self, centroid, var_list):
        y, x = self.get_nearest_grid_idx(centroid)
        ts = {}
        carra_time = pd.to_datetime(self.carra_ds['valid_time'].values)

        for var in var_list:
            if var in self.carra_ds:
                # Find nearest non-NaN if necessary
                if np.isnan(self.carra_ds[var][0, y, x].values):
                    y, x = self.find_valid_nearest_value(var, y, x)

                data = self.carra_ds[var][:, y, x].values
                series = pd.Series(data, index=carra_time)

                # Sum runoff, average other features daily
                if var in ["sro"]:
                    daily_series = series.resample('1D').sum()
                else:
                    daily_series = series.resample('1D').mean()

                ts[var] = daily_series
            else:
                print(f"Warning: {var} not found in CARRA dataset.")
        return ts


    def extract_sentinel_features(self, lake_idx):
        # Extract and resample Sentinel and Landsat features to daily
        time = pd.to_datetime(self.sentinel_ds["time"].values)
        features = {
            "HV_lake": pd.Series(self.sentinel_ds["HV_lake"][lake_idx].values, index=time),
            "HV_out": pd.Series(self.sentinel_ds["HV_out"][lake_idx].values, index=time),
            "S2_water": pd.Series(self.sentinel_ds["S2_water"][lake_idx].values, index=time),
            "S2_zenith": pd.Series(self.sentinel_ds["S2_zenith"][lake_idx].values, index=time),
            "LS_water": pd.Series(self.sentinel_ds["LS_water"][lake_idx].values, index=time),
            "LS_zenith": pd.Series(self.sentinel_ds["LS_zenith"][lake_idx].values, index=time),
        }

        daily_features = {k: v.resample('1D').mean() for k, v in features.items()}
        return daily_features


    def interpolate_smooth(self, df):
        # Interpolate missing values and smooth with 12-day rolling median
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.rolling(window=12, center=True, min_periods=1).median()
        return df


    def process_all_lakes(self, carra_vars):
        all_dfs = []
        self.load_data()

        sentinel_ids = self.sentinel_ds["ids"].values.astype(str)
        print(f"Processing {len(self.gdf)} lakes...")

        count = 0               # counter for demo dataset

        for idx, lake in self.gdf.iterrows():
            try:
                lake_id = str(lake["new_id"])

                matches = np.where(sentinel_ids == lake_id)[0]
                if len(matches) == 0:
                    raise ValueError(f"Lake new_id {lake_id} not found in Sentinel dataset")
                lake_idx = matches[0]

                centroid = lake["geometry"].centroid

                sentinel_ts = self.extract_sentinel_features(lake_idx)
                carra_ts = self.extract_carra_features(centroid, carra_vars)

                merged = {**sentinel_ts, **carra_ts}
                df = pd.DataFrame(merged).reindex(self.time_range)
                df = self.interpolate_smooth(df)

                # Add static lake attributes
                df["lake_id"] = lake_id
                df["label"] = lake["label"]
                df["region"] = lake["region"]
                df["elevation"] = lake["elevation"]
                df["area"] = lake["area"]
                df["year"] = lake["year"]

                all_dfs.append(df)

                if (idx + 1) % 20 == 0:
                    print(f"Processed {idx + 1}/{len(self.gdf)} lakes...")

            except Exception as e:
                print(f"Skipping lake {lake.get('new_id', 'UNKNOWN')} due to error: {e}")
                continue
            
            # count += 1
            # if count == 10:
            #     break

        if not all_dfs:
            raise ValueError("No lakes were successfully processed.")

        final_df = pd.concat(all_dfs)
        output_path = os.path.join(self.output_dir, "all_lakes_timeseries.csv")
        final_df.to_csv(output_path)
        print(f"Saved processed dataset to: {output_path}")
        return final_df


    def sanity_check_plot(self, csv_path, output_dir=None):
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)

        lake_ids = ["CW2019_1524", "SW2019_223", "CW2019_1707"]  # Representative lakes
        carra_features = ["t2m", "r2"]  # Climate variables

        for lake_id in lake_ids:
            lake_df = df[df['lake_id'] == lake_id].copy()

            if lake_df.empty:
                print(f"Lake ID {lake_id} not found!")
                continue
            if not {"HV_lake", "HV_out"}.issubset(lake_df.columns):
                print(f"Warning: HV_lake or HV_out missing for {lake_id}")
                continue
            if "S2_water" not in lake_df.columns:
                print(f"Warning: S2_water missing for {lake_id}")
                continue

            lake_df["HV_anom"] = lake_df["HV_lake"] - lake_df["HV_out"]
            lake_label = lake_df["label"].iloc[0] if "label" in lake_df.columns else "unknown"

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Left axis → HV anomaly
            hv_line, = ax1.plot(
                lake_df.index, lake_df["HV_anom"], color='black', label='HV anomaly', linewidth=1.8
            )
            ax1.set_ylabel('Backscatter Difference (dB)', color='black', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='black', labelsize=10)

            # Right axis → S2_water
            ax2 = ax1.twinx()
            s2_scatter = ax2.scatter(
                lake_df.index, lake_df["S2_water"], color='red', s=20, alpha=0.8
            )
            ax2.set_ylabel('Water Coverage (%)', fontsize=12)
            ax2.set_ylim(-0.05, 1.05)
            ax2.invert_yaxis()
            ax2.tick_params(axis='y', labelsize=10)
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

            # Third axis → Climate variables
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("outward", 60))
            climate_lines = []
            climate_colors = ["blue", "green"]

            for feature, color in zip(carra_features, climate_colors):
                if feature in lake_df.columns:
                    line, = ax3.plot(
                        lake_df.index, lake_df[feature],
                        label=feature,
                        color=color,
                        linestyle='--',
                        linewidth=1.8,
                        alpha=0.9
                    )
                    climate_lines.append(line)

            ax3.set_ylabel('Climate Variables', fontsize=12)
            ax3.tick_params(axis='y', labelsize=10)

            # Title
            plt.title(f"Lake ID: {lake_id} | Label: {lake_label}", fontsize=16, pad=20)

            # Manual clean legend
            custom_handles = [
                Line2D([0], [0], color='black', lw=2, label=r'$HV_{anom}$'),
                Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=6, label=r'$p_{water}$')
            ] + climate_lines

            fig.legend(
                handles=custom_handles,
                loc="lower right",        # ← Move legend inside lower-right
                bbox_to_anchor=(0.85, 0.20),  # Fine tune the position (x=85%, y=20%)
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                fontsize=11,
                markerscale=1.2
            )

            fig.tight_layout()

            # Save
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"{lake_id}_timeseries.png"), dpi=300)
                plt.close()
                print(f"Saved plot for lake {lake_id}")
            else:
                plt.show()


    def normalize_features(self, df, feature_cols):
        df = df.copy()
        for feat in feature_cols:
            df[feat] = (df[feat] - df[feat].mean()) / df[feat].std()
        return df


    def build_lake_tensor(self, df, feature_cols, label_col='label', seq_len=365):
        """
        Reshape daily time-series data into lake-level sequences.

        Args:
            df: Combined daily DataFrame with repeated lake entries
            feature_cols: List of dynamic causal features
            label_col: Column name for lake label
            seq_len: Expected length of each lake time series

        Returns:
            X: np.ndarray of shape (num_lakes, seq_len, num_features)
            y: np.ndarray of shape (num_lakes,)
            label_encoder: fitted LabelEncoder
        """
        df = df.sort_values(['lake_id', 'date'])
        lake_ids = df['lake_id'].unique()
        X = []
        y = []
        for lake_id in lake_ids:
            lake_df = df[df['lake_id'] == lake_id]
            if len(lake_df) != seq_len:
                continue
            X.append(lake_df[feature_cols].values)
            y.append(lake_df[label_col].iloc[0])
        X = np.stack(X)
        y = np.array(y)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        return X, y_encoded, label_encoder
