# preprocessing.py

from utils.preprocessing_utils import LakeDataPreprocessor
import os

def define_paths():
    # Define all input and output paths here
    sentinel_nc_path = "data/all_lakes_2019_LS.nc"
    carra_nc_path = "data/CARRA_GrIS_Dataset.nc"
    geojson_path = "data/GrIS_lakes.geojson"
    output_dir = "data/processed"
    return sentinel_nc_path, carra_nc_path, geojson_path, output_dir

def define_carra_variables():
    # Define CARRA variables to include, even weak ones (al, sp and sst) for causal discovery.
    # al: albedo, r2: relative humidity, sd: snow depth, sp: surface pressure, sro: runoff, sst: sea surface temperature, t2m: temperature at 2m
    return ["al", "r2", "sd", "sp", "sro", "sst", "t2m"]

def main():
    sentinel_nc_path, carra_nc_path, geojson_path, output_dir = define_paths()
    carra_vars = define_carra_variables()

    processor = LakeDataPreprocessor(
        sentinel_nc_path=sentinel_nc_path,
        carra_nc_path=carra_nc_path,
        geojson_path=geojson_path,
        output_dir=output_dir
    )

    # Step 1: Process and save all lakes
    final_df = processor.process_all_lakes(carra_vars=carra_vars)

    # Step 2: Run sanity check
    csv_path = os.path.join(output_dir, "all_lakes_timeseries.csv")
    sanity_output_dir = "figures/sanity_checks"
    processor.sanity_check_plot(csv_path, output_dir=sanity_output_dir)

if __name__ == "__main__":
    main()
