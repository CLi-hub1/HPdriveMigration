# %% [markdown] Heat-related policy data preprocessing
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

### Filter US heat-related policies and link to US counties
import numpy as np
def create_point(x):
    if isinstance(x, str):
        try:
            return Point(float(x.split()[1][1:]), float(x.split()[2][:-1]))
        except (IndexError, ValueError):
            return np.nan
    else:
        return np.nan
df = pd.read_excel(r'filepath\2017-2020CP.xlsx',sheet_name='Sheet1')
us_counties = gpd.read_file(r'filepath\tl_2019_us_county.shp')
df = df[df['Country'] == 'USA']
dfUSAhp = df.copy()
dfUSAhp['City Location'] = dfUSAhp['City Location'].apply(create_point)
dfUSAhp.dropna(subset=['City Location'], inplace=True)
# Create GeoDataFrame
gdf = gpd.GeoDataFrame(dfUSAhp, geometry='City Location')
points_in_counties = gpd.sjoin(gdf, us_counties, how='left', op='within')
# Add county FIPS and county names as last two columns
dfUSAhp['GEOID'] = points_in_counties['GEOID']
dfUSAhp['County'] = points_in_counties['NAME']
dfUSAhp.dropna(subset=['GEOID'], inplace=True)
dfUSAhp.to_excel(r'filepath\MP.xlsx', index=False)


# %% [markdown] built-up rate Preprocessing
import geopandas as gpd
from rasterstats import zonal_stats
import pandas as pd
import rasterio
import numpy as np

# Read shapefile
shapefile = r'filepath\UScounty.shp'
gdf = gpd.read_file(shapefile)

# Check coordinate system
print("Shapefile CRS:", gdf.crs)
# Raster file list
raster_files = [
    r'filepath\2016_impervious.img',
    r'filepath\2019_impervious.img',
    r'filepath\2021_impervious.img'
]
# Process each raster file
for img_file in raster_files:
    # Zonal statistics
    stats = zonal_stats(gdf, img_file, stats=['mean', 'std'], nodata=-9999)
    # Extract statistics and store in DataFrame
    for idx, stat in enumerate(stats):
        gdf.at[idx, f'mean_{img_file}'] = stat['mean']
        gdf.at[idx, f'std_{img_file}'] = stat['std']
df = gdf.drop('geometry', axis=1).copy()
# Export to Excel
gdf.to_excel(r'filepath\built-uprate.xlsx', index=False)
print("Exported zonal statistics with shapefile info to zonal_statistics_with_shp_info.xlsx")

# %% [markdown] Build-up Rate Preprocessing
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_excel(r'filepath\built-uprate.xlsx')

# Row-wise fitting and prediction
for index, row in df.iterrows():
    x_train = [1, 4, 6]  # X-coordinates
    y_train = [row['2016MEAN'], row['2019MEAN'], row['2021MEAN']]  # Y-coordinates

    # Perform OLS regression
    x_train = np.array(x_train).reshape(-1, 1)
    model = LinearRegression().fit(x_train, y_train)

    # Predict and fill 2017, 2018, 2020 values
    for col, x_pred in zip(['2017MEAN', '2018MEAN', '2020MEAN'], [2, 3, 5]):
        if pd.isnull(row[col]):  # Only fill missing values
            y_pred = model.predict(np.array(x_pred).reshape(1, -1))[0]
            df.at[index, col] = y_pred

df.to_excel(r'filepath\BUrate.xlsx', index=False)

# %% [markdown] WBGT Data Processing
import pandas as pd
## Generate three heat-related metrics from raw data: temperature, WBGT, UTCI
# Define input/output paths
input_file = r'filepath\Heatvars_County_2000-2020_v1.2.csv'
# Specify chunk size
chunksize = 10000
# Chunk reading and filtering
chunks = []
columns_to_keep = ['StCoFIPS', 'WBGTmax_C'] ##'Tmean_C', 'WBGTmean_C', 'UTCImean_C'
for chunk in pd.read_csv(input_file, chunksize=chunksize, usecols=['Date'] + columns_to_keep):
    filtered_chunk = chunk[chunk['Date'].astype(str).str.startswith('2020')]
    chunks.append(filtered_chunk)
# Merge chunks
filtered_data = pd.concat(chunks)
df_pivoted = filtered_data.pivot(index='StCoFIPS', columns='Date', values='WBGTmax_C').reset_index()
# Calculate days with WBGT >=30
df_pivoted['over30COUNT'] = df_pivoted.iloc[:, 1:].applymap(lambda x: 1 if x >= 30 else 0).sum(axis=1)
start_col = '2020-06-01'
end_col = '2020-08-31'
start_idx = df_pivoted.columns.get_loc(start_col)
end_idx = df_pivoted.columns.get_loc(end_col)
# Calculate June-August average
df_pivoted['meanWBGT_6-8'] = df_pivoted.iloc[:, start_idx:end_idx + 1].mean(axis=1)
dffiltered = df_pivoted.iloc[:, [0, -2, -1]].copy()
dffiltered.to_csv(r'filepath\WBGT2020.csv', index=False)

# %% [markdown] IV Variable Processing (20-year avg, 20-year max, 5-year avg, 5-year max)
import pandas as pd

countylist = pd.read_excel(r'filepath\2015_USclimatebrief_countylevel.xlsx')

tem = pd.read_csv(r'filepath\meantem2010-2014.csv')
tem['Value'] = tem['Value'].astype(float)
DF1 = tem.groupby('ID').mean().reset_index()
# Remove "-" in "ID" column
DF1['ID'] = DF1['ID'].str.replace('-', '')

# Extract state abbreviations
DF1['State_abb'] = DF1['ID'].str[:2]
DF1['State_FIPS'] = DF1['State_abb'].apply(lambda x: countylist.loc[countylist['State_abb'] == x, 'State_FIPS'].values[0] if x in countylist['State_abb'].values else None)
DF1.to_excel(r'filepath\meantem2010-2014.xlsx', index=False)

# %% [markdown] Determine County Regions
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

### Match GEOID and determine county coordinates
df = pd.read_excel(r'filepath\AAAmergenet.xlsx')
coords = gpd.read_file(r'filepath\UScounty_basemap(Removefore0&5lakes).shp')
df['GEOID'] = df['GEOID'].astype(str)
df = pd.merge(df, coords[['GEOID', 'INTPTLAT', 'INTPTLON']], on='GEOID', how='left')
df.to_excel(r'filepath\AAAmergenet.xlsx', index=False)

### Link to US regions
# Convert coordinates to Point objects
df = pd.read_excel(r'filepath\AAAmergenet.xlsx')
us_regions= gpd.read_file(r'filepath\cb_2018_us_region_500k.shp')
df['GEOID'] = df['GEOID'].astype(int)
df['INTPTLON'] = df['INTPTLON'].astype(float)
df['INTPTLAT'] = df['INTPTLAT'].astype(float)
geometry = [Point(xy) for xy in zip(df['INTPTLON'], df['INTPTLAT'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)
result = gpd.sjoin(gdf, us_regions, how="left", op='within')
# Extract values
region_values = result['REGIONCE']
df['REGION'] = region_values
df.to_excel(r'filepath\AAAmergenet.xlsx', index=False)

## Convert column to numeric
df = pd.read_excel(r'filepath\AAAmergenet.xlsx')
df['REGION'] = df['REGION'].astype(int)
df.to_excel(r'filepath\AAAmergenet.xlsx', index=False)

# %% [markdown] Generate SI Table S1 (County List with State and Region)
import pandas as pd
AD = pd.read_excel(r'filepath\AAAmergenet_new.xlsx')
AD = AD[['GEOID', 'REGION']]
ADq = AD.groupby('GEOID').mean().reset_index()

ST = pd.read_excel(r'filepath\2015_USclimatebrief_countylevel.xlsx')
ST = ST[['County_name', 'County_FIPS']]
ST = ST.rename(columns={'County_FIPS': 'GEOID'})

STq = ADq.merge(ST, on='GEOID', how='left')

# Map REGION values
region_map = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
STq['REGION'] = STq['REGION'].map(region_map)

# Split County_name column
STq[['County', 'State']] = STq['County_name'].str.split(', ', expand=True)

# Reorder columns
cols = list(STq.columns)
cols.remove('REGION')
cols.append('REGION')
STq = STq[cols]

STq = STq[['GEOID', 'County', 'State', 'REGION']]
STq = STq.dropna()
STq.to_excel(r'filepath\SI_tableS1.xlsx', index=False)