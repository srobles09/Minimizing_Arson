import pandas as pd
import numpy as np
import datetime
from geopy import distance
import geopandas as gpd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"
from scipy.optimize import linear_sum_assignment

### Read in data
crime_loc = 'https://www.denvergov.org/media/gis/DataCatalog/crime/csv/crime.csv'
#firestn_loc = 'https://www.denvergov.org/media/gis/DataCatalog/fire_stations/csv/fire_stations.csv'
firestn_loc = 'C:/Users/Sandy Oaks/Documents/Grad School/F20_MATH-5593/Project/fire_stations.csv'
firedisctrict_loc = 'https://www.denvergov.org/media/gis/DataCatalog/fire_districts/csv/fire_districts.csv'

crime = pd.read_csv(crime_loc)
fire_stns = pd.read_csv(firestn_loc)
fire_districts = pd.read_csv(firedisctrict_loc)


### Data Cleansing

# Crime
crime = crime[crime.OFFENSE_CATEGORY_ID=='arson']
crime.drop(['INCIDENT_ID', 'OFFENSE_ID','OFFENSE_CODE','OFFENSE_CODE_EXTENSION','FIRST_OCCURRENCE_DATE','LAST_OCCURRENCE_DATE','IS_CRIME','IS_TRAFFIC'], axis=1,inplace=True)
crime['REPORTED_DATE']= pd.to_datetime(crime['REPORTED_DATE']).dt.date
crime.reset_index(inplace=True,drop=True)

crime.OFFENSE_TYPE_ID.value_counts() # QC Point
#crime['coor'] = list(zip(crime['GEO_LAT'],crime['GEO_LON']))

# Fire Stations
fire_stns.drop(['ADDRESS_ID','HOUSE_PHONE','EMERGENCY_PHONE'], axis=1, inplace=True)


def distancer(df, coor):
    coords_1 = (df['GEO_LAT'], df['GEO_LON'])
    return distance.distance(coor,coords_1).miles



#Distances
for i in range(fire_stns.shape[0]):
    new_col_name = fire_stns.STATION_NUM.iloc[i]
    coor_stns = (fire_stns.GEO_LAT.iloc[i],fire_stns.GEO_LON.iloc[i])
    crime[new_col_name] = crime.apply(distancer,coor=coor_stns, axis=1)

#crime.to_csv("C:/Users/Sandy Oaks/Documents/Grad School/F20_MATH-5593/Project/crime_forLP.csv")


## ---------------- MINIMIZATION PROBLEM ---------------- ##
cost = crime.copy(deep=True)
cost.drop(['OFFENSE_TYPE_ID','OFFENSE_CATEGORY_ID','INCIDENT_ADDRESS','GEO_X','GEO_Y','GEO_LON','GEO_LAT','DISTRICT_ID','PRECINCT_ID','NEIGHBORHOOD_ID'], axis=1,inplace=True)


fs_dict ={c: i for i, c in enumerate(cost.drop(['REPORTED_DATE'],axis=1).columns)} #Tie index back to fire station
new_dict = {}
for k, v in fs_dict.items():
    new_dict[v] = k
fs_dict = new_dict; del new_dict

arson_df_list = []
for i in cost.REPORTED_DATE.unique():
    # Set up the data for date i
    tmp_cost = cost[cost['REPORTED_DATE']==i].copy(deep=True)
    tmp_cost.drop(['REPORTED_DATE'],axis=1,inplace=True)
    the_arson_indexes = tmp_cost.index #1 to many
    tmp_cost = np.array(tmp_cost)
    
    # Unbalanced Assignment Problem
    row_ind, col_ind = linear_sum_assignment(tmp_cost)
    tmp_cost[row_ind, col_ind].sum() # Total distance traveled
    

    # Append the data into main list    
    temp_df = pd.DataFrame(np.transpose(np.array([the_arson_indexes, col_ind])))
    temp_df["date"] = i
    arson_df_list.append(temp_df)
#End of loop

## Get final dataframe
optimized_arson = pd.concat(arson_df_list)
#optimized_arson.reset_index(inplace=True, drop=True)
optimized_arson.rename(columns={0: "arson_index", 1: "firestn_index"},inplace=True)
optimized_arson['firestation'] = optimized_arson['firestn_index'].map(fs_dict) 
crime['arson_index'] = crime.index

optimized_arson = pd.merge(optimized_arson.drop(['firestn_index'],axis=1),crime[['arson_index','OFFENSE_TYPE_ID','INCIDENT_ADDRESS','DISTRICT_ID','PRECINCT_ID','NEIGHBORHOOD_ID']],how='inner',on='arson_index')

## ---------------- HOW MANY TIMES DOES THE FIRE STATION EXIST OUTSIDE OF THE DISTRICT? ---------------- ##
optimized_arson.rename(columns={'DISTRICT_ID': 'arson_district'},inplace=True)
optimized_arson = pd.merge(optimized_arson, fire_stns[['STATION_NUM','DISTRICT']],how='left',left_on='firestation',right_on='STATION_NUM')
optimized_arson['equal_district'] = np.where(optimized_arson["arson_district"] == optimized_arson["DISTRICT"], True, False)

#optimized_arson.to_csv("C:/Users/Sandy Oaks/Documents/Grad School/F20_MATH-5593/Project/optimized_arson.csv")

## ---------------- VISUALIZE CRIME ---------------- ##

crime.hist(by=crime.OFFENSE_TYPE_ID)

gdf = gpd.read_file('C:/Users/Sandy Oaks/Documents/Grad School/F20_MATH-5593/Project/crime.shp')

pub_token = 'pk.eyJ1Ijoic3JvYmxlczA5IiwiYSI6ImNraHd3NDR1YjAwcXIyem96ZXZyOHByenYifQ.uWk1hO1nNQCJJ8zNQsHpvg'
px.set_mapbox_access_token(pub_token)
fig = px.scatter_mapbox(crime, lat="GEO_LAT", lon="GEO_LON",     color="OFFENSE_TYPE_ID",
                  color_continuous_scale=px.colors.cyclical.IceFire, zoom=11)
fig.scatter_mapbox(fire_stns, lat="GEO_LAT", lon="GEO_LON",     color="DISTRICT")
fig.update_layout(autosize=True, hovermode='closest',
                  mapbox = {'accesstoken': pub_token, 'style': "basic"})
fig.show()

crime['year'] = pd.DatetimeIndex(crime['REPORTED_DATE']).year
#crime['year'] = int(crime['year'])
fig2 = px.scatter_mapbox(crime, lat="GEO_LAT", lon="GEO_LON",     color='year',zoom=11)
fig2.show()