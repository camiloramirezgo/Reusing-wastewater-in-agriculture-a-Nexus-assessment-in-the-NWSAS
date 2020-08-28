#import required libraries
library(raster)

Data = "nwsas_1km_data - Results"
# Data = "nwsas_10km_data - Results"
# Scenario = "WWR by province"
Scenario = "Free agri water - High pop water"
# Data = "Youssef"
# Scenario = "IrrigationArea"
FilesDirectory <- paste0(Data, "/", Scenario, "/CSV")
fls <- list()

filenames <- list.files(FilesDirectory, pattern="*.gz", full.names=FALSE)
for (i in 1:length(filenames)){
  fls[[i]] <- read.csv(paste0(FilesDirectory,"/",filenames[i]), col.names = c("X","Y",strsplit(filenames[i],".gz")))
  try(dfr <- rasterFromXYZ(fls[[i]], crs = "+proj=lcc +lat_1=33.3 +lat_0=33.3 +lon_0=2.7 +k_0=0.999625769 +x_0=500135 +y_0=300090 +ellps=clrk80 +units=m +no_defs"), silent=TRUE)  #Convert first two columns as lon-lat and third as value                
  writeRaster(dfr, file = paste0(Data, "/", Scenario, "/Rasters/",strsplit(filenames[i],".gz"),".tif"), overwrite = TRUE)
}

# centroids = read.csv("nwsas_1km_data - Results/WWR per cluster/CSV/Centroid.gz")
# write.csv(centroids, 'centroids.csv')
