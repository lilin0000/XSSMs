#iMMStall.packages("terra")
#iMMStall.packages("terra", type = "source")
#iMMStall.packages("sf")

#library(terra)
library(raster)
#library(sf)
dt <- read.csv("D:/Users/lily/Desktop/Sample data/results/dt_b_1k.csv")
dt <- dt[,c("longititude","latitude","dt_b_1k")]
predraster <- rasterFromXYZ(dt)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/dt_b_1k.tif", format="GTiff", overwrite=TRUE)


nr <- read.csv("D:/Users/lily/Desktop/Sample data/results/nr_b_1k.csv")
nr <- nr[,c("longititude","latitude","nr_b_1k")]
predraster <- rasterFromXYZ(nr)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/nr_b_1k.tif", format="GTiff", overwrite=TRUE)


gb <- read.csv("D:/Users/lily/Desktop/Sample data/results/gb_b_1k.csv")
gb <- gb[,c("longititude","latitude","gb_b_1k")]
predraster <- rasterFromXYZ(gb)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/gb_b_1k.tif", format="GTiff", overwrite=TRUE)


rf <- read.csv("D:/Users/lily/Desktop/Sample data/results/rf_b_1k.csv")
rf <- rf[,c("longititude","latitude","rf_b_1k")]
predraster <- rasterFromXYZ(rf)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/rf_b_1k.tif", format="GTiff", overwrite=TRUE)


vr <- read.csv("D:/Users/lily/Desktop/Sample data/results/vr_b_1k.csv")
vr <- vr[,c("longititude","latitude","vr_b_1k")]
predraster <- rasterFromXYZ(vr)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/vr_b_1k.tif", format="GTiff", overwrite=TRUE)









dt <- read.csv("D:/Users/lily/Desktop/Sample data/results/dt_b_1k_MMS.csv")
dt <- dt[,c("longititude","latitude","dt_b_1k")]
predraster <- rasterFromXYZ(dt)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/dt_b_1k_MMS.tif", format="GTiff", overwrite=TRUE)


nr <- read.csv("D:/Users/lily/Desktop/Sample data/results/nr_b_1k_MMS.csv")
nr <- nr[,c("longititude","latitude","nr_b_1k")]
predraster <- rasterFromXYZ(nr)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/nr_b_1k_MMS.tif", format="GTiff", overwrite=TRUE)


gb <- read.csv("D:/Users/lily/Desktop/Sample data/results/gb_b_1k_MMS.csv")
gb <- gb[,c("longititude","latitude","gb_b_1k")]
predraster <- rasterFromXYZ(gb)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/gb_b_1k_MMS.tif", format="GTiff", overwrite=TRUE)


rf <- read.csv("D:/Users/lily/Desktop/Sample data/results/rf_b_1k_MMS.csv")
rf <- rf[,c("longititude","latitude","rf_b_1k")]
predraster <- rasterFromXYZ(rf)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/rf_b_1k_MMS.tif", format="GTiff", overwrite=TRUE)


vr <- read.csv("D:/Users/lily/Desktop/Sample data/results/vr_b_1k_MMS.csv")
vr <- vr[,c("longititude","latitude","vr_b_1k")]
predraster <- rasterFromXYZ(vr)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/vr_b_1k_MMS.tif", format="GTiff", overwrite=TRUE)






dt <- read.csv("D:/Users/lily/Desktop/Sample data/results/dt_b_1k_SS.csv")
dt <- dt[,c("longititude","latitude","dt_b_1k")]
predraster <- rasterFromXYZ(dt)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/dt_b_1k_SS.tif", format="GTiff", overwrite=TRUE)


nr <- read.csv("D:/Users/lily/Desktop/Sample data/results/nr_b_1k_SS.csv")
nr <- nr[,c("longititude","latitude","nr_b_1k")]
predraster <- rasterFromXYZ(nr)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/nr_b_1k_SS.tif", format="GTiff", overwrite=TRUE)


gb <- read.csv("D:/Users/lily/Desktop/Sample data/results/gb_b_1k_SS.csv")
gb <- gb[,c("longititude","latitude","gb_b_1k")]
predraster <- rasterFromXYZ(gb)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/gb_b_1k_SS.tif", format="GTiff", overwrite=TRUE)


rf <- read.csv("D:/Users/lily/Desktop/Sample data/results/rf_b_1k_SS.csv")
rf <- rf[,c("longititude","latitude","rf_b_1k")]
predraster <- rasterFromXYZ(rf)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/rf_b_1k_SS.tif", format="GTiff", overwrite=TRUE)


vr <- read.csv("D:/Users/lily/Desktop/Sample data/results/vr_b_1k_SS.csv")
vr <- vr[,c("longititude","latitude","vr_b_1k")]
predraster <- rasterFromXYZ(vr)
crs(predraster) <- crs("+init=EPSG:4326")
writeRaster(predraster, "D:/Users/lily/Desktop/Sample data/XSSMs/vr_b_1k_SS.tif", format="GTiff", overwrite=TRUE)

