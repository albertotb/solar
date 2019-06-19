library(reticulate)
library(ggplot2)
library(RColorBrewer)
library(ggmap)

data = read.csv("results/conv1D.csv")
names(data) = c("Location", "conv1D_lon", "conv1D_lat")

info = read.csv("/home/SHARED/SOLAR/data/info_new.csv")
info = info[,c('Location', 'Longitude', 'Latitude')]

df = merge(data, info, by = "Location")

target_sensor = "DH3"
predictors = c("AP1", "AP4",  "AP5",  "AP6",  "AP7",  "DH1",
               "DH10", "DH11", "DH2",  "DH4",  "DH5",
               "DH6", "DH7",  "DH8",  "DH9" )

plot_net = function(df, target_sensor, predictors){
  target = df[df$Location == target_sensor,]
  
  # To plot the network
  n = length(predictors)
  net <- data.frame(X = c(rep(df$Longitude[df$Location == target_sensor], n), df$Longitude[df$Location %in% predictors]),
                         Y = c(rep(df$Latitude[df$Location == target_sensor], n), df$Latitude[df$Location %in% predictors]) )
  net$grp <- as.factor(rep(1:n, times = 2))
  
  eps = 0.001
  hawaii <- get_stamenmap(bbox = c(left = min(df$Longitude) - eps, bottom = min(df$Latitude) - eps, 
                                   right = max(df$Longitude) + eps, top = max(df$Latitude) + eps), zoom = 16)
  
  
  #myPalette <- colorRampPalette(rev(brewer.pal(12, "Spectral")))
  #sc <- scale_colour_gradientn(colours = myPalette(10), limits=c(0, 0.1))
  
  p = ggmap(hawaii) + geom_point(aes(Longitude, Latitude), data=df, shape = 16, color="grey34", size=2) 
  #
  p = p + geom_line(aes(X, Y, group = grp), data=net, color="grey34", linetype = "dashed")
  #
  p = p + scale_color_gradient2(limits=c(0,0.1))
  #
  p = p + geom_point(aes(x=Longitude, y=Latitude, size=conv1D_lat, color=conv1D_lat), data=target) 
  #
  p = p + labs(color='MAE', size="Value") 
  #
  p = p + xlab("Longitude") + ylab("Latitude")
  #
  p = p + theme(axis.line=element_blank(),
                axis.text.x=element_blank(),
                axis.text.y=element_blank(),
                axis.ticks=element_blank(),
                #axis.title.x=element_blank(),
                #axis.title.y=element_blank(),
                #legend.position="none",
                panel.background=element_blank(),
                panel.border=element_blank(),
                panel.grid.major=element_blank(),
                panel.grid.minor=element_blank(),
                plot.background=element_blank())
  return(p)
}
