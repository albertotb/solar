library(reticulate)
library(ggplot2)
library(RColorBrewer)
library(ggmap)
library(reshape2)

data = read.csv("results/all.csv")

target_sensor = "DH3"
predictors = c("AP1", "AP4",  "AP5",  "AP6",  "AP7",  "DH1",
               "DH10", "DH11", "DH2",  "DH4",  "DH5",
               "DH6", "DH7",  "DH8",  "DH9" )

plot_all = function(df, name="model"){

  eps = 0.001
  hawaii <- get_stamenmap(bbox = c(left = min(df$Longitude) - eps, bottom = min(df$Latitude) - eps, 
                                   right = max(df$Longitude) + eps, top = max(df$Latitude) + eps), zoom = 15)
  
  
  #myPalette <- colorRampPalette(rev(brewer.pal(12, "Spectral")))
  #sc <- scale_colour_gradientn(colours = myPalette(10), limits=c(0, 0.1))
  
  p = ggmap(hawaii) + geom_point(aes(Longitude, Latitude,color=mae, size=mae), data=df, shape = 16) 
  #
  p = p + geom_text(aes(Longitude, Latitude, label=Location), data=df, size=3, hjust=0.001, vjust=0.001)
  #
  p = p + scale_color_gradient(limits=c(min(df$mae),max(df$mae)))
  #
  p = p + labs(title = name, color='MAE', size="Values") 
  #
  p = p + theme(plot.title = element_text(hjust = 0.5))
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
#
plot_facet = function(df){
  df_m = melt(data, id.vars = c("Location", "Latitude", "Longitude"))
  ## Remove wrong model
  df_m = df_m[df_m$variable != "GPconstant_Conv2D_LSTM_3times",]
  ##
  eps = 0.001
  hawaii <- get_stamenmap(bbox = c(left = min(df_m$Longitude) - eps, bottom = min(df_m$Latitude) - eps, 
                                   right = max(df_m$Longitude) + eps, top = max(df_m$Latitude) + eps), zoom = 15)
  
  
  #myPalette <- colorRampPalette(rev(brewer.pal(12, "Spectral")))
  #sc <- scale_colour_gradientn(colours = myPalette(10), limits=c(0, 0.1))
  
  p = ggmap(hawaii)
  p = p + geom_point(aes(Longitude, Latitude,color=value, size=value), data=df_m, shape = 16) 
  #
  p = p + geom_text(aes(Longitude, Latitude, label=Location), data=df, size=3, hjust=0.001, vjust=0.001)
  #
  p = p + facet_wrap(~ variable, ncol=2)
  #
  p = p + labs(color='MAE', size="Values") 
  #
  p = p + theme(plot.title = element_text(hjust = 0.5))
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
#
plot_net = function(df, target_sensor, predictors){
  target = df[df$Location == target_sensor,]
  
  # To plot the network
  n = length(predictors)
  net <- data.frame(X = c(rep(df$Longitude[df$Location == target_sensor], n), df$Longitude[df$Location %in% predictors]),
                         Y = c(rep(df$Latitude[df$Location == target_sensor], n), df$Latitude[df$Location %in% predictors]) )
  net$grp <- as.factor(rep(1:n, times = 2))
  
  eps = 0.001
  hawaii <- get_stamenmap(bbox = c(left = min(df$Longitude) - eps, bottom = min(df$Latitude) - eps, 
                                   right = max(df$Longitude) + eps, top = max(df$Latitude) + eps), zoom = 15)
  
  
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


models = names(data)[4:length(names(data))]
for(i in 1:length(models)){
  mod = as.character(models[i])
  df = data[, c("Location", "Latitude", "Longitude", mod )]
  names(df)[ncol(df)] = "mae"
  p = plot_all(df, name = mod)
  path = paste0("img/", mod, ".png")
  ggsave(p, filename = path, device="png", dpi = 600)
}




