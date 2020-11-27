library(ggplot2)
library(ggrepel)

scenario_folder = '1km_all_scenarios_dr5'
scenario = 'Baseline'

df = read.csv(paste0(scenario_folder, '/', scenario, '/', 'DataForNexusPlot.csv'))

p <- ggplot(df, aes(x=(IrrigationWaterAverage + PopulationWaterAverage)/1000000,
                    y=(FinalAveragePumpingEnergy + FinalAverageDesalinationEnergy + FinalAverageTreatmentEnergy)/1000000,)) + 
            geom_point(aes(size=TDS, fill=GroundwaterDepth), alpha=0.8, stroke=0.5, shape=21) + 
            geom_text_repel(data=subset(df, IrrigationWaterAverage+PopulationWaterAverage>100000000), aes(label = Cluster), alpha=0.5,
                            box.padding   = 0.1, 
                            point.padding = 0.3,
                            segment.color = 'grey50')  +
            scale_fill_viridis_c() +  theme_minimal() + 
            labs(x=expression(Water (Mm^3/yr)), y='Energy (GWh/yr)', fill='Depth to groundwater (m)',
                 size='TDS content (mg/l)')

ggsave(paste0(scenario_folder, '/', scenario, '/', 'NexusPlotWithReuse.pdf'), p, width = 7, height = 4, units='in')

p <- ggplot(df, aes(x=(IrrigationWaterAverage + PopulationWaterAverage)/1000000,
                    y=(IrrigationEnergyTotal + PopulationEnergyTotal)/1000000,)) + 
  geom_smooth(method = "lm", se=FALSE, size=0.4, alpha=0.8, color='black') +
  geom_point(aes(size=TDS, fill=GroundwaterDepth), alpha=0.8, stroke=0.5, shape=21) + 
  geom_text_repel(data=subset(df, IrrigationWaterAverage+PopulationWaterAverage>100000000), aes(label = Cluster), alpha=0.5,
                  box.padding   = 0.1, 
                  point.padding = 0.3,
                  segment.color = 'grey50') +
  scale_fill_viridis_c() +  theme_minimal() + 
  labs(x=expression(Water (Mm^3/yr)), y='Energy (GWh/yr)', fill='Depth to groundwater (m)',
       size='TDS content (mg/l)')

ggsave(paste0(scenario_folder, '/', scenario, '/', 'NexusPlot.pdf'), p, width = 7, height = 4, units='in')

df = read.csv(paste0(scenario_folder, '/', scenario, '/', 'DataForSavingsPlot.csv'))

p <- ggplot(df, aes(x=IrrigationAverageReusedWater/1000000,
                    y=PopulationAverageReusedWater/1000000,)) + 
  geom_point(aes(size=IrrigatedAreaAverage/1000, fill=PopulationAverage/1000), alpha=0.8, stroke=0.5, shape=21) + 
  geom_text_repel(data=subset(df, IrrigationAverageReusedWater>10000000), aes(label = Cluster), alpha=0.5,
                  box.padding   = 0.1, 
                  point.padding = 0.3,
                  segment.color = 'grey50')  +
  scale_fill_viridis_c(option = "plasma") +  theme_minimal() + 
  labs(x=expression('Water saving from irrigation ' (Mm^3/yr)), 
       y=expression('Water savings from population ' (Mm^3/yr)), 
       fill='Population (thousands)',
       size='Irrigated area (kHa)')

ggsave(paste0(scenario_folder, '/', scenario, '/', 'SavingsPlot.pdf'), p, width = 7, height = 4, units='in')
