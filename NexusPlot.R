library(ggplot2)
library(ggrepel)

scenario_folder = '1km_all_scenarios_dr5'
scenario = 'WWR per cluster'

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
p
ggsave(paste0(scenario_folder, '/', scenario, '/', 'NexusPlot.pdf'), p, width = 7, height = 4, units='in')
