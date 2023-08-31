library(ggplot2)
library(tidyverse)

# Input: ground truth
ground_truth_ria_formosa <- read.csv('./data/processed/datasets/ria_formosa_truth_data.csv')
colnames(ground_truth_ria_formosa)

# Output
spectral_signature_ria_formosa_3class <- './reports/spectral_signatures/ria_formosa_spectral_signatures_3class_R.png'
spectral_signature_ria_formosa_2class <- './reports/spectral_signatures/ria_formosa_spectral_signatures_2class_R.png'

features <- c('blue', 'green', 'red', 'nir', 'ndvi', 'dem')

ground_truth_ria_formosa <- transform(ground_truth_ria_formosa,
  for (feature in features) {
    feature <- as.numeric(feature)
  }) %>% pivot_longer(
    cols = features,
    names_to = "var",
    values_to = "value"
  ) %>% mutate(var=factor(var, levels = features))


# SPECTRAL SIGNATURES FOR 3 CLASSES

p <- ggplot(ground_truth_ria_formosa) +
    geom_density(
      aes(x = value, group = seagrass_class, fill = factor(seagrass_class)
      ),
      alpha = 0.7) +
    labs(
      title = "Density distribution of bands values by seagrass class"
    ) +
    facet_wrap(vars(var), scales = "free") +
      scale_fill_brewer(type="qual",
      name = "Seagrass class",
      labels = c("Seagrass absence",
              "Intertidal seagrass",
              "Subtidal seagrass"))

png(filename=spectral_signature_ria_formosa_3class,width=8, height=4, units="in", res=300)
plot(p)
dev.off()

# SPECTRAL SIGNATURES FOR 2 CLASSES 
ground_truth_ria_formosa_s <- filter(ground_truth_ria_formosa, seagrass_class != "0")

r <- ggplot(ground_truth_ria_formosa_s) +
      geom_density(
        aes(x = value, group = seagrass_class, fill = factor(seagrass_class)
        ),
        alpha = 0.7) +
      labs(
        title = "Density distribution of bands values by seagrass class"
      ) +
      facet_wrap(vars(var), scales = "free") +
        scale_fill_brewer(type="qual",
        name = "Seagrass class",
        labels = c("Intertidal seagrass",
                "Subtidal seagrass"))

png(filename=spectral_signature_ria_formosa_2class,width=8, height=4, units="in", res=300)
plot(r)
dev.off()
