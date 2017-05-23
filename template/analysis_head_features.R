Features = read.table("../Desktop/Lectures/Semester_2/Signal_image_processing/Homeworks/ImageProcessing_TeamProject/template/features.tsv", sep="\t", header=TRUE)

library(ggfortify)
features = Features[-c(20,8),] # remove lines with NA
rownames(features) = features[,c(1)]

# PCA
pca = prcomp(features[,c(2,3,4)], scale. = T)
autoplot(pca, data = features, colour = 'Direction_sight', label = T)

# Kmeans
set.seed(42)
kmeans = kmeans(features[,c(2,3,4)], 3)
autoplot(kmeans, data = features, label = TRUE)
