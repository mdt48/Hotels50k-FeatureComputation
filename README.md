# Abstract

The Hotels50k project is a large-scale database and image recognition project aimed at
combating sex trafficking. This is done by trying to recognize images from hotel rooms that
contain victims of sex trafficking. Image recognition is an extraordinarily challenging task due to
poor image quality, different cameras, perspectives, and many other factors. In addition to
creating a dataset of over 1 million images from booking websites and crowd-sourced through a
mobile app, the original authors developed a baseline model for image recognition based on the
Hotels50k dataset. In this paper, I will present a novel method of feature extraction and
recognition designed to improve the image recognition accuracy of the Hotels50k project.

# Introduction

There is an extraordinary number of photos taken of victims of sex trafficking online today, and
many of such pictures are taken in hotel rooms. Thus, the ability to recognize hotel rooms from
an image is of great interest and importance when trying to combat sex trafficking. This
relationship is what sparked the idea for the original Hotels50k dataset and image recognition
task. However, image recognition of hotels is a unique challenge in comparison to more generic
image recognition tasks that are placed in one of the following categories (Grauman and Leibe
2011):

1. Basic-level categories (e.g., ‘building’)
2. Specialized categories (e.g., ‘church’)
3. Exact instances (e.g., ‘the Notre-Dame’).


The Hotels50k domain spans the latter two of these generic categories. To learn what a specific
hotel looks like requires the learning of shared features across chains and what features make
certain chains identifiable.

This paper has three main contributions. First, providing a novel method of extracting features
from an image using state-of-the-art semantic segmentation networks. Second, collecting
important data on the similarities and differences amongst semantic classes to facilitate future
work on this problem. Third, proposing a unique method for choosing the "best" result from the
multi-feature kNN model built for this project.

# Methodology

First, I set up a state-of-the-art Resnet50+Pyramid Pooling model here. Several other models
were provided, however, I chose only to experiment with the Resnet50+Pyramid Pooling (PPM)
model due to its high level of reported accuracy and my familiarity with the model. The PPM
model is trained on the ADE20k dataset with 150 semantic classes - an issue considering the
semantically labeled Hotels50k data uses only 27 semantic classes.

To overcome this, I conducted two groups of training and evaluation experiments. The first
group was the model trained on the 150 semantic classes from the ADE20K dataset, and
evaluated on the Hotels50k data.

The Hotels50k data was adapted to these 150 semantic classes by by-hand mapping of Hotels50k
semantic classes to ADE20K semantic classes. To approach the second group, training on the 27


semantic classes from Hotels50k, I adapted the model above utilizing transfer learning. Each
layer of the encoder (Resnet50) was frozen, with the output layers for the decoder (PPM) adapted
to 27 semantic classes. Due to higher evaluation accuracy on the models trained with 150
classes, and because the models trained on 27 classes did not achieve the same separation in
distributions between the closest vector in the same and different semantic classes as in the 150
classes. Only the model trained on 150 classes was used to produce features.

In the original Hotels50k paper, a representative feature was computed over an entire image,
whereas the aim in this project is to create an average feature for each semantic class in each
image, which we call "semantic pooling". "Semantic pooling” is based on the assumption that the
average “bed” feature is similar from hotel to hotel in the same chain. To extract the average
semantic class features from each image, the image was first fed through the Resnet50+Pyramid
Pooling (PPM) semantic segmentation model. The predictions for the image were saved, along
with a 2048-dimensional feature map produced by the ResNet50 encoder. Each semantic class
present in the prediction, whose surface area also totaled over 10% (an arbitrary number) of the
total surface area of the image, was recorded. The coordinates of the location within each image
of each semantic class that met the surface area requirement were interpolated from the
prediction to the 2048-dimensional feature map. The average of the 2048-dimensional vectors
from each semantic class was calculated, giving a representative 2048-dimensional feature for
each semantic class in the image. The average semantic features were serialized and recorded in
an index file.

The semantically pooled features were computed for the entirety of the unconcluded test data set
(17k images), and the semantically pooled features for the training set are in the process of being
computed (~400k/1.1 million done).


In a change from regular kNN, the entire training dataset was not searched through. Instead, for
each testing image, the averaged semantic features were computed in an identical method as
described above for the training images. Then, instead of searching through the entire training
dataset for the closest semantic feature to each semantic feature in the test image, only the
training images with the same semantic features were compared. For example, if a bed and a wall
were found in the query image, the algorithm would conduct a kNN search through all the bed
features in the dataset and then all the wall features averaging the results, rather than the features
for every class. In line with the evaluation protocol created in the original Hotels50k paper, I
evaluated kNN accuracy on chains (Marriot, Days Inn, etc.) and instances (DC Marriot). The
average accuracy for each testing image over all of the semantic classes found in that image was
averaged and recorded. This was repeated for k=1,3,5. The higher values of k (k>10) could not
be computed as there were not enough features in each class.

# Results

In the first set of graphs shown on the right, there is a large separation between the closest of the
same class and the closest of a different class. In the second set on the right, some classes are
more unique to certain hotels. It was hoped that this would be seen naturally in the kNN,
however, in the final table, this is not the case.

# Conclusion

While the results were not as expected, there are several takeaways for continuing work on this
project and explanations for the results. There were several positive takeaways from this project.


First, there is now a method to extract semantic features from images. This method works
accurately and quickly, allowing for the large-scale extraction from datasets with millions of
photos. Second, as displayed in Figure 1, these features have clear separations supporting the
claim that distinct semantic classes are extracted from the images.

This is an important finding, as it gives a basis for the idea that we can search for the closest
feature in the same class to obtain the chain or instance that a query image originated.

However, the results of the kNN do not agree. There are some improvements or changes that
would yield better results in future experiments. First, there was not enough training data. There
was a large amount of unexpected server downtime leading to not all the training data features
being computed. This is an obvious problem, as now some chains and instances are represented
in the training data that are not represented or cannot be used to make an accurate inference.

Second, some classes were underrepresented or overrepresented in the training data. A solution
to this is recomputing features with a lower threshold for total surface area. In the experiments
run, the surface area threshold was set at 10%, meaning a semantic feature must take up 10% of
the picture to be computed and saved. While performing these experiments, it became clear that
even with this low margin, smaller, more distinct semantic classes such as lighting and art pieces
were underrepresented. The large semantic features like floors, walls, and ceilings were
overrepresented. In future experiments, I would reduce this surface area margin to obtain a more
representative distribution of the features.

Third, there needs to be a more robust scoring function to determine which k vectors are the
closest. The project needed a form of kNN that did not just minimize the distance between two
vectors but also considers how unique certain semantic classes are between hotel chains and


instances. This would most likely mean some form of weighting the distance between two
vectors, the unknown test feature and the known training image, by the likelihood that the
semantic class being computed are from the same class and chain(like the graphs to the left).
Instead of computing the Euclidean or cosine distance and taking the chain/instance with the
closest distance, we could look at how likely a semantic class is from a hotel chain and how
distinct that class is from that chain compared to other chains.

In addition to the flaws explored above, other paths would be prudent to explore in future
experiments. First, test different models for semantic segmentation. There were several models
given in the semantic segmentation suite, mentioned previously and they may be able to generate
more distinct features that would aid in searching.

Second, it would be worthwhile to explore the use of only the 27 classes from the Hotels50k
dataset. The 27 classes were not used because they did not achieve the same separation in
distributions between the closest vector in the same and different semantic classes as in the 150
classes. This could be due to the smaller number of semantic classes. There are several classes in
the Hotels50k dataset that could be further broken down into distinct micro classes rather than
more macro classes. Furthermore, more semantically labeled Hotels50k images would need to be
created. There were simply not enough to create a more accurate semantic segmentation module
that could be trained.

Finally, we could adapt what features are being computed. Instead of an average semantic class
feature for each image, a representative semantic class feature is created for each chain and
instance. Even within hotels, there can be variations in lighting and picture angle, which could


lead to different representations from one hotel room to another in the same hotel. Normalizing
the feature over the entire instance or chain would be more useful in catching the variations
within chains and instances.


