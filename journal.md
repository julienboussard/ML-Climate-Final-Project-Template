I downloaded datasets containing 3 videos of the Great Barrier Reef, with approximately 10000 images per video, with hand labels of the positions of crown of thornes. I only uploaded the first image of each video to github now as it seems too heavy (git push is not working)

As I was not very familiar with the detection and tracking litterature, I read several papers about it.
For detection, I plan on using a YoLO Neural Network pretrained on COCO dataset, as it seems to be the state-of-the-art and be working for different types of applications. 
For tracking, I plan on using Kalman Filter-like nmethods, and will think of more complex methods. 
I also read several papers on 2d pose estimation, as these models allow to enforce physical constraints into the tracking. 

I'm also thinking of augemnting the training data, as most images look similar and it does not seem too hard to take a crown of thrones, and "blend it" in other images (for example using Poisson blending)
