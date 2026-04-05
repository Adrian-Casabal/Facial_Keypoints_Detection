# Reflection

## Activity Description
Going into this project, I expected to do fairly well considering I have done several computer vision projects now. I wanted to do something more difficult than what I have done before, so instead of doing a simple computer vision project, I decided to do one where I needed to predict focused areas on an image. In the end, I trained a convolutional neural network to accurately predict keypoint positions on face images.

## Technical Decisions
Starting off, I used my own custom convolutional neural network because I wanted to build a complete baseline from scratch. I thought that creating a simple neural network would be good enough to get a good score. I based this thinking on the universal approximation theory, which states that neural networks with a certain structure can, in principle, approximate any continuous function to any desired degree of accuracy. I first started with simple preprocessing techniques, converting raw image strings into 96x96 grayscale arrays and normalizing the pixel values to be between 0 and 1. I also created a mask to track missing values and replaced missing coordinates with the center of the image, normalizing those coordinates. As I progressed through the competition, I began to see the limitations of my custom model. It gave me a good initial score, but it was still fairly bad compared to the scores on the leaderboard. In the following submissions after the first, I implemented more advanced preprocessing techniques such as horizontal flipping and brightness/contrast changes so that the model would be less sensitive to lighting differences across images. This did not initially improve my scores, and I had an idea that it might be because my custom model wasn't fit for the task anymore at that stage. So I switched from using my custom model to using a pretrained HRNet model, known for producing strong results on standard landmark benchmarks. I considered using a baseline ResNet model, but I figured that it would be out of date. After switching to a pretrained model, I got my best score, which put me in the top 20 scores on the leaderboard.

## Contributions
I worked alone so all contributions were made by me. Those contributions include:
- downloading the data and preprocessing it
- making and training a model
- converted the predictions into a kaggle-ready submission

## Quality Assessment
I would say that I did a lot of work for this event. Even though I am knowledgable in neural networks, I was able to learn a lot from this competition, mainly concerning the limitations of different models. If I had to do this event over again I would start with a pretrained model and spend more time fine-tuning parameters and data augmentation techniques.
