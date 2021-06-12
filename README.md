**Dataset description**

The dataset consists from a wide range of real-world environments sounds, like musical instruments, human sounds, domestic sounds, animal and so on. The categories distribution is unbalanced, the number of sample for each category is different:
![image](https://user-images.githubusercontent.com/71300410/121774414-0a61b400-cb8b-11eb-94b4-667c261266c5.png)

The distribution of the sound files length is also unbalanced:

![image](https://user-images.githubusercontent.com/71300410/121774531-793f0d00-cb8b-11eb-90f5-175e1f9d7249.png)


**What is the goal?**

I want to build a model that learns to analyze the sound pattern and assign it for the right category. I will user two model for achieving the goal, a 2D convolution model and a transformer model.

**The 2D convolution model**

I will build a 2D convolution model according to an architecture I chose.
2D convolution model expact 2D input. Therefor I will need to convert the 1D audio file to a 2D spectrogram.

Example of s random 1D audio - amlitude as function of time:

![image](https://user-images.githubusercontent.com/71300410/121774831-eacb8b00-cb8c-11eb-9e01-bd2da5b8c317.png)

A conversion to a 2D spectogram -  Amplitude as function of frequency and time:

![image](https://user-images.githubusercontent.com/71300410/121774959-7f35ed80-cb8d-11eb-8150-64bd6184318c.png)

I will treat the spectogram as an image, and feed it to the convolution model, that will learn to output the correct category.

Main problem with using convolution model here is that the dataset consist of audio files with various lenght, some are long and some are short. 
When I insert the image to the convolution model, I first shrink the image to fixed dimention, it causes long audio sequences loose alot of information, and it may damage the result. A way to improve it is to set a limit to the audio file length. Long files will still loose information because I cut part of the file, however when it enters the convolution network the amount of information loss won't be significant.

**The Transformer model**

My idea was that audio file composed of sequence, and the transfromer model is good with handeling sequences. I wanted to see how the transformer model perform for this data.
This is calssification problem, so I will only use the encoder module from the original transformer paper. The sequence that will be input to the transformer encoder will be the 1D audio file sequence and all the samples will be padded according to the sequence limit that I'll choose.
The lost of information from the conv2d is not a problem, because this time I use transformer model with a padding mask - it will not take the padded part into onsideration and it will be ignored using the attention mechanism and in the lost calculation. Therefore we don't have to limit the file length and lose information. However, for saving computation time I did choose to limit the audio file to a fixed length.

**Result**

I was able to reach around 50% accuracy in the conv2D model and 40% in the transformer model. 
I'm sure I can get much better result using the convolution model, by choosing different architecture. One problem is that because the image is a spectogram and not "real" image, pretrained conv2d model such as resnet doesn't really fit to my data so if I want to use them I'll need to train again all the layers, and it takes too much time. So for now I'll leave the conv2d like this.

For the transformer model, I've noticed that adding a positinal encoding doesn't change the result, it may suggests that for some reason the sequence for the audio file doesn't matter and can be ignored. It's a little bit weird, therefore I guess I have a problem somewhere in the model or in the data preperation.
