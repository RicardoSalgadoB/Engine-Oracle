# Engine Oracle

A *Deep Learning / Machine Learning Audio Classification* Project based on F1 Engine Data.

## Brief Technical Overview

This project uses Formula 1 engine audio data from Poles between 2017 and 2024 to predict the *year* the data comes from, which *driver* was in the car and which *circuit* was he driving. This is achieved with the aid of the audio preprocessing named **Librosa** and the implementation of a *Neural Network Multiclass Classifier* through **PyTorch**. Several techniques were used to achieve this, but the most significan are *Mel Frequency Cepstral Coefficients*, *DataLoaders* and *Cross Validation*.

## Introduction

First and foremost, this project is not in the same vein of my previous one (aka. [StreamStats](https://github.com/RicardoSalgadoB/streamstats)). To begin with I don't really see any production use of what I've created here, that is why I didn't containerize it. I also tried to keep AI assistance as far away as I could, that meant  I had to follow the steps described in some online articles which I will promptly link in the following sections.

Overall, this is an Deep Learning project that seeks to analyze and classify some motorsport data. In order to do so it makes use of several libraries (**PyTorch**, **Librosa**, **Numpy**, ...) and many other techniques (*Short Time Fourier Transform*, *Mel Frequency Cepstral Coefficients (MFCC)*, *DataLoaders*, *Cross validation*, ...). I will describe in the following sections.

The purpose of this project is to showcase my Deep Learning skills and my capacity to build reliable models from raw audio data.

## Reasoning Process

I don't feel like writing a structured breakdown of the project, so I will just be giving a general overview of my reasoning process and the technologies and techniques implemented from its beginning to its very end.

### 1) In a hole in the ground there lived a Deep Learning project.

I knew that I wanted to do a Machine/Deep Learning project with F1 data, however I didn't know what to do. For one reason or the other, I ended choosing this.

An immediate google search revealed a similiar [project](https://becominghuman.ai/signal-processing-engine-sound-detection-a88a8fa48344) by *data4help* (an organization dedicated to helping non-profits and NGOs harness the power of their data, which is cool), if anything that project lacked ambition as it only executed a binary classification between Mercedes and Ferrari engines for the 2019 season. I was going to do more I was going to classify the year of all engines since 2000.

Still, it gave me a good guide of what to do. Yet I was a necessity to **extract the data** from somewhere. *Where?* Well, youtube, the [official Formula 1 channel](https://www.youtube.com/@Formula1) uploads every Pole lap and I could find playlist with clean laps from 2017 onwards (guess I'll have to thank Liberty Media for that).

By obvious reasons I am not including the data here, but I guess it won't be very hard to figure out how to get them, is it? Anyway, the videos I got come from this 8 playlists:

* [https://www.youtube.com/watch?v=LqpkFMYYMxs&list=PLfoNZDHitwjXAOD6FUnDpwWDyuIbJZe__](https://www.youtube.com/watch?v=LqpkFMYYMxs&list=PLfoNZDHitwjXAOD6FUnDpwWDyuIbJZe__)
* [https://www.youtube.com/watch?v=xyX6aNxL9SQ&list=PLfoNZDHitwjVNx_OW8yp2smxjbbOuCZGp](https://www.youtube.com/watch?v=xyX6aNxL9SQ&list=PLfoNZDHitwjVNx_OW8yp2smxjbbOuCZGp)
* [https://www.youtube.com/watch?v=kmuKQ2JQK30&list=PLfoNZDHitwjUA9aqbPGKw1l4SIz2bACi_](https://www.youtube.com/watch?v=kmuKQ2JQK30&list=PLfoNZDHitwjUA9aqbPGKw1l4SIz2bACi_)
* [https://www.youtube.com/watch?v=qdf-7a4tPRk&list=PLfoNZDHitwjUG6Nq8W0XLC_ke3s90wb3M](https://www.youtube.com/watch?v=qdf-7a4tPRk&list=PLfoNZDHitwjUG6Nq8W0XLC_ke3s90wb3M)
* [https://www.youtube.com/watch?v=jwJOmeDjX8g&list=PLfoNZDHitwjWgczXBINGGl4mmkfas1_Pe](https://www.youtube.com/watch?v=jwJOmeDjX8g&list=PLfoNZDHitwjWgczXBINGGl4mmkfas1_Pe)
* [https://www.youtube.com/watch?v=jSIAT0UYotQ&list=PLCvaDWh6BegKHAs21Wg1wFcWsYn_081IB](https://www.youtube.com/watch?v=jSIAT0UYotQ&list=PLCvaDWh6BegKHAs21Wg1wFcWsYn_081IB)
* [https://www.youtube.com/watch?v=RMpWukELqCc&list=PLCvaDWh6BegKWKWT69oP0jxnHbLbWjgM7](https://www.youtube.com/watch?v=RMpWukELqCc&list=PLCvaDWh6BegKWKWT69oP0jxnHbLbWjgM7)
* [https://www.youtube.com/watch?v=r7Mikgrm52k&list=PL1HouA-yvTAWxnczWCWl-CJ4CZOm2TNHP](https://www.youtube.com/watch?v=r7Mikgrm52k&list=PL1HouA-yvTAWxnczWCWl-CJ4CZOm2TNHP)

All audio used in this project was downloaded from the [official Formula 1 channel](https://www.youtube.com/@Formula1) and all rigths belong to Formula One Management. It must be clear that this data wasn't used for any commercial purposes.

Getting back on track (see what I did), I used the [*Pytube*](https://github.com/pytube/pytube) library to download the data and read its documentation when I was in need of guidance. 

After downloading the data, I continued with the *data4help* article. Next step is **exploring the data**, aka. *Waveplots*, *Fast Fourier Transforms (FFT)*, *Short Time Fourier Transforms (STFT)*, *Mel Frequency Cepstral Coefficients (MFCC)* and *Spectograms*. The results of this exploration are mostly seen in the `generate_images.py` and the corresponding `Images` directory.

Once the *MFCCs* are extracted they can be feed into a Neural Network. But wait... there is not enough data, there is a necessity to **split the audios** this is done with the python library named [*pydub*](https://github.com/jiaaro/pydub) in the `split_chunks.py` script. The end result is a lot of data, more precisely around 10,000 files with a 1000 ms of engine audio each. I also have to mention that I was a bit conservative with the slices of audio I selected as I was wary of getting radio messages in the data.

Now with some processing the data (as seen in `Source.process_data.py`) is ready to be feed into the Neural Network Classifier. Instead of using *TensorFlow* as in the *data4help* article, I went with *PyTorch* as is more flexible and allow for more control. To train a multiclass classifier I used a Cross Entropy Loss Funcition and an Stochastic Gradient Descent (SGD) optimizer (also tried Adam, but had worse performance). I didn't copy, but I did really on the code I wrote when doing *Daniel Bourkes*'s amazing [PyTorch Tutorial](https://youtu.be/Z_ikDlimN6A?si=imyaEgiyZWcs2g8L).

And that was pretty much it, the model was ready it just need a little tuning.

### 2) But... is this enough?

Clearly not. As of right now, my project was only limited to classyfing years, but that was about to change as a drivers and circuits classifiers where soon added.

Yet the demons circling around my mind where very presenting. "Is this enough?", "Why no deployment?", ... It also didn't help that in spite of this (or perhaps beacuse of this), I was doing a little procrastinating.

When I recovered from that phase I decided to focus on showcasing all the ML & DL techniques I could with this project and that is exactly what I did.

### 3) Early Stoppping

I already was using dropout to deal with overfitting (as specified in *data4help*'s article), but I need a way to stop the training cycles before overfitting appeared. The answer to my quarrel was early stopping.

For this I replicated the work shown by *Amit Yadav* in his Medium article [A Practical Guide to Implementing Early Stopping in PyTorch for Model Training](https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d). So please, don't blame me for the OOP code, I just understood it, saw it was good and replicated it.

### 4) DataLoaders

At the start I was only testing what *DataLoaders* could do, but they ended up being extremely critical for making training more efficient. Not faster though, the models take more to train, but they all the classifier to reach smaller losses faster. After some testing I became adamant about their purposeful utilization (almost to the same level of my usage of flowery language).

### 5) Cross Validation

This was the final step of the project. It began when I realized that the training process could use a bit more data. It tried to follow only articles, but I ended up doing my own thing (aka. manually assigning subsets).

All things said, there are some important considerations in regards with cross validation. First, the validation metrics during training are a mirage as the model has been trained on the validation subset previously. Second, I still need to keep test data separated which means that not all training data is used.

Overall, cross validation had a positive impact in the results. This was particullarly significant wiht the *Driver Classifier* ramping up performances from 93 % to around 97 % as seen in the results section.

## Results

**Years**
* *Cross Entropy Loss*: 0.115874
* *Accuracy*: 96.7 %
* *Recall*: 96.9 %
* *Precision*: 96.6 %
* *F1-Score*: 96.7 %
* *Training Time*: 9:56.038

The model struggles the most with years 2020 and 2021, which is not a surprise. Due to the pandemic, 2021 new regulations where delayed by a year and therefore the 2021 cars were basically the same as those in 2020.

**Drivers**
* *Cross Entropy Loss*: 0.152287
* *Accuracy*: 96.3 %
* *Recall*: 96.0 %
* *Precision*: 97.8 %
* *F1-Score*: 96.8 %
* *Training Time*: 6:31.825

As seen in the confussion matrix, most of the unreliability occurs between Lewis Hamiliton and Valtteri Bottas. This to be expected as most of their poles come from when they were teammates. This means that the classifier struggles a bit when the drivers come from the same team, but particullarly with these 2 (maybe, they have a similar diriving style).

**Circuits**
* *Cross Entropy Loss*: 0.393853
* *Accuracy*: 88.2 %
* *Recall*: 85.1 %
* *Precision*: 90.3 %
* *F1-Score*: 86.9 %
* *Training Time*: 6:57.533

**Hardware**

I ran this on my local, real, physical laptop, which is a *M4 Macbook Pro* with 14 cores.

## How to use

***WARNING:*** Don't train the model if you think that your equipement will have problems. Even with my *M4*, there came a time when the model output just zeroed out. Either that or the model began predicting all poles belong to Hamilton, I would indeed expect Apple Silicon to be *teamLH*.

Without further ado, here is what you have to do to run this project:

1. Clone this repo
```sh
git clone [https://github/RicardoSalgadoB/EngineOracle.git](https://github/RicardoSalgadoB/EngineOracle.git)
cd EngineOracle
```

2. Create a virtual environment
With *venv*
```sh
python -m venv venv
source venv/bin/activate
```
or *conda*
```sh
conda create --name Engine-Oracle
conda activate Engine-Oracle
```

3. Install python dependencies
```sh
pip install -r requirements.txt
```

4. Run the entrypoint script. *Shell* only works on UNIX systems so... yeah. You can create a Vitual Machine if you want, is not a bad skill to have either way.

By executing this file you will get the audio data in your machine, already split into trunks.

Sorry for giving you an `entrypoint.sh` as if you were a Docker daemon.
```sh
chmod +x entrypoint.sh
sh entrypoint.sh
```

5. Either do `main.py`, `predict.py` or `test.py` depending on your desires.

*Main* will train the models again and test them.
```sh
python main.py
```

*Test* will test an existing model, but only that.
```sh
python test.py
```

*Predict* will make a year, driver and cicuti prediction on a file name specified in the code
```sh
python predict.py
```

*Generate images* use this to regenerate images, they are already generated though.
```sh
python generate_images
```


## Conclusion

The main limitation of this project is that it lacks any real world applicability. In other words, it is but a mere toy. I had chance to make a more useful project (in ), but I just didn't feel passionate about it. Even with that, I am a bit of a F1 purist, I don't think the model would be more valuable than the insights the mechanics and drivers can bring to the table nor would I like to see the massive use of this ML/DL models in the sport.

Another imporovement would be more finetuning of the model architectures. However, I believe this is good enough, as I don't want this project to focus on the results as much as the techniques I used to achieve those results. Though, I wouldn't mind exploration other ways to do circuit classification later on.

Concluding, I am satisfied with this project, could be more though. I did learning a lot about audio signal processing (the existing of a library like *Librosa* is extremely useful for audio processing) and made a solid show case of my deep learning skills (I'm might still market this as an ML project, as most skills are transferable and it is just a better strategy being fair).

## Contact
Ricardo Salgado Benítez - [ricardosabe2018@gmail.com] - [https://www.linkedin.com/in/ricardosalgadob/]