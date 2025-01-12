# Twitter sentiment analysis
---

## Contents
- [Introduction](#introduction)
- [Universal Sentence Encoder](#universal-sentence-encoder)
- [ML techniques](#ml-techniques)
- [Results](#results)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
---

## Introduction
This project aims to classify the sentiment of tweets from the dataset ["Twitter Sentiment Analysis"](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). This is done by utilizing Google's [Universal Sentence Encoder](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf) that encodes the tweets into an embedding vector that is fed into a fully connected feed forward neural network. 

---

## ML-Techniques
The network architecture is a simple feed forward network with two hidden layer. There are 512 input neurons, 256 neurons in the first hidden layer and 128 in the second hidden layer. ReLU is used as activation function between each layer. The output layer has 4 neuron representing the 4 different classes: "Positive", "Neutral", "Negative" and "Irrelevant".

---

## Results

---

## Prerequisites

---

## Usage


