This is my work for the [Kaggle competition](https://www.kaggle.com/competitions/text-based-geolocation/overview)
on predicting a geolocation from text data, specifically Tweets.
My solution uses features extracted by the pretrained large language model RoBERTa.
Afterwards a single linear layer is applied to classify datapoints to a location.
Directly letting the model predict coordinates turned out to be difficult to train.
This approach therefore uses a classification approach.
The model is still punished for predicting points that are far away from eachother
through a custom loss function which combines cross entropy and MSE.