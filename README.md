
# Sentiment Analysis with Transformers + BERT

Sentiment analysis is the interpretation and classification of emotions (positive, negative and neutral) within text data using text analysis techniques. Sentiment analysis tools allow businesses to identify customer sentiment toward products, brands or services in online feedback.
Sentiment analysis is a machine learning technique that detects polarity (e.g. a positive or negative opinion) within text, whether a whole document, paragraph, sentence, or clause.
Understanding people’s emotions is essential for businesses since customers are able to express their thoughts and feelings more openly than ever before. By automatically analyzing customer feedback, from survey responses to social media conversations, brands are able to listen attentively to their customers, and tailor products and services to meet their needs.

The most common use of The Sentiment Analysis API in the financial sector will be the analysis of financial news, in particular to predicting the behavior and possible trend of stock markets.
Traditional Technical Analysis of the Financial Market with the use of tools like Stochastics and Bollinger bands aside, sentiment analytics has been receiving a lot of attention as it allows the integration of both Fundamental Analysis (FA) and Technical Analysis (TA).
In real life, Financial Market Analysts make predictions on the stock market based on opinions and happenings in the news. Similarly, Sentiment Analysis API is making it possible for computers to do the same job now. Furthermore, with advance computational linguistic and machine learning techniques, the task of opinion mining proves to be more efficient than human analysts, having the capability to scan through huge chunk of text across various news channels within seconds.
A simple example of the real application of Sentiment Analysis API for the financial sector can be explained by the task of assigning positive, negative or neutral sentiment values to the words. For instance, words such as “good“, “benefit“, “positive“, and “growth” are all tagged with positive scores while words such as “risk“, “fall“, “bankruptcy“, and “loss” are tagged with negative scores. 


## Tech Used

`Torch`
`Transformers`
`Sklearn`
`Pandas`



## Outputs

In case of imbalanced data, it is necessary to balance them so that our training would not be biased.

![alt text](https://github.com/delaramhamraz73/Sentiment-Analysis-/blob/main/Balacing%20The%20Data.png)

How many tokens are there in each phrase? 

![alt text](https://github.com/delaramhamraz73/Sentiment-Analysis-/blob/main/Number%20of%20tokens%20in%20every%20sentence.png)

Training and evaluation process:

![alt text](https://github.com/delaramhamraz73/Sentiment-Analysis-/blob/main/Training%20History.png)

The Classification Report:

![alt text](https://github.com/delaramhamraz73/Sentiment-Analysis-/blob/main/The%20Classification%20Report.png)

Testing on a random finacial text:

![alt text](https://github.com/delaramhamraz73/Sentiment-Analysis-/blob/main/Output.png)



## Deployment

To deploy this project run

```bash
 python training.py
```
this file will train the model with the provided database in the same folder.

then run
```bash
 python Sentiment_Analysis.py
```
which will take the sample text, in our case "toys RU", analyze it and give you the results.

## Documentation

Which steps should we take in order to make a customize model?

![alt text](https://github.com/delaramhamraz73/Sentiment-Analysis-/blob/main/Steps%20to%20Building%20a%20Customized%20Model.png)

A Diagram to show how we use bert + transformers inorder to build a sentiment analysis model:

![alt text](https://github.com/delaramhamraz73/Sentiment-Analysis-/blob/main/Sentiment%20Analysis%20with%20Transformers%20and%20BERT%20Structure%20(Diagram).png)


