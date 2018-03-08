
## NewsMood

In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various news oulets, and to present your findings visually.

Your final output should provide a visualized summary of the sentiments expressed in Tweets sent out by the following news organizations: BBC, CBS, CNN, Fox, and New York times.

The first plot will be and/or feature the following:

Be a scatter plot of sentiments of the last 100 tweets sent out by each news organization, ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, and +1 the most positive sentiment possible.
Each plot point will reflect the compound sentiment of a tweet.
Sort each plot point by its relative timestamp.
The second plot will be a bar plot visualizing the overall sentiments of the last 100 tweets from each organization. For this plot, you will again aggregate the compound sentiments analyzed by VADER.

The tools of the trade you will need for your task as a data analyst include the following: tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

Your final Jupyter notebook must:

Pull last 100 tweets from each outlet.
Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet.
Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores.
Export the data in the DataFrame into a CSV file.
Save PNG images for each plot.

## Analysis

Trend 1: As of March 7th, 2018, 6:20pm (and in previous hours), BBCNews consistantly has a more negative polarity score compared to the other news sources.

Trend 2: As of March 7th, 2018, 6:20pm, New York Times is the only news source with a positive average polarity.

Trend 3: The average polarity is not consistant throughout, therefore the polarity scores fluctuate based on current news.


```python
#! pip install textblob
```


```python
# Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tweepy
import numpy as np
import seaborn as sns
import json

from datetime import datetime
from textblob import TextBlob, Word, Blobber

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API keys
import config 

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Search for the news stations
target_users = ["@BBCNews", "@CBSNews", "@CNN", "@FoxNews", "@nytimes"]

# Variables for holding lists
news_account = []
date = []
compound_list = []
positive_list = []
neutral_list = []
negative_list = []
tweet_count = []
```


```python
# Loop through each news station to get the recent 100 tweets
for user in target_users:
    
    #Tweet count
    tweet_number = 0

    # Loop through all news stations
    for x in range(5):
    
        public_tweets = api.user_timeline(user, page=x) 
        
        # Loop through all tweets
        for tweet in public_tweets:
            
            tweet_number += 1
            
            # Create columns with information about tweet
            news_account.append(user)
            date.append(tweet['created_at'])
            tweet_count.append(tweet_number)

            # Run Vader Analysis on each tweet
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            
            # Append the Vadar Analysis
            compound_list.append(compound)
            positive_list.append(pos)
            neutral_list.append(neu)
            negative_list.append(neg)

```


```python
#Generate Dataframe 

# Creating dictionary for DataFrame
tweet_summary = {
    "News Channel": news_account,
    "Date": date,
    "Compound": compound_list,
    "Positive": positive_list,
    "Neutral": neutral_list,
    "Negative": negative_list,
    "Tweets Ago": tweet_count}

df_sentiment_analysis = pd.DataFrame(tweet_summary, columns = ['News Channel', 'Date', 'Compound', 'Positive',
                                               'Neutral', 'Negative', 'Tweets Ago'])
df_sentiment_analysis.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>News Channel</th>
      <th>Date</th>
      <th>Compound</th>
      <th>Positive</th>
      <th>Neutral</th>
      <th>Negative</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCNews</td>
      <td>Wed Mar 07 22:36:52 +0000 2018</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@BBCNews</td>
      <td>Wed Mar 07 22:36:16 +0000 2018</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@BBCNews</td>
      <td>Wed Mar 07 22:35:39 +0000 2018</td>
      <td>-0.4939</td>
      <td>0.0</td>
      <td>0.802</td>
      <td>0.198</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@BBCNews</td>
      <td>Wed Mar 07 22:34:43 +0000 2018</td>
      <td>-0.4939</td>
      <td>0.0</td>
      <td>0.824</td>
      <td>0.176</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@BBCNews</td>
      <td>Wed Mar 07 22:16:54 +0000 2018</td>
      <td>-0.5719</td>
      <td>0.0</td>
      <td>0.764</td>
      <td>0.236</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create CSV from dataframe
df_sentiment_analysis.to_csv('NewsSentimentAnalysis.csv')
```


```python
# Plot the sentiments of the last 100 tweets

bbc = df_sentiment_analysis.loc[df_sentiment_analysis['News Channel'] == '@BBCNews']
cbs = df_sentiment_analysis.loc[df_sentiment_analysis['News Channel'] == '@CBSNews']
cnn = df_sentiment_analysis.loc[df_sentiment_analysis['News Channel'] == '@CNN']
fox = df_sentiment_analysis.loc[df_sentiment_analysis['News Channel'] == '@FoxNews']
nyt = df_sentiment_analysis.loc[df_sentiment_analysis['News Channel'] == '@nytimes']

# Plotting Twitter info
plt.scatter(bbc['Tweets Ago'], bbc['Compound'], c='lightskyblue', edgecolor='k', s=80, alpha=0.7, label='BBCNews')
plt.scatter(cbs['Tweets Ago'], cbs['Compound'], c='green', edgecolor='k', s=80, alpha=0.7, label='CBS')
plt.scatter(cnn['Tweets Ago'], cnn['Compound'], c='red', edgecolor='k', s=80, alpha=0.7, label='CNN')
plt.scatter(fox['Tweets Ago'], fox['Compound'], c='navy', edgecolor='k',s=80, alpha=0.7, label='Fox')
plt.scatter(nyt['Tweets Ago'], nyt['Compound'], c='yellow', edgecolor='k', s=80, alpha=0.7, label='New York Times')

now = datetime.now()
now = now.strftime("%m/%d/%Y")
plt.title('Sentiment Analysis of Media Tweets ({})'.format(now),fontsize=(14))

plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.xlim(105,-5)
plt.ylim(-1.1,1.1)

plt.legend(bbox_to_anchor=(1, 1),title='Media Sources')

plt.show()
plt.savefig("SentimentAnalysis.png") 
```


![png](output_9_0.png)



```python
# Create Overall Media Sentiment based on Twitter table
Twitter_summary = df_sentiment_analysis.groupby(["News Channel"])["Compound"].mean().reset_index()

# Renaming Columns
Twitter_summary.rename_axis({'News Channel': 'News Channel', 'Compound':  'Average Compound'},axis=1,inplace=True)

# Output Twitter Table dataframe
Twitter_summary
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>News Channel</th>
      <th>Average Compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBCNews</td>
      <td>-0.269421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@CBSNews</td>
      <td>-0.065931</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@CNN</td>
      <td>-0.078046</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@FoxNews</td>
      <td>-0.051749</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@nytimes</td>
      <td>0.033753</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot bar graph
plt.bar(0, bbc['Compound'].mean(), color='lightskyblue', width=1)
plt.bar(1, cbs['Compound'].mean(), color='green', width=1)
plt.bar(2, cnn['Compound'].mean(), color='red', width=1)
plt.bar(3, fox['Compound'].mean(), color='navy', width=1)
plt.bar(4, nyt['Compound'].mean(), color='yellow', width=1)

# Sets the y limits of the current chart
plt.ylim(-0.5,0.5)

# Give chart some labels and a tile
now = datetime.now()
now = now.strftime("%m/%d/%Y")
plt.title('Overall Media Sentiment Analysis based on Twitter ({})'.format(now),fontsize=(14))
plt.xlabel("Media Sources")
x_labels = ["BBC", "CBS", "CNN", "Fox", "NYT"]
x_locations = [value for value in np.arange(5)]
plt.xticks(x_locations, x_labels)
plt.ylabel("Tweet Polarity")
plt.grid()

# Print our chart to the screen and save image
plt.show()
plt.savefig("OverallSentimentAnalysisTwitter.png")
```


![png](output_11_0.png)

