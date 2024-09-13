# Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading


This is the source code for my master’s thesis.

## Intro
The project was to do some A/B testing on the potential profitability improvements to a stock trading bot with and without both:
 - sentiment analysis of twitter data
 - sentiment analysis of twitter data with topic clustering

To do this the model had to work with three scopes:
 - mutli-topics (sentiment with clustering, plus financial data)
 - no-topics (sentiment, no clustering, plus financial data)
 - no-sentiment (financial data only)
On top of this each model scope was also optimising its own hyperparameters. At the end the top ("optimal") parameter from each of the 3 scopes, were reran in their originating scenario and the other scenarios, multiple times. This was to produce a set of linked pairs for paired t-testing. This was done for a 5-mins and 30-mins prediction horizon.

## Limitations
Halfway through the project, Mr Musk’s famous takeover of twitter happened. Along with a change in the free student accessibility to live twitter data. This meant that Kaggle data had to be used, blocking me mimicking the technique used by Weng at al (see thesis section 4.4), to track the sentiment of a larger number of individual users on twitter as induvial features. Which was the original intent of the project. That with potentially creating a live system.

## Detailed Info
The following sections of the Fabio_James_Greenwood_MSc_Thesis.pdf will provide more information about:
 - The basic system overview -> 1.4 and 1.5
 - The conclusion -> 1.9
 - Discussion about the potential next steps -> 1.10
 - The NLP implemented -> 5.4
   - Text preparation
   - Topic clustering
   - Calculations (sentiment quantification)
 - Model training -> 5.5
   - Bit more interesting post section 5.5.1 (a straightforward explanation of RNNs)
   - Manual implementation of the deep random subspace ensembles implementation -> 5.5.2

I would note this is an academic document, I’m sure a technical document starting with a to-the-point introduction, followed closely by a technical specification and system overview would have been more interesting for any prospective employers, however this was the document I was requested to produce. Though in hindsight I admit that I should have produced more coding documentation. The PDF was produced in overleaf.







