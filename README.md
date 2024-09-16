# Social-Media-and-News-Article-Sentiment-Analysis-for-Stock-Market-Autotrading


This is the source code for my master’s thesis.

## Intro

The project was to perform A/B testing on the potential profitability improvements to a stock trading bot with added:
 - Sentiment analysis of twitter data
 - Sentiment analysis of twitter data with topic clustering

To achieve this the model was made to run with any of the following scopes:
 - Multi-topics (sentiment with topic clustering, plus financial data)
 - No-topics (linear sentiment, no clustering, plus financial data)
 - No-sentiment (financial data only) 

On top of this each model scope was also optimising its own hyperparameters. At the end, the top ("optimal") sets of parameters from each of the 3 scopes, were re-ran in their originating scenario and the other scenarios, multiple times. This was to produce a set of linked pairs for paired t-testing. This was done for a 5-mins and 30-mins prediction horizon.

## Limitations

Halfway through the project, Mr Musk’s famous takeover of twitter happened, along with the removal in the free student access to live twitter data. This meant that Kaggle data had to be used, blocking me mimicking the technique used by Weng at al (see the first half of thesis section 4.4), of the tracking sentiment of a larger number of individual users on twitter as individual features. Which was the original intent of the project. That and potentially creating a live system.

## Detailed Info

The following sections of the Fabio_James_Greenwood_MSc_Thesis.pdf will provide more information about (in order of importance):
 - The basic system overview -> 1.4 and 1.5
 - The conclusion -> 1.9
 - Discussion about the potential next steps -> 1.10
 - The NLP implemented -> 5.4
   - Text preparation
   - Topic clustering
   - Calculations (sentiment quantification)
 - Model training -> 5.5
   - Bit more interesting post section 5.5.1 (a straightforward explanation of RNNs)
   - Manual implementation of the deep random subspace ensembles method (not native to sci-kit learn and other packages used) -> 5.5.2
 - Analysis of results and performance
   - Analysis around potential issues around bias (potentially inflating profitability scores) -> 1.8.5
   - Performance of the hyperparameter optimisation -> 1.8.6

I would note this is an academic document, I’m sure a technical document starting with a to-the-point introduction, with technical specifications, system overviews etc would have been more interesting for any prospective employers, however this was the document I was requested to produce. After completing my first draft of the PDF I was keen to start coding, so I produced limited technical documentation and leant on the thesis. Though in hindsight I admit that this would have helped in development and presenting this project within my portfolio. However, after all the thesis writing I was just keen to start coding.

## Discussion

While the actual discussion, conclusion and future improvements can be found in the thesis document, below is a brief discussion about the results of the project:

Generally, the models that used twitter data (with market financial data) underperformed the models that just used the market financial data in a statistically significant fashion. In some cases, they were even in performance. 

This said the while the MAE error was larger than the average price change, denoting a large amount of noise in the prediction, the models were profitable overall. For the 30-mins prediction models this got up to 56.4% & 55.9%[^1] correct on guessing the next direction of stock movement and profitability scores of 5.62e-3 & 6.49e-3[^1] [^2]. This means that the model is more suited for the purpose of betting on the next direction of stock movement.

Caveats on the above statement about profitability scores. These values don’t consider spread [^3], forced liquidations [^4], consequences to sudden losses, scaling, market depth, lag and a range of other factors that would likely come to the fore in a real-world implementation. Much of this is discussed in various sections of the thesis. Also in a real-world scenario, leverage is likely to be used, which would multiply any potential profits/losses.

I would like to note that while the goal of the project was to see if I could improve a stock trader using twitter data, this was always an ambitious target and was selected more as a means to start working with twitter data and sentiment analysis. While I like to point to the limitations caused by the changes in twitter policy towards students halfway through the project, likely if I had access to all the data I originally planned to have, I would have still had mixed results.

Personally, I’m slightly surprised at the relatively strong performance of the models, however performance loss and real-world problems will likely crop up in a real-world implementation. Otherwise, you the reader and myself, could together, retire tomorrow. Thanks for taking the time to read this document, there was more undocumented work that went into his project, such as establishing a good range for each RNN parameter to be adjusted within, however I found it enjoyable. Happy to discuss this work or any job opportunities, my contact details are on my home page.


[^1]: Both pairs of scores being the validation and testing scores respectively for the results in table 9, row 3. These results are for a model produced with increased training epochs and component models.
[^2]: A profitability value of 6.49e-3 denotes that for every £100 staked in a trade an average of £0.649 is made
[^3]: Potentially negligible given the high profitability scores, discussed appendix J
[^4]: Discussed indirectly within appendix B but many more relevant resources available online.
