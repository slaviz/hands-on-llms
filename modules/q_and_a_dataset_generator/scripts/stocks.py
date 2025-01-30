import yfinance as yf
import pandas as pd
import requests
import logging
import traceback
from typing import Optional, Union, Dict

import numpy as np
import os
import constants
from base import SingletonMeta
import openai
import json
logger = logging.getLogger(__name__)


openai.api_key = os.environ["OPENAI_API_KEY"]
PROMPT_TEMPLATE = """
for the prompt below, extract relevant relevant names of companies that are traded in the stock market from the next prompt, any identity that appear in the prompt, or is related in some way should be extract too, the format of extraction should be:
{{"identity1": "appears in the prompt",
"identity2": "is main competitor of identity 1",
"identity3": "is main supplier of identity 2",
"identity4": "another relation with identity X"
}}

And so on, to maximum of 5 top related identities to the prompt, not that for each identity should describe it's relevance (it's not necessarily only what is provided in the example, but could be any relation that is deemed important),
- note that in the dict you should fill in the identity name  as the stock ticker and then its relation
- Note that in the prompt there may be no identities, at which point you should suggest identities yourself that are related to the question.

Your response should be sparse and contain only what is requested, be mindful of the amount of text you  respond with, it should be as minimal as possible and be only in the format above as this is part of a code pipeline
The prompt:
{PROMPT}
"""
EXAMPLE = {}
def build_prompt(prompt) -> str:
    return PROMPT_TEMPLATE.format(
        PROMPT=prompt
    )


def fetch_stock_prices(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval,progress=False)
    return data['Close'].iloc[:,-1:].values.flatten()

def post_process_prices(prices):
  prices = (prices - prices[0])/prices[0]
  return np.round(prices, 2)

# In case we for some reason can't ask for the company stock name from the chatgpt
def get_ticker (company_name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=url, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code


def get_performance_summary(company_tickers, sample_configs):
  results = {}
  for company, info in company_tickers.items():
    for sample_period_name, period, interval in sample_configs:
      try:
          prices = fetch_stock_prices(company, period=period, interval=interval)
          if company in results:
            results[company][sample_period_name] = post_process_prices(prices)
          else:
            results[company]={sample_period_name: post_process_prices(prices)}

      except Exception as e:
          print(f"Error fetching data for {company} in {sample_period_name, period, interval}: {e}")
          if company in results:
            results[company][sample_period_name] = "Not Available"
          else:
            results[company]={sample_period_name: "Not Available"}

  # Display results
  response_string = ""
  for company in results.keys():
    response_string += f"Company: {company}, {company_tickers[company]}\n"
    for period in results[company].keys():
      response_string +=f"{period}: {results[company][period]}\n"
  return response_string

def ask_gpt(question: str, version_to_use: str="mock"):
    response = {}
    if version_to_use=="mock":
        response = """{
            "MSFT": "appears in the prompt, a major technology company with strong financials and consistent growth",
            "AAPL": "is a main competitor of MSFT in technology and consumer products",
            "GOOGL": "is a competitor and collaborator with MSFT in cloud services and AI",
            "AMZN": "competes with MSFT in cloud computing through AWS",
            "INTC": "is a supplier for MSFT in hardware-related initiatives"
        }"""

    else:
        prompt = build_prompt(question)
        logger.info(f"{prompt=}")
        openai_response = openai.Completion.create(
            engine=version_to_use,
            # "text-davinci-003",    # See: https://github.com/iusztinpaul/hands-on-llms/issues/87
            prompt=prompt,
            temperature=0,
            max_tokens=100,
        )
        response = openai_response["choices"][0]["text"]
        print("###")
        print(response)
        print("###")
        logger.info(f"{response=}")
    return response


class StocksModule(metaclass=SingletonMeta):
    """
    A singleton class that provides stocks data from given string that contains identities.

    """

    def __init__(
        self,
        identity_extractor_model_name: str = constants.IDENTITY_EXTRACTION_MODEL,
        sample_configs: tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]=
                            (("Hourly changes (Last 24 Hours)", "1d", "1h"),
                            ("Daily changes (Last Week)", "5d", "1d"),
                            ("Monthly changes (Last 6 Month)", "6mo", "1mo")),
    ):
        """
        Initializes the EmbeddingModelSingleton instance.

        Args:
            model_id (str): The identifier of the gpt agent to use to extract identities
        """

        self._identity_extractor_model_name = identity_extractor_model_name
        self._sample_configs = sample_configs


    def __call__(
        self, input_text: str, to_list: bool = True
    ) -> str:
        """
        Extract identities and their stock prices

        Args:
            input_text (str): The input text to generate stocks from.

        Returns:
            Union[np.ndarray, list]: The embeddings generated for the input text.
        """

        try:
            response = ask_gpt(input_text,
                                self._identity_extractor_model_name
                                 )
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(f"Error when extracting identities from the following text: {input_text}")

            return ""
        try:
            identities = json.loads(response)
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(
                f"Error when parsing response from ask_gpt: {response}\n"
            )
            return ""
        try:
            result = get_performance_summary(identities, self._sample_configs)
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(
                f"Error when extracting stocks given identity dict embeddings for the following identities: {identities}\n"
                f" and the following input_text: {input_text}"
            )

            return ""


        return result
if __name__ == "__main__":
    sample_query = "should I invest in Microsoft"
    myStocksGenerator = StocksModule()
    #
    print(myStocksGenerator(sample_query))
    # run(/)
