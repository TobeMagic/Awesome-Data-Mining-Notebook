## World-map 可视化

使用轻量级sqlite 数据库存储数据实现，并使用pandas.read_query_data读取

https://www.kaggle.com/code/davidmezzetti/cord-19-analysis-with-sentence-embeddings/notebook

```python
# Workaround for mdv terminal width issue
os.environ["COLUMNS"] = "80"

from paperai.highlights import Highlights
from txtai.pipeline import Tokenizer

from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pycountry

# Use paperai + NLTK stop words
STOPWORDS = Highlights.STOP_WORDS | set(stopwords.words("english"))

# Tokenizes text and removes stopwords
def tokenize(text, case_sensitive=False):
    # Get list of accepted tokens
    tokens = [token for token in Tokenizer.tokenize(text) if token not in STOPWORDS]
    
    if case_sensitive:
        # Filter original tokens to preserve token casing
        return [token for token in text.split() if token.lower() in tokens]

    return tokens
    
# Country data
countries = [c.name for c in pycountry.countries]
countries = countries + ["USA"]

# Lookup country name for alpha code. If already an alpha code, return value
def countryname(x):
    country = pycountry.countries.get(alpha_3=x)
    return country.name if country else x
    
# Resolve alpha code for country name
def countrycode(x):
    return pycountry.countries.get(name=x).alpha_3

# Tokenize and filter only country names
def countrynames(x):
    return [countryname(country) for country in countries if country.lower() in x.lower()]

import pandas as pd
import sqlite3
# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

sections = pd.read_sql_query("select text from sections where tags is not null order by id desc LIMIT 500000", db)

# Filter tokens to only country names. Build dataframe of Country, Count, Code
mentions = pd.Series(np.concatenate([countrynames(x) for x in sections.Text])).value_counts()
mentions = mentions.rename_axis("Country").reset_index(name="Count")
mentions["Code"] = [countrycode(x) for x in mentions["Country"]]

# Set max to 5000 to allow shading for multiple countries
mentions["Count"] = mentions["Count"].clip(upper=5000)

mapplot(mentions, "Tagged Articles by Country Mentioned", "Articles by Country")
```



![image-20231030155822094](data visualization.assets/image-20231030155822094.png)

