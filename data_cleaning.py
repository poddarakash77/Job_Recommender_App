import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

df = pd.read_excel(r'tech_JD.xlsx')

df.drop(columns=['jobType', 'descriptionHTML', 'externalApplyLink', 'id', 'isExpired', 'postedAt', 'postingDateParsed', 'scrapedAt', 'searchInput/country', 'searchInput/position', 'url', 'urlInput'], inplace=True)

df = df[
    (df['jobType/0'].str.lower() == 'full-time') |
    (df['jobType/1'].str.lower() == 'full-time') |
    (df['jobType/2'].str.lower() == 'full-time') |
    (df['jobType/3'].str.lower() == 'full-time') ]

df.drop(columns=['jobType/3','jobType/2','jobType/1','jobType/0'], inplace=True)

def convert_salary_to_numeric(salary):
    if pd.isna(salary):
        return np.nan
    #salary = salary.replace('$', '').replace(',', '').replace('per', '').replace('class', '').replace('hour', '').replace('year', '').replace('a', '').strip()
    salary = re.sub(r'[^\d\.\-]', '', salary)

    if '-' in salary:
        low, high = salary.split('-')
        return (float(low.strip()) + float(high.strip())) / 2
    else:
        return float(salary)

df['salary_numeric'] = df['salary'].apply(convert_salary_to_numeric)

columns = ['company', 'ROLE', 'positionName',  'description', 'salary_numeric', 'rating', 'reviewsCount']
df_new = df[columns]

df_new.columns = df_new.columns.str.upper()

# Downloading NLTK data if not already available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
# Initializing stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Clean text by applying:
    - Lowercasing
    - Removing special characters and numbers
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    # Lowercasing
    text = text.lower()
    # Removing special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization
    words = word_tokenize(text)
    # Stopword removal and Lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Joining back into a single string
    return ' '.join(words)

# Applying the cleaning function to the job descriptions
df_new['CLEANED DESCRIPTION'] = df_new['DESCRIPTION'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')

df_new.drop(columns=['DESCRIPTION'], inplace=True)

df_new.to_csv(r'jd_cleaned.csv')