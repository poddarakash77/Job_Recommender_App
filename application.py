import re
import pandas as pd
import numpy as np
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from pyresparser import ResumeParser
import tempfile
import streamlit as st
import skills_extraction as skills_extraction
import time
import io

nltk.download("stopwords")
stopw = set(stopwords.words("english"))

# Load job dataset
jd_df = pd.read_csv(r'jd_cleaned.csv')

# Helper function for n-grams
def ngrams(string, n=3):
    string = fix_text(string)  # Fix text
    string = string.encode("ascii", errors="ignore").decode()  # Remove non-ASCII chars
    string = string.lower()
    chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
    rx = "[" + re.escape("".join(chars_to_remove)) + "]"
    string = re.sub(rx, "", string)
    string = string.replace("&", "and")
    string = string.replace(",", " ")
    string = string.replace("-", " ")
    string = string.title()  # Normalize case - capital at start of each word
    string = re.sub(" +", " ", string).strip()  # Remove multiple spaces
    string = " " + string + " "  # Pad names for n-grams
    string = re.sub(r"[,-./]|\sBD", r"", string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return ["".join(ngram) for ngram in ngrams]

# Streamlit app
def main():
    st.title("Job Recommendation System")
    st.write(
        "Upload your resume (PDF format), and we'll recommend the top jobs that match your skills."
    )

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Number input
    number = st.number_input(label="Number of Jobs to recommend", min_value=1, max_value=1000, value=10)

    # Submit button
    if st.button("Submit",use_container_width=True):
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Extract skills using uploaded resume
            skills = []
            extracted_skills = skills_extraction.skills_extractor(temp_file_path)
            skills.append(" ".join(word for word in extracted_skills))

            # Vectorization using TF-IDF
            vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
            tfidf = vectorizer.fit_transform(skills)

            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
            jd_test = jd_df["CLEANED DESCRIPTION"].values.astype("U")

            def getNearestN(query):
                queryTFIDF_ = vectorizer.transform(query)
                distances, indices = nbrs.kneighbors(queryTFIDF_)
                return distances, indices

            distances, indices = getNearestN(jd_test)
            test = list(jd_test)
            matches = []

            for i, j in enumerate(indices):
                dist = round(distances[i][0], 2)
                temp = [dist]
                matches.append(temp)

            matches = pd.DataFrame(matches, columns=["Match confidence"])

            # Recommend Top Jobs
            jd_df["match"] = matches["Match confidence"]
            recommended_jobs = jd_df.sort_values("match").head(number)
            recommended_jobs.reset_index(inplace=True, drop=True)
            recommended_jobs.index += 1

            with st.status("Analysing data...", expanded=True) as status:
                st.write("Searching for Jobs...")
                time.sleep(2)
                st.write("Found Jobs.")
                time.sleep(1)
                st.write("Loading...")
                time.sleep(1)
                status.update(
                    label="Analysis complete!", state="complete", expanded=False
                )

            # Display recommendations
            st.write(f"### Recommended Top {number} Jobs:")
            st.dataframe(recommended_jobs[['COMPANY', 'ROLE', 'POSITIONNAME', 'SALARY_NUMERIC', 'RATING', 'match']])

            csv_data = recommended_jobs.to_csv(index=True)  # Include the index starting from 1
            csv_buffer = io.StringIO(csv_data)

            st.download_button(
                label="Download Recommendations",
                data=csv_buffer.getvalue(),
                file_name="recommended_jobs.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.error("Please upload a resume before clicking Submit!")

if __name__ == "__main__":
    main()
