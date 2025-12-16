#Resume-Based Job Recommender System

Overview
This project provides an end-to-end solution for job recommendation. It consists of two main components: a Data Cleaning Pipeline that processes raw job descriptions (JDs) and a Streamlit Web Application that performs real-time matching between an uploaded resume (PDF) and the cleaned job data.

The system uses TF-IDF (Term Frequency-Inverse Document Frequency) combined with N-Gram analysis and K-Nearest Neighbors (KNN) to determine the similarity between a candidate's extracted skills and available job descriptions.

 How the Matching Works

The core of the recommender relies on finding the **closest distance** between two vector representations:

1.  **Resume Skills Vector:** Skills are extracted from the uploaded PDF, converted into a single text block, and then vectorized using the fitted TF-IDF model.
2.  **Job Description Vector :** Every job description in the `jd_cleaned.csv` is similarly vectorized.
3.  **Similarity Metric:** The `NearestNeighbors` model uses **Cosine Similarity** (which is calculated from $1 - \text{Cosine Distance}$) to measure how closely the skills in your resume $\mathbf{R}$ match the skills/description in each job $\mathbf{J}$.
4.  **Ranking:** Jobs are sorted by the lowest distance (highest similarity) to provide the best recommendations first.

Project Structure (Implied)

  * `application.py`: The main Streamlit web application code (live matching).
  * `data_cleaning_script.py`: The script used to transform `tech_JD.xlsx` into `jd_cleaned.csv`.
  * `jd_cleaned.csv`: The clean, processed job data used by the recommender.
  * `skills_extraction.py`: A custom module used for resume skill extraction (relies on `pyresparser`).
  * `tech_JD.xlsx`: Raw input job data file.
