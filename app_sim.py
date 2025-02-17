import streamlit as st
from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import textwrap
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os

# Import functions from glossary_similarity.py
from glossary_similarity import fetch_chunks_for_term_for_years, fetch_chunks_for_term_for_years, compute_term_dist_cosine

### **Neo4j Initialization**
def init_graph_DB():
    """Initialize the Neo4j connection"""
    load_dotenv()
    URI = os.getenv('NEO4J_URI')
    USERNAME = os.getenv('USERNAME')
    PASSWORD = os.getenv('DB_PASSWORD')

    # Suppress warnings from Neo4j logs
    import logging
    logging.getLogger("neo4j").setLevel(logging.ERROR)

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    return driver

### **Compute Similarity-Based Deviation**
def compute_deviation(df):
    """Computes deviation from the term embedding"""
    df["deviation"] = 1 - df["similarity"]  # Higher similarity = lower deviation
    return df

def scatterplot_from_multiple_terms(df, selected_terms):
    """Creates a single smooth scatter plot showing similarity over time for multiple terms, with extended hover data."""

    # Ensure 'chunk' column exists
    if "chunk" not in df.columns:
        if "chunk_text" in df.columns:
            df["chunk"] = df["chunk_text"]
        else:
            st.error("Missing 'chunk' column in DataFrame! Check data fetching.")
            st.write(df.head())
            st.stop()

    df["wrapped_chunk"] = df["chunk"].apply(lambda x: "<br>".join(textwrap.wrap(str(x), 50)))  # Ensure text format

    # Assign unique colors for each term
    term_colors = px.colors.qualitative.Set1 + px.colors.qualitative.Set3  # Large color palette
    term_color_map = {term: term_colors[i % len(term_colors)] for i, term in enumerate(selected_terms)}

    # Initialize figure
    fig = go.Figure()

    # Add similarity trend for each term in the SAME figure
    for term in selected_terms:
        term_df = df[df["term"] == term].copy()

        # Normalize similarity to be in range [0,1] so 1 is the term embedding
        term_df["normalized_similarity"] = term_df["similarity"]

        # Ensure 'month' exists and is properly formatted
        term_df["month"] = term_df["month"].fillna(1).astype(int)  # Default missing months to January
        term_df["date"] = term_df.apply(lambda row: f"{row['year']}-{str(row['month']).zfill(2)}", axis=1)

        # Sort values to make a continuous line graph
        term_df = term_df.sort_values(["year", "month"])

        fig.add_trace(go.Scatter(
            x=term_df['date'],  # Keep full year-month data for sorting
            y=term_df['normalized_similarity'],  
            mode='lines',  # Remove markers
            customdata=term_df[['id', 'year', 'month', 'similarity', 'wrapped_chunk', 'company', 'industry']],
            hovertemplate="<b>ID:</b> %{customdata[0]}<br>"
                        "<b>Year:</b> %{customdata[1]}<br>"
                        "<b>Month:</b> %{customdata[2]}<br>"
                        "<b>Similarity:</b> %{customdata[3]:.3f}<br>"
                        "<b>Statement:</b> %{customdata[4]}<br>"
                        "<b>Company:</b> %{customdata[5]}<br>"
                        "<b>Industry:</b> %{customdata[6]}<br>",
            line=dict(shape="spline", smoothing=0.3, width=2, color=term_color_map[term]),  # Smoother curves, thicker lines
            name=f"Similarity for {term}"
        ))

    # Update layout
    fig.update_layout(
        title="<b>Similarity Over Time for Multiple Terms</b>",
        title_font=dict(size=22, family="Arial", color="white"),
        xaxis=dict(
            title="<b>Year</b>", 
            tickmode="array",  # Show only specific year labels
            tickvals=[f"{year}-01" for year in df["year"].unique()],  # Display only first month of each year
            ticktext=[str(year) for year in df["year"].unique()],  # Show only year labels
            tickangle=0,  
            color="white",
            title_font=dict(size=18)
        ),
        yaxis=dict(title="<b>Similarity (1 = Term Embedding)</b>", range=[0, 1.1], color="white", title_font=dict(size=18)),
        hovermode="closest",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(family="Arial", size=14, color="white"),
        legend=dict(font=dict(color="white")),
        height=800,
        width=1200
    )

    st.plotly_chart(fig, use_container_width=True)

### **Streamlit UI**
st.set_page_config(layout="wide", page_title="Neo4j Term Similarity", page_icon='📖')

st.sidebar.header("Select Terms for Similarity Analysis")

# Load glossary terms from the embedded dataset
df_tfnd_glossary_2023 = pd.read_csv("data/df_tfnd_glossary_2023_embedded.csv")
df_tfnd_glossary_2023["embedding"] = df_tfnd_glossary_2023["embedding"].apply(lambda x: np.array(eval(x), dtype=np.float32))

glossary_terms = df_tfnd_glossary_2023['Term'].unique()

# Sidebar selection for multiple terms
selected_terms = st.sidebar.multiselect("Select one or more terms:", glossary_terms, default=["Biodiversity"])

start_year, end_year = st.sidebar.select_slider(
    "Please select the time range",
    options=list(range(2015, 2024)),  # Proper list of years
    value=(2015, 2023)  # Default selected range
)
years = list(range(start_year, end_year + 1))  # Generate all years in between

if st.sidebar.button("Analyze"):
    st.sidebar.write(f"Fetching embeddings for: {', '.join(selected_terms)}...")
    all_results = []
    term_embeddings = {}

    # Get embeddings for selected terms
    for term in selected_terms:
        term_array = df_tfnd_glossary_2023[df_tfnd_glossary_2023['Term'] == term]
        if term_array.empty:
            st.sidebar.error(f"No embedding found for term: {term}")
            continue
        
        term_embedding = term_array['embedding'].values[0]
        term_embeddings[term] = term_embedding

    if not term_embeddings:
        st.sidebar.error("No valid terms selected!")
        st.stop()

    # Fetch chunks **once per term, for all years at once**
    for term, term_embedding in term_embeddings.items():
        df_results = fetch_chunks_for_term_for_years(years, term_embedding)

        # Convert dictionary to DataFrame
        df_results = pd.DataFrame(df_results)

        # Compute cosine similarity for each term
        compute_term_dist_cosine(df_results, term_embedding)
        df_results["term"] = term  # Track term name
        df_results = compute_deviation(df_results)

        all_results.append(df_results)

    # Combine all results into a single DataFrame
    df_all_terms = pd.concat(all_results, ignore_index=True)

    if "month" not in df_all_terms.columns:
        st.write(df_all_terms.head())
        st.error("Month column is missing! Check data fetching.")
        st.stop()

    # Plot results
    scatterplot_from_multiple_terms(df_all_terms, selected_terms)

# if st.sidebar.button("Analyze"):
#     st.sidebar.write(f"Fetching embeddings for: {', '.join(selected_terms)}...")
#     df_results = None
#     all_results = []

#     for term in selected_terms:
#         # Get the embedding for the selected term
#         term_array = df_tfnd_glossary_2023[df_tfnd_glossary_2023['Term'] == term]
#         if term_array.empty:
#             st.sidebar.error(f"No embedding found for term: {term}")
#             continue
        
#         term_embedding = term_array['embedding'].values[0]

#         # Fetch statement chunks using `fetch_chunks_for_term_for_years`
#         df_results_dict = fetch_chunks_for_term_for_years(years, term_embedding)

#         # Flatten embeddings to a DataFrame
#         df_results = pd.DataFrame([
#             {
#                 "year": year,
#                 "month": record["month"],
#                 "id": record["id"],
#                 "company": record["company"],
#                 "industry": record["industry"],
#                 "chunk": record["chunk_text"],
#                 "embedding": record["embedding"],
#                 "term": term  # Track which term this data belongs to
#             }
#             for year, records in df_results_dict.items()
#             for record in records
#         ])

#         # Compute cosine similarity
#         compute_term_dist_cosine(df_results, term_embedding)

#         # Compute deviation
#         df_results = compute_deviation(df_results)

#         all_results.append(df_results)

#     # Combine all results into a single DataFrame
#     if all_results:
#         df_all_terms = pd.concat(all_results, ignore_index=True)
#         if "month" not in df_all_terms.columns:
#             st.write(df_all_terms.head())
#             st.error("Month column is missing! Check data fetching.")
#             st.stop()  # Stop execution if 'month' is missing
#         # Plot results
#         scatterplot_from_multiple_terms(df_all_terms, selected_terms)
#     else:
#         st.sidebar.error("No data found for the selected terms.")