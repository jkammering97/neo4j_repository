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
    """Creates a scatter plot showing similarity over time for multiple terms, adapting to Streamlit's theme."""

    # Detect Streamlit theme (light or dark)
    theme = st.get_option("theme.base")
    
    # Set colors based on theme
    if theme == "dark":
        bg_color = "black"
        text_color = "white"
        line_colors = px.colors.qualitative.Set1
    else:
        bg_color = "white"
        text_color = "black"
        line_colors = px.colors.qualitative.Set2

    # Assign unique colors for each term
    term_color_map = {term: line_colors[i % len(line_colors)] for i, term in enumerate(selected_terms)}

    fig = go.Figure()

    for term in selected_terms:
        term_df = df[df["term"] == term].copy()
        term_df["normalized_similarity"] = term_df["similarity"]

        term_df["month"] = term_df["month"].fillna(1).astype(int)
        term_df["date"] = term_df.apply(lambda row: f"{row['year']}-{str(row['month']).zfill(2)}", axis=1)

        term_df = term_df.sort_values(["year", "month"])

        fig.add_trace(go.Scatter(
            x=term_df['date'],
            y=term_df['normalized_similarity'],  
            mode='lines',
            customdata=term_df[['id', 'year', 'month', 'similarity', 'wrapped_chunk', 'company', 'industry']],
            hovertemplate="<b>ID:</b> %{customdata[0]}<br>"
                        "<b>Year:</b> %{customdata[1]}<br>"
                        "<b>Month:</b> %{customdata[2]}<br>"
                        "<b>Similarity:</b> %{customdata[3]:.3f}<br>"
                        "<b>Statement:</b> %{customdata[4]}<br>"
                        "<b>Company:</b> %{customdata[5]}<br>"
                        "<b>Industry:</b> %{customdata[6]}<br>",
            line=dict(shape="spline", smoothing=0.3, width=2, color=term_color_map[term]),
            name=f"Similarity for {term}"
        ))

    # Update layout dynamically
    fig.update_layout(
        title="<b>Similarity Over Time for Multiple Terms</b>",
        title_font=dict(size=22, family="Arial", color=text_color),
        xaxis=dict(title="<b>Year</b>", tickmode="array",
                   tickvals=[f"{year}-01" for year in df["year"].unique()],
                   ticktext=[str(year) for year in df["year"].unique()],
                   tickangle=0, color=text_color, title_font=dict(size=18)),
        yaxis=dict(title="<b>Similarity (1 = Term Embedding)</b>", range=[0, 1.1], 
                   color=text_color, title_font=dict(size=18)),
        hovermode="closest",
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(family="Arial", size=14, color=text_color),
        legend=dict(font=dict(color=text_color)),
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
contains = st.sidebar.radio("please select whether the results should contain the terms specifically or not.",options=['yes','no'])
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
        df_results = fetch_chunks_for_term_for_years(years, term, term_embedding, contains)

        # Convert dictionary to DataFrame
        df_results = pd.DataFrame(df_results)
        if df_results.empty:
            st.warning(f"No results found for '{term}' in the selected time range.")
            continue

        if "embedding" not in df_results.columns:
            st.error("No embedding data available for the selected term.")
            continue
        # Compute cosine similarity for each term
        compute_term_dist_cosine(df_results, term_embedding)
        df_results["term"] = term  # Track term name
        df_results = compute_deviation(df_results)

        all_results.append(df_results)
    if not all_results:        
        st.error(f"no results found for any terms.")
        st.stop()
    # Combine all results into a single DataFrame
    df_all_terms = pd.concat(all_results, ignore_index=True)

    if "chunk" in df_all_terms.columns:
        df_all_terms["wrapped_chunk"] = df_all_terms["chunk"].apply(lambda x: "<br>".join(textwrap.wrap(str(x), 50)))
    elif "chunk_text" in df_all_terms.columns:
        df_all_terms.rename(columns={"chunk_text": "chunk"}, inplace=True)
        df_all_terms["wrapped_chunk"] = df_all_terms["chunk"].apply(lambda x: "<br>".join(textwrap.wrap(str(x), 50)))
    else:
        st.error("Missing 'chunk' column in DataFrame! Check data fetching.")
        st.write(df_all_terms.head())
        st.stop()

    if "month" not in df_all_terms.columns:
        st.write(df_all_terms.head())
        st.error("Month column is missing! Check data fetching.")
        st.stop()

    # Plot results
    scatterplot_from_multiple_terms(df_all_terms, selected_terms)
