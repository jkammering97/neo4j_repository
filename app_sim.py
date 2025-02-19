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

def scatterplot_from_multiple_terms(df, selected_terms, mode):
    """Creates a scatter plot showing similarity over time for multiple terms, adapting to Streamlit's theme."""

    # Detect Streamlit theme (light or dark)
    theme = st.get_option("theme.base")
    if theme is None:
        theme = "dark"  # Default to dark mode
    # Set colors based on theme
    if theme == "dark":
        bg_color = "#1E1E1E"
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
            mode="lines" if mode == "Lines" else "markers",
            customdata=term_df[['id', 'year', 'month', 'similarity', 'wrapped_chunk', 'company', 'industry']],
            hovertemplate="<b>ID:</b> %{customdata[0]}<br>"
                        "<b>Year:</b> %{customdata[1]}<br>"
                        "<b>Month:</b> %{customdata[2]}<br>"
                        "<b>Similarity:</b> %{customdata[3]:.3f}<br>"
                        "<b>Statement:</b> %{customdata[4]}<br>"
                        "<b>Company:</b> %{customdata[5]}<br>"
                        "<b>Industry:</b> %{customdata[6]}<br>",
            line=dict(shape="spline", smoothing=0.3, width=2, color=term_color_map[term]) if mode == "Lines" else None,
            marker=dict(size=5, opacity=0.6) if mode == "Markers" else "Lines",
            name="<br>".join(textwrap.wrap(f"Similarity for {term}", width=30))  # Wrap legend text
        ))

    # Dynamically adjust width based on number of years
    num_years = df["year"].nunique()
    base_width = 300 + num_years * 50  # Adjust width dynamically
    width = min(base_width, 1200)  # Cap width to prevent excessive stretching

    # Update layout dynamically
    fig.update_layout(
        title="<b>Similarity Over Time for Multiple Terms</b>",
        title_font=dict(size=22, family="Arial", color=text_color),
        xaxis=dict(
            title="<b>Year</b>",
            tickmode="array",
            tickvals=[f"{y}-01" for y in df["year"].unique()],  # Use correct loop variable
            ticktext=[str(y) for y in df["year"].unique()],  # Fixed: Use 'y' instead of 'year'
            tickangle=0,
            color=text_color,
            title_font=dict(size=18),
        ),
        yaxis=dict(title="<b>Similarity (1 = Term Embedding)</b>", range=[0, 1.1], color=text_color, title_font=dict(size=18)),
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
if "term_results_cache" not in st.session_state:
    st.session_state.term_results_cache = {}  # Dictionary to store results

st.set_page_config(layout="wide", page_title="Neo4j Term Similarity", page_icon='📖')

st.sidebar.header("Select Terms for Similarity Analysis")

# Load glossary terms from the embedded dataset
df_tfnd_glossary_2023 = pd.read_json("data/df_tfnd_glossary_2023_embedded.json", orient="records")
df_tfnd_glossary_2023["embedding"] = df_tfnd_glossary_2023["embedding"].apply(lambda x: np.array(x, dtype=np.float32))

glossary_terms = df_tfnd_glossary_2023['Term'].unique()

# Sidebar selection for multiple terms
selected_terms = st.sidebar.multiselect("Select one or more terms:", glossary_terms, default=["Biodiversity"])
contains = st.sidebar.radio("please select whether the results should contain the terms specifically or not.",options=['yes','no'])
start_year, end_year = st.sidebar.select_slider(
    "Please select the time range",
    options=list(range(2015, 2024)),  # Proper list of years
    value=(2015, 2023)  # Default selected range
)
n_chunks_per_year = st.sidebar.selectbox("please select the number of chunks to be evaluated per year",options=[25,50,75,100,125,150,250])

years = list(range(start_year, end_year + 1))  # Generate all years in between

if "plot_mode" not in st.session_state:
    st.session_state.plot_mode = "Markers"

    plot_mode = st.radio(
        "Select Plot Mode:", 
        ["Lines", "Markers"], 
        index=1 if st.session_state.plot_mode == "Markers" else 0
    )

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

    # Stop if no valid terms
    if not term_embeddings:
        st.sidebar.error("No valid terms selected!")
        st.stop()

    # Fetch chunks once per term, only if not already in session state
    for term, term_embedding in term_embeddings.items():
        # Define cache key for the term
        cache_key = f"{term}_{contains}_{start_year}_{end_year}_{n_chunks_per_year}"

        # Check if cached data exists and matches all parameters
        cached_data = st.session_state.term_results_cache.get(cache_key)

        if cached_data:
            df_results = cached_data["data"]
        else:
            st.sidebar.write(f"Fetching new data for: {term}...")
            df_results = fetch_chunks_for_term_for_years(years, term, term_embedding, contains, chunks_per_year=n_chunks_per_year)

            # Store in session cache
            st.session_state.term_results_cache[cache_key] = {
                "data": df_results,
                "contains": contains,
                "start_year": start_year,
                "end_year": end_year,
                "n_chunks_per_year": n_chunks_per_year
            }

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
        st.error(f"No results found for any terms.")
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

if "plot_mode" not in st.session_state:
    st.session_state.plot_mode = "Markers"

# UI: Radio button for selecting plot mode
plot_mode = st.radio(
    "Select Plot Mode:", 
    ["Lines", "Markers"], 
    index=1 if st.session_state.plot_mode == "Markers" else 0
)

# If the mode changes, update session state and rerun
if plot_mode != st.session_state.plot_mode:
    st.session_state.plot_mode = plot_mode
    st.rerun()

# Check if we have stored data before rendering
if "df_all_terms" in st.session_state and st.session_state.df_all_terms is not None:
    # Render graph with cached data
    scatterplot_from_multiple_terms(st.session_state.df_all_terms, selected_terms, st.session_state.plot_mode)
else:
    st.warning("Click 'Analyze' to fetch data before visualizing.")