#%%
from neo4j_access import *
from dotenv import load_dotenv
from neo4j import GraphDatabase
import numpy as np
from numpy.linalg import norm
import pandas as pd
from sentence_transformers import SentenceTransformer

import streamlit as st

import umap

import plotly.graph_objects as go
import plotly.express as px
import textwrap
#%%

def initialize(streamlit_secret=True):
    """Initialize the Neo4j driver using Streamlit secrets for deployment."""
    if streamlit_secret == False:
        URI = os.getenv("NEO4J_URI")
        USERNAME = os.getenv("USERNAME")
        PASSWORD = os.getenv("DB_PASSWORD")
    else: # if using streamlit secrets
        # First, check if credentials are in Streamlit Secrets (for Streamlit Cloud)
        if "connections" in st.secrets:
            URI = st.secrets["connections"]["NEO4J_URI"]
            USERNAME = st.secrets["connections"]["USERNAME"]
            PASSWORD = st.secrets["connections"]["DB_PASSWORD"]
        else:
            # If not using Streamlit Cloud, fall back to environment variables
            URI = os.environ.get("NEO4J_URI")
            USERNAME = os.environ.get("USERNAME")
            PASSWORD = os.environ.get("DB_PASSWORD")
    # Suppress warnings from Neo4j logs
    import logging
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    if not URI or not USERNAME or not PASSWORD:
        raise ValueError("Neo4j credentials are missing! Ensure they are set in GitHub Secrets or your .env file.")

    return GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def compare_glossary_to_statements(glossary_embedding, n=10):
    """Finds the most similar database statements for a given glossary term embedding."""

    driver = initialize()
    similar_statements = []

    with driver.session() as session:
        # Neo4j vector similarity query
        result = session.run("""
            CALL db.index.vector.queryNodes('chunk_embeddings', $n, $glossary_embedding)
            YIELD node AS similarStatement, score
            MATCH (similarStatement)<-[:INCLUDES]-(s:Statement)-[:WAS_GIVEN_AT]->(e:ECC)
            RETURN id(similarStatement) AS id, s.text AS statement, e.year AS year, score
            ORDER BY score DESC
            """, n=n, glossary_embedding=glossary_embedding.tolist())

        # Collect results
        for record in result:
            similar_statements.append({
                'id': record['id'],
                'statement': record['statement'],
                'score': record['score'],
                'year': record['year']
            })

    driver.close()
    return similar_statements

def make_yrl_query_old(year, month, glossary_embedding, chunks_per_month=400, batch_size=100):

    driver = initialize()

    def fetch_batch(session, batch_ids):
        batch_query = """
        MATCH (c:Chunk) WHERE elementId(c) IN $batch_ids
        MATCH (c)<-[:INCLUDES]-(s:Statement)-[:WAS_GIVEN_AT]->(e:ECC)<-[:ARRANGED]-(co:Company)-[:IN_INDUSTRY]->(i:Industry)
        RETURN elementId(c) AS id, c.embedding AS embedding, 
               substring(s.text, c.start_index, c.end_index - c.start_index+1) AS chunk_text,
               s.name AS name, datetime(e.time).year AS year, datetime(e.time).month AS month, 
               co.name AS company, i.name AS industry
        ORDER BY year DESC, month DESC
        """
        results = session.run(batch_query, batch_ids=batch_ids)
        return [dict(record) for record in results]

    with driver.session() as session:
        # Get IDs of all chunks containing the specific year and month
        id_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $n, $glossary_embedding)
        YIELD node AS similarChunk, score
        MATCH (similarChunk)<-[:INCLUDES]-(s:Statement)-[:WAS_GIVEN_AT]->(e:ECC)
        WHERE datetime(e.time).year = $year  // Fetch ALL data for the year
        RETURN elementId(similarChunk) AS chunk_id, elementId(s) AS statement_id, 
            s.text AS statement, datetime(e.time).year AS year, 
            datetime(e.time).month AS month, score
        ORDER BY year DESC, month ASC, score DESC
        """
        ids = [record["chunk_id"] for record in session.run(
            id_query, year=year, month=month, n=chunks_per_month, glossary_embedding=glossary_embedding
        )]

        total_ids = len(ids)
        chunks = []
        for i in range(0, total_ids, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_chunks = fetch_batch(session, batch_ids)

            for chunk in batch_chunks:
                chunk['embedding'] = np.array(chunk['embedding'], dtype=np.float32)
                chunk['year'] = year
                chunk['month'] = month
                chunks.append(chunk)
            
    driver.close()
    print(f'Processing size {len(chunks)} for {year}-{month}')
    return chunks

def make_yrl_query(year, glossary_embedding, chunks_per_year=250, batch_size=100):
    driver = initialize()

    def fetch_batch(session, batch_ids):
        batch_query = """
        MATCH (c:Chunk) WHERE elementId(c) IN $batch_ids
        MATCH (c)<-[:INCLUDES]-(s:Statement)-[:WAS_GIVEN_AT]->(e:ECC)<-[:ARRANGED]-(co:Company)-[:IN_INDUSTRY]->(i:Industry)
        RETURN elementId(c) AS id, c.embedding AS embedding, 
               substring(s.text, c.start_index, c.end_index - c.start_index+1) AS chunk_text,
               s.name AS name, datetime(e.time).year AS year, datetime(e.time).month AS month, 
               co.name AS company, i.name AS industry
        ORDER BY year DESC, month ASC
        """
        results = session.run(batch_query, batch_ids=batch_ids)
        return [dict(record) for record in results]

    with driver.session() as session:
        # Fetch all chunk IDs for the **entire year**, but keep month information
        id_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $n, $glossary_embedding)
        YIELD node AS similarChunk, score
        MATCH (similarChunk)<-[:INCLUDES]-(s:Statement)-[:WAS_GIVEN_AT]->(e:ECC)
        WHERE datetime(e.time).year = $year  // Fetch all data for the year
        RETURN elementId(similarChunk) AS chunk_id, elementId(s) AS statement_id, 
               s.text AS statement, datetime(e.time).year AS year, 
               datetime(e.time).month AS month, score
        ORDER BY year DESC, month ASC, score DESC
        """
        ids = [record["chunk_id"] for record in session.run(
            id_query, year=year, n=chunks_per_year, glossary_embedding=glossary_embedding
        )]

        total_ids = len(ids)
        chunks = []
        for i in range(0, total_ids, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_chunks = fetch_batch(session, batch_ids)

            for chunk in batch_chunks:
                chunk['embedding'] = np.array(chunk['embedding'], dtype=np.float32)
                chunk['year'] = year
                chunk['month'] = chunk['month']  # Keep month data
                chunks.append(chunk)
    driver.close()
    print(f'Processing size {len(chunks)} for year {year}')
    return chunks

# @st.cache_data(show_spinner=True)
def fetch_chunks_for_term_for_years(years, term, glossary_embedding, contains, streamlit_secret=True, chunks_per_year=50, batch_size=200):
    """makes two calls to the NEO4J database based on the term and its glossary embedding:
    * _id_query:_ query for chunks of the ECC transcripts that are semantically similar to the search term:
        * uses vector search to find similar text chunks either by 
            * cosine similarity or 
            * using CONTAINS for an explicit search query
    * _fetch_batch:_ gets the additional information about the queried data including: 
        * [company, industry, year and month of the conference]
    """
    
    driver = initialize(streamlit_secret=streamlit_secret)

    with driver.session() as session:
        term_filter = "AND s.text CONTAINS $term" if contains == "yes" else ""

        id_query = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $n, $glossary_embedding)
        YIELD node AS similarChunk, score
        MATCH (similarChunk)<-[:INCLUDES]-(s:Statement)-[:WAS_GIVEN_AT]->(e:ECC)
        WHERE datetime(e.time).year IN $years
        AND size(s.text) > 45  // Adjust length requirement here
        {term_filter}  
        RETURN elementId(similarChunk) AS chunk_id, elementId(s) AS statement_id, 
            s.text AS statement, datetime(e.time).year AS year, 
            datetime(e.time).month AS month, similarChunk.embedding AS embedding, score
        ORDER BY year DESC, month ASC, score DESC
        """

        ids = [record["chunk_id"] for record in session.run(
            id_query, years=years, term=term, n=chunks_per_year, glossary_embedding=glossary_embedding
        )]

        total_ids = len(ids)
        chunks = []

        def fetch_batch(session, batch_ids):
            batch_query = """
            MATCH (c:Chunk) WHERE elementId(c) IN $batch_ids
            MATCH (c)<-[:INCLUDES]-(s:Statement)-[:WAS_GIVEN_AT]->(e:ECC)<-[:ARRANGED]-(co:Company)-[:IN_INDUSTRY]->(i:Industry)
            RETURN elementId(c) AS id, c.embedding AS embedding, 
                substring(s.text, c.start_index, c.end_index - c.start_index+1) AS chunk_text,
                s.name AS name, datetime(e.time).year AS year, datetime(e.time).month AS month, 
                co.name AS company, i.name AS industry
            ORDER BY year DESC, month ASC
            """
            results = session.run(batch_query, batch_ids=batch_ids)
            return [dict(record) for record in results]

        for i in range(0, total_ids, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_chunks = fetch_batch(session, batch_ids)

            for chunk in batch_chunks:
                chunk['embedding'] = np.array(chunk['embedding'], dtype=np.float32)  # Ensure embedding is included
                chunks.append(chunk)

    driver.close()

    return chunks

def flatten_embeddings(dict_yrl_results: dict):
    flattened_data = []

    for year, records in dict_yrl_results.items():
        for record in records:
            flattened_data.append({
                "year": year,
                "month": record.get("month", 1),  # Default to January if missing
                "id": record["id"],
                "company": record["company"],
                "industry": record["industry"],
                "chunk": record["chunk_text"],
                "embedding": record["embedding"]
            })

    df_yrl_terms = pd.DataFrame(flattened_data)
    return df_yrl_terms

def compute_term_dist_cosine(df: pd.DataFrame, term_embedding: np.array):
    # Convert list of arrays to a 2D NumPy array
    embedding_matrix = np.vstack(df["embedding"].values)
    # Compute norms in a vectorized way
    embedding_norms = np.linalg.norm(embedding_matrix, axis=1)
    term_norm = np.linalg.norm(term_embedding)

    # Compute cosine similarity for all rows in one go
    cosine_similarities = np.dot(embedding_matrix, term_embedding) / (embedding_norms * term_norm)

    # Assign results directly to the DataFrame
    df["similarity"] = cosine_similarities

def wrap_text(text, width=50):
    """Wraps text into multiple lines for better readability in hover text."""
    return "<br>".join(textwrap.wrap(text, width))

def fit_umap_model(embeddings, n_neighbors=5, n_components=10, min_dist=0.11, metric='euclidean', random_state=42):
    umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric=metric, random_state=random_state)
    umap_model.fit(embeddings)
    return umap_model

def scatterplot_from_embeddings(yrl_df, term, term_embedding, normalized_sizes):
    # Apply text wrapping to chunks (statements)
    yrl_df["wrapped_chunk"] = yrl_df["chunk"].apply(lambda x: wrap_text(x, width=50))

    # Scatter plot for all embeddings
    trace_all = go.Scattergl(
        x=yrl_df['year'],
        y=yrl_df['embedding_reduced'],
        mode='markers',
        customdata=yrl_df[['id', 'deviation', 'similarity', 'company', 'industry', 'wrapped_chunk']],  # Use wrapped text
        hovertemplate=(
            "<b>ID:</b> %{customdata[0]}<br>" +
            "<b>Deviation:</b> %{customdata[1]}<br>" +
            "<b>Similarity (cosine):</b> %{customdata[2]}<br>" +
            "<b>Company:</b> %{customdata[3]}<br>" +
            "<b>Industry:</b> %{customdata[4]}<br>" +
            "<b>Statement:</b> %{customdata[5]}<br>"  # Wrapped text ensures better readability
        ),
        marker=dict(
            size=normalized_sizes,
            opacity=0.7,
            color='royalblue',  # Color theme
            showscale=False
        ),
        name=""
    )

    # Scatter plot for the actual term embedding
    trace_term = go.Scattergl(
        x=yrl_df["year"],
        y=[term_embedding] * len(yrl_df["year"]),  # Repeat the term embedding value for all years
        mode="markers",
        marker=dict(
            size=14,
            color="limegreen",
            symbol="diamond",
            line=dict(width=2, color="black")
        ),
        name=f"{term} Term Embedding"
    )

    # Create final figure
    fig = go.Figure(data=[trace_all, trace_term])

    # Update layout with dark background and improved font styles
    fig.update_layout(
        title=f"<b>Chunk Embeddings Over Time vs. {term}</b>",
        title_font=dict(size=18, family="Arial", color="white"),
        xaxis=dict(title="<b>Year</b>", tickmode="linear", dtick=1, color="white"),
        yaxis=dict(title="<b>Projected Embedding Value</b>", color="white"),
        hovermode="closest",
        plot_bgcolor="lightgrey",  # Dark background
        paper_bgcolor="lightgrey",
        font=dict(family="Arial", size=12, color="white"),
        legend=dict(font=dict(color="white"))
    )

    fig.show()

def scatterplot_from_cosine_similarity(yrl_df, term, term_embedding, normalized_sizes):
    # Apply text wrapping to chunks (statements) for better readability
    yrl_df["wrapped_chunk"] = yrl_df["chunk"].apply(lambda x: wrap_text(x, width=50))

    # Compute deviation as `1 - similarity` to measure how far embeddings are from the term
    yrl_df["deviation"] = 1 - yrl_df["similarity"]

    # Scatter plot for all embeddings
    trace_all = go.Scattergl(
        x=yrl_df['year'],
        y=yrl_df['deviation'],  # Use deviation instead of UMAP-reduced embedding
        mode='markers',
        customdata=yrl_df[['id', 'similarity', 'company', 'industry', 'wrapped_chunk']],  # Include relevant data
        hovertemplate=(
            "<b>ID:</b> %{customdata[0]}<br>" +
            "<b>Similarity (cosine):</b> %{customdata[1]:.3f}<br>" +
            "<b>Company:</b> %{customdata[2]}<br>" +
            "<b>Industry:</b> %{customdata[3]}<br>" +
            "<b>Statement:</b> %{customdata[4]}<br>"  # Wrapped text for readability
        ),
        marker=dict(
            size=normalized_sizes,
            opacity=0.7,
            color='royalblue',  # Color theme
            showscale=False
        ),
        name="Chunk Embeddings"
    )

    # Scatter plot for the actual term embedding
    trace_term = go.Scattergl(
        x=yrl_df["year"],
        y=[0] * len(yrl_df["year"]),  # Term embedding is at `deviation = 0`
        mode="markers",
        marker=dict(
            size=14,
            color="limegreen",
            symbol="diamond",
            line=dict(width=2, color="black")
        ),
        name=f"{term} Term Embedding"
    )

    # Create final figure
    fig = go.Figure(data=[trace_all, trace_term])

    # Update layout with a dark theme and better font styles
    fig.update_layout(
        title=f"<b>Similarity-Based Deviation Over Time vs. {term}</b>",
        title_font=dict(size=18, family="Arial", color="white"),
        xaxis=dict(title="<b>Year</b>", tickmode="linear", dtick=1, color="white"),
        yaxis=dict(title="<b>Deviation from Term (1 - Similarity)</b>", color="white"),
        hovermode="closest",
        plot_bgcolor="lightgrey",  # Dark background
        paper_bgcolor="lightgrey",
        font=dict(family="Arial", size=12, color="white"),
        legend=dict(font=dict(color="white"))
    )

    fig.show()

#%%
