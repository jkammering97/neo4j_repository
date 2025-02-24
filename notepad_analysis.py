#%%
from legacy.neo4j_access import *
from dotenv import load_dotenv
from neo4j import GraphDatabase
import numpy as np
from numpy.linalg import norm
import pandas as pd
#from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px
import textwrap
#%%
from glossary_similarity import fetch_chunks_for_term_for_years_biodiv_subset, get_biodiversity_subset
#%%
#%%
terms = ['Nature-related systemic risks', 
        'Nature-related physical risks',
        'Nature-related transition risks',
        'Nature-related opportunities',
        'Ecosystem protection, restoration and regeneration opportunity']
sub_tnfd_glossary = df_tfnd_glossary_2023[df_tfnd_glossary_2023['Term'].isin(terms)]
# %%
time_frame = [2015,2016,2017,2018,2019,2020,2021,2022,2023]
all_terms_similar_embeddings = pd.DataFrame()

# fetch biodiv. subset from database
driver, biodiversity_subset = get_biodiversity_subset(time_frame,chunks_per_year=100000,streamlit_secret=False)

for i, row in sub_tnfd_glossary.iterrows():
    
    term = row['Term']
    embedding = row['embedding']

    print(f'processing term {term} [{i} out of {len(df_tfnd_glossary_2023)}] ..')

    chunks = fetch_chunks_for_term_for_years_biodiv_subset(driver,
                                                           time_frame,
                                                           term,
                                                           embedding,
                                                           biodiversity_subset, 
                                                           streamlit_secret=False,
                                                           chunks_per_year=100)
    results = pd.DataFrame(chunks)
    results['term_embedding'] = [np.array(embedding, dtype=np.float32)] * len(results)
    results['term'] = term

    print(f'results for {term}: {len(results)}')
    
    all_terms_similar_embeddings = pd.concat([all_terms_similar_embeddings, results])

#%%
all_terms_similar_embeddings.to_pickle('data/alltermssimilarembeddings_n100_nobiodiv_subset.pkl')
#%%
all_terms_similar_embeddings_ = pd.read_pickle('data/cosine_similarlity_all_terms_n100_biodiv_subset.pkl')
#%%
similar_terms_cosine_n100 = pd.read_pickle('data/cosine_similarlity_all_terms_n100_mpnet_years15-23.pkl')

#%%
# %%
# Earlier, you reduced the embeddings using UMAP before calculating similarities.
# UMAP transforms the embeddings to fewer dimensions, which may make distances appear smaller (higher similarity).
similar_terms_cosine_n50['similarity'].describe()

#%%
similar_terms_cosine_n100['similarity'].describe()
# %%
# Compute summary statistics and ranking for similarity scores
similarity_stats_n100 = similar_terms_cosine_n100.groupby(["term"])["similarity"].agg(["mean", "std", "max", "min", "count"]).reset_index()

# Rank terms based on mean similarity score (descending order)
similarity_stats_n100 = similarity_stats_n100.sort_values(by="mean", ascending=False)

# %%
# which terms to use 
terms = ['Biodiversity', 'Double materiality', 'Biodiversity offsets', 'Cut off dates (related to no- deforestation and no-conversion commitments)']
df = similar_terms_cosine_n100[similar_terms_cosine_n100['term'].isin(terms)]

#%%
#%%
df["month"] = df["month"].fillna(1).astype(int)

# Convert year & month into a continuous time format
df["time"] = df["year"] + (df["month"] - 1) / 12
#%%
# Function to wrap long text
def wrap_text(text, width=50):
    return "<br>".join(textwrap.wrap(text, width=width))
#%%
# Apply text wrapping to chunk text

df["wrapped_chunk"] = df["chunk_text"].apply(lambda x: wrap_text(str(x), width=50))
#%%
# Create scatter plot
fig = px.scatter(
    df,
    x="time",
    y="score",
    hover_data=["term", "wrapped_chunk", "company"],
    title="score Trends Over Time",
    labels={"time": "Year", "similarity": "Similarity Score (1 = Term Embedding)"}
)

# Adjust marker size & opacity for readability
fig.update_traces(marker=dict(size=5, opacity=0.5))

# Improve hover template with line breaks
fig.update_traces(hovertemplate="<b>Year:</b> %{x:.1f}<br>"
                                "<b>Similarity Score:</b> %{y:.3f}<br>"
                                "<b>Term:</b> %{customdata[0]}<br>"
                                "<b>Statement:</b> %{customdata[1]}<br>"
                                "<b>Company:</b> %{customdata[2]}")

# Format x-axis to show only years
fig.update_layout(
    showlegend=False,  # Remove legend
    xaxis=dict(
        tickmode="array",
        tickvals=df["year"].unique(),  # Show only year markers
        ticktext=[str(y) for y in df["year"].unique()],
        title="<b>Year</b>"
    ),
    yaxis=dict(range=[0, 1.1]),
    height=700
)

fig.show()
# %%
risk_glossary_terms = ['Ecosystem stability risk', 
                       'Nature-related systemic risks',
                       'Nature-related transition risks']
opportunities_terms = ['Sustainable use of natural resources opportunity',
                       'Ecosystem protection, restoration and regeneration opportunity']

df_opportunities = all_terms_similar_embeddings_[all_terms_similar_embeddings_["term"].isin(opportunities_terms)]
df_risks = all_terms_similar_embeddings_[all_terms_similar_embeddings_["term"].isin(risk_glossary_terms)]
# %%
# Combine data for grouped visualization
df_opportunities["Group"] = "TNFD - Opportunities"
df_risks["Group"] = "TNFD - Risks"
df_combined = pd.concat([df_opportunities, df_risks])

# Plot
fig = px.violin(
    df_combined, 
    x="Group", 
    y="score", 
    color="Group",
    box=True,  # Show boxplot inside the violin
    points="all",  # Show individual data points
    title="Similarity Score Distribution for Opportunities vs. Risks"
)

# Display the figure
fig.show()
# %%



def plot_similarity_density(df):
    """
    Plots the Gaussian-like similarity distribution for all terms across different years.
    Parameters:
        df (pd.DataFrame): DataFrame containing 'year' and 'score' columns.
    """
    # Ensure necessary columns exist
    if "year" not in df or "score" not in df:
        raise ValueError("DataFrame must contain 'year' and 'score' columns")

    plt.figure(figsize=(10, 6))

    # Plot KDE distribution per year
    unique_years = sorted(df["year"].unique())
    for year in unique_years:
        subset = df[df["year"] == year]
        sns.kdeplot(subset["score"], label=str(year), fill=True, alpha=0.4)

    # Formatting
    plt.xlabel("Similarity Score")
    plt.ylabel("Density")
    plt.title("Similarity Score Distribution Over Time")
    plt.xlim(0.5, 1)  # Restrict similarity score range
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
#%%
def scatterplot_by_terms(df, terms):
    # Ensure necessary columns exist
    required_cols = ["year", "score", "term", "chunk_text", "company", "industry"]
    df["time"] = df["year"] + (df["month"] - 1) / 12
    # Function to wrap long text
    def wrap_text(text, width=50):
        return "<br>".join(textwrap.wrap(str(text), width=width))

    # Apply text wrapping to statements
    df["wrapped_chunk"] = df["chunk_text"].apply(lambda x: wrap_text(x, width=50))

    # Get unique terms
    if not terms:
        unique_terms = df["term"].unique()
    else:
        unique_terms = terms

    fig = go.Figure()

    # Assign colors dynamically
    colors = px.colors.qualitative.Set1  # Pick a predefined color palette

    for _, term in enumerate(unique_terms):
        term_df = df[df["term"] == term]
        color = "black"

        trace = go.Scattergl(
            x=term_df["time"],
            y=term_df["score"],
            mode="markers",
            customdata=term_df[["id", "score", "company", "industry", "wrapped_chunk"]],
            hovertemplate=(
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>Similarity score (cosine):</b> %{customdata[1]:.3f}<br>" +
                "<b>Company:</b> %{customdata[2]}<br>" +
                "<b>Industry:</b> %{customdata[3]}<br>" +
                "<b>Statement:</b> %{customdata[4]}<br>"
            ),
            marker=dict(
                size=6,
                opacity=0.7,
                color=color,
                showscale=False
            ),
            name=term
        )
                    

        fig.add_trace(trace)


    # Update layout
    fig.update_layout(
        title="<b>Similarity-Based score (cosine) Over Time by Term</b>",
        xaxis=dict(title="<b>Year</b>", tickmode="linear", dtick=1),
        yaxis=dict(title="<b>Similarity Score</b>"),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        legend_title="Terms",
        height=700
    )

    fig.show()

#%%
def scatterplot_compare_terms(df1, df2, label1="Dataset 1", label2="Dataset 2"):
    """
    Plots similarity scores over time from two different datasets with distinct colors,
    including a subtle mean trend line for each dataset.
    """
    required_cols = ["year", "score", "term", "chunk_text", "company", "industry", "month"]
    for df in [df1, df2]:
        for col in required_cols:
            if col not in df:
                raise ValueError(f"DataFrame is missing required column: {col}")

    df1["time"] = df1["year"] + (df1["month"] - 1) / 12
    df2["time"] = df2["year"] + (df2["month"] - 1) / 12

    def wrap_text(text, width=50):
        return "<br>".join(textwrap.wrap(str(text), width=width))

    df1["wrapped_chunk"] = df1["chunk_text"].apply(lambda x: wrap_text(x, width=50))
    df2["wrapped_chunk"] = df2["chunk_text"].apply(lambda x: wrap_text(x, width=50))

    unique_terms = set(df1["term"].unique()).union(set(df2["term"].unique()))

    fig = go.Figure()

    # Assign colors
    color1 = "black"
    color2 = "lightgreen"
    mean_color1 = "gray"
    mean_color2 = "green"

    for term in unique_terms:
        term_df1 = df1[df1["term"] == term]
        term_df2 = df2[df2["term"] == term]

        if not term_df1.empty:
            fig.add_trace(go.Scattergl(
                x=term_df1["time"], y=term_df1["score"], mode="markers",
                marker=dict(size=8, opacity=0.7, color=color1),
                customdata=term_df1[["id", "score", "company", "industry", "wrapped_chunk"]],
                hovertemplate="<b>ID:</b> %{customdata[0]}<br>"
                              "<b>Similarity score (cosine):</b> %{customdata[1]:.3f}<br>"
                              "<b>Company:</b> %{customdata[2]}<br>"
                              "<b>Industry:</b> %{customdata[3]}<br>"
                              "<b>Statement:</b> %{customdata[4]}<br>",
                name=f"{label1}: {term}"
            ))

        if not term_df2.empty:
            fig.add_trace(go.Scattergl(
                x=term_df2["time"], y=term_df2["score"], mode="markers",
                marker=dict(size=6, opacity=0.9, color=color2),
                customdata=term_df2[["id", "score", "company", "industry", "wrapped_chunk"]],
                hovertemplate="<b>ID:</b> %{customdata[0]}<br>"
                              "<b>Similarity score (cosine):</b> %{customdata[1]:.3f}<br>"
                              "<b>Company:</b> %{customdata[2]}<br>"
                              "<b>Industry:</b> %{customdata[3]}<br>"
                              "<b>Statement:</b> %{customdata[4]}<br>",
                name=f"{label2}: {term}"
            ))

    # Compute mean similarity per year
    mean_df1 = df1.groupby("year")["score"].mean().reset_index()
    mean_df2 = df2.groupby("year")["score"].mean().reset_index()

    # Add trend lines
    fig.add_trace(go.Scatter(
        x=mean_df1["year"], y=mean_df1["score"], mode="lines",
        line=dict(color=mean_color1, width=2, dash="dash"),
        name=f"Mean {label1}"
    ))

    fig.add_trace(go.Scatter(
        x=mean_df2["year"], y=mean_df2["score"], mode="lines",
        line=dict(color=mean_color2, width=2, dash="dash"),
        name=f"Mean {label2}"
    ))

    # Update layout
    fig.update_layout(
        title="<b>Comparison of Similarity Scores Over Time</b>",
        xaxis=dict(title="<b>Year</b>", tickmode="linear", dtick=1),
        yaxis=dict(title="<b>Similarity Score</b>"),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        legend_title="Dataset & Terms",
        height=700
    )

    fig.show()
# %%
def scatterplot_compare_terms_jitter(df1, df2, label1="Dataset 1", label2="Dataset 2"):
    """
    Plots similarity scores over time from two different datasets with distinct colors.

    Parameters:
        df1 (pd.DataFrame): First dataset containing 'year', 'score', 'term', 'chunk_text', 'company', and 'industry'.
        df2 (pd.DataFrame): Second dataset with the same structure as df1.
        label1 (str): Label for the first dataset in the legend.
        label2 (str): Label for the second dataset in the legend.
    """
    # Ensure necessary columns exist
    required_cols = ["year", "score", "term", "chunk_text", "company", "industry", "month"]
    for df in [df1, df2]:
        for col in required_cols:
            if col not in df:
                raise ValueError(f"DataFrame is missing required column: {col}")

    # Convert year & month into a continuous time format
    df1["time"] = df1["year"] + (df1["month"] - 1) / 12
    df2["time"] = df2["year"] + (df2["month"] - 1) / 12

    # Add small jitter to reduce overlap
    df1["time"] += np.random.normal(0, 0.001, size=len(df1))  # Small randomness in x-axis
    df2["time"] += np.random.normal(0, 0.001, size=len(df2))

    # Function to wrap long text
    def wrap_text(text, width=50):
        return "<br>".join(textwrap.wrap(str(text), width=width))

    # Apply text wrapping
    df1["wrapped_chunk"] = df1["chunk_text"].apply(lambda x: wrap_text(x))
    df2["wrapped_chunk"] = df2["chunk_text"].apply(lambda x: wrap_text(x))

    # Get unique terms across both datasets
    unique_terms = set(df1["term"].unique()).union(set(df2["term"].unique()))

    fig = go.Figure()

    # Assign two distinct colors
    color1 = "black"
    color2 = "springgreen"

    for term in unique_terms:
        # Filter for the term in each dataset
        term_df1 = df1[df1["term"] == term]
        term_df2 = df2[df2["term"] == term]

        # Plot first dataset points (higher opacity for better visibility)
        if not term_df1.empty:
            fig.add_trace(go.Scattergl(
                x=term_df1["time"],
                y=term_df1["score"],
                mode="markers",
                marker=dict(size=6, opacity=0.7, color=color1),  # Increased opacity and size
                customdata=term_df1[["id", "score", "company", "industry", "wrapped_chunk"]],
                hovertemplate=(
                    "<b>ID:</b> %{customdata[0]}<br>"
                    "<b>Similarity score (cosine):</b> %{customdata[1]:.3f}<br>"
                    "<b>Company:</b> %{customdata[2]}<br>"
                    "<b>Industry:</b> %{customdata[3]}<br>"
                    "<b>Statement:</b> %{customdata[4]}<br>"
                ),
                name=f"{label1}: {term}"
            ))

        # Plot second dataset points (lower opacity to reveal overlap)
        if not term_df2.empty:
            fig.add_trace(go.Scattergl(
                x=term_df2["time"],
                y=term_df2["score"],
                mode="markers",
                marker=dict(size=8, opacity=0.5, color=color2),  # Lower opacity to reduce clutter
                customdata=term_df2[["id", "score", "company", "industry", "wrapped_chunk"]],
                hovertemplate=(
                    "<b>ID:</b> %{customdata[0]}<br>"
                    "<b>Similarity score (cosine):</b> %{customdata[1]:.3f}<br>"
                    "<b>Company:</b> %{customdata[2]}<br>"
                    "<b>Industry:</b> %{customdata[3]}<br>"
                    "<b>Statement:</b> %{customdata[4]}<br>"
                ),
                name=f"{label2}: {term}"
            ))

    # Update layout
    fig.update_layout(
        title="<b>Comparison of Similarity Scores Over Time</b>",
        xaxis=dict(title="<b>Year</b>", tickmode="linear", dtick=1),
        yaxis=dict(title="<b>Similarity Score</b>"),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        legend=dict(
            title="Dataset & Terms",
            x=1.05,  # Move legend slightly right
            y=1, 
            tracegroupgap=5,  # Reduce gap between legend items
            font=dict(size=10),  # Reduce font size
        ),
        height=700
    )

    fig.show()
# %%
def boxplot_terms_over_time(df, terms, bin_size=2):
    """
    Creates boxplots of similarity score density over time for selected terms.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'year', 'score', and 'term'.
        terms (list): List of terms to plot.
        bin_size (int): The number of years per bin (default: 2 years).
    """
    # Ensure necessary columns exist
    required_cols = ["year", "score", "term"]
    for col in required_cols:
        if col not in df:
            raise ValueError(f"Missing required column: {col}")

    # Filter dataframe for selected terms
    df_filtered = df[df["term"].isin(terms)].copy()

    # Create bins for time periods
    df_filtered["year_bin"] = (df_filtered["year"] // bin_size) * bin_size  # Group into bins of `bin_size` years

    # Assign distinct colors dynamically
    color_palette = px.colors.qualitative.Set1  # Choose a distinct color set
    term_colors = {term: color_palette[i % len(color_palette)] for i, term in enumerate(terms)}

    # Create boxplot using Plotly
    fig = px.box(
        df_filtered,
        x="year_bin",
        y="score",
        color="term",
        points="all",  # Show all individual points in the boxplot
        title="Similarity Score Distribution Over Time",
        labels={"year_bin": "Time Period", "score": "Similarity Score"},
        category_orders={"year_bin": sorted(df_filtered["year_bin"].unique())},
        color_discrete_map=term_colors  # Assign distinct colors per term
    )

    # Update layout for better readability
    fig.update_layout(
        xaxis=dict(title="Year (Binned)", tickmode="linear", dtick=bin_size),
        yaxis=dict(title="Similarity Score"),
        boxmode="group",  # Group boxplots per time period
        height=700
    )

    fig.show()

# %%
def plot_term_trends_over_time(df, top_n=20, bin_size=2):
    """
    Plots how terms have increased over time in similarity score.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'year_bin', 'term', and 'score' columns.
        top_n (int): Number of terms to display with the **highest increase**.
    """
    

    df["year_bin"] = (df["year"] // bin_size) * bin_size  # Group into bins of `bin_size` years

    # Compute mean similarity score per term per time period
    term_trends = df.groupby(["year_bin", "term"])["score"].mean().reset_index()

    # Pivot to get terms as rows, time bins as columns
    term_pivot = term_trends.pivot(index="term", columns="year_bin", values="score")

    # Compute change from first to last available time period
    term_pivot["score_change"] = term_pivot.iloc[:, -1] - term_pivot.iloc[:, 0]  # Last column - first column

    # Get top N increasing terms
    top_terms = term_pivot.nlargest(top_n, "score_change").index.tolist()

    # Filter dataset to include only the top increasing terms
    df_filtered = term_trends[term_trends["term"].isin(top_terms)]

    # Plot with Plotly
    fig = px.line(
        df_filtered, x="year_bin", y="score", color="term",
        markers=True, title=f"Top {top_n} Terms with Highest Score Increase Over Time",
        labels={"year_bin": "Time Period", "score": "Mean Similarity Score"},
    )

    # Improve layout
    fig.update_layout(
        xaxis=dict(title="Year Bin", tickmode="array", tickvals=df["year_bin"].unique()),
        yaxis=dict(title="Mean Similarity Score"),
        height=700,
        legend_title="Term",
    )

    fig.show()
# %%
