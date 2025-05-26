#%%
%matplotlib inline
from glossary_similarity import fetch_chunks_for_term_for_years_biodiv_subset, get_biodiversity_subset, initialize, get_biodiversity_subset_year_bins, get_significance_marker, format_regression_output,fit_and_compute_regression_by_terms, fit_and_compute_regression
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
from IPython.core.display import display, HTML
time_frame = [2014,2015,2016,2017,2018,2019,2020,2021,2022]
#%%
data_nature_related_opportunities = pd.read_csv("data/chunks_w_nature_related_risks_opportunities.csv",sep=";")


#%%
# terms specifically mentioned in 2021 technical report
terms = ['Nature-related opportunities']
         #,'Nature-related systemic risks', 
        # 'Nature-related physical risks',
        # 'Nature-related transition risks'
        # 'Direct impacts',
        # 'Impacts (on nature)']
#         Dependencies (on
# nature)
# double materiality as central idea through impacts and risks
#%%
df_tfnd_glossary_2023 = pd.read_json("data/df_tfnd_glossary_2023_embedded.json", orient="records")
df_tfnd_glossary_2023["term_embedding"] = df_tfnd_glossary_2023["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
#%%

#%%
def fetch_chunks_by_year_bins(years, term, term_embedding, chunks_per_year=1000, batch_size=200):
    """
    Queries the database for chunk embeddings dynamically per year, ensuring
    correct retrieval of metadata.
    """
    driver = initialize(streamlit_secret=False, with_bio=False, custom_definition="")
    with driver.session() as session:
        chunks = []
        for year in years:
            index_name = f"chunk_embeddings_{year}"  # Dynamically select correct index
            print(f"Fetching data from index: {index_name}")

            id_query = f"""
            CALL db.index.vector.queryNodes('{index_name}',$n,$term_embedding) YIELD node AS similarChunk, score
            WITH similarChunk, score
            MATCH (c:Company)-[:ARRANGED]-(e:ECC)<-[:WAS_GIVEN_AT]-(s:Statement)-[:INCLUDES]->(similarChunk)
            OPTIONAL MATCH (i:Industry)<-[:IN_INDUSTRY]-(c)
            OPTIONAL MATCH (se:Sector)<-[:IN_SECTOR]-(c)
            RETURN c.name as company,datetime(e.time).year AS year, 
                    datetime(e.time).month AS month, datetime(e.time).day AS day,
                    i.name as industry,se.name as sector, score
                    //similarChunk.embedding as embedding, score 
            """

            id_results = session.run(
                id_query, term=term, n=chunks_per_year, term_embedding=term_embedding
            )
            # Directly process and store results
            for record in id_results:
                chunk = dict(record)
                #chunk['chunk_embedding_float32'] = np.array(chunk['embedding'], dtype=np.float32)
                #chunk['chunk_embedding_float64'] = chunk.pop('embedding')  # Keep original
                chunks.append(chunk)
    driver.close()
    return chunks
# %%
data_nature_related_opportunities = pd.DataFrame()

for i, row in df_tfnd_glossary_2023[df_tfnd_glossary_2023['Term'].isin(terms)].iterrows():
    term = row['Term']
    term_embedding = row['term_embedding']

    print(f'processing term {term} [{i} out of {len(df_tfnd_glossary_2023)}] ..')

    chunks = fetch_chunks_by_year_bins(
                                        time_frame,
                                        term,
                                        term_embedding,
                                        chunks_per_year=10000)
    results = pd.DataFrame(chunks)
    # results['term_embedding'] = [np.array(term_embedding, dtype=np.float32)] * len(results)
    results['term'] = term

    print(f'results for {term}: {len(results)}')
    
    data_nature_related_opportunities = pd.concat([data_nature_related_opportunities, results])
# %%
df_opportunities_risks['year'].value_counts(normalize=True).sort_index()
# they are a bit skewed towards the early years
#%%
# label by affected, non-affected industry
with open("data/affected_industries.json", "r") as f:
    affected_industries = json.load(f)
print(affected_industries)
#%%
# label the industries by affected/nonaffected
data_nature_related_opportunities['tnfd_treated_industry'] = data_nature_related_opportunities['industry'].isin(affected_industries['affected']).astype(int)
#%%
data_nature_related_opportunities['tnfd_treated_industry'].value_counts()
# -> wrongly-sampled control group
#%%         SECTOR AFFECTS
tnfd_affected_sectors = ['Industrials', 'Financial Services', 'Utilities', 'Basic Materials', 'Energy','Consumer Cyclical','Financial']
data_nature_related_opportunities['tnfd_treated_sector'] = data_nature_related_opportunities['sector'].isin(tnfd_affected_sectors).astype(int)
#%%         ADD DATE
data_nature_related_opportunities["date"] = pd.to_datetime(data_nature_related_opportunities[["year", "month", "day"]])
tnfd_inception = pd.Timestamp("2021-06-01")
data_nature_related_opportunities["delta_since_TNFD"] = (
    data_nature_related_opportunities["date"] - tnfd_inception
).dt.days

data_nature_related_opportunities['after_tnfd'] = (data_nature_related_opportunities["date"] >= tnfd_inception).astype(int)
#%%         DID COLUMNS (SECTOR & INDUSTRY)
data_nature_related_opportunities['did_term_sector'] = (data_nature_related_opportunities['after_tnfd'] * data_nature_related_opportunities['tnfd_treated_sector'])
data_nature_related_opportunities['did_term_industry'] = (data_nature_related_opportunities['after_tnfd'] * data_nature_related_opportunities['tnfd_treated_industry'])
#%%
# Create a new column for binned years (grouping in 2-year intervals)
data_nature_related_opportunities["year_binned"] = (data_nature_related_opportunities["year"] // 2) * 2

# Compute the mean and IQR (Q1, Q3) for each term by binned year
summary = (
    data_nature_related_opportunities
    .groupby(['term', 'year_binned'])['score']
    .describe(percentiles=[0.25, 0.75])[['mean', '25%', '75%']]
    .rename(columns={'25%': 'Q1', '75%': 'Q3'}))

# Format each row into a compact string: "Mean: value | IQR: [Q1 - Q3]"
summary_formatted = summary.apply(lambda row: f"{row['mean']:.3f} ({round(row['Q3'] - row['Q1'],3)})", axis=1)

# Pivot table: Terms as rows, Binned Years as columns
summary_pivot = summary_formatted.unstack(level='year_binned')


#%%         TRIM OUTLIERS
def trim_outliers_iqr(group):
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return group[(group >= lower_bound) & (group <= upper_bound)]

# Apply IQR trimming per term & year
data_nature_related_opportunities['score_trimmed'] = data_nature_related_opportunities.groupby(['term', 'year'])['score'].transform(lambda x: trim_outliers_iqr(x))

# Remove rows where 'score_trimmed' is NaN (i.e., outliers removed)
data_nature_related_opportunities.dropna(subset=['score_trimmed'], inplace=True)

# Display results
print(data_nature_related_opportunities.head())
#%%         REMOVE MISSING VALUES
data_nature_related_opportunities.dropna(inplace=True)




#%%
regression_results = fit_and_compute_regression(data_nature_related_opportunities)
#%%
display(HTML(regression_results.to_html(escape=False)))



#%%
import matplotlib.pyplot as plt

# Aggregate mean scores per year
df_trends = df_oversampled.groupby(["year", "affected"])["score"].mean().reset_index()

# Plot trends for treated vs. control
plt.figure(figsize=(8, 5))
for label, group in df_trends.groupby("affected"):
    plt.plot(group["year"], group["score"], label=f'Treatment={label}', marker='o')
plt.axvline(x=2021, color='r', linestyle='--', label="TNFD Introduction")
plt.xlabel("Year")
plt.ylabel("Mean Log Score")
plt.legend()
plt.title("Parallel Trends Assumption Check")
#%%
df_oversampled_short = df_oversampled[df_oversampled['year'] > 2018].copy()
#%%#OVERSAMPLING OF CONTROL GROUP
# Step 1: Compute mean and standard deviation of control group (Treatment=0) per year
control_stats = df_oversampled_short[df_oversampled_short['affected'] == 0].groupby("year")["score"].agg(['mean', 'std', 'count']).reset_index()

# Step 2: Define the oversampling factor
oversampling_factor = df_oversampled_short['affected'].value_counts()[1] // df_oversampled_short['affected'].value_counts()[0]  # Match treated sample size

# Step 3: Generate synthetic samples
synthetic_samples = []
for _, row in control_stats.iterrows():
    year = row['year']
    mean_score = row['mean']
    std_score = row['std']
    count = row['count']

    # Generate new samples with normal distribution
    synthetic_scores = np.random.normal(loc=mean_score, scale=std_score, size=int(count * oversampling_factor))
    
    # Create synthetic DataFrame
    synthetic_df = pd.DataFrame({
        'year': [year] * len(synthetic_scores),
        'score': synthetic_scores,
        'affected': [0] * len(synthetic_scores)  # Control group
    })
    
    synthetic_samples.append(synthetic_df)

# Step 4: Concatenate new synthetic samples
df_oversampled_short = pd.concat([df_oversampled_short, *synthetic_samples], ignore_index=True)

# Step 5: Verify new distribution
print(df_oversampled_short['affected'].value_counts(normalize=True))
#%%
# Define post-TNFD introduction variable
df_oversampled_short['after_tnfd'] = (df_oversampled_short['year'] >= 2021
                                      ).astype(int)

# Define interaction term for DiD
df_oversampled_short['interaction_term'] = df_oversampled_short['affected'] * df_oversampled_short['after_tnfd']
#%%
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Run Difference-in-Differences (DiD) regression
model = smf.ols("score ~ affected + after_tnfd + interaction_term", data=df_oversampled_short).fit(cov_type='HC3')

# Display results
print(model.summary())
#%%