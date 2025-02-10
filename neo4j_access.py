#%%
from langchain_community.vectorstores import Neo4jVector
import pandas as pd
from neo4j import GraphDatabase
import os
import time
from dotenv import load_dotenv
#%%
class QueryNeo4J:
    """can query the db with fixed class attributes URI, USERNAME and PASSWORD
    """
    URI = None
    USERNAME = None
    PASSWORD = None
    driver = None

    @classmethod
    def initialize(cls):
        load_dotenv()
        cls.URI = os.getenv('NEO4J_URI')
        cls.USERNAME = os.getenv('USERNAME')
        cls.PASSWORD = os.getenv('DB_PASSWORD')
        cls.driver = GraphDatabase.driver(cls.URI, auth=(cls.USERNAME, cls.PASSWORD))

    @classmethod
    def commit_query(cls, query=None, params={}):
        """Fetch data from Neo4j interactively from the terminal"""
        try:
            if cls.driver is None:
                raise RuntimeError("neo4j driver is not initialized. Call QueryNeo4J.initialize() first.")

            with cls.driver.session(default_access_mode="READ") as session:

                accept = cls.execution_plan(session, query)
                if accept:
                    # Measure actual execution time
                    start_time = time.time()
                    result = session.run(query, params)
                    cls.result_data_dict = [record.data() for record in result]
                    end_time = time.time()

                    print(f"\nquery Execution Time: {end_time - start_time:.6f} seconds")

                    return cls.create_dataframe()
                else:
                    print("query execution canceled.")
                    return None  # **Stops execution if the user rejects**

        except Exception as e:
            print(f"query execution error: {e}")
            return None

    @classmethod
    def execution_plan(cls, query):
        _, summary, _ = cls.driver.execute_query(f"EXPLAIN {query}")
        if summary:
            print(summary.plan['args']['string-representation'])
        else:
            print("No execution plan available. (EXPLAIN did not return metadata)")

        while True:
            accept = input("accept execution plan? (type: y/n)")
            if accept.lower() == "n":
                return False
            elif accept.lower() == "y":
                return True
            else:
                print("invalid input. Please type 'y' for Yes or 'n' for No.")

    @classmethod
    def close_access(cls):
        try:
            cls.driver.close()
            cls.driver._closed()
            print('connection closed successfully')
        except BaseException as e:
            print(e)

    @classmethod
    def create_dataframe(cls):
        """Convert Neo4j result into a clean Pandas DataFrame with proper column names"""

        if not cls.result_data_dict:
            print("⚠️ No data retrieved from Neo4j.")
            return pd.DataFrame()  # Return empty DataFrame if no results
        
        # Detect if results are wrapped in a node key (e.g., {'p': {...}})
        first_record = cls.result_data_dict[0] if cls.result_data_dict else {}
        node_keys = list(first_record.keys())  # Get top-level keys (e.g., ['p'])

        if len(node_keys) == 1 and isinstance(first_record[node_keys[0]], dict):
            cls.result_data_dict = [record[node_keys[0]] for record in cls.result_data_dict]
            print("🛠 Extracted node properties successfully.")

        columns = list(cls.result_data_dict[0].keys()) if cls.result_data_dict else []
        cls.df = pd.DataFrame(cls.result_data_dict, columns=columns)

        print("\n📊 Query Result as DataFrame:")

        print(cls.df)

        return cls.df


if __name__ == "__main__":
    QueryNeo4J.initialize()
    query = "MATCH (s:Statement)-[:WAS_GIVEN_AT]->(e:ECC) WHERE toLower(s.text) CONTAINS 'biodiversity' RETURN e.time as datetime, e.title as conference_name, s.text as statement"
    QueryNeo4J.commit_query(query=query)
    
