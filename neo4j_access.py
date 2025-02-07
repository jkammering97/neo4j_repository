from langchain_community.vectorstores import Neo4jVector
import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

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
    def fetch_data(cls, query=None, params={}):
        """Fetch data from Neo4j interactively from the terminal"""
        try:
            # If no query is provided, ask the user in the terminal
            while not query:
                try:
                    query = input("🔍 Please input a Cypher query: ").strip()
                    if not query:
                        print("⚠️ Query cannot be empty. Try again.")
                except KeyboardInterrupt:
                    print("\n❌ Query input interrupted. Exiting...")
                    return
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue

            if cls.driver is None:
                raise RuntimeError("Neo4j driver is not initialized. Call QueryNeo4J.initialize() first.")

            with cls.driver.session(default_access_mode="READ") as session:
                result = session.run(query, params)
                cls.result_data_dict =  [record.data() for record in result]

            cls.close_access()
            cls.return_dataframe()
        
        except BaseException as e:
            print(e)
            cls.close_access()

    @classmethod
    def close_access(cls):
        try:
            cls.driver.close()
            cls.driver._closed()
            print('connection closed successfully')
        except BaseException as e:
            print(e)

    @classmethod
    def return_dataframe(cls):
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
        df = pd.DataFrame(cls.result_data_dict, columns=columns)

        print("\n📊 Query Result as DataFrame:")
        print(df)

        return df

if __name__ == "__main__":
    QueryNeo4J.initialize()
    QueryNeo4J.fetch_data()
