# subagents/knowledgeDB_agent/tools/neo4j.py

from neo4j import GraphDatabase
import logging
from datetime import datetime

# Configure your Neo4j connection details here
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # Replace with your Neo4j password


class Neo4jTool:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        logging.info("Initialized Neo4j driver")

    def close(self):
        self.driver.close()
        logging.info("Closed Neo4j driver")

    def create_node(self, tx, node):
        """
        Creates or updates a node in Neo4j.
        Always overwrites the created_at timestamp to current UTC time.
        """
        query = """
            MERGE (n:Entity {name: $name})
            SET n.type = $type,
                n.description = $description,
                n.created_at = $created_at
        """
        tx.run(
            query,
            name=node["name"],
            type=node["type"],
            description=node.get("summary", ""),
            created_at=node.get("created_at") or datetime.utcnow().isoformat()
        )

    def create_relationship(self, tx, rel):
        """
        Creates a relationship between two existing nodes.
        """
        query = """
            MATCH (a:Entity {name: $from_name}), (b:Entity {name: $to_name})
            MERGE (a)-[r:RELATION {type: $type}]->(b)
            RETURN r
        """
        tx.run(
            query,
            from_name=rel["from_node"],
            to_name=rel["to_node"],
            type=rel["type"]
        )

    def save_knowledge_graph(self, knowledge_graph: dict):
        """
        Saves nodes and relationships to Neo4j.
        Automatically adds/updates created_at for each node.
        """
        with self.driver.session() as session:
            nodes = knowledge_graph.get("nodes", [])
            relationships = knowledge_graph.get("relationships", [])

            # Add timestamp to each node
            for node in nodes:
                node["created_at"] = datetime.utcnow().isoformat()

            def create_nodes(tx):
                for node in nodes:
                    self.create_node(tx, node)

            def create_relationships(tx):
                for rel in relationships:
                    self.create_relationship(tx, rel)

            session.execute_write(create_nodes)
            session.execute_write(create_relationships)
            logging.info(f"Saved {len(nodes)} nodes and {len(relationships)} relationships to Neo4j.")
