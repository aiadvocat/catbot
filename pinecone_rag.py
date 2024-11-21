from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import json
from tqdm.auto import tqdm

BATCH_SIZE = 128
VECTOR_LIMIT = 1024

class PineconeRAG:
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initializes a Pinecone RAG instance.
        
        Args:
            api_key (str): API key for Pinecone.
            environment (str): Pinecone environment.
            index_name (str): Name of the Pinecone index.
        """

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # Check if the index exists, otherwise create it
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        if index_name not in self.pc.list_indexes().names():
        # Define the spec (index configuration)
            spec = ServerlessSpec(
                cloud='aws',  # Cloud provider
                region='us-east-1',  # Region
            )
            
            # Create the index with the necessary spec
            self.pc.create_index(
                name=index_name,
                dimension=self.model.get_sentence_embedding_dimension(),  # Ensure this matches the dimension of your embeddings 1536
                metric='cosine',  # Distance metric to use (e.g., 'cosine', 'dotproduct', 'euclidean')
                spec=spec
            )
    
        self.index = self.pc.Index(index_name)

        print(f"Changed index to...{index_name}")
        self.vector_count = int(self.index.describe_index_stats().total_vector_count) 

    def change_index(self, index_name):
        self.index_name = index_name
        if index_name not in self.pc.list_indexes().names():
         # Define the spec (index configuration)
            spec = ServerlessSpec(
                cloud='aws',  # Cloud provider
                region='us-east-1',  # Region
            )
            
            # Create the index with the necessary spec
            self.pc.create_index(
                name=index_name,
                dimension=self.model.get_sentence_embedding_dimension(),  # Ensure this matches the dimension of your embeddings 1536
                metric='cosine',  # Distance metric to use (e.g., 'cosine', 'dotproduct', 'euclidean')
                spec=spec
            )
        else:
            print(f"Found existing index {index_name}")
        
        self.index = self.pc.Index(index_name) 
        print(f"Changed index to...{index_name}")
        self.vector_count = int(self.index.describe_index_stats().total_vector_count)        


    def _private_upsert_text(self, rag_data):
        sentences = []

        lines = rag_data.splitlines('\n')

        for line in lines:
            if line.strip() != '':
                sentences.append(line.strip())

        sentences = sentences[:VECTOR_LIMIT]

        self.vector_count = len(sentences)

        for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
            # find end of batch
            i_end = min(i+BATCH_SIZE, len(sentences))
            # create IDs batch
            ids = [str(x) for x in range(i, i_end)]
            # create metadata batch
            # creates a 'text' field in the data
            metadata = [{'text': text} for text in sentences[i:i_end]]
            # create embeddings
            xc = self.model.encode(sentences[i:i_end])
            # create records list for upsert
            records = zip(ids, xc, metadata)
            # upsert to Pinecone
            self.index.upsert(vectors=records)

    def get_vector_count(self):
        return self.vector_count
    
    def get_index_list(self):
        return self.pc.list_indexes().names()    
    
    def delete_index(self, index_name):
        if index_name in self.pc.list_indexes().names():
            self.pc.delete_index(index_name)
            print(f"Index '{index_name}' deleted successfully.")
        else:
            print(f"Index '{index_name}' does not exist.")    

    def describe_index_stats(self):
        return self.index.describe_index_stats()        

    def upsert_data(self, new_index, rag_data):
        """
        Inserts or updates vectors in the Pinecone index.
        
        Args:
            rag_data (str): data to be indexed.
        """
        self.change_index(new_index)
        if isinstance(rag_data, str):
            self._private_upsert_text(rag_data)
        # Process the lines and sentences


    def query(self, query, top_k: int = 3):
        """
        Queries the index with a vector and retrieves the top_k results.
        
        Args:
            query: the text prompt.
            top_k (int): Number of nearest neighbors to retrieve.
            
        Returns: Text results to augment query
        """
        xq = self.model.encode(query).tolist()
    
        # Retrieve top 3 most relevant vectors from Pinecone
        result = self.index.query(vector=xq, top_k=top_k, include_metadata=True)
 
        output = ""
        olen = min(5, len(result['matches']))
        for i in range(olen):
            # this assumes the vector database has a 'text' field to match the upsert.
            # currently experimenting with other formats
    #        text = result['matches'][i]['metadata']['text']
    #        text = result['matches'][i]['metadata']['description']
            text = json.dumps(result['matches'][i]['metadata'])
            output = output + str(i+1) + ")" + text
        return output        


    def delete_index(self):
        """
        Deletes the Pinecone index.
        """
        try:
            index = self.pc.Index(self.index_name)
            index.delete(delete_all=True)
        except Exception as e:
        # index may not exist, ok to pass
            print(str(e))
