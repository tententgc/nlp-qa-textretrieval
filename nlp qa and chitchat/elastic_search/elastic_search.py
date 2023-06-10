from elasticsearch import Elasticsearch

def preprocess_text(text):
    return text


class DocumentRetrieval:
    def __init__(self, hosts: str, user, password, http_ca_path):
        self.hosts = hosts
        self.user = user
        self.password = password,
        self.http_ca_path = http_ca_path
        
        self.client = Elasticsearch(
            hosts=hosts,
            basic_auth=(user, password),
            verify_certs=True,
            ca_certs=http_ca_path,
        )
    
    def list_index(self):
        return list(self.client.indices.get_alias(index="*").keys())
    
    def delete_index(self, index_name):
        self.client.indices.delete(index=index_name)
    
    def search(self, query: str, index_name: str="doc", top_n: int=1, thresh: float=10):
        res = self.client.search(
            index=index_name,
            # can be change
            body={
                "query": {
                    "match": {
                        "context": preprocess_text(query)
                    }
                }
            }
        )
        
        filename_search = {}
        have_search = len(res["hits"]["hits"])
        if have_search:
            max_score = res["hits"]["max_score"]
            
            if max_score < thresh:
                return filename_search
            
            for i, hit in enumerate(res["hits"]["hits"]):
                if i == top_n:
                    break
                
                score = hit["_score"]
                if score < thresh:
                    continue
                
                search_file = hit["_source"]["sub_filename"]
                filename_search[search_file] = score
        
        return filename_search
    
if __name__ == "__main__":
    ES_HOSTS = "https://localhost:9200"
    ES_USER = "elastic"
    ES_PASS = "Pic1zs71iL1RWB3By=DC"
    HTTP_CA_PATH = "http_ca.crt"
    index_name = "document"
    
    dr = DocumentRetrieval(ES_HOSTS, ES_USER, ES_PASS, HTTP_CA_PATH)
    
    print(dr.search("ใครเป็นผู้ออกตราสารหนี้ภาคเอกชน ไร้ใบตราสาร", index_name="ตราสารหนี้3"))