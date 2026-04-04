from models import embedding_model
from supabase_client import supabase
from milvus import collection

# run this script to seed data from supabase to milvus

def supabase_to_milvus():
    resp = supabase.table('menu').select('*').execute()
    items = resp.data
    for menu_item in items:
        vector = embedding_model.feature_extraction(menu_item['description'])
        milvus_row = {'embedding': vector, 'sql_id': menu_item['id']}
        collection.insert(milvus_row)
        
     
        
        
    



supabase_to_milvus()

 

