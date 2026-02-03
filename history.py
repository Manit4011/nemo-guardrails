import pymongo
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

client = pymongo.MongoClient(CONN_STRING)

# col = db["Test"]
# cursor = col.find({})#'conv_id': 'a:1hOMvx-E5nzjFVDC06VJUUJXHYTFkr3w-DCxxvNLI1ObIcxvC8zK_zh3nCyP3GivTCbl2OlfcNmRX1CwgQE71u8SmYBkrl18yi3HGlYM-zoYC3Wn4hN7OQESwKsseQhgc'})#metadata.filename': 'Chat Bot - Feedbacks.xlsx'})
# data = list(cursor)
# print(len(data))
# if not data:
#     print("No records found.")
# for i in range(1,10):
#     print(f"Conv {i}:")
#     pprint(data[-i])
# print(data)

def get_databases():
    databases = client.list_database_names()
    return databases
db = client["PolicyDB"]
# db = client["LargeDocDB"]

def get_collections():
    collections = db.list_collection_names()
    return collections

def empty_collection(col_name):
    col = db[col_name]
    print(col)
    res = col.delete_many({})
    print(f"Deleted {res.deleted_count} from {col_name}")

def duplicate_data(col_name, col_name_2, db_name="PolicyDB", db_name_2="PolicyDB"):
    db=client[db_name]
    db2=client[db_name_2]
    col = db[col_name]
    col2 = db2[col_name_2]
    docs = list(col.find())
    print(len(docs))

    for doc in docs:
        doc.pop('_id', None)
        doc['upload_source'] = 'teams'
        doc['datetime'] =  doc.get('metadata',{}).get('timestamp', None)
        doc['doc_type'] = 'conv'
        doc['metadata']['file_name'] = "conv_file"

    res = col2.insert_many(docs)
    print(f"Copied {len(res.inserted_ids)} documents from {col_name} to {col_name_2}")

def delete_files(col_name, file_name):
    col = db[col_name]
    res = col.delete_many({'metadata.file_name':file_name})
    print(f"Deleted {res.deleted_count} from {col_name} under {file_name}")

def get_unique_files(col_name):
    col = db[col_name]
    cursor = col.find({})
    data = list(cursor)
    print(len(data))
    unique_file_names = set()
    # Iterate over the cursor to collect unique 'metadata.file_name' values
    for row in data:
        try:
            file_name = row['metadata'].get('file_name', None)  # Safely get 'file_name'
            if file_name:
                # print(file_name + " " + row['metadata'].get('sheet_name', None))
                unique_file_names.add(file_name)
        except KeyError as e:
            print(f"KeyError: Missing 'metadata' or 'file_name' in record")
        except Exception as e:
            print(f"Unexpected error: {e}")

    # Convert the set of unique file names to a list
    return list(unique_file_names)

def get_file_entries(col_name, file_name):
    col = db[col_name]
    cursor = col.find({'metadata.file_name':file_name})
    data = list(cursor)
    return data

if __name__== '__main__':
    print(get_databases())
    print(get_collections())
    empty_collection('Summary')
    # duplicate_data("Messages", "Test", "HelpDesk", "PolicyDB")
    # # len(get_unique_files("Test"))
    # unique_file_names_list = get_unique_files("Summary")
    # print(len(unique_file_names_list))
    # if unique_file_names_list:
    #     print(f"Total count: {len(unique_file_names_list)} and files are:")
    #     for file_name in unique_file_names_list:
    #         print(file_name)
    # else:
    #     print("No files found.")
    # for file_name in unique_file_names_list:
    #     if(file_name.startswith("Gemini-ISMS-")):
    #         col = db["Old_Policies"]
    #         res = col.delete_many({'metadata.file_name':file_name})
    #         print(f"Deleted {res.deleted_count} under {file_name}")
    # print(len(unique_file_names_list))