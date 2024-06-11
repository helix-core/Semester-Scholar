import os
from pymongo import MongoClient
import urllib.parse
from bson.binary import Binary

username = 'ahmedmustafan'
password = 'mydbboss321@'
encoded_username = urllib.parse.quote_plus(username)
encoded_password = urllib.parse.quote_plus(password)

# MongoDB Atlas connection details
mongo_uri = f"mongodb+srv://{encoded_username}:{encoded_password}@cluster0.w7vw3w2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # Replace with your MongoDB URI
client = MongoClient(mongo_uri)
db = client.pdf_storage  # Replace with your database name

# Function to upload file
def upload_file(file_path, collection_name):
    with open(file_path, 'rb') as file:
        binary_data = Binary(file.read())
    file_name = os.path.basename(file_path)
    collection = db[collection_name]
    collection.insert_one({
        "file_name": file_name,
        "file_data": binary_data
    })

# Directory structure: base_dir/semester_<n>/files
base_dir = 'C:\\Users\\ahmed\\Downloads\\testsql'

# Loop through semesters
for semester in range(1, 7):
    semester_dir = os.path.join(base_dir, f'semester_{semester}')
    collection_name = f'semester_{semester}'
    if os.path.isdir(semester_dir):
        # Loop through files
        for file_name in os.listdir(semester_dir):
            file_path = os.path.join(semester_dir, file_name)
            if os.path.isfile(file_path):
                file_type = os.path.splitext(file_name)[1].lower()
                if file_type in ['.pdf', '.doc', '.docx', '.ppt', '.pptx']:
                    upload_file(file_path, collection_name)

# Close the connection
client.close()
