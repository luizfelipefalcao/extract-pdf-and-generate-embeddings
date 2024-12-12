import boto3
import PyPDF2
import requests
import json

from aws_requests_auth.aws_auth import AWSRequestsAuth

s3_client = boto3.client('s3')

bucket_name = 'chatbot-pdf-storage'
object_name = 'Evolution_of_the_Internet_Detailed.pdf'
download_path = '/Users/felipe/Desktop/Evolution_of_the_Internet_Detailed.pdf'  # Local file path to save the PDF

# Download the file from S3 <---- STEP 1
try:
    s3_client.download_file(bucket_name, object_name, download_path)
    print(f"Downloaded {object_name} to {download_path}")
except Exception as e:
    print(f"Error downloading file: {e}")

# Extract text from the download_path file <---- STEP 2
def extract_text_from_pdf(pdf_path): 
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_text = extract_text_from_pdf(download_path)

# Embedding from the pdf_text <---- STEP 3
def get_embeddings_from_bedrock(pdfText):
    modelId = 'amazon.titan-embed-text-v2:0'
    body = json.dumps({
        "inputText": pdfText,
        "dimensions": 512,
        "normalize": True 
    })
    boto3_bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    try:
        response = boto3_bedrock_client.invoke_model(
            body=body,
            modelId=modelId
        )
        response_body = json.loads(response.get('body').read())
        embeddings = response_body.get('embedding')
        return embeddings

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

embeddings = get_embeddings_from_bedrock(pdf_text)
print(f"Embeddings: {embeddings}")







