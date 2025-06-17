# utils.py - Fixed version with better error handling
import re
import os
import glob
import docx
from pathlib import Path
from tempfile import NamedTemporaryFile
from collections import defaultdict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader
import io

SERVICE_ACCOUNT_FILE = "summarizer-chatbot-1ec53e920015.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def extract_drive_file_id(url):
    """Extract file/folder ID from Google Drive URL"""
    if not url or not isinstance(url, str):
        return None
    
    # Matches both /file/d/... and /folders/...
    match = re.search(r"/(?:folders|d)/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)
    elif "id=" in url:
        return url.split("id=")[-1].split("&")[0]
    return None

def download_all_files_from_drive_folder(folder_id):
    """Download all supported files from a Google Drive folder"""
    if not folder_id:
        print("Error: No folder ID provided")
        return []
        
    try:
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            print(f"Error: Service account file not found: {SERVICE_ACCOUNT_FILE}")
            return []
            
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        service = build("drive", "v3", credentials=creds)
        
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = results.get("files", [])
        
        if not files:
            print("No files found in the folder")
            return []
        
        file_paths = []
        supported_extensions = (".pdf", ".txt", ".docx")
        
        for file in files:
            file_name = file.get("name", "")
            if not file_name.lower().endswith(supported_extensions):
                print(f"Skipping unsupported file: {file_name}")
                continue
            
            try:
                request = service.files().get_media(fileId=file["id"])
                fh = NamedTemporaryFile(delete=False, suffix="." + file_name.split(".")[-1])
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                fh.close()
                
                file_paths.append((file_name, fh.name))
                print(f"Downloaded: {file_name}")
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
                continue
        
        return file_paths
    except Exception as e:
        print(f"Error accessing Google Drive: {e}")
        return []

def extract_text_from_file(file_obj):
    """Extract text from various file types with better error handling"""
    if not file_obj:
        return ""
    
    # Handle different input types
    if isinstance(file_obj, str):
        # If it's a file path
        if os.path.exists(file_obj):
            with open(file_obj, 'rb') as f:
                return extract_text_from_file(f)
        else:
            return file_obj  # Return as text if it's not a file path
    
    file_name = getattr(file_obj, 'name', '').lower()
    text = ""
    
    try:
        if file_name.endswith(".pdf"):
            # Reset file pointer if possible
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            
            # Handle both file objects and file paths
            if hasattr(file_obj, 'read'):
                pdf_data = file_obj.read()
                if isinstance(pdf_data, bytes):
                    reader = PdfReader(io.BytesIO(pdf_data))
                else:
                    reader = PdfReader(file_obj)
            else:
                reader = PdfReader(file_obj)
                
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        elif file_name.endswith(".txt"):
            if hasattr(file_obj, 'read'):
                content = file_obj.read()
                if isinstance(content, bytes):
                    text = content.decode("utf-8", errors="ignore")
                else:
                    text = str(content)
            else:
                text = str(file_obj)
                
        elif file_name.endswith(".docx"):
            # Reset file pointer if possible
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
                
            doc = docx.Document(file_obj)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
        else:
            print(f"Unsupported file type: {file_name}")
            return ""
            
    except Exception as e:
        print(f"Error extracting text from {file_name}: {e}")
        return ""
    
    # Clean up the text
    text = text.strip()
    if not text:
        print(f"Warning: No text extracted from {file_name}")
    
    return text
