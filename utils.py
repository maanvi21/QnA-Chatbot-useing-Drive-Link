from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tempfile import NamedTemporaryFile
from pypdf import PdfReader
import docx

SERVICE_ACCOUNT_FILE = "summarizer-chatbot-1ec53e920015.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def download_all_files_from_drive_folder(folder_id):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)

    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    file_paths = []

    for file in files:
        if not file["name"].endswith((".pdf", ".txt", ".docx")):
            continue

        request = service.files().get_media(fileId=file["id"])
        fh = NamedTemporaryFile(delete=False, suffix="." + file["name"].split(".")[-1])
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        file_paths.append(fh.name)

    return file_paths

def extract_text_from_file(file_obj):
    file_name = file_obj.name.lower()
    text = ""

    if file_name.endswith(".pdf"):
        reader = PdfReader(file_obj)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file_name.endswith(".txt"):
        text = file_obj.read().decode("utf-8", errors="ignore")
    elif file_name.endswith(".docx"):
        doc = docx.Document(file_obj)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        raise ValueError("Unsupported file type.")

    return text
