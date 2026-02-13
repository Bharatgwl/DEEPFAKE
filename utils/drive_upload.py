from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

def upload_to_drive(local_file, drive_folder_name):
    """
    Upload a file to Google Drive
    
    Args:
        local_file: Path to file to upload
        drive_folder_name: Name of folder in Drive to upload to
    
    Returns:
        folder_id: ID of the Drive folder
    """
    gauth = GoogleAuth()

    # Configure authentication
    gauth.settings['client_config_backend'] = 'file'
    gauth.settings['client_config_file'] = 'credentials.json'
    gauth.settings['save_credentials'] = True
    gauth.settings['save_credentials_backend'] = 'file'
    gauth.settings['save_credentials_file'] = 'token.json'
    gauth.settings['get_refresh_token'] = True
    gauth.settings['oauth_scope'] = ['https://www.googleapis.com/auth/drive.file']

    # Authenticate
    gauth.LocalWebserverAuth(host_name='localhost', port_numbers=[8080])
    drive = GoogleDrive(gauth)

    # Find or create folder
    folder_list = drive.ListFile({
        'q': f"title='{drive_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()

    if folder_list:
        folder = folder_list[0]
    else:
        folder_metadata = {
            "title": drive_folder_name,
            "mimeType": "application/vnd.google-apps.folder"
        }
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()

    # Upload file
    file = drive.CreateFile({
        "title": os.path.basename(local_file),
        "parents": [{"id": folder["id"]}]
    })
    file.SetContentFile(local_file)
    file.Upload()

    return folder["id"]