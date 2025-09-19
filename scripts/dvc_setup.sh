#!/usr/bin/env bash
set -e
echo "Configure DVC remote - edit this script for your remote details"
echo "Example: dvc remote add -d mygdrive gdrive://<folder-id>"

# Example (UNCOMMENT and fill your folder-id if you want to run)
# dvc remote add -d gdrive_remote gdrive://<your-google-drive-folder-id>
# dvc remote modify gdrive_remote gdrive_use_service_account true
# dvc push

echo "Edit 'scripts/dvc_setup.sh' and uncomment your preferred remote command then run it."
