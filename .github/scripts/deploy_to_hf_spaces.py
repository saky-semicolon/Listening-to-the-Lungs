import os
import sys
from huggingface_hub import HfApi, create_repo, whoami
try:
    from huggingface_hub import SpaceSdk  # type: ignore
except Exception:
    SpaceSdk = None  # older huggingface_hub


def main():
    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("HF_SPACE_ID")  # e.g., "username/listening-to-the-lungs"
    if not token or not space_id:
        print("HF_TOKEN and HF_SPACE_ID must be set in secrets.")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        who = whoami(token=token)
        print("Authenticated as:", who.get("name") or who.get("email"))
    except Exception as e:
        print("Authentication failed:", e)
        sys.exit(1)

    # Ensure repo exists as a Space; always request Streamlit SDK, but don't fail hard
    try:
        if SpaceSdk is not None:
            create_repo(space_id, repo_type="space", exist_ok=True, space_sdk=SpaceSdk.STREAMLIT)
        else:
            create_repo(space_id, repo_type="space", exist_ok=True, space_sdk="streamlit")
    except Exception as e:
        print("Warning: could not ensure Space with Streamlit SDK (continuing):", e)

    # Upload files needed for the app: app.py, requirements.txt, src/
    files_to_upload = [
        ("app.py", "app.py"),
        ("requirements.txt", "requirements.txt"),
    ]

    # Walk src directory
    for root, _, files in os.walk("src"):
        for f in files:
            local_path = os.path.join(root, f)
            remote_path = local_path  # keep same structure
            files_to_upload.append((local_path, remote_path))

    # Upload
    for local_path, remote_path in files_to_upload:
        print(f"Uploading {local_path} -> {space_id}:{remote_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=space_id,
            repo_type="space",
            commit_message=f"Update {remote_path} via CI/CD",
        )

    print("Deployment complete. Space should rebuild automatically.")


if __name__ == "__main__":
    main()


