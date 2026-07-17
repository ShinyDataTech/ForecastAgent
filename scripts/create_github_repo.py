import os
import sys
import subprocess
import urllib.request
import urllib.error
import json
import getpass

def run_git(args):
    """Helper to run git commands and return output."""
    try:
        res = subprocess.run(
            ["git"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: git {' '.join(args)}")
        print(f"Error: {e.stderr.strip()}")
        return None

def create_github_repo(token, name="ForecastAgent"):
    """Creates a repository on GitHub using the REST API."""
    url = "https://api.github.com/user/repos"
    payload = {
        "name": name,
        "description": "ForecastAgent 1.0: Zero-shot and fine-tuned time series forecasting SDK & serving API",
        "private": False,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "ForecastAgent-CLI"
        },
        method="POST"
    )
    
    try:
        print(f"Sending request to create GitHub repository '{name}'...")
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print(f"Successfully created remote repository: {res_data['html_url']}")
            return res_data["clone_url"], res_data["owner"]["login"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(error_body)
            # Handle case where repo already exists
            if any(err.get("message") == "name already exists on this account" for err in err_json.get("errors", [])):
                print(f"Repository '{name}' already exists on your account.")
                # We need to fetch the username to construct the clone url
                user_req = urllib.request.Request(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "ForecastAgent-CLI"
                    }
                )
                with urllib.request.urlopen(user_req) as user_resp:
                    user_data = json.loads(user_resp.read().decode("utf-8"))
                    username = user_data["login"]
                    clone_url = f"https://github.com/{username}/{name}.git"
                    return clone_url, username
            print(f"API Error ({e.code}): {err_json.get('message')}")
        except Exception:
            print(f"HTTP Error ({e.code}): {error_body}")
        sys.exit(1)
    except Exception as e:
        print(f"Connection Error: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("           ForecastAgent GitHub Repository Creator")
    print("=" * 60)
    
    # 1. Check if token is in environment, else prompt
    token = os.getenv("GITHUB_PAT")
    if not token:
        print("A GitHub Personal Access Token (PAT) with 'repo' scope is required.")
        token = getpass.getpass("Enter your GitHub Personal Access Token (PAT): ").strip()
    
    if not token:
        print("Error: No GitHub token provided.")
        sys.exit(1)
        
    # 2. Create the repository
    clone_url, owner = create_github_repo(token)
    
    # Authenticated clone URL to allow headless push
    auth_clone_url = clone_url.replace("https://", f"https://{token}@")
    
    print("\nConfiguring local Git repository...")
    
    # 3. Check git initialization
    if not os.path.exists(".git"):
        print("Initializing new Git repository locally...")
        run_git(["init"])
    else:
        print("Git repository already initialized.")
        
    # 4. Set local git user configuration
    print("Setting local repository Git configuration...")
    run_git(["config", "user.name", "ShinyDataTech"])
    run_git(["config", "user.email", "shinydatatech@gmail.com"])
    
    # 5. Add remote origin
    print("Setting remote origin...")
    run_git(["remote", "remove", "origin"])  # Clear if exists
    run_git(["remote", "add", "origin", auth_clone_url])
    
    # 6. Commit and push
    print("Staging files...")
    run_git(["add", "."])
    
    print("Creating initial commit...")
    # Check if there is anything to commit
    status_out = run_git(["status", "--porcelain"])
    if not status_out:
        print("No changes to commit.")
    else:
        run_git(["commit", "-m", "Initial commit of ForecastAgent 1.0 package"])
        
    print("Setting branch name to 'main'...")
    run_git(["branch", "-M", "main"])
    
    print("Pushing to GitHub...")
    # Run git push directly to show progress/errors
    try:
        subprocess.run(
            ["git", "push", "-u", "origin", "main", "--force"],
            check=True
        )
        print(f"\nSUCCESS: ForecastAgent 1.0 successfully pushed to https://github.com/{owner}/ForecastAgent")
    except subprocess.CalledProcessError as e:
        print("\nFailed to push code to GitHub. Please verify your token permissions.")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure working directory is workspace root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
