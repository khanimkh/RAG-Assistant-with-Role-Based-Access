import getpass
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.auth_service import hash_password, USERS_PATH

# ============================================================================
# FILE OVERVIEW: add_user.py
# ============================================================================
# PURPOSE
#   The only supported way to add or update a user. Run locally by
#   whoever administers the app - never through the API, since there's no
#   "create the first admin" endpoint (that would be an unauthenticated
#   way to grant yourself admin, which defeats the point of having auth).
#
# GENERAL ALGORITHM
#   1. prompt for username, password (hidden input via getpass), and role
#   2. CALLS hash_password(password) -> src/services/auth_service.py -
#      the plaintext password never touches disk, only its hash does
#   3. load the existing data/users.yaml (or start empty if it doesn't
#      exist yet)
#   4. add/overwrite this username's entry with {password_hash, role}
#   5. write the file back
#
# USAGE
#   python scripts/add_user.py
# ============================================================================


def main():
    username = input("Username: ").strip()
    password = getpass.getpass("Password: ")
    role = input("Role (general/admin) [general]: ").strip() or "general"

    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    users = config.get("users") or {}
    users[username] = {"password_hash": hash_password(password), "role": role}
    config["users"] = users

    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    print(f"User {username!r} added/updated with role {role!r}.")


if __name__ == "__main__":
    main()
