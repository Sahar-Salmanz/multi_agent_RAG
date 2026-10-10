"""
auth.py
───────
API key authentication for FastAPI.
Maps an API key in the Authorization header to a user_id.

For now, valid keys are stored in .env as a comma-separated list:
  API_KEYS=key1:user1,key2:user2

When you add more users, just add their key:user_id pair to .env.
No code changes required.
"""

import os
from dotenv import load_dotenv
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

# Parse API_KEYS from .env into a dict: {key: user_id}
# Format in .env:  API_KEYS=abc123:sahar,def456:user2
def _load_api_keys() -> dict:
    raw = os.getenv("API_KEYS", "")
    keys = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, user_id = pair.split(":", 1)
            keys[key.strip()] = user_id.strip()
    return keys


API_KEYS = _load_api_keys()
bearer   = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer),
) -> str:
    """
    FastAPI dependency that validates the Bearer token and returns the user_id.

    Usage in a route:
        @router.post("/chat")
        def chat(user_id: str = Depends(get_current_user)):
            ...

    Raises:
        HTTPException 401 if the key is missing or invalid.
    """
    token = credentials.credentials
    user_id = API_KEYS.get(token)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id