import os
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()

# Access them securely
db_password = os.getenv("DB_PASS")
db_user = os.getenv("DB_USER")

print(f"Connecting to database as {db_user}...")