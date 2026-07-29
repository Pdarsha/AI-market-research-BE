import os

class Config:
    BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000/v1")
    TEST_USER_EMAIL = os.getenv("TEST_USER_EMAIL", "testuser@example.com")
    TEST_USER_PASSWORD = os.getenv("TEST_USER_PASSWORD", "TestPassword123!")
    TEST_USER_NAME = os.getenv("TEST_USER_NAME", "Test User")
    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

config = Config()
