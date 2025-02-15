# app/config.py

from pydantic import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str = "model.pth"
    EPOCHS: int = 100
    LEARNING_RATE: float = 0.01
    INPUT_DIM: int = 2
    HIDDEN_DIM: int = 10
    OUTPUT_DIM: int = 1

    class Config:
        env_file = ".env"


settings = Settings()
