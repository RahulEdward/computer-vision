"""Configuration management for Computer Genie"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class GenieConfig(BaseSettings):
    """Main configuration class"""
    
    # API Keys
    genie_api_key: Optional[str] = Field(None, env="GENIE_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    huggingface_token: Optional[str] = Field(None, env="HUGGINGFACE_TOKEN")
    
    # Model Settings
    default_model: str = Field("genie-vision", env="GENIE_DEFAULT_MODEL")
    timeout: int = Field(30, env="GENIE_MODEL_TIMEOUT")
    max_retries: int = Field(3, env="GENIE_MAX_RETRIES")
    retry_delay: float = Field(1.0, env="GENIE_RETRY_DELAY")
    
    # System Settings
    screenshot_quality: int = Field(95, env="GENIE_SCREENSHOT_QUALITY")
    display_number: int = Field(1, env="GENIE_DISPLAY")
    debug_mode: bool = Field(False, env="GENIE_DEBUG")
    log_level: str = Field("INFO", env="GENIE_LOG_LEVEL")
    
    # Performance Settings
    cache_enabled: bool = Field(True, env="GENIE_CACHE_ENABLED")
    cache_ttl: int = Field(3600, env="GENIE_CACHE_TTL")
    parallel_execution: bool = Field(True, env="GENIE_PARALLEL")
    max_workers: int = Field(4, env="GENIE_MAX_WORKERS")
    
    # Storage Settings
    data_dir: Path = Field(Path.home() / ".computer_genie", env="GENIE_DATA_DIR")
    temp_dir: Path = Field(Path("/tmp/computer_genie"), env="GENIE_TEMP_DIR")
    log_dir: Path = Field(Path.home() / ".computer_genie/logs", env="GENIE_LOG_DIR")
    
    # Telemetry Settings
    telemetry_enabled: bool = Field(True, env="GENIE_TELEMETRY_ENABLED")
    telemetry_endpoint: str = Field("https://telemetry.computergenie.ai", env="GENIE_TELEMETRY_ENDPOINT")
    
    # Security Settings
    secure_mode: bool = Field(False, env="GENIE_SECURE_MODE")
    allowed_domains: list[str] = Field([], env="GENIE_ALLOWED_DOMAINS")
    blocked_domains: list[str] = Field([], env="GENIE_BLOCKED_DOMAINS")
    
    # API Server Settings
    api_host: str = Field("127.0.0.1", env="GENIE_API_HOST")
    api_port: int = Field(9261, env="GENIE_API_PORT")
    api_workers: int = Field(1, env="GENIE_API_WORKERS")
    
    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'protected_namespaces': ('settings_',)
    }
        
    @validator("data_dir", "temp_dir", "log_dir")
    def create_directories(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v

config = GenieConfig()