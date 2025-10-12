"""Clipboard tool for Computer Genie"""

import asyncio
from typing import Optional
from .base import BaseTool

class Clipboard(BaseTool):
    """Tool for clipboard operations"""
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__("clipboard", config)
    
    async def initialize(self):
        """Initialize clipboard tool"""
        pass
    
    async def cleanup(self):
        """Cleanup clipboard tool"""
        pass
    
    async def execute(self, action: str, *args, **kwargs):
        """Execute clipboard operation"""
        if action == "copy":
            return await self.copy(args[0] if args else "")
        elif action == "paste":
            return await self.paste()
        elif action == "clear":
            return await self.clear()
        else:
            raise ValueError(f"Unknown clipboard action: {action}")
    
    async def copy(self, text: str) -> bool:
        """Copy text to clipboard"""
        try:
            # Use PowerShell to copy to clipboard on Windows
            process = await asyncio.create_subprocess_exec(
                "powershell", "-Command", f"Set-Clipboard -Value '{text}'",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False
    
    async def paste(self) -> str:
        """Get text from clipboard"""
        try:
            # Use PowerShell to get clipboard content on Windows
            process = await asyncio.create_subprocess_exec(
                "powershell", "-Command", "Get-Clipboard",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                return stdout.decode().strip()
            return ""
        except Exception:
            return ""
    
    async def clear(self) -> bool:
        """Clear clipboard"""
        return await self.copy("")

__all__ = ['Clipboard']