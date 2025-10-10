"""
Memory-mapped file system for efficient screenshot handling.
Provides zero-copy access to large image files with minimal memory overhead.
"""

import asyncio
import hashlib
import logging
import mmap
import os
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FileHeader:
    """Header structure for memory-mapped files."""
    magic: bytes = b'CGMF'  # Computer Genie Memory File
    version: int = 1
    file_type: int = 0  # 0=image, 1=data, 2=cache
    width: int = 0
    height: int = 0
    channels: int = 0
    dtype: int = 0  # numpy dtype code
    compression: int = 0  # 0=none, 1=lz4, 2=zstd
    timestamp: float = 0.0
    checksum: int = 0
    data_offset: int = 128  # Header size
    data_size: int = 0
    metadata_size: int = 0
    reserved: bytes = b'\x00' * 64
    
    def pack(self) -> bytes:
        """Pack header to bytes."""
        return struct.pack(
            '<4sIIIIIIIfIIII64s',
            self.magic, self.version, self.file_type,
            self.width, self.height, self.channels, self.dtype,
            self.compression, self.timestamp, self.checksum,
            self.data_offset, self.data_size, self.metadata_size,
            0, self.reserved
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'FileHeader':
        """Unpack header from bytes."""
        values = struct.unpack('<4sIIIIIIIfIIII64s', data[:128])
        return cls(
            magic=values[0],
            version=values[1],
            file_type=values[2],
            width=values[3],
            height=values[4],
            channels=values[5],
            dtype=values[6],
            compression=values[7],
            timestamp=values[8],
            checksum=values[9],
            data_offset=values[10],
            data_size=values[11],
            metadata_size=values[12],
            reserved=values[14]
        )
    
    def is_valid(self) -> bool:
        """Check if header is valid."""
        return self.magic == b'CGMF' and self.version == 1


class MemoryMappedFile:
    """Memory-mapped file wrapper for efficient access."""
    
    def __init__(self, file_path: Path, mode: str = 'r'):
        self.file_path = file_path
        self.mode = mode
        self.file_handle = None
        self.mmap_handle = None
        self.header: Optional[FileHeader] = None
        self.is_open = False
        
        # Performance tracking
        self.access_count = 0
        self.last_access_time = 0.0
        self.total_read_bytes = 0
    
    def open(self):
        """Open memory-mapped file."""
        try:
            if self.mode == 'r':
                self.file_handle = open(self.file_path, 'rb')
                self.mmap_handle = mmap.mmap(
                    self.file_handle.fileno(), 
                    0, 
                    access=mmap.ACCESS_READ
                )
            elif self.mode == 'w':
                self.file_handle = open(self.file_path, 'r+b')
                self.mmap_handle = mmap.mmap(
                    self.file_handle.fileno(), 
                    0, 
                    access=mmap.ACCESS_WRITE
                )
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            
            # Read header
            if self.mmap_handle.size() >= 128:
                header_data = self.mmap_handle[:128]
                self.header = FileHeader.unpack(header_data)
                
                if not self.header.is_valid():
                    raise ValueError("Invalid file header")
            
            self.is_open = True
            logger.debug(f"Opened memory-mapped file: {self.file_path}")
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"Failed to open memory-mapped file {self.file_path}: {e}")
    
    def close(self):
        """Close memory-mapped file."""
        if self.mmap_handle:
            self.mmap_handle.close()
            self.mmap_handle = None
        
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        
        self.is_open = False
    
    def read_data(self, offset: int = None, size: int = None) -> memoryview:
        """Read data with zero-copy access."""
        if not self.is_open or not self.header:
            raise RuntimeError("File not open")
        
        self.access_count += 1
        self.last_access_time = time.time()
        
        if offset is None:
            offset = self.header.data_offset
        
        if size is None:
            size = self.header.data_size
        
        self.total_read_bytes += size
        
        # Return memory view for zero-copy access
        return memoryview(self.mmap_handle)[offset:offset + size]
    
    def read_image_array(self) -> np.ndarray:
        """Read image data as numpy array."""
        if not self.header or self.header.file_type != 0:
            raise ValueError("Not an image file")
        
        data_view = self.read_data()
        
        # Convert to numpy array
        dtype = np.dtype(np.uint8 if self.header.dtype == 0 else np.float32)
        array = np.frombuffer(data_view, dtype=dtype)
        
        # Reshape to image dimensions
        if self.header.channels == 1:
            shape = (self.header.height, self.header.width)
        else:
            shape = (self.header.height, self.header.width, self.header.channels)
        
        return array.reshape(shape)
    
    def write_data(self, data: bytes, offset: int = None):
        """Write data to memory-mapped file."""
        if not self.is_open or self.mode != 'w':
            raise RuntimeError("File not open for writing")
        
        if offset is None:
            offset = self.header.data_offset if self.header else 128
        
        # Write data
        self.mmap_handle[offset:offset + len(data)] = data
        self.mmap_handle.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file access statistics."""
        return {
            "file_path": str(self.file_path),
            "file_size_mb": self.file_path.stat().st_size / (1024 * 1024) if self.file_path.exists() else 0,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "total_read_mb": self.total_read_bytes / (1024 * 1024),
            "is_open": self.is_open,
            "header_valid": self.header.is_valid() if self.header else False
        }
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MemoryMappedFileSystem:
    """High-performance file system using memory mapping."""
    
    def __init__(self, base_path: Path, max_open_files: int = 100):
        self.base_path = Path(base_path)
        self.max_open_files = max_open_files
        
        # File management
        self.open_files: Dict[str, MemoryMappedFile] = {}
        self.file_access_order: List[str] = []
        
        # Directory structure
        self.screenshots_dir = self.base_path / "screenshots"
        self.cache_dir = self.base_path / "cache"
        self.temp_dir = self.base_path / "temp"
        
        # Performance tracking
        self.stats = {
            "files_created": 0,
            "files_opened": 0,
            "files_closed": 0,
            "total_size_mb": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize directories
        self._initialize_directories()
    
    def _initialize_directories(self):
        """Initialize directory structure."""
        for directory in [self.screenshots_dir, self.cache_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized memory-mapped file system at {self.base_path}")
    
    def _generate_file_id(self, content_hash: str, file_type: str) -> str:
        """Generate unique file ID."""
        return f"{file_type}_{content_hash[:16]}"
    
    def _get_file_path(self, file_id: str, file_type: str) -> Path:
        """Get file path for given ID and type."""
        if file_type == "screenshot":
            return self.screenshots_dir / f"{file_id}.cgmf"
        elif file_type == "cache":
            return self.cache_dir / f"{file_id}.cgmf"
        elif file_type == "temp":
            return self.temp_dir / f"{file_id}.cgmf"
        else:
            raise ValueError(f"Unknown file type: {file_type}")
    
    async def store_screenshot(self, image_data: np.ndarray, metadata: Dict[str, Any] = None) -> str:
        """Store screenshot with memory mapping."""
        try:
            # Generate content hash
            content_hash = hashlib.md5(image_data.tobytes()).hexdigest()
            file_id = self._generate_file_id(content_hash, "screenshot")
            file_path = self._get_file_path(file_id, "screenshot")
            
            # Check if file already exists
            if file_path.exists():
                logger.debug(f"Screenshot already exists: {file_id}")
                return file_id
            
            # Create header
            header = FileHeader(
                file_type=0,  # Image
                width=image_data.shape[1] if len(image_data.shape) > 1 else image_data.shape[0],
                height=image_data.shape[0],
                channels=image_data.shape[2] if len(image_data.shape) == 3 else 1,
                dtype=0 if image_data.dtype == np.uint8 else 1,
                timestamp=time.time(),
                data_size=image_data.nbytes,
                metadata_size=len(str(metadata)) if metadata else 0
            )
            
            # Calculate checksum
            header.checksum = hash(image_data.tobytes()) & 0xFFFFFFFF
            
            # Create file
            with open(file_path, 'wb') as f:
                # Write header
                f.write(header.pack())
                
                # Write metadata if present
                if metadata:
                    metadata_bytes = str(metadata).encode('utf-8')
                    f.write(metadata_bytes)
                    # Pad to align data
                    padding = (128 - (len(metadata_bytes) % 128)) % 128
                    f.write(b'\x00' * padding)
                    header.data_offset = 128 + len(metadata_bytes) + padding
                
                # Write image data
                f.write(image_data.tobytes())
            
            self.stats["files_created"] += 1
            self.stats["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
            
            logger.debug(f"Stored screenshot: {file_id} ({image_data.shape})")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store screenshot: {e}")
            raise
    
    async def load_screenshot(self, file_id: str) -> Optional[np.ndarray]:
        """Load screenshot with memory mapping."""
        try:
            file_path = self._get_file_path(file_id, "screenshot")
            
            if not file_path.exists():
                self.stats["cache_misses"] += 1
                return None
            
            # Check if file is already open
            if file_id in self.open_files:
                mmf = self.open_files[file_id]
                self.stats["cache_hits"] += 1
            else:
                # Open new memory-mapped file
                mmf = await self._open_file(file_id, file_path, 'r')
            
            # Read image array
            image_array = mmf.read_image_array()
            
            logger.debug(f"Loaded screenshot: {file_id} ({image_array.shape})")
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to load screenshot {file_id}: {e}")
            return None
    
    async def _open_file(self, file_id: str, file_path: Path, mode: str) -> MemoryMappedFile:
        """Open memory-mapped file with LRU management."""
        # Check if we need to close old files
        while len(self.open_files) >= self.max_open_files:
            await self._close_lru_file()
        
        # Open new file
        mmf = MemoryMappedFile(file_path, mode)
        mmf.open()
        
        # Add to tracking
        self.open_files[file_id] = mmf
        self.file_access_order.append(file_id)
        self.stats["files_opened"] += 1
        
        return mmf
    
    async def _close_lru_file(self):
        """Close least recently used file."""
        if not self.file_access_order:
            return
        
        # Find LRU file
        lru_file_id = self.file_access_order[0]
        
        # Close and remove
        if lru_file_id in self.open_files:
            self.open_files[lru_file_id].close()
            del self.open_files[lru_file_id]
            self.stats["files_closed"] += 1
        
        self.file_access_order.remove(lru_file_id)
    
    async def get_screenshot_roi(self, file_id: str, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """Get region of interest from screenshot without loading entire image."""
        try:
            file_path = self._get_file_path(file_id, "screenshot")
            
            if not file_path.exists():
                return None
            
            # Open file if not already open
            if file_id not in self.open_files:
                await self._open_file(file_id, file_path, 'r')
            
            mmf = self.open_files[file_id]
            
            # Load full image (optimization: could implement partial loading)
            full_image = mmf.read_image_array()
            
            # Extract ROI
            if len(full_image.shape) == 3:
                roi = full_image[y:y+height, x:x+width, :]
            else:
                roi = full_image[y:y+height, x:x+width]
            
            return roi
            
        except Exception as e:
            logger.error(f"Failed to get ROI from {file_id}: {e}")
            return None
    
    async def store_cache_data(self, key: str, data: Any, ttl_seconds: float = 3600.0) -> str:
        """Store arbitrary data in cache with memory mapping."""
        try:
            # Serialize data
            import pickle
            serialized_data = pickle.dumps(data)
            
            # Generate file ID
            content_hash = hashlib.md5(serialized_data).hexdigest()
            file_id = self._generate_file_id(content_hash, "cache")
            file_path = self._get_file_path(file_id, "cache")
            
            # Check if already exists
            if file_path.exists():
                return file_id
            
            # Create header
            header = FileHeader(
                file_type=1,  # Data
                timestamp=time.time(),
                data_size=len(serialized_data),
                checksum=hash(serialized_data) & 0xFFFFFFFF
            )
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(header.pack())
                f.write(serialized_data)
            
            self.stats["files_created"] += 1
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store cache data: {e}")
            raise
    
    async def load_cache_data(self, file_id: str) -> Optional[Any]:
        """Load cached data with memory mapping."""
        try:
            file_path = self._get_file_path(file_id, "cache")
            
            if not file_path.exists():
                return None
            
            # Open file
            if file_id not in self.open_files:
                await self._open_file(file_id, file_path, 'r')
            
            mmf = self.open_files[file_id]
            
            # Check TTL
            if mmf.header and time.time() - mmf.header.timestamp > 3600:  # Default TTL
                await self.delete_file(file_id, "cache")
                return None
            
            # Read and deserialize data
            data_view = mmf.read_data()
            import pickle
            data = pickle.loads(data_view.tobytes())
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load cache data {file_id}: {e}")
            return None
    
    async def delete_file(self, file_id: str, file_type: str):
        """Delete file and clean up resources."""
        try:
            # Close if open
            if file_id in self.open_files:
                self.open_files[file_id].close()
                del self.open_files[file_id]
                if file_id in self.file_access_order:
                    self.file_access_order.remove(file_id)
            
            # Delete file
            file_path = self._get_file_path(file_id, file_type)
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_path.unlink()
                self.stats["total_size_mb"] -= file_size / (1024 * 1024)
                logger.debug(f"Deleted file: {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
    
    async def cleanup_expired_files(self, max_age_hours: float = 24.0):
        """Clean up expired files."""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for directory in [self.screenshots_dir, self.cache_dir, self.temp_dir]:
            for file_path in directory.glob("*.cgmf"):
                try:
                    # Check file age
                    file_age = current_time - file_path.stat().st_mtime
                    
                    if file_age > max_age_seconds:
                        file_id = file_path.stem
                        file_type = "screenshot" if directory == self.screenshots_dir else "cache"
                        await self.delete_file(file_id, file_type)
                        
                except Exception as e:
                    logger.warning(f"Failed to check file age {file_path}: {e}")
    
    async def get_file_info(self, file_id: str, file_type: str) -> Optional[Dict[str, Any]]:
        """Get file information without loading data."""
        try:
            file_path = self._get_file_path(file_id, file_type)
            
            if not file_path.exists():
                return None
            
            # Read header only
            with open(file_path, 'rb') as f:
                header_data = f.read(128)
                header = FileHeader.unpack(header_data)
            
            if not header.is_valid():
                return None
            
            file_stats = file_path.stat()
            
            return {
                "file_id": file_id,
                "file_type": file_type,
                "file_path": str(file_path),
                "file_size_mb": file_stats.st_size / (1024 * 1024),
                "created_time": header.timestamp,
                "modified_time": file_stats.st_mtime,
                "width": header.width,
                "height": header.height,
                "channels": header.channels,
                "data_size_mb": header.data_size / (1024 * 1024),
                "is_open": file_id in self.open_files
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info {file_id}: {e}")
            return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get file system statistics."""
        # Calculate directory sizes
        total_size = 0
        file_counts = {"screenshot": 0, "cache": 0, "temp": 0}
        
        for directory, file_type in [(self.screenshots_dir, "screenshot"), 
                                   (self.cache_dir, "cache"), 
                                   (self.temp_dir, "temp")]:
            for file_path in directory.glob("*.cgmf"):
                total_size += file_path.stat().st_size
                file_counts[file_type] += 1
        
        return {
            **self.stats,
            "total_size_mb": total_size / (1024 * 1024),
            "open_files": len(self.open_files),
            "max_open_files": self.max_open_files,
            "file_counts": file_counts,
            "directories": {
                "screenshots": str(self.screenshots_dir),
                "cache": str(self.cache_dir),
                "temp": str(self.temp_dir)
            },
            "performance": {
                "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["cache_hits"] + self.stats["cache_misses"], 1),
                "avg_file_size_mb": total_size / (1024 * 1024) / max(sum(file_counts.values()), 1),
                "memory_efficiency": len(self.open_files) / max(self.max_open_files, 1)
            }
        }
    
    async def benchmark_performance(self, num_screenshots: int = 100) -> Dict[str, Any]:
        """Benchmark memory-mapped file system performance."""
        # Generate test data
        test_images = []
        for i in range(num_screenshots):
            # Create random image
            image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            test_images.append(image)
        
        # Benchmark storage
        storage_times = []
        file_ids = []
        
        for image in test_images:
            start_time = time.time()
            file_id = await self.store_screenshot(image)
            storage_time = (time.time() - start_time) * 1000
            storage_times.append(storage_time)
            file_ids.append(file_id)
        
        # Benchmark loading
        loading_times = []
        
        for file_id in file_ids:
            start_time = time.time()
            loaded_image = await self.load_screenshot(file_id)
            loading_time = (time.time() - start_time) * 1000
            loading_times.append(loading_time)
        
        # Benchmark ROI extraction
        roi_times = []
        
        for file_id in file_ids[:10]:  # Test subset for ROI
            start_time = time.time()
            roi = await self.get_screenshot_roi(file_id, 100, 100, 200, 200)
            roi_time = (time.time() - start_time) * 1000
            roi_times.append(roi_time)
        
        # Clean up test files
        for file_id in file_ids:
            await self.delete_file(file_id, "screenshot")
        
        return {
            "storage_performance": {
                "avg_time_ms": np.mean(storage_times),
                "min_time_ms": np.min(storage_times),
                "max_time_ms": np.max(storage_times),
                "target_100ms_met": np.mean(storage_times) < 100
            },
            "loading_performance": {
                "avg_time_ms": np.mean(loading_times),
                "min_time_ms": np.min(loading_times),
                "max_time_ms": np.max(loading_times),
                "target_100ms_met": np.mean(loading_times) < 100
            },
            "roi_performance": {
                "avg_time_ms": np.mean(roi_times),
                "min_time_ms": np.min(roi_times),
                "max_time_ms": np.max(roi_times),
                "target_100ms_met": np.mean(roi_times) < 100
            },
            "overall_performance": {
                "total_operations": len(storage_times) + len(loading_times) + len(roi_times),
                "avg_operation_time_ms": np.mean(storage_times + loading_times + roi_times),
                "memory_mapped_efficiency": "High" if np.mean(loading_times) < 50 else "Medium"
            }
        }
    
    async def close_all_files(self):
        """Close all open files."""
        for file_id, mmf in list(self.open_files.items()):
            mmf.close()
        
        self.open_files.clear()
        self.file_access_order.clear()
    
    async def cleanup(self):
        """Clean up file system resources."""
        await self.close_all_files()
        logger.info("Memory-mapped file system cleaned up")


# Factory function for easy integration
def create_memory_mapped_filesystem(base_path: Union[str, Path], max_open_files: int = 100) -> MemoryMappedFileSystem:
    """Create memory-mapped file system."""
    return MemoryMappedFileSystem(Path(base_path), max_open_files)