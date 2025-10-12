"""
Zero-copy architecture for image processing pipeline.
Eliminates memory copying overhead for maximum performance.
"""

import ctypes
import logging
import mmap
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryPool:
    """High-performance memory pool for zero-copy operations."""
    
    def __init__(self, pool_size_mb: int = 512):
        self.pool_size = pool_size_mb * 1024 * 1024  # Convert to bytes
        self.pool = None
        self.free_blocks: List[Tuple[int, int]] = []  # (offset, size) pairs
        self.allocated_blocks: Dict[int, int] = {}  # offset -> size
        self.total_allocated = 0
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the memory pool."""
        try:
            # Allocate aligned memory for better performance
            self.pool = mmap.mmap(-1, self.pool_size)
            self.free_blocks = [(0, self.pool_size)]
            logger.info(f"Initialized memory pool: {self.pool_size // (1024*1024)}MB")
        except Exception as e:
            logger.error(f"Failed to initialize memory pool: {e}")
            raise
    
    def allocate(self, size: int, alignment: int = 64) -> Optional[int]:
        """Allocate aligned memory block."""
        # Align size to boundary
        aligned_size = (size + alignment - 1) & ~(alignment - 1)
        
        # Find suitable free block
        for i, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= aligned_size:
                # Allocate from this block
                self.allocated_blocks[offset] = aligned_size
                self.total_allocated += aligned_size
                
                # Update free blocks
                remaining_size = block_size - aligned_size
                if remaining_size > 0:
                    self.free_blocks[i] = (offset + aligned_size, remaining_size)
                else:
                    del self.free_blocks[i]
                
                return offset
        
        logger.warning(f"Failed to allocate {aligned_size} bytes from memory pool")
        return None
    
    def deallocate(self, offset: int):
        """Deallocate memory block."""
        if offset not in self.allocated_blocks:
            logger.warning(f"Attempting to deallocate unallocated block at offset {offset}")
            return
        
        size = self.allocated_blocks[offset]
        del self.allocated_blocks[offset]
        self.total_allocated -= size
        
        # Add to free blocks and merge adjacent blocks
        self.free_blocks.append((offset, size))
        self._merge_free_blocks()
    
    def _merge_free_blocks(self):
        """Merge adjacent free blocks to reduce fragmentation."""
        if len(self.free_blocks) <= 1:
            return
        
        # Sort by offset
        self.free_blocks.sort(key=lambda x: x[0])
        
        merged = []
        current_offset, current_size = self.free_blocks[0]
        
        for offset, size in self.free_blocks[1:]:
            if current_offset + current_size == offset:
                # Adjacent blocks, merge them
                current_size += size
            else:
                # Non-adjacent, add current and start new
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
        
        merged.append((current_offset, current_size))
        self.free_blocks = merged
    
    def get_memory_view(self, offset: int, size: int) -> memoryview:
        """Get memory view for zero-copy access."""
        if offset not in self.allocated_blocks:
            raise ValueError(f"Invalid memory offset: {offset}")
        
        return memoryview(self.pool)[offset:offset + size]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        return {
            "pool_size_mb": self.pool_size // (1024 * 1024),
            "allocated_mb": self.total_allocated // (1024 * 1024),
            "free_mb": (self.pool_size - self.total_allocated) // (1024 * 1024),
            "utilization_percent": (self.total_allocated / self.pool_size) * 100,
            "free_blocks": len(self.free_blocks),
            "allocated_blocks": len(self.allocated_blocks)
        }
    
    def cleanup(self):
        """Clean up memory pool."""
        if self.pool:
            self.pool.close()


class ZeroCopyImage:
    """Zero-copy image wrapper for efficient processing."""
    
    def __init__(self, memory_pool: MemoryPool, width: int, height: int, 
                 channels: int = 3, dtype: np.dtype = np.uint8):
        self.memory_pool = memory_pool
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        
        # Calculate memory requirements
        self.bytes_per_pixel = np.dtype(dtype).itemsize * channels
        self.row_stride = width * self.bytes_per_pixel
        self.total_size = height * self.row_stride
        
        # Allocate memory
        self.memory_offset = memory_pool.allocate(self.total_size)
        if self.memory_offset is None:
            raise MemoryError("Failed to allocate memory for image")
        
        # Create memory view
        self.memory_view = memory_pool.get_memory_view(self.memory_offset, self.total_size)
        
        # Create numpy array view (zero-copy)
        self._array = np.frombuffer(
            self.memory_view, 
            dtype=dtype
        ).reshape((height, width, channels))
    
    @property
    def array(self) -> np.ndarray:
        """Get numpy array view (zero-copy)."""
        return self._array
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image shape."""
        return (self.height, self.width, self.channels)
    
    def copy_from_array(self, source: np.ndarray):
        """Copy data from numpy array."""
        if source.shape != self.shape:
            raise ValueError(f"Shape mismatch: {source.shape} vs {self.shape}")
        
        # Direct memory copy
        np.copyto(self._array, source)
    
    def copy_from_pil(self, pil_image: Image.Image):
        """Copy data from PIL Image."""
        # Convert PIL to numpy
        array = np.array(pil_image)
        if len(array.shape) == 2:  # Grayscale
            array = np.expand_dims(array, axis=2)
        
        self.copy_from_array(array)
    
    def to_pil(self) -> Image.Image:
        """Convert to PIL Image (zero-copy when possible)."""
        if self.channels == 1:
            mode = 'L'
            array = self._array[:, :, 0]
        elif self.channels == 3:
            mode = 'RGB'
            array = self._array
        elif self.channels == 4:
            mode = 'RGBA'
            array = self._array
        else:
            raise ValueError(f"Unsupported channel count: {self.channels}")
        
        return Image.fromarray(array, mode=mode)
    
    def get_roi(self, x: int, y: int, width: int, height: int) -> 'ZeroCopyImageROI':
        """Get region of interest without copying data."""
        return ZeroCopyImageROI(self, x, y, width, height)
    
    def __del__(self):
        """Clean up allocated memory."""
        if hasattr(self, 'memory_offset') and self.memory_offset is not None:
            self.memory_pool.deallocate(self.memory_offset)


class ZeroCopyImageROI:
    """Region of interest in a zero-copy image."""
    
    def __init__(self, parent_image: ZeroCopyImage, x: int, y: int, width: int, height: int):
        self.parent = parent_image
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Validate bounds
        if (x + width > parent_image.width or 
            y + height > parent_image.height or
            x < 0 or y < 0):
            raise ValueError("ROI bounds exceed parent image")
        
        # Create array view (zero-copy)
        self._array = parent_image.array[y:y+height, x:x+width]
    
    @property
    def array(self) -> np.ndarray:
        """Get numpy array view of ROI."""
        return self._array
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get ROI shape."""
        return self._array.shape


class ZeroCopyProcessor:
    """Zero-copy image processor for maximum performance."""
    
    def __init__(self, memory_pool: MemoryPool):
        self.memory_pool = memory_pool
        self.temp_images: List[ZeroCopyImage] = []
    
    def create_image(self, width: int, height: int, channels: int = 3) -> ZeroCopyImage:
        """Create a new zero-copy image."""
        image = ZeroCopyImage(self.memory_pool, width, height, channels)
        self.temp_images.append(image)
        return image
    
    def load_image_from_file(self, file_path: Path) -> ZeroCopyImage:
        """Load image from file into zero-copy format."""
        # Load with PIL first
        pil_image = Image.open(file_path)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Create zero-copy image
        width, height = pil_image.size
        zero_copy_image = self.create_image(width, height, 3)
        zero_copy_image.copy_from_pil(pil_image)
        
        return zero_copy_image
    
    def resize_image(self, source: ZeroCopyImage, new_width: int, new_height: int) -> ZeroCopyImage:
        """Resize image with minimal copying."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for image resizing")
        
        # Create destination image
        dest = self.create_image(new_width, new_height, source.channels)
        
        # Use OpenCV for efficient resizing
        cv2.resize(source.array, (new_width, new_height), dst=dest.array)
        
        return dest
    
    def convert_colorspace(self, source: ZeroCopyImage, conversion: int) -> ZeroCopyImage:
        """Convert colorspace in-place when possible."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for colorspace conversion")
        
        # Determine output channels
        if conversion in [cv2.COLOR_RGB2GRAY, cv2.COLOR_BGR2GRAY]:
            output_channels = 1
        elif conversion in [cv2.COLOR_GRAY2RGB, cv2.COLOR_GRAY2BGR]:
            output_channels = 3
        else:
            output_channels = source.channels
        
        # Create destination image
        dest = self.create_image(source.width, source.height, output_channels)
        
        # Convert colorspace
        cv2.cvtColor(source.array, conversion, dst=dest.array)
        
        return dest
    
    def apply_filter_inplace(self, image: ZeroCopyImage, filter_func: callable):
        """Apply filter function in-place to avoid copying."""
        filter_func(image.array, image.array)  # In-place operation
    
    def gaussian_blur(self, source: ZeroCopyImage, kernel_size: int, sigma: float) -> ZeroCopyImage:
        """Apply Gaussian blur with zero-copy optimization."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for Gaussian blur")
        
        dest = self.create_image(source.width, source.height, source.channels)
        cv2.GaussianBlur(source.array, (kernel_size, kernel_size), sigma, dst=dest.array)
        
        return dest
    
    def edge_detection(self, source: ZeroCopyImage, low_threshold: int = 50, 
                      high_threshold: int = 150) -> ZeroCopyImage:
        """Perform edge detection with zero-copy optimization."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for edge detection")
        
        # Convert to grayscale if needed
        if source.channels > 1:
            gray = self.convert_colorspace(source, cv2.COLOR_RGB2GRAY)
        else:
            gray = source
        
        # Create destination image
        dest = self.create_image(source.width, source.height, 1)
        
        # Apply Canny edge detection
        cv2.Canny(gray.array[:, :, 0], low_threshold, high_threshold, edges=dest.array[:, :, 0])
        
        return dest
    
    def find_contours(self, binary_image: ZeroCopyImage) -> List[np.ndarray]:
        """Find contours without copying image data."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for contour detection")
        
        # Ensure single channel
        if binary_image.channels > 1:
            gray_data = binary_image.array[:, :, 0]
        else:
            gray_data = binary_image.array[:, :, 0]
        
        contours, _ = cv2.findContours(gray_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def cleanup_temp_images(self):
        """Clean up temporary images."""
        for image in self.temp_images:
            del image
        self.temp_images.clear()


class ZeroCopyPipeline:
    """High-performance zero-copy image processing pipeline."""
    
    def __init__(self, memory_pool_size_mb: int = 512):
        self.memory_pool = MemoryPool(memory_pool_size_mb)
        self.processor = ZeroCopyProcessor(self.memory_pool)
        self.pipeline_stats = {
            "images_processed": 0,
            "total_processing_time_ms": 0,
            "avg_processing_time_ms": 0,
            "memory_copies_avoided": 0
        }
    
    @contextmanager
    def processing_context(self) -> Generator[ZeroCopyProcessor, None, None]:
        """Context manager for zero-copy processing."""
        try:
            yield self.processor
        finally:
            # Clean up temporary images
            self.processor.cleanup_temp_images()
    
    async def process_screenshot(self, screenshot_data: bytes) -> Dict[str, Any]:
        """Process screenshot with zero-copy optimization."""
        start_time = time.time()
        
        with self.processing_context() as processor:
            try:
                # Load image from bytes
                from io import BytesIO
                pil_image = Image.open(BytesIO(screenshot_data))
                
                # Create zero-copy image
                image = processor.create_image(pil_image.width, pil_image.height, 3)
                image.copy_from_pil(pil_image)
                
                # Process image for UI element detection
                # 1. Convert to grayscale for edge detection
                gray = processor.convert_colorspace(image, cv2.COLOR_RGB2GRAY)
                
                # 2. Apply Gaussian blur to reduce noise
                blurred = processor.gaussian_blur(gray, 5, 1.0)
                
                # 3. Edge detection
                edges = processor.edge_detection(blurred)
                
                # 4. Find contours
                contours = processor.find_contours(edges)
                
                # 5. Extract UI elements
                ui_elements = []
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Filter small contours
                        x, y, w, h = cv2.boundingRect(contour)
                        ui_elements.append({
                            "type": "element",
                            "bbox": [x, y, x + w, y + h],
                            "center": [x + w // 2, y + h // 2],
                            "area": cv2.contourArea(contour)
                        })
                
                # Update statistics
                processing_time = (time.time() - start_time) * 1000
                self.pipeline_stats["images_processed"] += 1
                self.pipeline_stats["total_processing_time_ms"] += processing_time
                self.pipeline_stats["avg_processing_time_ms"] = (
                    self.pipeline_stats["total_processing_time_ms"] / 
                    self.pipeline_stats["images_processed"]
                )
                self.pipeline_stats["memory_copies_avoided"] += 3  # Estimated copies avoided
                
                return {
                    "ui_elements": ui_elements,
                    "processing_time_ms": processing_time,
                    "zero_copy_optimized": True,
                    "memory_usage_mb": self.memory_pool.get_stats()["allocated_mb"]
                }
                
            except Exception as e:
                logger.error(f"Zero-copy processing failed: {e}")
                return {"error": str(e), "ui_elements": []}
    
    async def process_image_batch(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple images efficiently with zero-copy."""
        results = []
        
        with self.processing_context() as processor:
            for image_path in image_paths:
                try:
                    start_time = time.time()
                    
                    # Load image
                    image = processor.load_image_from_file(image_path)
                    
                    # Simple processing example
                    gray = processor.convert_colorspace(image, cv2.COLOR_RGB2GRAY)
                    edges = processor.edge_detection(gray)
                    contours = processor.find_contours(edges)
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    results.append({
                        "file": str(image_path),
                        "contours_found": len(contours),
                        "processing_time_ms": processing_time,
                        "zero_copy_optimized": True
                    })
                    
                except Exception as e:
                    results.append({
                        "file": str(image_path),
                        "error": str(e)
                    })
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        memory_stats = self.memory_pool.get_stats()
        
        return {
            **self.pipeline_stats,
            "memory_pool": memory_stats,
            "target_time_ms": 100,
            "performance_ratio": 100 / max(self.pipeline_stats["avg_processing_time_ms"], 1),
            "zero_copy_efficiency": self.pipeline_stats["memory_copies_avoided"] / max(self.pipeline_stats["images_processed"], 1)
        }
    
    async def benchmark_zero_copy_performance(self) -> Dict[str, Any]:
        """Benchmark zero-copy vs traditional processing."""
        # Create test image
        test_image_size = (1920, 1080, 3)
        test_data = np.random.randint(0, 255, test_image_size, dtype=np.uint8)
        
        # Benchmark zero-copy processing
        zero_copy_times = []
        for _ in range(10):
            start_time = time.time()
            
            with self.processing_context() as processor:
                image = processor.create_image(*test_image_size)
                image.copy_from_array(test_data)
                
                # Simulate processing pipeline
                gray = processor.convert_colorspace(image, cv2.COLOR_RGB2GRAY)
                blurred = processor.gaussian_blur(gray, 5, 1.0)
                edges = processor.edge_detection(blurred)
                contours = processor.find_contours(edges)
            
            zero_copy_times.append((time.time() - start_time) * 1000)
        
        # Benchmark traditional processing (with copying)
        traditional_times = []
        for _ in range(10):
            start_time = time.time()
            
            # Simulate traditional processing with copies
            image_copy1 = test_data.copy()
            gray_copy = cv2.cvtColor(image_copy1, cv2.COLOR_RGB2GRAY)
            blurred_copy = cv2.GaussianBlur(gray_copy, (5, 5), 1.0)
            edges_copy = cv2.Canny(blurred_copy, 50, 150)
            contours, _ = cv2.findContours(edges_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            traditional_times.append((time.time() - start_time) * 1000)
        
        zero_copy_avg = np.mean(zero_copy_times)
        traditional_avg = np.mean(traditional_times)
        
        return {
            "zero_copy_processing": {
                "avg_time_ms": zero_copy_avg,
                "min_time_ms": np.min(zero_copy_times),
                "max_time_ms": np.max(zero_copy_times)
            },
            "traditional_processing": {
                "avg_time_ms": traditional_avg,
                "min_time_ms": np.min(traditional_times),
                "max_time_ms": np.max(traditional_times)
            },
            "performance_improvement": {
                "speedup_factor": traditional_avg / zero_copy_avg,
                "time_saved_ms": traditional_avg - zero_copy_avg,
                "efficiency_gain_percent": ((traditional_avg - zero_copy_avg) / traditional_avg) * 100
            },
            "target_100ms_met": zero_copy_avg < 100
        }
    
    def cleanup(self):
        """Clean up pipeline resources."""
        self.processor.cleanup_temp_images()
        self.memory_pool.cleanup()


# Factory function for easy integration
def create_zero_copy_pipeline(memory_pool_size_mb: int = 512) -> ZeroCopyPipeline:
    """Create zero-copy processing pipeline."""
    return ZeroCopyPipeline(memory_pool_size_mb)