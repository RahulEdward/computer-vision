"""
SIMD (Single Instruction, Multiple Data) optimizations for parallel pixel processing.
Provides vectorized operations for high-performance image processing.
"""

import logging
import platform
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from numba import jit, prange, vectorize
from numba.types import float32, uint8

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SIMDCapabilities:
    """Detect and manage SIMD capabilities."""
    
    def __init__(self):
        self.capabilities = self._detect_capabilities()
        self.optimal_chunk_size = self._determine_optimal_chunk_size()
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect available SIMD instruction sets."""
        capabilities = {
            "sse": False,
            "sse2": False,
            "sse3": False,
            "ssse3": False,
            "sse4_1": False,
            "sse4_2": False,
            "avx": False,
            "avx2": False,
            "avx512": False,
            "neon": False,  # ARM NEON
            "cuda": CUDA_AVAILABLE,
            "opencl": False
        }
        
        # Check CPU features (simplified detection)
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            flags = cpu_info.get('flags', [])
            
            for flag in flags:
                flag_lower = flag.lower()
                if flag_lower in capabilities:
                    capabilities[flag_lower] = True
                elif 'avx512' in flag_lower:
                    capabilities['avx512'] = True
                elif 'avx2' in flag_lower:
                    capabilities['avx2'] = True
                elif 'avx' in flag_lower:
                    capabilities['avx'] = True
        except ImportError:
            logger.warning("cpuinfo not available, using basic detection")
            
            # Basic platform detection
            if platform.machine().lower() in ['amd64', 'x86_64']:
                capabilities.update({
                    "sse": True,
                    "sse2": True,
                    "sse3": True,
                    "avx": True
                })
            elif 'arm' in platform.machine().lower():
                capabilities["neon"] = True
        
        # Check OpenCV SIMD support
        try:
            if hasattr(cv2, 'useOptimized') and cv2.useOptimized():
                capabilities["opencv_simd"] = True
        except:
            capabilities["opencv_simd"] = False
        
        return capabilities
    
    def _determine_optimal_chunk_size(self) -> int:
        """Determine optimal chunk size for SIMD operations."""
        if self.capabilities.get("avx512"):
            return 64  # 512 bits / 8 bits per byte
        elif self.capabilities.get("avx2") or self.capabilities.get("avx"):
            return 32  # 256 bits / 8 bits per byte
        elif any(self.capabilities.get(f"sse{i}") for i in ["", "2", "3", "4_1", "4_2"]):
            return 16  # 128 bits / 8 bits per byte
        elif self.capabilities.get("neon"):
            return 16  # ARM NEON 128-bit
        else:
            return 8   # Fallback
    
    def get_best_implementation(self) -> str:
        """Get the best available SIMD implementation."""
        if self.capabilities.get("cuda"):
            return "cuda"
        elif self.capabilities.get("avx512"):
            return "avx512"
        elif self.capabilities.get("avx2"):
            return "avx2"
        elif self.capabilities.get("avx"):
            return "avx"
        elif self.capabilities.get("sse4_2"):
            return "sse4_2"
        elif self.capabilities.get("neon"):
            return "neon"
        else:
            return "scalar"


# Numba-optimized functions for CPU SIMD
@jit(nopython=True, parallel=True, fastmath=True)
def simd_rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using SIMD-optimized operations."""
    height, width = rgb_image.shape[:2]
    grayscale = np.empty((height, width), dtype=np.uint8)
    
    for i in prange(height):
        for j in prange(width):
            # Standard luminance formula with integer arithmetic
            r, g, b = rgb_image[i, j, 0], rgb_image[i, j, 1], rgb_image[i, j, 2]
            grayscale[i, j] = (77 * r + 150 * g + 29 * b) >> 8
    
    return grayscale


@jit(nopython=True, parallel=True, fastmath=True)
def simd_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur using SIMD-optimized convolution."""
    height, width = image.shape[:2]
    
    # Generate Gaussian kernel
    kernel = np.empty((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    sum_kernel = 0.0
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
            sum_kernel += kernel[i, j]
    
    # Normalize kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] /= sum_kernel
    
    # Apply convolution
    if len(image.shape) == 3:
        channels = image.shape[2]
        result = np.empty_like(image, dtype=np.uint8)
        
        for c in prange(channels):
            for i in prange(center, height - center):
                for j in prange(center, width - center):
                    value = 0.0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            value += image[i + ki - center, j + kj - center, c] * kernel[ki, kj]
                    result[i, j, c] = min(255, max(0, int(value)))
    else:
        result = np.empty_like(image, dtype=np.uint8)
        for i in prange(center, height - center):
            for j in prange(center, width - center):
                value = 0.0
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        value += image[i + ki - center, j + kj - center] * kernel[ki, kj]
                result[i, j] = min(255, max(0, int(value)))
    
    return result


@jit(nopython=True, parallel=True, fastmath=True)
def simd_edge_detection(image: np.ndarray) -> np.ndarray:
    """Sobel edge detection using SIMD optimization."""
    height, width = image.shape[:2]
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    edges = np.empty((height, width), dtype=np.uint8)
    
    for i in prange(1, height - 1):
        for j in prange(1, width - 1):
            gx = 0.0
            gy = 0.0
            
            for ki in range(3):
                for kj in range(3):
                    pixel = float(image[i + ki - 1, j + kj - 1])
                    gx += pixel * sobel_x[ki, kj]
                    gy += pixel * sobel_y[ki, kj]
            
            magnitude = np.sqrt(gx * gx + gy * gy)
            edges[i, j] = min(255, max(0, int(magnitude)))
    
    return edges


@jit(nopython=True, parallel=True, fastmath=True)
def simd_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Histogram equalization using SIMD optimization."""
    height, width = image.shape[:2]
    
    # Calculate histogram
    histogram = np.zeros(256, dtype=np.int32)
    for i in prange(height):
        for j in prange(width):
            histogram[image[i, j]] += 1
    
    # Calculate cumulative distribution
    cdf = np.empty(256, dtype=np.float32)
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i]
    
    # Normalize CDF
    total_pixels = height * width
    for i in range(256):
        cdf[i] = (cdf[i] * 255.0) / total_pixels
    
    # Apply equalization
    result = np.empty_like(image, dtype=np.uint8)
    for i in prange(height):
        for j in prange(width):
            result[i, j] = int(cdf[image[i, j]])
    
    return result


@vectorize([uint8(uint8, uint8)], target='parallel')
def simd_blend_pixels(pixel1: np.uint8, pixel2: np.uint8) -> np.uint8:
    """Blend two pixels using vectorized operations."""
    return (pixel1 + pixel2) // 2


@jit(nopython=True, parallel=True, fastmath=True)
def simd_resize_bilinear(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Bilinear image resizing using SIMD optimization."""
    old_height, old_width = image.shape[:2]
    
    # Calculate scaling factors
    scale_x = old_width / new_width
    scale_y = old_height / new_height
    
    if len(image.shape) == 3:
        channels = image.shape[2]
        result = np.empty((new_height, new_width, channels), dtype=np.uint8)
        
        for c in prange(channels):
            for i in prange(new_height):
                for j in prange(new_width):
                    # Calculate source coordinates
                    src_y = i * scale_y
                    src_x = j * scale_x
                    
                    # Get integer and fractional parts
                    y1 = int(src_y)
                    x1 = int(src_x)
                    y2 = min(y1 + 1, old_height - 1)
                    x2 = min(x1 + 1, old_width - 1)
                    
                    dy = src_y - y1
                    dx = src_x - x1
                    
                    # Bilinear interpolation
                    top_left = image[y1, x1, c]
                    top_right = image[y1, x2, c]
                    bottom_left = image[y2, x1, c]
                    bottom_right = image[y2, x2, c]
                    
                    top = top_left * (1 - dx) + top_right * dx
                    bottom = bottom_left * (1 - dx) + bottom_right * dx
                    value = top * (1 - dy) + bottom * dy
                    
                    result[i, j, c] = int(value)
    else:
        result = np.empty((new_height, new_width), dtype=np.uint8)
        
        for i in prange(new_height):
            for j in prange(new_width):
                src_y = i * scale_y
                src_x = j * scale_x
                
                y1 = int(src_y)
                x1 = int(src_x)
                y2 = min(y1 + 1, old_height - 1)
                x2 = min(x1 + 1, old_width - 1)
                
                dy = src_y - y1
                dx = src_x - x1
                
                top_left = image[y1, x1]
                top_right = image[y1, x2]
                bottom_left = image[y2, x1]
                bottom_right = image[y2, x2]
                
                top = top_left * (1 - dx) + top_right * dx
                bottom = bottom_left * (1 - dx) + bottom_right * dx
                value = top * (1 - dy) + bottom * dy
                
                result[i, j] = int(value)
    
    return result


# CUDA kernels for GPU acceleration
if CUDA_AVAILABLE:
    @cuda.jit
    def cuda_rgb_to_grayscale_kernel(rgb_image, grayscale):
        """CUDA kernel for RGB to grayscale conversion."""
        i, j = cuda.grid(2)
        if i < rgb_image.shape[0] and j < rgb_image.shape[1]:
            r, g, b = rgb_image[i, j, 0], rgb_image[i, j, 1], rgb_image[i, j, 2]
            grayscale[i, j] = (77 * r + 150 * g + 29 * b) >> 8
    
    @cuda.jit
    def cuda_gaussian_blur_kernel(image, result, kernel, kernel_size):
        """CUDA kernel for Gaussian blur."""
        i, j = cuda.grid(2)
        height, width = image.shape[:2]
        center = kernel_size // 2
        
        if center <= i < height - center and center <= j < width - center:
            if len(image.shape) == 3:
                channels = image.shape[2]
                for c in range(channels):
                    value = 0.0
                    for ki in range(kernel_size):
                        for kj in range(kernel_size):
                            value += image[i + ki - center, j + kj - center, c] * kernel[ki, kj]
                    result[i, j, c] = min(255, max(0, int(value)))
            else:
                value = 0.0
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        value += image[i + ki - center, j + kj - center] * kernel[ki, kj]
                result[i, j] = min(255, max(0, int(value)))


class SIMDProcessor:
    """High-performance SIMD image processor."""
    
    def __init__(self, use_gpu: bool = True):
        self.capabilities = SIMDCapabilities()
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.implementation = self.capabilities.get_best_implementation()
        
        # Performance statistics
        self.stats = {
            "operations_count": 0,
            "total_processing_time_ms": 0.0,
            "avg_processing_time_ms": 0.0,
            "pixels_processed": 0,
            "throughput_mpixels_per_sec": 0.0
        }
        
        logger.info(f"SIMD Processor initialized with {self.implementation} implementation")
        logger.info(f"Capabilities: {self.capabilities.capabilities}")
    
    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using SIMD optimization."""
        start_time = time.time()
        
        try:
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU implementation
                gpu_image = cp.asarray(image)
                gpu_result = cp.empty((image.shape[0], image.shape[1]), dtype=cp.uint8)
                
                # Use CuPy's built-in optimized functions when possible
                if len(image.shape) == 3:
                    weights = cp.array([0.299, 0.587, 0.114], dtype=cp.float32)
                    gpu_result = cp.dot(gpu_image, weights).astype(cp.uint8)
                else:
                    gpu_result = gpu_image.copy()
                
                result = cp.asnumpy(gpu_result)
            
            elif self.use_gpu and CUDA_AVAILABLE:
                # CUDA implementation
                gpu_image = cuda.to_device(image)
                gpu_result = cuda.device_array((image.shape[0], image.shape[1]), dtype=np.uint8)
                
                threads_per_block = (16, 16)
                blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
                blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
                blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
                
                cuda_rgb_to_grayscale_kernel[blocks_per_grid, threads_per_block](gpu_image, gpu_result)
                result = gpu_result.copy_to_host()
            
            else:
                # CPU SIMD implementation
                if len(image.shape) == 3:
                    result = simd_rgb_to_grayscale(image)
                else:
                    result = image.copy()
            
            self._update_stats(start_time, image.shape[0] * image.shape[1])
            return result
            
        except Exception as e:
            logger.error(f"SIMD RGB to grayscale failed: {e}")
            # Fallback to OpenCV
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                return image.copy()
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur using SIMD optimization."""
        start_time = time.time()
        
        try:
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU implementation using CuPy
                gpu_image = cp.asarray(image)
                
                # Use CuPy's optimized Gaussian filter
                from cupyx.scipy import ndimage
                result = ndimage.gaussian_filter(gpu_image, sigma=sigma)
                result = cp.asnumpy(result).astype(np.uint8)
            
            else:
                # CPU SIMD implementation
                result = simd_gaussian_blur(image, kernel_size, sigma)
            
            self._update_stats(start_time, image.shape[0] * image.shape[1])
            return result
            
        except Exception as e:
            logger.error(f"SIMD Gaussian blur failed: {e}")
            # Fallback to OpenCV
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Perform edge detection using SIMD optimization."""
        start_time = time.time()
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = self.rgb_to_grayscale(image)
            else:
                gray_image = image
            
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU implementation
                gpu_image = cp.asarray(gray_image)
                
                # Sobel filters
                sobel_x = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
                sobel_y = cp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=cp.float32)
                
                from cupyx.scipy import ndimage
                grad_x = ndimage.convolve(gpu_image.astype(cp.float32), sobel_x)
                grad_y = ndimage.convolve(gpu_image.astype(cp.float32), sobel_y)
                
                magnitude = cp.sqrt(grad_x**2 + grad_y**2)
                result = cp.clip(magnitude, 0, 255).astype(cp.uint8)
                result = cp.asnumpy(result)
            
            else:
                # CPU SIMD implementation
                result = simd_edge_detection(gray_image)
            
            self._update_stats(start_time, image.shape[0] * image.shape[1])
            return result
            
        except Exception as e:
            logger.error(f"SIMD edge detection failed: {e}")
            # Fallback to OpenCV
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            return cv2.Canny(gray, 50, 150)
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Perform histogram equalization using SIMD optimization."""
        start_time = time.time()
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = self.rgb_to_grayscale(image)
            else:
                gray_image = image
            
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU implementation
                gpu_image = cp.asarray(gray_image)
                
                # Calculate histogram
                hist = cp.histogram(gpu_image, bins=256, range=(0, 256))[0]
                
                # Calculate CDF
                cdf = cp.cumsum(hist)
                cdf_normalized = (cdf * 255 / cdf[-1]).astype(cp.uint8)
                
                # Apply equalization
                result = cdf_normalized[gpu_image]
                result = cp.asnumpy(result)
            
            else:
                # CPU SIMD implementation
                result = simd_histogram_equalization(gray_image)
            
            self._update_stats(start_time, image.shape[0] * image.shape[1])
            return result
            
        except Exception as e:
            logger.error(f"SIMD histogram equalization failed: {e}")
            # Fallback to OpenCV
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            return cv2.equalizeHist(gray)
    
    def resize_image(self, image: np.ndarray, new_size: Tuple[int, int], 
                    interpolation: str = 'bilinear') -> np.ndarray:
        """Resize image using SIMD optimization."""
        start_time = time.time()
        new_height, new_width = new_size
        
        try:
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU implementation
                gpu_image = cp.asarray(image)
                
                # Use CuPy's optimized zoom function
                from cupyx.scipy import ndimage
                zoom_factors = (new_height / image.shape[0], new_width / image.shape[1])
                if len(image.shape) == 3:
                    zoom_factors = zoom_factors + (1,)
                
                result = ndimage.zoom(gpu_image, zoom_factors, order=1)
                result = cp.asnumpy(result).astype(np.uint8)
            
            else:
                # CPU SIMD implementation
                if interpolation == 'bilinear':
                    result = simd_resize_bilinear(image, new_height, new_width)
                else:
                    # Fallback to OpenCV for other interpolations
                    cv2_interpolation = {
                        'nearest': cv2.INTER_NEAREST,
                        'bilinear': cv2.INTER_LINEAR,
                        'bicubic': cv2.INTER_CUBIC,
                        'lanczos': cv2.INTER_LANCZOS4
                    }.get(interpolation, cv2.INTER_LINEAR)
                    
                    result = cv2.resize(image, (new_width, new_height), 
                                      interpolation=cv2_interpolation)
            
            self._update_stats(start_time, new_height * new_width)
            return result
            
        except Exception as e:
            logger.error(f"SIMD resize failed: {e}")
            # Fallback to OpenCV
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def blend_images(self, image1: np.ndarray, image2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend two images using SIMD optimization."""
        start_time = time.time()
        
        try:
            if image1.shape != image2.shape:
                raise ValueError("Images must have the same shape")
            
            if self.use_gpu and CUPY_AVAILABLE:
                # GPU implementation
                gpu_img1 = cp.asarray(image1)
                gpu_img2 = cp.asarray(image2)
                
                result = (alpha * gpu_img1 + (1 - alpha) * gpu_img2).astype(cp.uint8)
                result = cp.asnumpy(result)
            
            else:
                # CPU SIMD implementation using vectorized operations
                result = (alpha * image1.astype(np.float32) + 
                         (1 - alpha) * image2.astype(np.float32)).astype(np.uint8)
            
            self._update_stats(start_time, image1.shape[0] * image1.shape[1])
            return result
            
        except Exception as e:
            logger.error(f"SIMD blend failed: {e}")
            # Fallback to OpenCV
            return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
    
    def _update_stats(self, start_time: float, pixels_processed: int):
        """Update performance statistics."""
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        self.stats["operations_count"] += 1
        self.stats["total_processing_time_ms"] += processing_time
        self.stats["pixels_processed"] += pixels_processed
        
        # Update averages
        self.stats["avg_processing_time_ms"] = (
            self.stats["total_processing_time_ms"] / self.stats["operations_count"]
        )
        
        # Calculate throughput (megapixels per second)
        if self.stats["total_processing_time_ms"] > 0:
            self.stats["throughput_mpixels_per_sec"] = (
                self.stats["pixels_processed"] / 
                (self.stats["total_processing_time_ms"] / 1000) / 1_000_000
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processor performance statistics."""
        return {
            **self.stats,
            "implementation": self.implementation,
            "capabilities": self.capabilities.capabilities,
            "optimal_chunk_size": self.capabilities.optimal_chunk_size,
            "gpu_enabled": self.use_gpu,
            "target_performance": {
                "target_throughput_mpixels_per_sec": 100,  # Target: 100 MP/s
                "target_avg_processing_time_ms": 10,
                "performance_score": min(
                    self.stats["throughput_mpixels_per_sec"] / 100,
                    10 / max(self.stats["avg_processing_time_ms"], 1)
                )
            }
        }
    
    async def benchmark_operations(self, test_image: np.ndarray, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark SIMD operations."""
        operations = {
            "rgb_to_grayscale": lambda img: self.rgb_to_grayscale(img),
            "gaussian_blur": lambda img: self.gaussian_blur(img),
            "edge_detection": lambda img: self.edge_detection(img),
            "histogram_equalization": lambda img: self.histogram_equalization(img),
            "resize": lambda img: self.resize_image(img, (img.shape[0]//2, img.shape[1]//2)),
        }
        
        results = {}
        
        for op_name, op_func in operations.items():
            times = []
            
            for _ in range(iterations):
                start_time = time.time()
                try:
                    _ = op_func(test_image)
                    processing_time = (time.time() - start_time) * 1000
                    times.append(processing_time)
                except Exception as e:
                    logger.error(f"Benchmark {op_name} failed: {e}")
                    times.append(float('inf'))
            
            results[op_name] = {
                "avg_time_ms": np.mean(times),
                "min_time_ms": np.min(times),
                "max_time_ms": np.max(times),
                "std_time_ms": np.std(times),
                "throughput_mpixels_per_sec": (
                    test_image.shape[0] * test_image.shape[1] / 
                    (np.mean(times) / 1000) / 1_000_000
                ) if np.mean(times) > 0 else 0
            }
        
        return {
            "benchmark_results": results,
            "test_image_size": test_image.shape,
            "iterations": iterations,
            "implementation": self.implementation,
            "overall_performance": {
                "avg_throughput_mpixels_per_sec": np.mean([
                    r["throughput_mpixels_per_sec"] for r in results.values()
                ]),
                "target_achieved": all(
                    r["avg_time_ms"] < 50 for r in results.values()
                )
            }
        }


# Factory function for easy integration
def create_simd_processor(use_gpu: bool = True) -> SIMDProcessor:
    """Create SIMD processor instance."""
    return SIMDProcessor(use_gpu)