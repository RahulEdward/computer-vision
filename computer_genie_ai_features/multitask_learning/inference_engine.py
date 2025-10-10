"""
Inference Engine for Multi-Task Computer Genie
मल्टी-टास्क इन्फरेंस इंजन - Computer Genie के लिए

High-performance inference engine for real-time multi-task predictions
with optimization, caching, and batch processing capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import time
import logging
from pathlib import Path
import json
from collections import deque, defaultdict
import threading
import queue
import cv2
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .multi_task_model import MultiTaskModel, MultiTaskConfig


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""
    # Model settings
    model_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # Optimization settings
    enable_tensorrt: bool = False
    enable_onnx: bool = False
    optimize_for_mobile: bool = False
    
    # Batch processing
    max_batch_size: int = 8
    batch_timeout_ms: int = 50  # Maximum wait time for batching
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 300  # 5 minutes
    
    # Performance monitoring
    enable_profiling: bool = False
    log_inference_time: bool = True
    
    # Post-processing
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    
    # Multi-threading
    num_worker_threads: int = 4
    async_inference: bool = True


class CacheEntry:
    """Cache entry with TTL support"""
    
    def __init__(self, data: Any, ttl_seconds: int):
        self.data = data
        self.timestamp = time.time()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > self.ttl_seconds


class InferenceCache:
    """LRU cache with TTL for inference results"""
    
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.Lock()
    
    def _generate_key(self, image: np.ndarray, tasks: List[str]) -> str:
        """Generate cache key from image and tasks"""
        # Use image hash and tasks as key
        image_hash = hash(image.tobytes())
        tasks_str = "_".join(sorted(tasks))
        return f"{image_hash}_{tasks_str}"
    
    def get(self, image: np.ndarray, tasks: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        key = self._generate_key(image, tasks)
        
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[key]
                    self.access_order.remove(key)
                    return None
                
                # Update access order
                self.access_order.remove(key)
                self.access_order.append(key)
                
                return entry.data
        
        return None
    
    def put(self, image: np.ndarray, tasks: List[str], result: Dict[str, Any]) -> None:
        """Cache result"""
        key = self._generate_key(image, tasks)
        
        with self.lock:
            # Remove oldest entries if cache is full
            while len(self.cache) >= self.max_size:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
            
            # Add new entry
            self.cache[key] = CacheEntry(result, self.ttl_seconds)
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


class BatchProcessor:
    """Batch processor for efficient inference"""
    
    def __init__(self, model: MultiTaskModel, config: InferenceConfig):
        self.model = model
        self.config = config
        self.batch_queue = queue.Queue()
        self.result_futures = {}
        self.processing = False
        self.lock = threading.Lock()
    
    def add_request(self, request_id: str, image: torch.Tensor, 
                   tasks: List[str]) -> 'BatchFuture':
        """Add inference request to batch"""
        future = BatchFuture(request_id)
        
        with self.lock:
            self.batch_queue.put({
                'id': request_id,
                'image': image,
                'tasks': tasks,
                'future': future
            })
            
            if not self.processing:
                self.processing = True
                threading.Thread(target=self._process_batch, daemon=True).start()
        
        return future
    
    def _process_batch(self) -> None:
        """Process batched requests"""
        batch_requests = []
        start_time = time.time()
        
        # Collect requests for batch
        while len(batch_requests) < self.config.max_batch_size:
            try:
                # Wait for requests with timeout
                timeout = max(0, self.config.batch_timeout_ms / 1000 - (time.time() - start_time))
                request = self.batch_queue.get(timeout=timeout)
                batch_requests.append(request)
            except queue.Empty:
                break
        
        if not batch_requests:
            with self.lock:
                self.processing = False
            return
        
        try:
            # Prepare batch
            batch_images = torch.stack([req['image'] for req in batch_requests])
            
            # Run inference
            with torch.no_grad():
                batch_results = self.model(batch_images)
            
            # Distribute results
            for i, request in enumerate(batch_requests):
                # Extract result for this request
                result = self._extract_single_result(batch_results, i)
                request['future'].set_result(result)
        
        except Exception as e:
            # Set error for all requests
            for request in batch_requests:
                request['future'].set_exception(e)
        
        # Continue processing if more requests are waiting
        with self.lock:
            if not self.batch_queue.empty():
                threading.Thread(target=self._process_batch, daemon=True).start()
            else:
                self.processing = False
    
    def _extract_single_result(self, batch_results: Dict[str, Any], 
                              index: int) -> Dict[str, Any]:
        """Extract single result from batch results"""
        single_result = {}
        
        # Extract shared features
        if "shared_features" in batch_results:
            single_result["shared_features"] = batch_results["shared_features"][index]
        
        # Extract task outputs
        if "task_outputs" in batch_results:
            single_result["task_outputs"] = {}
            for task_name, task_output in batch_results["task_outputs"].items():
                single_result["task_outputs"][task_name] = {}
                for key, value in task_output.items():
                    if isinstance(value, torch.Tensor):
                        single_result["task_outputs"][task_name][key] = value[index]
                    else:
                        single_result["task_outputs"][task_name][key] = value
        
        return single_result


class BatchFuture:
    """Future object for batch processing results"""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self._result = None
        self._exception = None
        self._done = False
        self._condition = threading.Condition()
    
    def set_result(self, result: Any) -> None:
        """Set result"""
        with self._condition:
            self._result = result
            self._done = True
            self._condition.notify_all()
    
    def set_exception(self, exception: Exception) -> None:
        """Set exception"""
        with self._condition:
            self._exception = exception
            self._done = True
            self._condition.notify_all()
    
    def result(self, timeout: Optional[float] = None) -> Any:
        """Get result"""
        with self._condition:
            if not self._done:
                self._condition.wait(timeout)
            
            if self._exception:
                raise self._exception
            
            return self._result
    
    def done(self) -> bool:
        """Check if done"""
        return self._done


class PostProcessor:
    """Post-processing for inference results"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
    
    def process_element_detection(self, predictions: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Post-process element detection results"""
        class_logits = predictions["class_logits"]
        bboxes = predictions["bboxes"]
        confidence = predictions["confidence"]
        
        # Apply confidence threshold
        conf_mask = confidence > self.config.confidence_threshold
        
        if not conf_mask.any():
            return []
        
        # Filter predictions
        filtered_logits = class_logits[conf_mask]
        filtered_bboxes = bboxes[conf_mask]
        filtered_confidence = confidence[conf_mask]
        
        # Get class predictions
        class_preds = torch.argmax(filtered_logits, dim=1)
        
        # Apply NMS
        keep_indices = self._apply_nms(filtered_bboxes, filtered_confidence)
        
        # Format results
        results = []
        for idx in keep_indices[:self.config.max_detections]:
            results.append({
                "class_id": class_preds[idx].item(),
                "bbox": filtered_bboxes[idx].cpu().numpy().tolist(),
                "confidence": filtered_confidence[idx].item()
            })
        
        return results
    
    def process_ocr(self, predictions: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Post-process OCR results"""
        char_logits = predictions["char_logits"]
        text_regions = predictions["text_regions"]
        language_logits = predictions["language_logits"]
        
        # Decode character sequences
        char_preds = torch.argmax(char_logits, dim=-1)
        
        # Get language predictions
        language_preds = torch.argmax(language_logits, dim=1)
        
        results = []
        for i in range(len(char_preds)):
            # Decode text (simplified - would need proper tokenizer)
            text = self._decode_text(char_preds[i])
            
            if text.strip():  # Only include non-empty text
                results.append({
                    "text": text,
                    "bbox": text_regions[i].cpu().numpy().tolist(),
                    "language_id": language_preds[i].item(),
                    "confidence": torch.softmax(char_logits[i], dim=-1).max(dim=-1)[0].mean().item()
                })
        
        return results
    
    def process_intent_classification(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Post-process intent classification results"""
        intent_logits = predictions["intent_logits"]
        urgency = predictions["urgency"]
        
        # Get intent prediction
        intent_probs = torch.softmax(intent_logits, dim=1)
        intent_pred = torch.argmax(intent_probs, dim=1)
        intent_confidence = intent_probs.max(dim=1)[0]
        
        return {
            "intent_id": intent_pred.item(),
            "intent_confidence": intent_confidence.item(),
            "urgency": urgency.item(),
            "intent_probabilities": intent_probs.cpu().numpy().tolist()
        }
    
    def process_action_prediction(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Post-process action prediction results"""
        action_logits = predictions["action_logits"]
        action_params = predictions["action_params"]
        success_probability = predictions["success_probability"]
        
        # Get action prediction
        action_probs = torch.softmax(action_logits, dim=1)
        action_pred = torch.argmax(action_probs, dim=1)
        action_confidence = action_probs.max(dim=1)[0]
        
        return {
            "action_id": action_pred.item(),
            "action_confidence": action_confidence.item(),
            "action_parameters": action_params.cpu().numpy().tolist(),
            "success_probability": success_probability.item(),
            "action_probabilities": action_probs.cpu().numpy().tolist()
        }
    
    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor) -> List[int]:
        """Apply Non-Maximum Suppression"""
        # Convert to (x1, y1, x2, y2) format
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i.item())
            
            if len(order) == 1:
                break
            
            # Compute IoU
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = torch.where(iou <= self.config.nms_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def _decode_text(self, char_sequence: torch.Tensor) -> str:
        """Decode character sequence to text (simplified)"""
        # This is a simplified version - in practice, you'd use a proper tokenizer
        # that maps token IDs back to characters/words
        
        # Remove padding tokens (assuming -1 or 0 is padding)
        valid_chars = char_sequence[char_sequence > 0]
        
        # Convert to string (this is a placeholder - implement proper decoding)
        text = "".join([chr(min(max(c.item(), 32), 126)) for c in valid_chars[:50]])
        
        return text.strip()


class PerformanceMonitor:
    """Monitor inference performance"""
    
    def __init__(self):
        self.inference_times = deque(maxlen=1000)
        self.task_times = defaultdict(lambda: deque(maxlen=1000))
        self.throughput_history = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def record_inference_time(self, total_time: float, 
                             task_times: Dict[str, float] = None) -> None:
        """Record inference timing"""
        with self.lock:
            self.inference_times.append(total_time)
            
            if task_times:
                for task_name, task_time in task_times.items():
                    self.task_times[task_name].append(task_time)
    
    def record_throughput(self, requests_per_second: float) -> None:
        """Record throughput"""
        with self.lock:
            self.throughput_history.append(requests_per_second)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self.lock:
            stats = {}
            
            if self.inference_times:
                stats["inference_time"] = {
                    "mean": np.mean(self.inference_times),
                    "median": np.median(self.inference_times),
                    "p95": np.percentile(self.inference_times, 95),
                    "p99": np.percentile(self.inference_times, 99)
                }
            
            if self.throughput_history:
                stats["throughput"] = {
                    "mean_rps": np.mean(self.throughput_history),
                    "max_rps": np.max(self.throughput_history)
                }
            
            # Task-specific times
            stats["task_times"] = {}
            for task_name, times in self.task_times.items():
                if times:
                    stats["task_times"][task_name] = {
                        "mean": np.mean(times),
                        "median": np.median(times)
                    }
            
            return stats


class MultiTaskInferenceEngine:
    """Main inference engine for multi-task model"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize components
        self.cache = InferenceCache(config.cache_size, config.cache_ttl_seconds) if config.enable_caching else None
        self.batch_processor = BatchProcessor(self.model, config) if config.max_batch_size > 1 else None
        self.post_processor = PostProcessor(config)
        self.performance_monitor = PerformanceMonitor() if config.enable_profiling else None
        
        # Thread pool for async inference
        self.executor = ThreadPoolExecutor(max_workers=config.num_worker_threads) if config.async_inference else None
        
        self.logger.info(f"Inference engine initialized on {self.device}")
    
    def _load_model(self) -> MultiTaskModel:
        """Load model from checkpoint"""
        if not self.config.model_path or not Path(self.config.model_path).exists():
            # Create default model for demo
            model_config = MultiTaskConfig()
            model = MultiTaskModel(model_config)
            self.logger.warning("Using default model - no checkpoint loaded")
            return model.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        
        # Extract model config from checkpoint
        if "config" in checkpoint and hasattr(checkpoint["config"], "model_config"):
            model_config = checkpoint["config"].model_config
        else:
            model_config = MultiTaskConfig()
        
        # Create and load model
        model = MultiTaskModel(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Optimize model
        model = self._optimize_model(model)
        
        return model.to(self.device)
    
    def _optimize_model(self, model: MultiTaskModel) -> MultiTaskModel:
        """Optimize model for inference"""
        # TorchScript compilation
        try:
            model = torch.jit.script(model)
            self.logger.info("Model compiled with TorchScript")
        except Exception as e:
            self.logger.warning(f"TorchScript compilation failed: {e}")
        
        # Additional optimizations can be added here
        # - TensorRT optimization
        # - ONNX conversion
        # - Mobile optimization
        
        return model
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        """Preprocess input image"""
        # Convert to numpy array
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize and normalize
        image = cv2.resize(image, (224, 224))  # Standard input size
        image = image.astype(np.float32) / 255.0
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image: Union[np.ndarray, Image.Image, str],
                tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run inference on single image"""
        if tasks is None:
            tasks = ["element_detection", "ocr", "intent_classification", "action_prediction"]
        
        # Check cache
        if self.cache:
            if isinstance(image, str):
                image_array = cv2.imread(image)
            elif isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            cached_result = self.cache.get(image_array, tasks)
            if cached_result is not None:
                return cached_result
        
        # Preprocess
        start_time = time.time()
        image_tensor = self.preprocess_image(image)
        preprocess_time = time.time() - start_time
        
        # Run inference
        inference_start = time.time()
        
        if self.batch_processor and self.config.max_batch_size > 1:
            # Use batch processing
            request_id = f"req_{int(time.time() * 1000000)}"
            future = self.batch_processor.add_request(request_id, image_tensor.squeeze(0), tasks)
            raw_output = future.result(timeout=5.0)  # 5 second timeout
        else:
            # Direct inference
            with torch.no_grad():
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        raw_output = self.model(image_tensor)
                else:
                    raw_output = self.model(image_tensor)
        
        inference_time = time.time() - inference_start
        
        # Post-process
        postprocess_start = time.time()
        result = self._postprocess_output(raw_output, tasks)
        postprocess_time = time.time() - postprocess_start
        
        total_time = time.time() - start_time
        
        # Add timing information
        result["timing"] = {
            "preprocess_time": preprocess_time,
            "inference_time": inference_time,
            "postprocess_time": postprocess_time,
            "total_time": total_time
        }
        
        # Record performance
        if self.performance_monitor:
            task_times = {
                "preprocess": preprocess_time,
                "inference": inference_time,
                "postprocess": postprocess_time
            }
            self.performance_monitor.record_inference_time(total_time, task_times)
        
        # Cache result
        if self.cache and isinstance(image, np.ndarray):
            self.cache.put(image, tasks, result)
        
        if self.config.log_inference_time:
            self.logger.info(f"Inference completed in {total_time:.3f}s")
        
        return result
    
    def _postprocess_output(self, raw_output: Dict[str, Any], 
                           tasks: List[str]) -> Dict[str, Any]:
        """Post-process model output"""
        result = {}
        task_outputs = raw_output.get("task_outputs", {})
        
        # Process each requested task
        if "element_detection" in tasks and "element_detection" in task_outputs:
            result["element_detection"] = self.post_processor.process_element_detection(
                task_outputs["element_detection"]
            )
        
        if "ocr" in tasks and "ocr" in task_outputs:
            result["ocr"] = self.post_processor.process_ocr(
                task_outputs["ocr"]
            )
        
        if "intent_classification" in tasks and "intent_classification" in task_outputs:
            result["intent_classification"] = self.post_processor.process_intent_classification(
                task_outputs["intent_classification"]
            )
        
        if "action_prediction" in tasks and "action_prediction" in task_outputs:
            result["action_prediction"] = self.post_processor.process_action_prediction(
                task_outputs["action_prediction"]
            )
        
        return result
    
    async def predict_async(self, image: Union[np.ndarray, Image.Image, str],
                           tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Async inference"""
        if not self.executor:
            raise RuntimeError("Async inference not enabled")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.predict, image, tasks)
    
    def predict_batch(self, images: List[Union[np.ndarray, Image.Image, str]],
                     tasks: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Batch inference"""
        results = []
        
        if self.config.async_inference and self.executor:
            # Parallel processing
            futures = []
            for image in images:
                future = self.executor.submit(self.predict, image, tasks)
                futures.append(future)
            
            for future in futures:
                results.append(future.result())
        else:
            # Sequential processing
            for image in images:
                results.append(self.predict(image, tasks))
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_monitor:
            return {}
        
        return self.performance_monitor.get_stats()
    
    def clear_cache(self) -> None:
        """Clear inference cache"""
        if self.cache:
            self.cache.clear()
    
    def shutdown(self) -> None:
        """Shutdown inference engine"""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.logger.info("Inference engine shutdown complete")


# Example usage
if __name__ == "__main__":
    # Create inference configuration
    config = InferenceConfig(
        model_path="",  # Will use default model
        device="cpu",
        max_batch_size=4,
        enable_caching=True,
        async_inference=True
    )
    
    # Create inference engine
    engine = MultiTaskInferenceEngine(config)
    
    # Create dummy image for testing
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run inference
    result = engine.predict(dummy_image)
    
    print("Inference completed!")
    print(f"Tasks processed: {list(result.keys())}")
    print(f"Timing: {result.get('timing', {})}")
    
    # Get performance stats
    stats = engine.get_performance_stats()
    if stats:
        print(f"Performance stats: {stats}")
    
    # Shutdown
    engine.shutdown()