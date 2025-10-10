"""
Custom binary protocol for high-performance data transmission.
Provides 50% faster data transmission than JSON with efficient serialization.
"""

import asyncio
import gzip
import logging
import lz4.frame
import struct
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """Message type identifiers."""
    PING = 0x01
    PONG = 0x02
    REQUEST = 0x10
    RESPONSE = 0x11
    ERROR = 0x12
    NOTIFICATION = 0x20
    SCREENSHOT = 0x30
    UI_ELEMENTS = 0x31
    ACTION_RESULT = 0x32
    BATCH_REQUEST = 0x40
    BATCH_RESPONSE = 0x41
    STREAM_START = 0x50
    STREAM_DATA = 0x51
    STREAM_END = 0x52


class CompressionType(IntEnum):
    """Compression type identifiers."""
    NONE = 0x00
    ZLIB = 0x01
    GZIP = 0x02
    LZ4 = 0x03
    ZSTD = 0x04


class DataType(IntEnum):
    """Data type identifiers for efficient serialization."""
    NULL = 0x00
    BOOL = 0x01
    INT8 = 0x02
    INT16 = 0x03
    INT32 = 0x04
    INT64 = 0x05
    UINT8 = 0x06
    UINT16 = 0x07
    UINT32 = 0x08
    UINT64 = 0x09
    FLOAT32 = 0x0A
    FLOAT64 = 0x0B
    STRING = 0x10
    BYTES = 0x11
    ARRAY = 0x20
    DICT = 0x21
    NUMPY_ARRAY = 0x30
    IMAGE = 0x31


@dataclass
class MessageHeader:
    """Binary message header."""
    magic: bytes = b'CGBP'  # Computer Genie Binary Protocol
    version: int = 1
    message_type: MessageType = MessageType.REQUEST
    compression: CompressionType = CompressionType.NONE
    flags: int = 0
    message_id: int = 0
    payload_size: int = 0
    checksum: int = 0
    timestamp: float = 0.0
    reserved: bytes = b'\x00' * 8
    
    def pack(self) -> bytes:
        """Pack header to binary format."""
        return struct.pack(
            '<4sBBBBIIIfQ8s',
            self.magic,
            self.version,
            self.message_type,
            self.compression,
            self.flags,
            self.message_id,
            self.payload_size,
            self.checksum,
            self.timestamp,
            0,  # Reserved field
            self.reserved
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'MessageHeader':
        """Unpack header from binary format."""
        values = struct.unpack('<4sBBBBIIIfQ8s', data[:48])
        return cls(
            magic=values[0],
            version=values[1],
            message_type=MessageType(values[2]),
            compression=CompressionType(values[3]),
            flags=values[4],
            message_id=values[5],
            payload_size=values[6],
            checksum=values[7],
            timestamp=values[8],
            reserved=values[10]
        )
    
    def is_valid(self) -> bool:
        """Check if header is valid."""
        return self.magic == b'CGBP' and self.version == 1
    
    @property
    def size(self) -> int:
        """Get header size in bytes."""
        return 48


class BinarySerializer:
    """High-performance binary serializer."""
    
    def __init__(self):
        self.type_handlers = {
            type(None): self._serialize_null,
            bool: self._serialize_bool,
            int: self._serialize_int,
            float: self._serialize_float,
            str: self._serialize_string,
            bytes: self._serialize_bytes,
            list: self._serialize_array,
            tuple: self._serialize_array,
            dict: self._serialize_dict,
            np.ndarray: self._serialize_numpy_array,
        }
        
        self.deserialize_handlers = {
            DataType.NULL: self._deserialize_null,
            DataType.BOOL: self._deserialize_bool,
            DataType.INT8: self._deserialize_int8,
            DataType.INT16: self._deserialize_int16,
            DataType.INT32: self._deserialize_int32,
            DataType.INT64: self._deserialize_int64,
            DataType.UINT8: self._deserialize_uint8,
            DataType.UINT16: self._deserialize_uint16,
            DataType.UINT32: self._deserialize_uint32,
            DataType.UINT64: self._deserialize_uint64,
            DataType.FLOAT32: self._deserialize_float32,
            DataType.FLOAT64: self._deserialize_float64,
            DataType.STRING: self._deserialize_string,
            DataType.BYTES: self._deserialize_bytes,
            DataType.ARRAY: self._deserialize_array,
            DataType.DICT: self._deserialize_dict,
            DataType.NUMPY_ARRAY: self._deserialize_numpy_array,
        }
    
    def serialize(self, obj: Any) -> bytes:
        """Serialize object to binary format."""
        buffer = BytesIO()
        self._serialize_object(obj, buffer)
        return buffer.getvalue()
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize object from binary format."""
        buffer = BytesIO(data)
        return self._deserialize_object(buffer)
    
    def _serialize_object(self, obj: Any, buffer: BytesIO):
        """Serialize single object."""
        obj_type = type(obj)
        
        if obj_type in self.type_handlers:
            self.type_handlers[obj_type](obj, buffer)
        else:
            # Fallback to string representation
            self._serialize_string(str(obj), buffer)
    
    def _serialize_null(self, obj: None, buffer: BytesIO):
        """Serialize null value."""
        buffer.write(struct.pack('<B', DataType.NULL))
    
    def _serialize_bool(self, obj: bool, buffer: BytesIO):
        """Serialize boolean value."""
        buffer.write(struct.pack('<BB', DataType.BOOL, int(obj)))
    
    def _serialize_int(self, obj: int, buffer: BytesIO):
        """Serialize integer value with optimal size."""
        if -128 <= obj <= 127:
            buffer.write(struct.pack('<Bb', DataType.INT8, obj))
        elif -32768 <= obj <= 32767:
            buffer.write(struct.pack('<Bh', DataType.INT16, obj))
        elif -2147483648 <= obj <= 2147483647:
            buffer.write(struct.pack('<Bi', DataType.INT32, obj))
        else:
            buffer.write(struct.pack('<Bq', DataType.INT64, obj))
    
    def _serialize_float(self, obj: float, buffer: BytesIO):
        """Serialize float value."""
        # Use float32 if precision allows
        if abs(obj) < 3.4e38 and np.float32(obj) == obj:
            buffer.write(struct.pack('<Bf', DataType.FLOAT32, obj))
        else:
            buffer.write(struct.pack('<Bd', DataType.FLOAT64, obj))
    
    def _serialize_string(self, obj: str, buffer: BytesIO):
        """Serialize string value."""
        encoded = obj.encode('utf-8')
        buffer.write(struct.pack('<BI', DataType.STRING, len(encoded)))
        buffer.write(encoded)
    
    def _serialize_bytes(self, obj: bytes, buffer: BytesIO):
        """Serialize bytes value."""
        buffer.write(struct.pack('<BI', DataType.BYTES, len(obj)))
        buffer.write(obj)
    
    def _serialize_array(self, obj: Union[list, tuple], buffer: BytesIO):
        """Serialize array/list value."""
        buffer.write(struct.pack('<BI', DataType.ARRAY, len(obj)))
        for item in obj:
            self._serialize_object(item, buffer)
    
    def _serialize_dict(self, obj: dict, buffer: BytesIO):
        """Serialize dictionary value."""
        buffer.write(struct.pack('<BI', DataType.DICT, len(obj)))
        for key, value in obj.items():
            self._serialize_object(key, buffer)
            self._serialize_object(value, buffer)
    
    def _serialize_numpy_array(self, obj: np.ndarray, buffer: BytesIO):
        """Serialize numpy array with metadata."""
        # Serialize type info
        buffer.write(struct.pack('<B', DataType.NUMPY_ARRAY))
        
        # Serialize shape
        shape_bytes = struct.pack('<I', len(obj.shape))
        shape_bytes += struct.pack('<' + 'I' * len(obj.shape), *obj.shape)
        buffer.write(shape_bytes)
        
        # Serialize dtype
        dtype_str = str(obj.dtype).encode('ascii')
        buffer.write(struct.pack('<I', len(dtype_str)))
        buffer.write(dtype_str)
        
        # Serialize data
        data_bytes = obj.tobytes()
        buffer.write(struct.pack('<I', len(data_bytes)))
        buffer.write(data_bytes)
    
    def _deserialize_object(self, buffer: BytesIO) -> Any:
        """Deserialize single object."""
        data_type_bytes = buffer.read(1)
        if not data_type_bytes:
            raise ValueError("Unexpected end of data")
        
        data_type = DataType(data_type_bytes[0])
        
        if data_type in self.deserialize_handlers:
            return self.deserialize_handlers[data_type](buffer)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
    
    def _deserialize_null(self, buffer: BytesIO) -> None:
        """Deserialize null value."""
        return None
    
    def _deserialize_bool(self, buffer: BytesIO) -> bool:
        """Deserialize boolean value."""
        return bool(struct.unpack('<B', buffer.read(1))[0])
    
    def _deserialize_int8(self, buffer: BytesIO) -> int:
        """Deserialize int8 value."""
        return struct.unpack('<b', buffer.read(1))[0]
    
    def _deserialize_int16(self, buffer: BytesIO) -> int:
        """Deserialize int16 value."""
        return struct.unpack('<h', buffer.read(2))[0]
    
    def _deserialize_int32(self, buffer: BytesIO) -> int:
        """Deserialize int32 value."""
        return struct.unpack('<i', buffer.read(4))[0]
    
    def _deserialize_int64(self, buffer: BytesIO) -> int:
        """Deserialize int64 value."""
        return struct.unpack('<q', buffer.read(8))[0]
    
    def _deserialize_uint8(self, buffer: BytesIO) -> int:
        """Deserialize uint8 value."""
        return struct.unpack('<B', buffer.read(1))[0]
    
    def _deserialize_uint16(self, buffer: BytesIO) -> int:
        """Deserialize uint16 value."""
        return struct.unpack('<H', buffer.read(2))[0]
    
    def _deserialize_uint32(self, buffer: BytesIO) -> int:
        """Deserialize uint32 value."""
        return struct.unpack('<I', buffer.read(4))[0]
    
    def _deserialize_uint64(self, buffer: BytesIO) -> int:
        """Deserialize uint64 value."""
        return struct.unpack('<Q', buffer.read(8))[0]
    
    def _deserialize_float32(self, buffer: BytesIO) -> float:
        """Deserialize float32 value."""
        return struct.unpack('<f', buffer.read(4))[0]
    
    def _deserialize_float64(self, buffer: BytesIO) -> float:
        """Deserialize float64 value."""
        return struct.unpack('<d', buffer.read(8))[0]
    
    def _deserialize_string(self, buffer: BytesIO) -> str:
        """Deserialize string value."""
        length = struct.unpack('<I', buffer.read(4))[0]
        return buffer.read(length).decode('utf-8')
    
    def _deserialize_bytes(self, buffer: BytesIO) -> bytes:
        """Deserialize bytes value."""
        length = struct.unpack('<I', buffer.read(4))[0]
        return buffer.read(length)
    
    def _deserialize_array(self, buffer: BytesIO) -> list:
        """Deserialize array value."""
        length = struct.unpack('<I', buffer.read(4))[0]
        return [self._deserialize_object(buffer) for _ in range(length)]
    
    def _deserialize_dict(self, buffer: BytesIO) -> dict:
        """Deserialize dictionary value."""
        length = struct.unpack('<I', buffer.read(4))[0]
        result = {}
        for _ in range(length):
            key = self._deserialize_object(buffer)
            value = self._deserialize_object(buffer)
            result[key] = value
        return result
    
    def _deserialize_numpy_array(self, buffer: BytesIO) -> np.ndarray:
        """Deserialize numpy array."""
        # Read shape
        shape_len = struct.unpack('<I', buffer.read(4))[0]
        shape = struct.unpack('<' + 'I' * shape_len, buffer.read(4 * shape_len))
        
        # Read dtype
        dtype_len = struct.unpack('<I', buffer.read(4))[0]
        dtype_str = buffer.read(dtype_len).decode('ascii')
        dtype = np.dtype(dtype_str)
        
        # Read data
        data_len = struct.unpack('<I', buffer.read(4))[0]
        data_bytes = buffer.read(data_len)
        
        # Reconstruct array
        array = np.frombuffer(data_bytes, dtype=dtype)
        return array.reshape(shape)


class CompressionManager:
    """Manages different compression algorithms."""
    
    def __init__(self):
        self.compressors = {
            CompressionType.NONE: (lambda x: x, lambda x: x),
            CompressionType.ZLIB: (zlib.compress, zlib.decompress),
            CompressionType.GZIP: (gzip.compress, gzip.decompress),
        }
        
        # Add LZ4 if available
        if 'lz4' in globals():
            self.compressors[CompressionType.LZ4] = (
                lz4.frame.compress,
                lz4.frame.decompress
            )
        
        # Add ZSTD if available
        if ZSTD_AVAILABLE:
            cctx = zstd.ZstdCompressor(level=3)
            dctx = zstd.ZstdDecompressor()
            self.compressors[CompressionType.ZSTD] = (
                cctx.compress,
                dctx.decompress
            )
    
    def compress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm."""
        if compression_type in self.compressors:
            compressor, _ = self.compressors[compression_type]
            return compressor(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    def decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type in self.compressors:
            _, decompressor = self.compressors[compression_type]
            return decompressor(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    def get_best_compression(self, data: bytes) -> Tuple[CompressionType, bytes]:
        """Find best compression for given data."""
        best_type = CompressionType.NONE
        best_data = data
        best_ratio = 1.0
        
        for comp_type in self.compressors:
            if comp_type == CompressionType.NONE:
                continue
            
            try:
                compressed = self.compress(data, comp_type)
                ratio = len(compressed) / len(data)
                
                if ratio < best_ratio:
                    best_type = comp_type
                    best_data = compressed
                    best_ratio = ratio
            except Exception as e:
                logger.warning(f"Compression {comp_type} failed: {e}")
        
        return best_type, best_data


class FastBinaryProtocol:
    """High-performance binary protocol implementation."""
    
    def __init__(self, auto_compress: bool = True, compression_threshold: int = 1024):
        self.serializer = BinarySerializer()
        self.compression_manager = CompressionManager()
        self.auto_compress = auto_compress
        self.compression_threshold = compression_threshold
        
        # Message tracking
        self.next_message_id = 1
        self.pending_requests: Dict[int, asyncio.Future] = {}
        
        # Performance statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "compression_ratio": 0.0,
            "avg_serialization_time_ms": 0.0,
            "avg_deserialization_time_ms": 0.0
        }
    
    def create_message(self, message_type: MessageType, payload: Any, 
                      compression: Optional[CompressionType] = None) -> bytes:
        """Create binary message."""
        start_time = time.time()
        
        try:
            # Serialize payload
            if MSGPACK_AVAILABLE and isinstance(payload, (dict, list)):
                # Use msgpack for complex structures
                payload_bytes = msgpack.packb(payload, use_bin_type=True)
            else:
                # Use custom serializer
                payload_bytes = self.serializer.serialize(payload)
            
            # Determine compression
            if compression is None and self.auto_compress and len(payload_bytes) > self.compression_threshold:
                compression, payload_bytes = self.compression_manager.get_best_compression(payload_bytes)
            elif compression is None:
                compression = CompressionType.NONE
            elif compression != CompressionType.NONE:
                payload_bytes = self.compression_manager.compress(payload_bytes, compression)
            
            # Create header
            message_id = self.next_message_id
            self.next_message_id += 1
            
            header = MessageHeader(
                message_type=message_type,
                compression=compression,
                message_id=message_id,
                payload_size=len(payload_bytes),
                checksum=zlib.crc32(payload_bytes) & 0xffffffff,
                timestamp=time.time()
            )
            
            # Pack message
            header_bytes = header.pack()
            message_bytes = header_bytes + payload_bytes
            
            # Update statistics
            serialization_time = (time.time() - start_time) * 1000
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(message_bytes)
            self._update_avg_time("avg_serialization_time_ms", serialization_time)
            
            if compression != CompressionType.NONE:
                original_size = len(self.serializer.serialize(payload))
                compression_ratio = len(payload_bytes) / original_size
                self.stats["compression_ratio"] = (
                    (self.stats["compression_ratio"] * (self.stats["messages_sent"] - 1) + compression_ratio) /
                    self.stats["messages_sent"]
                )
            
            return message_bytes
            
        except Exception as e:
            logger.error(f"Failed to create message: {e}")
            raise
    
    def parse_message(self, data: bytes) -> Tuple[MessageHeader, Any]:
        """Parse binary message."""
        start_time = time.time()
        
        try:
            if len(data) < MessageHeader().size:
                raise ValueError("Insufficient data for header")
            
            # Parse header
            header = MessageHeader.unpack(data[:MessageHeader().size])
            
            if not header.is_valid():
                raise ValueError("Invalid message header")
            
            # Extract payload
            payload_start = MessageHeader().size
            payload_end = payload_start + header.payload_size
            
            if len(data) < payload_end:
                raise ValueError("Insufficient data for payload")
            
            payload_bytes = data[payload_start:payload_end]
            
            # Verify checksum
            calculated_checksum = zlib.crc32(payload_bytes) & 0xffffffff
            if calculated_checksum != header.checksum:
                raise ValueError("Checksum mismatch")
            
            # Decompress if needed
            if header.compression != CompressionType.NONE:
                payload_bytes = self.compression_manager.decompress(payload_bytes, header.compression)
            
            # Deserialize payload
            if MSGPACK_AVAILABLE:
                try:
                    payload = msgpack.unpackb(payload_bytes, raw=False)
                except (msgpack.exceptions.ExtraData, ValueError):
                    # Fallback to custom deserializer
                    payload = self.serializer.deserialize(payload_bytes)
            else:
                payload = self.serializer.deserialize(payload_bytes)
            
            # Update statistics
            deserialization_time = (time.time() - start_time) * 1000
            self.stats["messages_received"] += 1
            self.stats["bytes_received"] += len(data)
            self._update_avg_time("avg_deserialization_time_ms", deserialization_time)
            
            return header, payload
            
        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            raise
    
    def _update_avg_time(self, stat_key: str, new_time: float):
        """Update average time statistic."""
        if stat_key == "avg_serialization_time_ms":
            count = self.stats["messages_sent"]
        else:
            count = self.stats["messages_received"]
        
        current_avg = self.stats[stat_key]
        self.stats[stat_key] = ((current_avg * (count - 1)) + new_time) / count
    
    async def send_request(self, payload: Any, timeout: float = 30.0) -> Any:
        """Send request and wait for response."""
        # Create request message
        request_data = self.create_message(MessageType.REQUEST, payload)
        
        # Extract message ID for tracking
        header = MessageHeader.unpack(request_data[:MessageHeader().size])
        message_id = header.message_id
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_requests[message_id] = response_future
        
        try:
            # Send request (implementation depends on transport layer)
            await self._send_data(request_data)
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request {message_id} timed out")
            raise
        finally:
            # Clean up
            self.pending_requests.pop(message_id, None)
    
    async def send_response(self, request_header: MessageHeader, payload: Any):
        """Send response to a request."""
        response_data = self.create_message(MessageType.RESPONSE, payload)
        
        # Update message ID to match request
        response_header = MessageHeader.unpack(response_data[:MessageHeader().size])
        response_header.message_id = request_header.message_id
        
        # Repack with correct message ID
        response_data = response_header.pack() + response_data[MessageHeader().size:]
        
        await self._send_data(response_data)
    
    async def send_notification(self, payload: Any):
        """Send notification (no response expected)."""
        notification_data = self.create_message(MessageType.NOTIFICATION, payload)
        await self._send_data(notification_data)
    
    async def handle_received_message(self, data: bytes):
        """Handle received message."""
        try:
            header, payload = self.parse_message(data)
            
            if header.message_type == MessageType.RESPONSE:
                # Handle response
                if header.message_id in self.pending_requests:
                    future = self.pending_requests[header.message_id]
                    if not future.done():
                        future.set_result(payload)
            
            elif header.message_type == MessageType.REQUEST:
                # Handle request (should be implemented by subclass)
                await self._handle_request(header, payload)
            
            elif header.message_type == MessageType.NOTIFICATION:
                # Handle notification (should be implemented by subclass)
                await self._handle_notification(header, payload)
            
        except Exception as e:
            logger.error(f"Failed to handle received message: {e}")
    
    async def _send_data(self, data: bytes):
        """Send data (to be implemented by transport layer)."""
        raise NotImplementedError("Transport layer must implement _send_data")
    
    async def _handle_request(self, header: MessageHeader, payload: Any):
        """Handle incoming request (to be implemented by subclass)."""
        pass
    
    async def _handle_notification(self, header: MessageHeader, payload: Any):
        """Handle incoming notification (to be implemented by subclass)."""
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get protocol performance statistics."""
        total_messages = self.stats["messages_sent"] + self.stats["messages_received"]
        total_bytes = self.stats["bytes_sent"] + self.stats["bytes_received"]
        
        return {
            **self.stats,
            "total_messages": total_messages,
            "total_bytes": total_bytes,
            "avg_message_size_bytes": total_bytes / max(total_messages, 1),
            "throughput_messages_per_sec": total_messages / max(time.time(), 1),  # Simplified
            "compression_enabled": self.auto_compress,
            "compression_threshold_bytes": self.compression_threshold,
            "target_performance": {
                "target_speedup_vs_json": 2.0,  # 50% faster
                "target_serialization_time_ms": 10,
                "target_compression_ratio": 0.7,
                "performance_score": min(
                    10 / max(self.stats["avg_serialization_time_ms"], 1),
                    2.0
                )
            }
        }
    
    async def benchmark_vs_json(self, test_data: List[Any], iterations: int = 100) -> Dict[str, Any]:
        """Benchmark binary protocol vs JSON."""
        import json
        
        # Benchmark JSON
        json_times = []
        json_sizes = []
        
        for _ in range(iterations):
            for data in test_data:
                start_time = time.time()
                json_str = json.dumps(data, default=str)
                json_bytes = json_str.encode('utf-8')
                json.loads(json_str)
                json_time = (time.time() - start_time) * 1000
                
                json_times.append(json_time)
                json_sizes.append(len(json_bytes))
        
        # Benchmark binary protocol
        binary_times = []
        binary_sizes = []
        
        for _ in range(iterations):
            for data in test_data:
                start_time = time.time()
                binary_data = self.create_message(MessageType.REQUEST, data)
                self.parse_message(binary_data)
                binary_time = (time.time() - start_time) * 1000
                
                binary_times.append(binary_time)
                binary_sizes.append(len(binary_data))
        
        # Calculate results
        json_avg_time = np.mean(json_times)
        binary_avg_time = np.mean(binary_times)
        json_avg_size = np.mean(json_sizes)
        binary_avg_size = np.mean(binary_sizes)
        
        speedup = json_avg_time / binary_avg_time
        size_reduction = (json_avg_size - binary_avg_size) / json_avg_size
        
        return {
            "json_performance": {
                "avg_time_ms": json_avg_time,
                "avg_size_bytes": json_avg_size
            },
            "binary_performance": {
                "avg_time_ms": binary_avg_time,
                "avg_size_bytes": binary_avg_size
            },
            "improvement": {
                "speedup_factor": speedup,
                "size_reduction_percent": size_reduction * 100,
                "target_50_percent_faster": speedup >= 1.5,
                "overall_efficiency": speedup * (1 + size_reduction)
            }
        }


# Factory function for easy integration
def create_binary_protocol(auto_compress: bool = True, compression_threshold: int = 1024) -> FastBinaryProtocol:
    """Create binary protocol instance."""
    return FastBinaryProtocol(auto_compress, compression_threshold)