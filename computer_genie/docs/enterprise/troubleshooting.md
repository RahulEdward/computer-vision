# 🔧 Computer Genie Enterprise Troubleshooting & Support Guide

<div align="center">

![Troubleshooting](https://img.shields.io/badge/Enterprise-Troubleshooting-red?style=for-the-badge&logo=tools)

**Comprehensive Troubleshooting & Support Documentation**

</div>

---

## 📋 **Table of Contents**

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues & Solutions](#common-issues--solutions)
3. [Installation Troubleshooting](#installation-troubleshooting)
4. [API Troubleshooting](#api-troubleshooting)
5. [Performance Issues](#performance-issues)
6. [Security Issues](#security-issues)
7. [Integration Problems](#integration-problems)
8. [Error Code Reference](#error-code-reference)
9. [Log Analysis](#log-analysis)
10. [Support Resources](#support-resources)

---

## 🚀 **Quick Diagnostics**

### **System Health Check**
```bash
# Run comprehensive system diagnostics
python -m computer_genie.diagnostics --full-check

# Quick health check
python -m computer_genie.diagnostics --health

# Network connectivity test
python -m computer_genie.diagnostics --network

# API connectivity test
python -m computer_genie.diagnostics --api-test
```

### **Environment Verification**
```bash
# Check Python environment
python --version
pip list | grep computer-genie

# Check system requirements
python -m computer_genie.diagnostics --system-requirements

# Check GPU availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check disk space
python -m computer_genie.diagnostics --disk-space
```

### **Configuration Validation**
```bash
# Validate configuration files
python -m computer_genie.config --validate

# Check API credentials
python -m computer_genie.auth --test-credentials

# Verify database connection
python -m computer_genie.db --test-connection

# Check file permissions
python -m computer_genie.diagnostics --permissions
```

---

## 🔍 **Common Issues & Solutions**

### **Issue 1: Installation Failures**

#### **Problem**: Package installation fails with dependency conflicts
```bash
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

#### **Solution**:
```bash
# Method 1: Clean installation
pip uninstall computer-genie -y
pip cache purge
pip install computer-genie --no-cache-dir

# Method 2: Use virtual environment
python -m venv computer_genie_env
source computer_genie_env/bin/activate  # Linux/Mac
# or
computer_genie_env\Scripts\activate  # Windows
pip install computer-genie

# Method 3: Install with specific versions
pip install computer-genie==1.0.0 --force-reinstall
```

### **Issue 2: API Authentication Failures**

#### **Problem**: 401 Unauthorized errors
```json
{
  "error": {
    "code": "INVALID_API_KEY",
    "message": "The provided API key is invalid"
  }
}
```

#### **Solution**:
```python
# Verify API key format
import re

def validate_api_key(api_key):
    # API keys should start with 'cg_' and be 64 characters long
    pattern = r'^cg_[a-zA-Z0-9]{61}$'
    return bool(re.match(pattern, api_key))

# Check API key
api_key = "your_api_key_here"
if not validate_api_key(api_key):
    print("Invalid API key format")

# Test API connectivity
import requests

response = requests.get(
    'https://api.computer-genie.com/v1/auth/verify',
    headers={'Authorization': f'Bearer {api_key}'}
)

if response.status_code == 200:
    print("API key is valid")
else:
    print(f"API key validation failed: {response.status_code}")
```

### **Issue 3: File Processing Failures**

#### **Problem**: Documents fail to process with unclear errors
```json
{
  "error": {
    "code": "PROCESSING_FAILED",
    "message": "Document processing failed"
  }
}
```

#### **Solution**:
```python
# File validation before upload
import os
import mimetypes

def validate_file_for_processing(file_path):
    """Comprehensive file validation"""
    
    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check file size (max 50MB)
    file_size = os.path.getsize(file_path)
    max_size = 50 * 1024 * 1024  # 50MB
    if file_size > max_size:
        raise ValueError(f"File too large: {file_size} bytes (max: {max_size})")
    
    # Check file type
    mime_type, _ = mimetypes.guess_type(file_path)
    supported_types = [
        'application/pdf',
        'image/jpeg',
        'image/png',
        'image/tiff',
        'image/bmp'
    ]
    
    if mime_type not in supported_types:
        raise ValueError(f"Unsupported file type: {mime_type}")
    
    # Check file is not corrupted (basic check)
    try:
        with open(file_path, 'rb') as f:
            # Read first few bytes to ensure file is readable
            f.read(1024)
    except Exception as e:
        raise ValueError(f"File appears to be corrupted: {e}")
    
    return True

# Usage
try:
    validate_file_for_processing('document.pdf')
    print("File validation passed")
except Exception as e:
    print(f"File validation failed: {e}")
```

### **Issue 4: Memory Issues**

#### **Problem**: Out of memory errors during processing
```
MemoryError: Unable to allocate array with shape (10000, 10000) and data type float64
```

#### **Solution**:
```python
# Memory optimization techniques

# 1. Process files in batches
def process_files_in_batches(file_paths, batch_size=5):
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        for file_path in batch:
            try:
                result = process_document(file_path)
                yield result
            except MemoryError:
                # Reduce batch size and retry
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(f"Reducing batch size to {batch_size}")
                else:
                    raise

# 2. Clear memory between operations
import gc

def process_with_memory_management(file_path):
    try:
        result = process_document(file_path)
        return result
    finally:
        # Force garbage collection
        gc.collect()

# 3. Use streaming for large files
def process_large_file_streaming(file_path):
    with open(file_path, 'rb') as f:
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            # Process chunk
            process_chunk(chunk)
```

### **Issue 5: Network Connectivity Problems**

#### **Problem**: Timeout errors and connection failures
```
requests.exceptions.ConnectTimeout: HTTPSConnectionPool(host='api.computer-genie.com', port=443)
```

#### **Solution**:
```python
# Network troubleshooting and resilience

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

def create_resilient_session():
    """Create HTTP session with retry logic and timeouts"""
    
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    
    # Configure adapter
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set timeouts
    session.timeout = (10, 30)  # (connect, read) timeout
    
    return session

# Network connectivity test
def test_network_connectivity():
    """Test network connectivity to Computer Genie API"""
    
    endpoints = [
        'https://api.computer-genie.com/v1/health',
        'https://api.computer-genie.com/v1/status',
        'https://auth.computer-genie.com/health'
    ]
    
    session = create_resilient_session()
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = session.get(endpoint, timeout=10)
            end_time = time.time()
            
            print(f"✅ {endpoint}: {response.status_code} ({end_time - start_time:.2f}s)")
            
        except requests.exceptions.Timeout:
            print(f"❌ {endpoint}: Timeout")
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint}: Connection Error")
        except Exception as e:
            print(f"❌ {endpoint}: {type(e).__name__}: {e}")

# Usage
test_network_connectivity()
```

---

## 🛠️ **Installation Troubleshooting**

### **Windows Installation Issues**

#### **Problem**: PowerShell execution policy prevents script execution
```
install.bat : File cannot be loaded because running scripts is disabled on this system
```

#### **Solution**:
```powershell
# Check current execution policy
Get-ExecutionPolicy

# Set execution policy for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Alternative: Run with bypass
powershell -ExecutionPolicy Bypass -File install.ps1
```

#### **Problem**: Visual C++ build tools missing
```
Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools"
```

#### **Solution**:
```bash
# Option 1: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Option 2: Use pre-compiled wheels
pip install --only-binary=all computer-genie

# Option 3: Use conda instead of pip
conda install -c conda-forge computer-genie
```

### **Linux Installation Issues**

#### **Problem**: Permission denied errors
```
PermissionError: [Errno 13] Permission denied: '/usr/local/lib/python3.9/site-packages/'
```

#### **Solution**:
```bash
# Option 1: Install for current user only
pip install --user computer-genie

# Option 2: Use virtual environment
python3 -m venv computer_genie_env
source computer_genie_env/bin/activate
pip install computer-genie

# Option 3: Use sudo (not recommended)
sudo pip install computer-genie
```

#### **Problem**: Missing system dependencies
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

#### **Solution**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0

# CentOS/RHEL
sudo yum install -y \
    mesa-libGL \
    glib2 \
    libSM \
    libXext \
    libXrender \
    libgomp \
    gtk3

# Check installation
python -c "import cv2; print('OpenCV imported successfully')"
```

### **macOS Installation Issues**

#### **Problem**: Xcode command line tools missing
```
xcrun: error: invalid active developer path, missing xcrun
```

#### **Solution**:
```bash
# Install Xcode command line tools
xcode-select --install

# Verify installation
xcode-select -p

# If still having issues, reset
sudo xcode-select --reset
```

#### **Problem**: Apple Silicon compatibility issues
```
ImportError: dlopen failed: cannot load library
```

#### **Solution**:
```bash
# Use Rosetta 2 for Intel compatibility
arch -x86_64 pip install computer-genie

# Or use native ARM64 build
pip install --no-binary :all: computer-genie

# Check architecture
python -c "import platform; print(platform.machine())"
```

---

## 🌐 **API Troubleshooting**

### **Authentication Issues**

#### **Debug Authentication Flow**
```python
import requests
import json

def debug_authentication(api_key):
    """Debug API authentication step by step"""
    
    print("🔍 Debugging API Authentication...")
    
    # Step 1: Check API key format
    print(f"1. API Key format: {'✅ Valid' if api_key.startswith('cg_') else '❌ Invalid'}")
    
    # Step 2: Test auth endpoint
    try:
        response = requests.get(
            'https://api.computer-genie.com/v1/auth/verify',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=10
        )
        
        print(f"2. Auth endpoint: {response.status_code}")
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"   ✅ Authenticated as: {user_info.get('email', 'Unknown')}")
            print(f"   ✅ Plan: {user_info.get('plan', 'Unknown')}")
            print(f"   ✅ Rate limit: {user_info.get('rate_limit', 'Unknown')}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Step 3: Test API endpoint
    try:
        response = requests.get(
            'https://api.computer-genie.com/v1/documents',
            headers={'Authorization': f'Bearer {api_key}'},
            params={'limit': 1},
            timeout=10
        )
        
        print(f"3. API endpoint: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ API access confirmed")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

# Usage
debug_authentication('your_api_key_here')
```

### **Rate Limiting Issues**

#### **Handle Rate Limits Gracefully**
```python
import time
import random
from functools import wraps

def handle_rate_limits(max_retries=3, base_delay=1):
    """Decorator to handle rate limiting with exponential backoff"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    response = func(*args, **kwargs)
                    
                    # Check for rate limiting
                    if hasattr(response, 'status_code') and response.status_code == 429:
                        if attempt < max_retries:
                            # Get retry-after header or use exponential backoff
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                delay = int(retry_after)
                            else:
                                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                            
                            print(f"Rate limited. Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                            continue
                        else:
                            raise Exception("Max retries exceeded for rate limiting")
                    
                    return response
                    
                except Exception as e:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Request failed. Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        raise e
            
        return wrapper
    return decorator

# Usage
@handle_rate_limits(max_retries=3, base_delay=2)
def make_api_request(url, headers, **kwargs):
    return requests.request(url=url, headers=headers, **kwargs)
```

### **Request/Response Debugging**

#### **Debug API Requests**
```python
import requests
import json
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

def debug_api_request(method, url, headers=None, data=None, files=None):
    """Debug API request with detailed logging"""
    
    print(f"🔍 Debugging {method.upper()} request to {url}")
    
    # Log headers (mask sensitive data)
    if headers:
        safe_headers = headers.copy()
        if 'Authorization' in safe_headers:
            safe_headers['Authorization'] = 'Bearer ***MASKED***'
        print(f"Headers: {json.dumps(safe_headers, indent=2)}")
    
    # Log request data
    if data:
        print(f"Data: {json.dumps(data, indent=2) if isinstance(data, dict) else str(data)}")
    
    if files:
        print(f"Files: {list(files.keys()) if isinstance(files, dict) else 'Binary data'}")
    
    try:
        # Make request
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data if isinstance(data, dict) else None,
            data=data if not isinstance(data, dict) else None,
            files=files,
            timeout=30
        )
        
        # Log response
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            print(f"Response Body: {json.dumps(response_json, indent=2)}")
        except:
            print(f"Response Body: {response.text[:500]}...")
        
        return response
        
    except Exception as e:
        print(f"❌ Request failed: {type(e).__name__}: {e}")
        raise

# Usage
response = debug_api_request(
    method='POST',
    url='https://api.computer-genie.com/v1/documents',
    headers={'Authorization': 'Bearer your_api_key'},
    files={'file': open('document.pdf', 'rb')}
)
```

---

## ⚡ **Performance Issues**

### **Slow Processing Times**

#### **Performance Monitoring**
```python
import time
import psutil
import threading
from contextlib import contextmanager

@contextmanager
def performance_monitor(operation_name):
    """Monitor performance metrics during operation"""
    
    print(f"🔍 Monitoring performance for: {operation_name}")
    
    # Initial metrics
    start_time = time.time()
    start_cpu = psutil.cpu_percent()
    start_memory = psutil.virtual_memory().percent
    process = psutil.Process()
    start_process_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Monitor in background
    metrics = {'peak_cpu': start_cpu, 'peak_memory': start_memory}
    monitoring = True
    
    def monitor():
        while monitoring:
            metrics['peak_cpu'] = max(metrics['peak_cpu'], psutil.cpu_percent())
            metrics['peak_memory'] = max(metrics['peak_memory'], psutil.virtual_memory().percent)
            time.sleep(0.1)
    
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    
    try:
        yield metrics
    finally:
        monitoring = False
        monitor_thread.join()
        
        # Final metrics
        end_time = time.time()
        end_process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_used = end_process_memory - start_process_memory
        
        print(f"📊 Performance Report for {operation_name}:")
        print(f"   ⏱️  Duration: {duration:.2f} seconds")
        print(f"   🧠 Peak CPU: {metrics['peak_cpu']:.1f}%")
        print(f"   💾 Peak Memory: {metrics['peak_memory']:.1f}%")
        print(f"   📈 Process Memory Used: {memory_used:.1f} MB")

# Usage
with performance_monitor("Document Processing"):
    result = process_document('large_document.pdf')
```

#### **Optimization Strategies**
```python
# 1. Parallel processing for multiple files
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def process_files_parallel(file_paths, max_workers=None):
    """Process multiple files in parallel"""
    
    if max_workers is None:
        max_workers = min(len(file_paths), multiprocessing.cpu_count())
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_document, file_path): file_path 
            for file_path in file_paths
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append({'file': file_path, 'result': result, 'status': 'success'})
            except Exception as e:
                results.append({'file': file_path, 'error': str(e), 'status': 'failed'})
    
    return results

# 2. Caching for repeated operations
from functools import lru_cache
import hashlib

def get_file_hash(file_path):
    """Get MD5 hash of file for caching"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

@lru_cache(maxsize=100)
def process_document_cached(file_hash, file_path):
    """Process document with caching based on file hash"""
    return process_document(file_path)

def process_with_cache(file_path):
    file_hash = get_file_hash(file_path)
    return process_document_cached(file_hash, file_path)

# 3. Memory-efficient processing for large files
def process_large_file_efficiently(file_path, chunk_size=1024*1024):
    """Process large files in chunks to manage memory"""
    
    file_size = os.path.getsize(file_path)
    
    if file_size > 100 * 1024 * 1024:  # 100MB
        print(f"Large file detected ({file_size/1024/1024:.1f}MB). Using chunked processing.")
        
        # Process in chunks
        results = []
        with open(file_path, 'rb') as f:
            chunk_num = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Process chunk
                chunk_result = process_chunk(chunk, chunk_num)
                results.append(chunk_result)
                chunk_num += 1
                
                # Force garbage collection
                import gc
                gc.collect()
        
        # Combine results
        return combine_chunk_results(results)
    else:
        # Process normally
        return process_document(file_path)
```

### **Memory Optimization**

#### **Memory Usage Monitoring**
```python
import tracemalloc
import psutil
from functools import wraps

def monitor_memory(func):
    """Decorator to monitor memory usage of functions"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start tracing
        tracemalloc.start()
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            # Get peak memory
            current, peak = tracemalloc.get_traced_memory()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"🧠 Memory usage for {func.__name__}:")
            print(f"   Initial: {initial_memory:.1f} MB")
            print(f"   Final: {final_memory:.1f} MB")
            print(f"   Peak traced: {peak / 1024 / 1024:.1f} MB")
            print(f"   Difference: {final_memory - initial_memory:.1f} MB")
            
            return result
            
        finally:
            tracemalloc.stop()
    
    return wrapper

# Usage
@monitor_memory
def process_document_with_monitoring(file_path):
    return process_document(file_path)
```

#### **Memory Cleanup Strategies**
```python
import gc
import weakref

class MemoryManager:
    """Manage memory usage during processing"""
    
    def __init__(self, max_memory_mb=1000):
        self.max_memory_mb = max_memory_mb
        self.objects = []
    
    def add_object(self, obj):
        """Add object to memory tracking"""
        self.objects.append(weakref.ref(obj))
    
    def cleanup(self):
        """Force cleanup of tracked objects"""
        # Remove dead references
        self.objects = [ref for ref in self.objects if ref() is not None]
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.max_memory_mb:
            print(f"⚠️ High memory usage: {memory_mb:.1f} MB")
            
            # More aggressive cleanup
            for ref in self.objects:
                obj = ref()
                if obj is not None:
                    del obj
            
            self.objects.clear()
            gc.collect()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# Usage
with MemoryManager(max_memory_mb=500) as mem_mgr:
    for file_path in large_file_list:
        result = process_document(file_path)
        mem_mgr.add_object(result)
        
        # Periodic cleanup
        if len(mem_mgr.objects) > 10:
            mem_mgr.cleanup()
```

---

## 🔒 **Security Issues**

### **SSL/TLS Certificate Issues**

#### **Certificate Verification Problems**
```python
import ssl
import socket
import requests
from urllib3.exceptions import InsecureRequestWarning

def diagnose_ssl_issues(hostname, port=443):
    """Diagnose SSL/TLS certificate issues"""
    
    print(f"🔍 Diagnosing SSL for {hostname}:{port}")
    
    try:
        # Test basic connection
        context = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                print(f"✅ SSL connection successful")
                print(f"   Certificate subject: {cert.get('subject')}")
                print(f"   Certificate issuer: {cert.get('issuer')}")
                print(f"   Certificate expires: {cert.get('notAfter')}")
                
    except ssl.SSLError as e:
        print(f"❌ SSL Error: {e}")
        
        # Try with different SSL context
        try:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock) as ssock:
                    print("⚠️ Connection successful with SSL verification disabled")
                    
        except Exception as e2:
            print(f"❌ Connection failed even with SSL verification disabled: {e2}")
    
    except Exception as e:
        print(f"❌ Connection error: {e}")

# Test Computer Genie API SSL
diagnose_ssl_issues('api.computer-genie.com')

# Workaround for SSL issues (not recommended for production)
def create_unverified_session():
    """Create session that bypasses SSL verification (use with caution)"""
    
    session = requests.Session()
    session.verify = False
    
    # Suppress SSL warnings
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    
    return session
```

### **API Key Security**

#### **Secure API Key Management**
```python
import os
import keyring
import getpass
from cryptography.fernet import Fernet

class SecureAPIKeyManager:
    """Secure API key management"""
    
    def __init__(self, service_name="computer-genie"):
        self.service_name = service_name
    
    def store_api_key(self, username, api_key):
        """Store API key securely in system keyring"""
        try:
            keyring.set_password(self.service_name, username, api_key)
            print(f"✅ API key stored securely for {username}")
        except Exception as e:
            print(f"❌ Failed to store API key: {e}")
    
    def get_api_key(self, username):
        """Retrieve API key from system keyring"""
        try:
            api_key = keyring.get_password(self.service_name, username)
            if api_key:
                return api_key
            else:
                print(f"❌ No API key found for {username}")
                return None
        except Exception as e:
            print(f"❌ Failed to retrieve API key: {e}")
            return None
    
    def delete_api_key(self, username):
        """Delete API key from system keyring"""
        try:
            keyring.delete_password(self.service_name, username)
            print(f"✅ API key deleted for {username}")
        except Exception as e:
            print(f"❌ Failed to delete API key: {e}")

# Usage
key_manager = SecureAPIKeyManager()

# Store API key
username = "user@company.com"
api_key = getpass.getpass("Enter your API key: ")
key_manager.store_api_key(username, api_key)

# Retrieve API key
api_key = key_manager.get_api_key(username)
```

### **Data Privacy & Compliance**

#### **Data Sanitization**
```python
import re
import hashlib

class DataSanitizer:
    """Sanitize sensitive data before logging or transmission"""
    
    def __init__(self):
        # Patterns for sensitive data
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'api_key': r'\bcg_[a-zA-Z0-9]{61}\b'
        }
    
    def sanitize_text(self, text, replacement='***REDACTED***'):
        """Sanitize sensitive information in text"""
        
        sanitized = text
        
        for pattern_name, pattern in self.patterns.items():
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def hash_sensitive_data(self, data):
        """Hash sensitive data for logging purposes"""
        
        if isinstance(data, str):
            return hashlib.sha256(data.encode()).hexdigest()[:8] + "..."
        else:
            return hashlib.sha256(str(data).encode()).hexdigest()[:8] + "..."
    
    def sanitize_dict(self, data_dict, sensitive_keys=None):
        """Sanitize dictionary containing sensitive data"""
        
        if sensitive_keys is None:
            sensitive_keys = ['password', 'api_key', 'token', 'secret', 'ssn', 'credit_card']
        
        sanitized = {}
        
        for key, value in data_dict.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                sanitized[key] = self.hash_sensitive_data(value)
            elif isinstance(value, str):
                sanitized[key] = self.sanitize_text(value)
            else:
                sanitized[key] = value
        
        return sanitized

# Usage
sanitizer = DataSanitizer()

# Sanitize text
text = "Contact John at john.doe@company.com or call 555-123-4567"
sanitized_text = sanitizer.sanitize_text(text)
print(sanitized_text)  # "Contact John at ***REDACTED*** or call ***REDACTED***"

# Sanitize dictionary
user_data = {
    'name': 'John Doe',
    'email': 'john.doe@company.com',
    'api_key': 'cg_1234567890abcdef...',
    'phone': '555-123-4567'
}
sanitized_data = sanitizer.sanitize_dict(user_data)
print(sanitized_data)
```

---

## 🔗 **Integration Problems**

### **Database Connection Issues**

#### **Database Connectivity Troubleshooting**
```python
import psycopg2
import pymongo
import mysql.connector
from sqlalchemy import create_engine, text
import time

class DatabaseTroubleshooter:
    """Troubleshoot database connectivity issues"""
    
    def test_postgresql(self, connection_string):
        """Test PostgreSQL connection"""
        try:
            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            cursor.close()
            conn.close()
            print(f"✅ PostgreSQL connection successful: {version[0]}")
            return True
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            return False
    
    def test_mongodb(self, connection_string):
        """Test MongoDB connection"""
        try:
            client = pymongo.MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Force connection
            client.server_info()
            print(f"✅ MongoDB connection successful")
            client.close()
            return True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return False
    
    def test_mysql(self, host, user, password, database, port=3306):
        """Test MySQL connection"""
        try:
            conn = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
                port=port,
                connection_timeout=10
            )
            cursor = conn.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            cursor.close()
            conn.close()
            print(f"✅ MySQL connection successful: {version[0]}")
            return True
        except Exception as e:
            print(f"❌ MySQL connection failed: {e}")
            return False
    
    def test_sqlalchemy(self, connection_string):
        """Test SQLAlchemy connection"""
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print(f"✅ SQLAlchemy connection successful")
            return True
        except Exception as e:
            print(f"❌ SQLAlchemy connection failed: {e}")
            return False

# Usage
troubleshooter = DatabaseTroubleshooter()

# Test different database connections
troubleshooter.test_postgresql("postgresql://user:password@localhost:5432/computer_genie")
troubleshooter.test_mongodb("mongodb://localhost:27017/computer_genie")
troubleshooter.test_mysql("localhost", "user", "password", "computer_genie")
```

### **Third-party Service Integration**

#### **Service Health Monitoring**
```python
import requests
import time
from datetime import datetime, timedelta

class ServiceHealthMonitor:
    """Monitor health of third-party services"""
    
    def __init__(self):
        self.services = {
            'computer_genie_api': 'https://api.computer-genie.com/v1/health',
            'auth_service': 'https://auth.computer-genie.com/health',
            'storage_service': 'https://storage.computer-genie.com/health'
        }
        self.history = {}
    
    def check_service(self, service_name, url, timeout=10):
        """Check health of a single service"""
        
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=timeout)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # ms
            
            status = {
                'service': service_name,
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'status_code': response.status_code,
                'response_time': response_time,
                'timestamp': datetime.now(),
                'error': None
            }
            
            if response.status_code == 200:
                print(f"✅ {service_name}: Healthy ({response_time:.1f}ms)")
            else:
                print(f"⚠️ {service_name}: Unhealthy (HTTP {response.status_code})")
            
        except requests.exceptions.Timeout:
            status = {
                'service': service_name,
                'status': 'timeout',
                'status_code': None,
                'response_time': timeout * 1000,
                'timestamp': datetime.now(),
                'error': 'Request timeout'
            }
            print(f"❌ {service_name}: Timeout")
            
        except Exception as e:
            status = {
                'service': service_name,
                'status': 'error',
                'status_code': None,
                'response_time': None,
                'timestamp': datetime.now(),
                'error': str(e)
            }
            print(f"❌ {service_name}: Error - {e}")
        
        # Store in history
        if service_name not in self.history:
            self.history[service_name] = []
        
        self.history[service_name].append(status)
        
        # Keep only last 100 entries
        self.history[service_name] = self.history[service_name][-100:]
        
        return status
    
    def check_all_services(self):
        """Check health of all configured services"""
        
        print(f"🔍 Checking health of {len(self.services)} services...")
        
        results = {}
        for service_name, url in self.services.items():
            results[service_name] = self.check_service(service_name, url)
        
        return results
    
    def get_service_uptime(self, service_name, hours=24):
        """Calculate service uptime over specified period"""
        
        if service_name not in self.history:
            return None
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            check for check in self.history[service_name]
            if check['timestamp'] > cutoff_time
        ]
        
        if not recent_checks:
            return None
        
        healthy_checks = [
            check for check in recent_checks
            if check['status'] == 'healthy'
        ]
        
        uptime_percentage = (len(healthy_checks) / len(recent_checks)) * 100
        
        return {
            'uptime_percentage': uptime_percentage,
            'total_checks': len(recent_checks),
            'healthy_checks': len(healthy_checks),
            'period_hours': hours
        }

# Usage
monitor = ServiceHealthMonitor()

# Check all services
results = monitor.check_all_services()

# Monitor continuously
def continuous_monitoring(interval_minutes=5):
    while True:
        print(f"\n📊 Service Health Check - {datetime.now()}")
        monitor.check_all_services()
        
        # Print uptime statistics
        for service_name in monitor.services.keys():
            uptime = monitor.get_service_uptime(service_name)
            if uptime:
                print(f"   {service_name}: {uptime['uptime_percentage']:.1f}% uptime")
        
        time.sleep(interval_minutes * 60)

# Run continuous monitoring (uncomment to use)
# continuous_monitoring()
```

---

## 📊 **Error Code Reference**

### **HTTP Status Codes**

| Code | Meaning | Description | Solution |
|------|---------|-------------|----------|
| 200 | OK | Request successful | No action needed |
| 201 | Created | Resource created successfully | No action needed |
| 400 | Bad Request | Invalid request format | Check request syntax and parameters |
| 401 | Unauthorized | Invalid or missing authentication | Verify API key and authentication |
| 403 | Forbidden | Insufficient permissions | Check user permissions and plan limits |
| 404 | Not Found | Resource not found | Verify resource ID and endpoint URL |
| 409 | Conflict | Resource conflict | Check for duplicate resources |
| 422 | Unprocessable Entity | Validation errors | Fix validation errors in request |
| 429 | Too Many Requests | Rate limit exceeded | Implement rate limiting and retry logic |
| 500 | Internal Server Error | Server error | Contact support if persistent |
| 502 | Bad Gateway | Gateway error | Temporary issue, retry later |
| 503 | Service Unavailable | Service temporarily unavailable | Check service status, retry later |
| 504 | Gateway Timeout | Request timeout | Increase timeout or retry |

### **Application Error Codes**

#### **Authentication Errors (AUTH_XXX)**
```yaml
AUTH_001: "Invalid API key format"
AUTH_002: "API key not found"
AUTH_003: "API key expired"
AUTH_004: "API key suspended"
AUTH_005: "Invalid OAuth token"
AUTH_006: "OAuth token expired"
AUTH_007: "Insufficient permissions"
AUTH_008: "Account suspended"
AUTH_009: "MFA required"
AUTH_010: "Invalid credentials"
```

#### **Validation Errors (VAL_XXX)**
```yaml
VAL_001: "Missing required field"
VAL_002: "Invalid field format"
VAL_003: "Field value out of range"
VAL_004: "Invalid file type"
VAL_005: "File too large"
VAL_006: "File corrupted"
VAL_007: "Invalid JSON format"
VAL_008: "Schema validation failed"
VAL_009: "Duplicate resource"
VAL_010: "Invalid parameter combination"
```

#### **Processing Errors (PROC_XXX)**
```yaml
PROC_001: "Document processing failed"
PROC_002: "OCR extraction failed"
PROC_003: "Computer vision analysis failed"
PROC_004: "Unsupported document format"
PROC_005: "Document too complex"
PROC_006: "Processing timeout"
PROC_007: "Insufficient quality"
PROC_008: "Language not supported"
PROC_009: "Model not available"
PROC_010: "Processing queue full"
```

#### **System Errors (SYS_XXX)**
```yaml
SYS_001: "Internal server error"
SYS_002: "Database connection failed"
SYS_003: "Storage service unavailable"
SYS_004: "Memory allocation failed"
SYS_005: "Disk space insufficient"
SYS_006: "Service dependency unavailable"
SYS_007: "Configuration error"
SYS_008: "Network connectivity issue"
SYS_009: "Resource exhausted"
SYS_010: "Maintenance mode"
```

---

## 📝 **Log Analysis**

### **Log Collection and Analysis**

#### **Enable Debug Logging**
```python
import logging
import sys
from datetime import datetime

def setup_debug_logging(log_file=None):
    """Setup comprehensive debug logging"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"computer_genie_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # HTTP request logging
    import urllib3
    urllib3.disable_warnings()
    
    # Enable requests logging
    logging.getLogger("requests.packages.urllib3").setLevel(logging.DEBUG)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.DEBUG)
    
    print(f"Debug logging enabled. Log file: {log_file}")
    
    return log_file

# Usage
log_file = setup_debug_logging()
```

#### **Log Analysis Tools**
```python
import re
import json
from collections import defaultdict, Counter
from datetime import datetime

class LogAnalyzer:
    """Analyze Computer Genie logs for troubleshooting"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.entries = []
        self.parse_logs()
    
    def parse_logs(self):
        """Parse log file into structured entries"""
        
        log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+?) - (.+)'
        
        with open(self.log_file, 'r') as f:
            for line in f:
                match = re.match(log_pattern, line.strip())
                if match:
                    timestamp_str, logger_name, level, location, message = match.groups()
                    
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    except:
                        timestamp = None
                    
                    entry = {
                        'timestamp': timestamp,
                        'logger': logger_name,
                        'level': level,
                        'location': location,
                        'message': message
                    }
                    
                    self.entries.append(entry)
    
    def get_error_summary(self):
        """Get summary of errors and warnings"""
        
        errors = [entry for entry in self.entries if entry['level'] in ['ERROR', 'CRITICAL']]
        warnings = [entry for entry in self.entries if entry['level'] == 'WARNING']
        
        error_counter = Counter([entry['message'][:100] for entry in errors])
        warning_counter = Counter([entry['message'][:100] for entry in warnings])
        
        return {
            'total_errors': len(errors),
            'total_warnings': len(warnings),
            'top_errors': error_counter.most_common(5),
            'top_warnings': warning_counter.most_common(5)
        }
    
    def get_api_errors(self):
        """Extract API-related errors"""
        
        api_errors = []
        
        for entry in self.entries:
            if entry['level'] in ['ERROR', 'WARNING']:
                message = entry['message'].lower()
                if any(keyword in message for keyword in ['api', 'http', 'request', 'response', '401', '403', '429', '500']):
                    api_errors.append(entry)
        
        return api_errors
    
    def get_performance_issues(self):
        """Identify performance-related issues"""
        
        performance_issues = []
        
        for entry in self.entries:
            message = entry['message'].lower()
            if any(keyword in message for keyword in ['timeout', 'slow', 'memory', 'performance', 'took']):
                performance_issues.append(entry)
        
        return performance_issues
    
    def generate_report(self):
        """Generate comprehensive troubleshooting report"""
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'log_file': self.log_file,
            'total_entries': len(self.entries),
            'error_summary': self.get_error_summary(),
            'api_errors': len(self.get_api_errors()),
            'performance_issues': len(self.get_performance_issues())
        }
        
        # Time range
        if self.entries:
            timestamps = [entry['timestamp'] for entry in self.entries if entry['timestamp']]
            if timestamps:
                report['time_range'] = {
                    'start': min(timestamps).isoformat(),
                    'end': max(timestamps).isoformat()
                }
        
        # Log level distribution
        level_counter = Counter([entry['level'] for entry in self.entries])
        report['log_level_distribution'] = dict(level_counter)
        
        return report

# Usage
analyzer = LogAnalyzer('computer_genie_debug.log')
report = analyzer.generate_report()

print("📊 Log Analysis Report:")
print(json.dumps(report, indent=2, default=str))

# Print top errors
error_summary = analyzer.get_error_summary()
if error_summary['top_errors']:
    print("\n🔥 Top Errors:")
    for error, count in error_summary['top_errors']:
        print(f"   {count}x: {error}")
```

---

## 📞 **Support Resources**

### **Getting Help**

#### **1. Self-Service Resources**
- 📚 **Documentation**: [docs.computer-genie.com](https://docs.computer-genie.com)
- 🔍 **Knowledge Base**: [help.computer-genie.com](https://help.computer-genie.com)
- 💬 **Community Forum**: [community.computer-genie.com](https://community.computer-genie.com)
- 📖 **API Reference**: [api-docs.computer-genie.com](https://api-docs.computer-genie.com)
- 🎥 **Video Tutorials**: [tutorials.computer-genie.com](https://tutorials.computer-genie.com)

#### **2. Direct Support Channels**

| Channel | Availability | Response Time | Best For |
|---------|-------------|---------------|----------|
| 📧 Email Support | 24/7 | 4-24 hours | Non-urgent issues |
| 💬 Live Chat | Business hours | < 5 minutes | Quick questions |
| 📞 Phone Support | Business hours | Immediate | Urgent issues |
| 🎫 Support Tickets | 24/7 | 1-8 hours | Complex issues |

#### **3. Enterprise Support**
- 🏢 **Dedicated Account Manager**: For Enterprise customers
- 🔧 **Technical Architect**: For implementation guidance
- 📊 **Success Manager**: For optimization and best practices
- 🚨 **24/7 Emergency Support**: For critical issues

### **Contact Information**

#### **Abhishek Technologies Pvt Ltd**
```
🏢 Headquarters:
   Abhishek Technologies Pvt Ltd
   Technology Park, Sector 5
   Bangalore, Karnataka 560001
   India

📧 Email Contacts:
   General Support: support@abhishektech.com
   Technical Support: tech-support@abhishektech.com
   API Support: api-support@abhishektech.com
   Sales Inquiries: sales@abhishektech.com
   Enterprise Support: enterprise@abhishektech.com

📞 Phone Numbers:
   India: +91-80-XXXX-XXXX
   US: +1-XXX-XXX-XXXX
   UK: +44-XXX-XXX-XXXX
   
🌐 Online:
   Website: https://www.abhishektech.com
   Support Portal: https://support.computer-genie.com
   Status Page: https://status.computer-genie.com
```

### **Before Contacting Support**

#### **Information to Gather**
1. **Environment Details**:
   - Operating system and version
   - Python version
   - Computer Genie version
   - Installation method

2. **Error Information**:
   - Complete error messages
   - Error codes
   - Stack traces
   - Log files

3. **Reproduction Steps**:
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Frequency of occurrence

4. **System Information**:
   - Hardware specifications
   - Network configuration
   - Security software
   - Other installed software

#### **Support Request Template**
```
Subject: [URGENT/HIGH/MEDIUM/LOW] Brief description of issue

Environment:
- OS: [Windows 10/Ubuntu 20.04/macOS 12.0]
- Python: [3.8.10]
- Computer Genie: [1.0.0]
- Installation: [pip/conda/source]

Issue Description:
[Detailed description of the problem]

Steps to Reproduce:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Expected Behavior:
[What you expected to happen]

Actual Behavior:
[What actually happened]

Error Messages:
[Complete error messages and stack traces]

Additional Context:
[Any additional information that might be helpful]

Attachments:
- Log files
- Screenshots
- Configuration files
```

### **Emergency Support Procedures**

#### **Critical Issue Definition**
- Production system completely down
- Data loss or corruption
- Security breach
- Service unavailable for > 1 hour

#### **Emergency Contact Process**
1. **Call Emergency Hotline**: +91-80-XXXX-XXXX (24/7)
2. **Send Emergency Email**: emergency@abhishektech.com
3. **Create High-Priority Ticket**: [support.computer-genie.com](https://support.computer-genie.com)
4. **Follow up within 15 minutes** if no response

#### **Escalation Matrix**
```
Level 1: Technical Support (0-2 hours)
Level 2: Senior Engineer (2-4 hours)
Level 3: Engineering Manager (4-8 hours)
Level 4: CTO/VP Engineering (8+ hours)
```

---

## 🔄 **Continuous Improvement**

### **Feedback and Feature Requests**
- 💡 **Feature Requests**: [features.computer-genie.com](https://features.computer-genie.com)
- 🐛 **Bug Reports**: [bugs.computer-genie.com](https://bugs.computer-genie.com)
- 📝 **Documentation Feedback**: [docs-feedback.computer-genie.com](https://docs-feedback.computer-genie.com)

### **Stay Updated**
- 📰 **Release Notes**: [releases.computer-genie.com](https://releases.computer-genie.com)
- 📧 **Newsletter**: [newsletter.computer-genie.com](https://newsletter.computer-genie.com)
- 🐦 **Twitter**: [@ComputerGenie](https://twitter.com/ComputerGenie)
- 📱 **LinkedIn**: [Abhishek Technologies](https://linkedin.com/company/abhishektech)

---

**© 2024 Abhishek Technologies Pvt Ltd. All rights reserved.**

*This troubleshooting guide is regularly updated. For the latest version, visit [docs.computer-genie.com/troubleshooting](https://docs.computer-genie.com/troubleshooting)*