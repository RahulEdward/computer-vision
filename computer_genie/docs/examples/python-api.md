# 🐍 Computer Genie Python API Examples

<div align="center">

![Python API](https://img.shields.io/badge/Python-API-green?style=for-the-badge&logo=python)

**Comprehensive Python Integration Examples**

</div>

---

## 📋 **Table of Contents**

1. [Installation & Setup](#installation--setup)
2. [Basic Usage](#basic-usage)
3. [Vision Operations](#vision-operations)
4. [Document Processing](#document-processing)
5. [Automation Scripts](#automation-scripts)
6. [Advanced Features](#advanced-features)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)

---

## 🚀 **Installation & Setup**

### **Installation**
```python
# Install Computer Genie
pip install computer-genie

# Install with additional dependencies
pip install computer-genie[full]

# Development installation
pip install computer-genie[dev]
```

### **Basic Setup**
```python
import computer_genie as cg
from computer_genie import Vision, Config, Automation

# Initialize Computer Genie
genie = cg.ComputerGenie()

# Configure API settings
config = Config(
    model="gpt-4-vision",
    timeout=30,
    quality="high"
)

# Initialize with configuration
genie = cg.ComputerGenie(config=config)
```

### **Authentication Setup**
```python
import os
from computer_genie import ComputerGenie

# Set API key via environment variable
os.environ['COMPUTER_GENIE_API_KEY'] = 'your-api-key-here'

# Or pass directly to constructor
genie = ComputerGenie(api_key='your-api-key-here')

# Enterprise authentication
genie = ComputerGenie(
    api_key='your-api-key',
    endpoint='https://enterprise.computer-genie.com',
    organization='your-org-id'
)
```

---

## 🔧 **Basic Usage**

### **Simple Vision Tasks**
```python
from computer_genie import ComputerGenie

# Initialize
genie = ComputerGenie()

# Take a screenshot
screenshot = genie.vision.screenshot()
print(f"Screenshot saved: {screenshot.path}")

# Analyze current screen
analysis = genie.vision.analyze("What do you see on the screen?")
print(analysis.description)

# Extract text from screen
text = genie.vision.extract_text()
print(f"Extracted text: {text}")
```

### **File Operations**
```python
# Analyze an image file
result = genie.vision.analyze_image("path/to/image.jpg", "Describe this image")
print(result.description)

# Extract text from image
text = genie.vision.extract_text_from_image("document.png")
print(text)

# Process PDF document
pdf_data = genie.vision.process_pdf("invoice.pdf")
print(pdf_data.extracted_text)
```

### **Basic Automation**
```python
# Click on an element
genie.automation.click("Submit button")

# Type text
genie.automation.type("Hello, World!")

# Press keyboard shortcut
genie.automation.key_combination("ctrl+c")

# Wait for element
genie.automation.wait_for_element("loading spinner", timeout=10)
```

---

## 👁️ **Vision Operations**

### **Screenshot Management**
```python
from computer_genie import ComputerGenie
from datetime import datetime

genie = ComputerGenie()

# Take screenshot with custom settings
screenshot = genie.vision.screenshot(
    region=(0, 0, 1920, 1080),  # x, y, width, height
    quality="high",
    format="png"
)

# Take screenshot of specific window
window_screenshot = genie.vision.screenshot_window("Google Chrome")

# Take screenshot with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
screenshot = genie.vision.screenshot(
    filename=f"screenshot_{timestamp}.png"
)

# Capture multiple monitors
for monitor in genie.vision.get_monitors():
    screenshot = genie.vision.screenshot(monitor=monitor.id)
    print(f"Monitor {monitor.id}: {screenshot.path}")
```

### **Screen Analysis**
```python
# Detailed screen analysis
analysis = genie.vision.analyze_screen(
    prompt="Identify all interactive elements on this page",
    include_coordinates=True,
    confidence_threshold=0.8
)

for element in analysis.elements:
    print(f"Element: {element.type} at {element.coordinates}")

# Find specific elements
buttons = genie.vision.find_elements("button")
for button in buttons:
    print(f"Button: {button.text} at {button.position}")

# Check if element exists
if genie.vision.element_exists("login form"):
    print("Login form found on screen")

# Wait for element to appear
element = genie.vision.wait_for_element(
    "success message",
    timeout=30,
    check_interval=1
)
```

### **Text Recognition (OCR)**
```python
# Extract all text from screen
text_data = genie.vision.extract_text(
    language="en",
    confidence_threshold=0.7
)

print(f"Extracted text: {text_data.text}")
print(f"Confidence: {text_data.confidence}")

# Extract text from specific region
region_text = genie.vision.extract_text_from_region(
    region=(100, 100, 500, 300),
    language="en"
)

# Extract structured data
table_data = genie.vision.extract_table_data()
for row in table_data.rows:
    print(row)

# Extract form fields
form_data = genie.vision.extract_form_fields()
for field, value in form_data.items():
    print(f"{field}: {value}")
```

---

## 📄 **Document Processing**

### **PDF Processing**
```python
from computer_genie import ComputerGenie

genie = ComputerGenie()

# Process single PDF
pdf_result = genie.document.process_pdf("invoice.pdf")
print(f"Pages: {pdf_result.page_count}")
print(f"Text: {pdf_result.text}")
print(f"Metadata: {pdf_result.metadata}")

# Extract specific information
invoice_data = genie.document.extract_invoice_data("invoice.pdf")
print(f"Invoice Number: {invoice_data.invoice_number}")
print(f"Total Amount: {invoice_data.total_amount}")
print(f"Due Date: {invoice_data.due_date}")

# Process multiple PDFs
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = genie.document.batch_process_pdfs(pdf_files)

for file, result in results.items():
    print(f"{file}: {result.summary}")
```

### **Image Document Processing**
```python
# Process scanned documents
scanned_doc = genie.document.process_scanned_document("scan.jpg")
print(f"Extracted text: {scanned_doc.text}")
print(f"Document type: {scanned_doc.document_type}")

# Extract form data from image
form_data = genie.document.extract_form_data("form_image.png")
for field, value in form_data.items():
    print(f"{field}: {value}")

# Process business cards
business_card = genie.document.process_business_card("card.jpg")
print(f"Name: {business_card.name}")
print(f"Company: {business_card.company}")
print(f"Email: {business_card.email}")
print(f"Phone: {business_card.phone}")

# Extract receipt information
receipt = genie.document.process_receipt("receipt.jpg")
print(f"Store: {receipt.store_name}")
print(f"Total: {receipt.total_amount}")
print(f"Items: {receipt.items}")
```

### **Document Classification**
```python
# Classify document type
doc_type = genie.document.classify_document("unknown_doc.pdf")
print(f"Document type: {doc_type.category}")
print(f"Confidence: {doc_type.confidence}")

# Batch classification
documents = ["doc1.pdf", "doc2.jpg", "doc3.png"]
classifications = genie.document.batch_classify(documents)

for doc, classification in classifications.items():
    print(f"{doc}: {classification.category} ({classification.confidence})")

# Custom classification
custom_classifier = genie.document.create_classifier([
    "invoice", "receipt", "contract", "report"
])

result = custom_classifier.classify("document.pdf")
print(f"Classification: {result.category}")
```

---

## 🤖 **Automation Scripts**

### **Web Automation**
```python
from computer_genie import ComputerGenie
import time

genie = ComputerGenie()

def automate_login(username, password):
    """Automate login process"""
    try:
        # Wait for page to load
        genie.automation.wait_for_element("login form", timeout=10)
        
        # Fill username
        genie.automation.click("username field")
        genie.automation.type(username)
        
        # Fill password
        genie.automation.click("password field")
        genie.automation.type(password)
        
        # Submit form
        genie.automation.click("login button")
        
        # Wait for success
        success = genie.automation.wait_for_element(
            "dashboard", timeout=15
        )
        
        return success is not None
        
    except Exception as e:
        print(f"Login failed: {e}")
        return False

# Use the function
if automate_login("john.doe", "password123"):
    print("Login successful!")
else:
    print("Login failed!")
```

### **Data Entry Automation**
```python
def automate_data_entry(data_file):
    """Automate data entry from CSV file"""
    import csv
    
    with open(data_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            try:
                # Click new entry button
                genie.automation.click("new entry button")
                
                # Fill form fields
                for field, value in row.items():
                    genie.automation.click(f"{field} field")
                    genie.automation.clear_field()
                    genie.automation.type(value)
                
                # Submit entry
                genie.automation.click("save button")
                
                # Wait for confirmation
                genie.automation.wait_for_element("success message")
                
                print(f"Entry saved: {row['name']}")
                
            except Exception as e:
                print(f"Failed to save entry {row}: {e}")
                continue

# Run data entry automation
automate_data_entry("customer_data.csv")
```

### **File Management Automation**
```python
def organize_screenshots():
    """Organize screenshots by date and content"""
    import os
    from datetime import datetime
    
    screenshot_dir = "screenshots"
    organized_dir = "organized_screenshots"
    
    for filename in os.listdir(screenshot_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(screenshot_dir, filename)
            
            # Analyze screenshot content
            analysis = genie.vision.analyze_image(
                file_path, 
                "What type of content is this? (web, desktop, app, document)"
            )
            
            content_type = analysis.category.lower()
            
            # Get file date
            file_date = datetime.fromtimestamp(
                os.path.getctime(file_path)
            ).strftime("%Y-%m")
            
            # Create organized directory structure
            target_dir = os.path.join(organized_dir, file_date, content_type)
            os.makedirs(target_dir, exist_ok=True)
            
            # Move file
            target_path = os.path.join(target_dir, filename)
            os.rename(file_path, target_path)
            
            print(f"Moved {filename} to {target_dir}")

# Run organization
organize_screenshots()
```

---

## 🔧 **Advanced Features**

### **Custom Models and Configuration**
```python
from computer_genie import ComputerGenie, Config

# Custom configuration
config = Config(
    model="gpt-4-vision-preview",
    temperature=0.7,
    max_tokens=1000,
    timeout=60,
    retry_attempts=3,
    cache_enabled=True
)

genie = ComputerGenie(config=config)

# Use different models for different tasks
vision_config = Config(model="gpt-4-vision")
text_config = Config(model="gpt-3.5-turbo")

vision_genie = ComputerGenie(config=vision_config)
text_genie = ComputerGenie(config=text_config)

# Switch models dynamically
genie.set_model("claude-3-vision")
result = genie.vision.analyze("complex image analysis")

genie.set_model("gpt-4-vision")
result2 = genie.vision.analyze("detailed description needed")
```

### **Batch Processing**
```python
from concurrent.futures import ThreadPoolExecutor
import os

def process_image_batch(image_folder, prompt):
    """Process multiple images in parallel"""
    
    image_files = [
        f for f in os.listdir(image_folder) 
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    def process_single_image(filename):
        file_path = os.path.join(image_folder, filename)
        try:
            result = genie.vision.analyze_image(file_path, prompt)
            return {
                'filename': filename,
                'analysis': result.description,
                'confidence': result.confidence,
                'status': 'success'
            }
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e),
                'status': 'failed'
            }
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_image, image_files))
    
    return results

# Process batch of images
results = process_image_batch(
    "product_images", 
    "Describe this product and identify its category"
)

for result in results:
    if result['status'] == 'success':
        print(f"{result['filename']}: {result['analysis']}")
    else:
        print(f"{result['filename']}: Error - {result['error']}")
```

### **Workflow Automation**
```python
class DocumentWorkflow:
    """Advanced document processing workflow"""
    
    def __init__(self):
        self.genie = ComputerGenie()
        self.processed_docs = []
    
    def process_document_pipeline(self, document_path):
        """Complete document processing pipeline"""
        
        # Step 1: Classify document
        classification = self.genie.document.classify_document(document_path)
        
        # Step 2: Extract content based on type
        if classification.category == "invoice":
            content = self.genie.document.extract_invoice_data(document_path)
        elif classification.category == "receipt":
            content = self.genie.document.process_receipt(document_path)
        elif classification.category == "contract":
            content = self.genie.document.extract_contract_terms(document_path)
        else:
            content = self.genie.document.extract_general_content(document_path)
        
        # Step 3: Validate extracted data
        validation = self.validate_extracted_data(content)
        
        # Step 4: Store results
        result = {
            'document_path': document_path,
            'classification': classification,
            'content': content,
            'validation': validation,
            'processed_at': datetime.now()
        }
        
        self.processed_docs.append(result)
        return result
    
    def validate_extracted_data(self, content):
        """Validate extracted data quality"""
        validation_result = {
            'is_valid': True,
            'confidence_score': 0.0,
            'issues': []
        }
        
        # Check confidence scores
        if hasattr(content, 'confidence') and content.confidence < 0.8:
            validation_result['issues'].append("Low confidence in extraction")
            validation_result['is_valid'] = False
        
        # Check for required fields
        required_fields = ['text', 'metadata']
        for field in required_fields:
            if not hasattr(content, field) or not getattr(content, field):
                validation_result['issues'].append(f"Missing {field}")
                validation_result['is_valid'] = False
        
        validation_result['confidence_score'] = getattr(content, 'confidence', 0.0)
        return validation_result
    
    def generate_report(self):
        """Generate processing report"""
        total_docs = len(self.processed_docs)
        successful_docs = sum(1 for doc in self.processed_docs if doc['validation']['is_valid'])
        
        report = {
            'total_processed': total_docs,
            'successful': successful_docs,
            'failed': total_docs - successful_docs,
            'success_rate': successful_docs / total_docs if total_docs > 0 else 0,
            'documents': self.processed_docs
        }
        
        return report

# Use the workflow
workflow = DocumentWorkflow()

# Process multiple documents
documents = ["invoice1.pdf", "receipt1.jpg", "contract1.pdf"]
for doc in documents:
    result = workflow.process_document_pipeline(doc)
    print(f"Processed {doc}: {result['validation']['is_valid']}")

# Generate report
report = workflow.generate_report()
print(f"Success rate: {report['success_rate']:.2%}")
```

---

## ⚠️ **Error Handling**

### **Basic Error Handling**
```python
from computer_genie import ComputerGenie, ComputerGenieError
from computer_genie.exceptions import (
    VisionError, 
    AutomationError, 
    DocumentError,
    APIError
)

genie = ComputerGenie()

try:
    result = genie.vision.analyze("What's on the screen?")
    print(result.description)
    
except VisionError as e:
    print(f"Vision processing error: {e}")
    
except AutomationError as e:
    print(f"Automation error: {e}")
    
except APIError as e:
    print(f"API error: {e}")
    
except ComputerGenieError as e:
    print(f"General Computer Genie error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

### **Retry Logic**
```python
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=3, delay=2)
def robust_screenshot():
    """Take screenshot with retry logic"""
    return genie.vision.screenshot()

@retry_on_failure(max_retries=5, delay=1)
def robust_click(element):
    """Click element with retry logic"""
    return genie.automation.click(element)

# Use robust functions
screenshot = robust_screenshot()
robust_click("submit button")
```

### **Comprehensive Error Handling**
```python
class RobustComputerGenie:
    """Wrapper class with comprehensive error handling"""
    
    def __init__(self, config=None):
        self.genie = ComputerGenie(config=config)
        self.error_log = []
    
    def safe_execute(self, operation, *args, **kwargs):
        """Safely execute any Computer Genie operation"""
        try:
            method = getattr(self.genie, operation)
            result = method(*args, **kwargs)
            return {
                'success': True,
                'result': result,
                'error': None
            }
        except Exception as e:
            error_info = {
                'operation': operation,
                'args': args,
                'kwargs': kwargs,
                'error': str(e),
                'timestamp': datetime.now()
            }
            self.error_log.append(error_info)
            
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    def get_error_summary(self):
        """Get summary of all errors"""
        if not self.error_log:
            return "No errors recorded"
        
        error_counts = {}
        for error in self.error_log:
            operation = error['operation']
            error_counts[operation] = error_counts.get(operation, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_by_operation': error_counts,
            'recent_errors': self.error_log[-5:]  # Last 5 errors
        }

# Use robust wrapper
robust_genie = RobustComputerGenie()

# Safe operations
screenshot_result = robust_genie.safe_execute('vision.screenshot')
if screenshot_result['success']:
    print(f"Screenshot taken: {screenshot_result['result'].path}")
else:
    print(f"Screenshot failed: {screenshot_result['error']}")

# Check error summary
error_summary = robust_genie.get_error_summary()
print(f"Total errors: {error_summary['total_errors']}")
```

---

## 🎯 **Best Practices**

### **Performance Optimization**
```python
from computer_genie import ComputerGenie, Config

# Optimized configuration
config = Config(
    cache_enabled=True,
    parallel_processing=True,
    max_workers=4,
    timeout=30,
    quality="medium"  # Balance between speed and quality
)

genie = ComputerGenie(config=config)

# Use caching for repeated operations
@lru_cache(maxsize=100)
def cached_analysis(image_path, prompt):
    """Cache analysis results for repeated queries"""
    return genie.vision.analyze_image(image_path, prompt)

# Batch similar operations
def batch_process_images(image_paths, prompt):
    """Process multiple images efficiently"""
    return genie.vision.batch_analyze_images(image_paths, prompt)

# Use appropriate quality settings
def quick_screenshot():
    """Take quick screenshot for monitoring"""
    return genie.vision.screenshot(quality="low", format="jpg")

def detailed_screenshot():
    """Take detailed screenshot for analysis"""
    return genie.vision.screenshot(quality="high", format="png")
```

### **Resource Management**
```python
from contextlib import contextmanager

@contextmanager
def computer_genie_session(config=None):
    """Context manager for Computer Genie sessions"""
    genie = ComputerGenie(config=config)
    try:
        yield genie
    finally:
        # Cleanup resources
        genie.cleanup()

# Use context manager
with computer_genie_session() as genie:
    screenshot = genie.vision.screenshot()
    analysis = genie.vision.analyze("Describe the screen")
    # Resources automatically cleaned up

# Memory management for large batches
def process_large_batch(file_list, batch_size=10):
    """Process large batches with memory management"""
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i + batch_size]
        
        with computer_genie_session() as genie:
            results = []
            for file_path in batch:
                result = genie.vision.analyze_image(file_path, "Analyze content")
                results.append(result)
            
            # Process results
            yield results
        
        # Memory is freed between batches
```

### **Security Best Practices**
```python
import os
from pathlib import Path

class SecureComputerGenie:
    """Security-focused Computer Genie wrapper"""
    
    def __init__(self):
        # Use environment variables for sensitive data
        api_key = os.getenv('COMPUTER_GENIE_API_KEY')
        if not api_key:
            raise ValueError("API key not found in environment variables")
        
        self.genie = ComputerGenie(api_key=api_key)
        self.allowed_paths = [
            Path.home() / "Documents",
            Path.home() / "Pictures",
            Path.cwd()
        ]
    
    def validate_file_path(self, file_path):
        """Validate file path for security"""
        path = Path(file_path).resolve()
        
        # Check if path is within allowed directories
        for allowed_path in self.allowed_paths:
            try:
                path.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        
        raise SecurityError(f"Access denied to path: {file_path}")
    
    def safe_analyze_image(self, file_path, prompt):
        """Safely analyze image with path validation"""
        self.validate_file_path(file_path)
        
        # Sanitize prompt
        safe_prompt = self.sanitize_prompt(prompt)
        
        return self.genie.vision.analyze_image(file_path, safe_prompt)
    
    def sanitize_prompt(self, prompt):
        """Sanitize user input prompts"""
        # Remove potentially harmful content
        forbidden_terms = ['system', 'admin', 'password', 'secret']
        
        for term in forbidden_terms:
            if term.lower() in prompt.lower():
                raise SecurityError(f"Forbidden term in prompt: {term}")
        
        return prompt[:1000]  # Limit prompt length

# Use secure wrapper
secure_genie = SecureComputerGenie()

try:
    result = secure_genie.safe_analyze_image(
        "~/Documents/image.jpg", 
        "Describe this image"
    )
    print(result.description)
except SecurityError as e:
    print(f"Security error: {e}")
```

---

## 📊 **Real-World Use Cases**

### **Document Processing Pipeline**
```python
def enterprise_document_processor():
    """Enterprise-grade document processing system"""
    
    class DocumentProcessor:
        def __init__(self):
            self.genie = ComputerGenie()
            self.processed_count = 0
            self.failed_count = 0
        
        def process_invoice_batch(self, invoice_folder):
            """Process batch of invoices"""
            results = []
            
            for invoice_file in Path(invoice_folder).glob("*.pdf"):
                try:
                    # Extract invoice data
                    invoice_data = self.genie.document.extract_invoice_data(
                        str(invoice_file)
                    )
                    
                    # Validate data
                    if self.validate_invoice_data(invoice_data):
                        # Store in database
                        self.store_invoice_data(invoice_data)
                        results.append({
                            'file': invoice_file.name,
                            'status': 'success',
                            'data': invoice_data
                        })
                        self.processed_count += 1
                    else:
                        results.append({
                            'file': invoice_file.name,
                            'status': 'validation_failed',
                            'data': invoice_data
                        })
                        self.failed_count += 1
                        
                except Exception as e:
                    results.append({
                        'file': invoice_file.name,
                        'status': 'processing_failed',
                        'error': str(e)
                    })
                    self.failed_count += 1
            
            return results
        
        def validate_invoice_data(self, data):
            """Validate extracted invoice data"""
            required_fields = ['invoice_number', 'total_amount', 'date']
            return all(hasattr(data, field) and getattr(data, field) for field in required_fields)
        
        def store_invoice_data(self, data):
            """Store invoice data in database"""
            # Implementation would connect to your database
            print(f"Storing invoice {data.invoice_number}: ${data.total_amount}")
        
        def generate_report(self):
            """Generate processing report"""
            total = self.processed_count + self.failed_count
            success_rate = self.processed_count / total if total > 0 else 0
            
            return {
                'total_processed': total,
                'successful': self.processed_count,
                'failed': self.failed_count,
                'success_rate': f"{success_rate:.2%}"
            }
    
    # Use the processor
    processor = DocumentProcessor()
    results = processor.process_invoice_batch("invoices/")
    report = processor.generate_report()
    
    print(f"Processing complete: {report}")
    return results

# Run enterprise processing
enterprise_document_processor()
```

### **Quality Assurance Automation**
```python
def automated_qa_testing():
    """Automated QA testing with Computer Genie"""
    
    class QAAutomation:
        def __init__(self):
            self.genie = ComputerGenie()
            self.test_results = []
        
        def test_login_flow(self, username, password):
            """Test complete login flow"""
            test_case = {
                'name': 'Login Flow Test',
                'steps': [],
                'status': 'running',
                'start_time': datetime.now()
            }
            
            try:
                # Step 1: Navigate to login page
                self.genie.automation.navigate_to("https://app.example.com/login")
                test_case['steps'].append({'step': 'Navigate to login', 'status': 'passed'})
                
                # Step 2: Verify login form exists
                if self.genie.vision.element_exists("login form"):
                    test_case['steps'].append({'step': 'Login form visible', 'status': 'passed'})
                else:
                    test_case['steps'].append({'step': 'Login form visible', 'status': 'failed'})
                    raise Exception("Login form not found")
                
                # Step 3: Fill credentials
                self.genie.automation.click("username field")
                self.genie.automation.type(username)
                self.genie.automation.click("password field")
                self.genie.automation.type(password)
                test_case['steps'].append({'step': 'Fill credentials', 'status': 'passed'})
                
                # Step 4: Submit form
                self.genie.automation.click("login button")
                test_case['steps'].append({'step': 'Submit form', 'status': 'passed'})
                
                # Step 5: Verify successful login
                if self.genie.automation.wait_for_element("dashboard", timeout=10):
                    test_case['steps'].append({'step': 'Login successful', 'status': 'passed'})
                    test_case['status'] = 'passed'
                else:
                    test_case['steps'].append({'step': 'Login successful', 'status': 'failed'})
                    test_case['status'] = 'failed'
                
            except Exception as e:
                test_case['status'] = 'failed'
                test_case['error'] = str(e)
            
            test_case['end_time'] = datetime.now()
            test_case['duration'] = (test_case['end_time'] - test_case['start_time']).total_seconds()
            
            self.test_results.append(test_case)
            return test_case
        
        def test_form_validation(self, form_selector):
            """Test form validation"""
            test_case = {
                'name': 'Form Validation Test',
                'steps': [],
                'status': 'running',
                'start_time': datetime.now()
            }
            
            try:
                # Test empty form submission
                self.genie.automation.click("submit button")
                
                if self.genie.vision.element_exists("validation error"):
                    test_case['steps'].append({'step': 'Empty form validation', 'status': 'passed'})
                else:
                    test_case['steps'].append({'step': 'Empty form validation', 'status': 'failed'})
                
                # Test invalid email format
                self.genie.automation.click("email field")
                self.genie.automation.type("invalid-email")
                self.genie.automation.click("submit button")
                
                if self.genie.vision.element_exists("email validation error"):
                    test_case['steps'].append({'step': 'Email validation', 'status': 'passed'})
                else:
                    test_case['steps'].append({'step': 'Email validation', 'status': 'failed'})
                
                test_case['status'] = 'passed'
                
            except Exception as e:
                test_case['status'] = 'failed'
                test_case['error'] = str(e)
            
            test_case['end_time'] = datetime.now()
            test_case['duration'] = (test_case['end_time'] - test_case['start_time']).total_seconds()
            
            self.test_results.append(test_case)
            return test_case
        
        def generate_test_report(self):
            """Generate comprehensive test report"""
            total_tests = len(self.test_results)
            passed_tests = sum(1 for test in self.test_results if test['status'] == 'passed')
            failed_tests = total_tests - passed_tests
            
            report = {
                'summary': {
                    'total_tests': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'pass_rate': f"{passed_tests / total_tests:.2%}" if total_tests > 0 else "0%"
                },
                'test_details': self.test_results,
                'generated_at': datetime.now()
            }
            
            return report
    
    # Run QA tests
    qa = QAAutomation()
    
    # Test login flow
    login_result = qa.test_login_flow("testuser", "testpass")
    print(f"Login test: {login_result['status']}")
    
    # Test form validation
    validation_result = qa.test_form_validation("contact-form")
    print(f"Validation test: {validation_result['status']}")
    
    # Generate report
    report = qa.generate_test_report()
    print(f"QA Report: {report['summary']}")
    
    return report

# Run QA automation
automated_qa_testing()
```

---

## 📞 **Support and Resources**

### **Getting Help**
```python
# Check Computer Genie version and status
print(f"Computer Genie version: {cg.__version__}")
print(f"API status: {genie.get_api_status()}")

# Get available models
models = genie.get_available_models()
print(f"Available models: {models}")

# Check system requirements
requirements = genie.check_system_requirements()
print(f"System check: {requirements}")

# Get usage statistics
stats = genie.get_usage_stats()
print(f"API usage: {stats}")
```

### **Debug Information**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get debug information
debug_info = genie.get_debug_info()
print(f"Debug info: {debug_info}")

# Test API connection
connection_test = genie.test_connection()
print(f"Connection test: {connection_test}")
```

---

## 📚 **Additional Resources**

- 📖 **Full Documentation**: [docs.computer-genie.com](https://docs.computer-genie.com)
- 🐍 **Python Package**: [pypi.org/project/computer-genie](https://pypi.org/project/computer-genie)
- 💬 **Community Forum**: [community.computer-genie.com](https://community.computer-genie.com)
- 🐛 **Issue Tracker**: [github.com/computer-genie/issues](https://github.com/computer-genie/issues)
- 📧 **Support Email**: support@abhishektech.com

---

**© 2024 Abhishek Technologies Pvt Ltd. All rights reserved.**

*For more Python examples and tutorials, visit [python.computer-genie.com](https://python.computer-genie.com)*