# 🔌 Computer Genie Enterprise API & Integration Guide

<div align="center">

![API Integration](https://img.shields.io/badge/Enterprise-API%20Integration-blue?style=for-the-badge&logo=api)

**Comprehensive API Reference & Integration Documentation**

</div>

---

## 📋 **Table of Contents**

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Core APIs](#core-apis)
4. [Integration Patterns](#integration-patterns)
5. [SDKs & Libraries](#sdks--libraries)
6. [Webhooks](#webhooks)
7. [Enterprise Integrations](#enterprise-integrations)
8. [Rate Limiting](#rate-limiting)
9. [Error Handling](#error-handling)
10. [Best Practices](#best-practices)

---

## 🎯 **API Overview**

### **API Architecture**
Computer Genie Enterprise provides a comprehensive RESTful API built on modern standards:

- **REST Architecture**: RESTful design principles
- **OpenAPI 3.0**: Complete API specification
- **JSON Format**: All requests and responses in JSON
- **HTTP/2**: High-performance protocol support
- **GraphQL**: Flexible data querying (Enterprise feature)
- **Real-time**: WebSocket support for live updates

### **Base URLs**
```yaml
# Environment URLs
production: https://api.computer-genie.com/v1
staging: https://staging-api.computer-genie.com/v1
sandbox: https://sandbox-api.computer-genie.com/v1

# Regional Endpoints
us_east: https://us-east-api.computer-genie.com/v1
eu_west: https://eu-west-api.computer-genie.com/v1
asia_pacific: https://ap-api.computer-genie.com/v1
```

### **API Versioning**
```http
# Header-based versioning (Recommended)
GET /documents
Accept: application/vnd.computer-genie.v1+json

# URL-based versioning
GET /v1/documents

# Query parameter versioning
GET /documents?version=1
```

### **Content Types**
```http
# Request Content Types
Content-Type: application/json
Content-Type: multipart/form-data  # For file uploads
Content-Type: application/x-www-form-urlencoded

# Response Content Types
Accept: application/json
Accept: application/vnd.computer-genie.v1+json
Accept: application/xml  # Limited support
```

---

## 🔐 **Authentication**

### **API Key Authentication**
```bash
# API Key in Header (Recommended)
curl -X GET "https://api.computer-genie.com/v1/documents" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"

# API Key in Query Parameter (Not recommended for production)
curl -X GET "https://api.computer-genie.com/v1/documents?api_key=YOUR_API_KEY"
```

### **OAuth 2.0 Authentication**
```bash
# Step 1: Get Authorization Code
https://auth.computer-genie.com/oauth/authorize?
  response_type=code&
  client_id=YOUR_CLIENT_ID&
  redirect_uri=YOUR_REDIRECT_URI&
  scope=documents:read documents:write&
  state=random_state_string

# Step 2: Exchange Code for Token
curl -X POST "https://auth.computer-genie.com/oauth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=authorization_code" \
  -d "code=AUTHORIZATION_CODE" \
  -d "client_id=YOUR_CLIENT_ID" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "redirect_uri=YOUR_REDIRECT_URI"

# Response
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "scope": "documents:read documents:write"
}

# Step 3: Use Access Token
curl -X GET "https://api.computer-genie.com/v1/documents" \
  -H "Authorization: Bearer ACCESS_TOKEN"
```

### **JWT Token Authentication**
```bash
# Login to get JWT token
curl -X POST "https://api.computer-genie.com/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@company.com",
    "password": "secure_password",
    "mfa_code": "123456"
  }'

# Response
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600,
  "token_type": "Bearer"
}

# Use JWT token
curl -X GET "https://api.computer-genie.com/v1/documents" \
  -H "Authorization: Bearer JWT_TOKEN"
```

### **API Scopes**
```yaml
# Available API Scopes
scopes:
  documents:read: "Read document information"
  documents:write: "Create and update documents"
  documents:delete: "Delete documents"
  
  processing:read: "Read processing status"
  processing:write: "Submit processing jobs"
  
  analytics:read: "Access analytics data"
  
  admin:users: "Manage users"
  admin:system: "System administration"
  
  webhooks:read: "Read webhook configurations"
  webhooks:write: "Manage webhooks"
```

---

## 🔧 **Core APIs**

### **Document Management API**

#### **Upload Document**
```bash
# Upload single document
curl -X POST "https://api.computer-genie.com/v1/documents" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@document.pdf" \
  -F "metadata={\"category\":\"invoice\",\"priority\":\"high\"}"

# Response
{
  "id": "doc_1234567890",
  "filename": "document.pdf",
  "size": 1048576,
  "mime_type": "application/pdf",
  "status": "uploaded",
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "category": "invoice",
    "priority": "high"
  },
  "urls": {
    "download": "https://api.computer-genie.com/v1/documents/doc_1234567890/download",
    "thumbnail": "https://api.computer-genie.com/v1/documents/doc_1234567890/thumbnail"
  }
}
```

#### **Batch Upload Documents**
```bash
# Upload multiple documents
curl -X POST "https://api.computer-genie.com/v1/documents/batch" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files[]=@document1.pdf" \
  -F "files[]=@document2.jpg" \
  -F "files[]=@document3.png" \
  -F "batch_metadata={\"project\":\"Q1_2024\",\"department\":\"finance\"}"

# Response
{
  "batch_id": "batch_1234567890",
  "status": "processing",
  "total_files": 3,
  "uploaded_files": 3,
  "failed_files": 0,
  "documents": [
    {
      "id": "doc_1234567891",
      "filename": "document1.pdf",
      "status": "uploaded"
    },
    {
      "id": "doc_1234567892",
      "filename": "document2.jpg",
      "status": "uploaded"
    },
    {
      "id": "doc_1234567893",
      "filename": "document3.png",
      "status": "uploaded"
    }
  ]
}
```

#### **Get Document Information**
```bash
# Get single document
curl -X GET "https://api.computer-genie.com/v1/documents/doc_1234567890" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response
{
  "id": "doc_1234567890",
  "filename": "document.pdf",
  "size": 1048576,
  "mime_type": "application/pdf",
  "status": "processed",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z",
  "metadata": {
    "category": "invoice",
    "priority": "high",
    "pages": 3,
    "language": "en"
  },
  "processing": {
    "ocr_completed": true,
    "cv_completed": true,
    "confidence_score": 0.95
  },
  "extracted_data": {
    "text": "Invoice #12345...",
    "entities": [
      {
        "type": "amount",
        "value": "$1,234.56",
        "confidence": 0.98
      }
    ]
  }
}
```

#### **List Documents**
```bash
# List documents with filtering
curl -X GET "https://api.computer-genie.com/v1/documents?status=processed&category=invoice&limit=50&offset=0" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response
{
  "documents": [
    {
      "id": "doc_1234567890",
      "filename": "invoice1.pdf",
      "status": "processed",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 50,
    "offset": 0,
    "has_more": true
  },
  "filters": {
    "status": "processed",
    "category": "invoice"
  }
}
```

### **Computer Vision API**

#### **Image Analysis**
```bash
# Analyze image
curl -X POST "https://api.computer-genie.com/v1/vision/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@image.jpg" \
  -F "features=[\"objects\",\"text\",\"faces\",\"landmarks\"]"

# Response
{
  "analysis_id": "analysis_1234567890",
  "image_info": {
    "width": 1920,
    "height": 1080,
    "format": "JPEG",
    "size": 512000
  },
  "features": {
    "objects": [
      {
        "name": "person",
        "confidence": 0.95,
        "bounding_box": {
          "x": 100,
          "y": 150,
          "width": 200,
          "height": 300
        }
      }
    ],
    "text": [
      {
        "text": "Computer Genie",
        "confidence": 0.98,
        "bounding_box": {
          "x": 50,
          "y": 50,
          "width": 150,
          "height": 30
        }
      }
    ],
    "faces": [
      {
        "confidence": 0.92,
        "emotions": {
          "happy": 0.8,
          "neutral": 0.2
        },
        "age_range": {
          "min": 25,
          "max": 35
        },
        "bounding_box": {
          "x": 120,
          "y": 160,
          "width": 80,
          "height": 100
        }
      }
    ]
  }
}
```

#### **Custom Model Inference**
```bash
# Use custom trained model
curl -X POST "https://api.computer-genie.com/v1/vision/models/custom_model_123/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@custom_image.jpg" \
  -F "parameters={\"threshold\":0.8,\"max_results\":10}"

# Response
{
  "model_id": "custom_model_123",
  "model_version": "1.2.0",
  "predictions": [
    {
      "class": "defect_type_a",
      "confidence": 0.92,
      "bounding_box": {
        "x": 200,
        "y": 300,
        "width": 50,
        "height": 75
      }
    }
  ],
  "processing_time": 0.15
}
```

### **OCR Processing API**

#### **Text Extraction**
```bash
# Extract text from document
curl -X POST "https://api.computer-genie.com/v1/ocr/extract" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "document=@document.pdf" \
  -F "options={\"language\":\"en\",\"output_format\":\"structured\"}"

# Response
{
  "extraction_id": "ocr_1234567890",
  "status": "completed",
  "pages": [
    {
      "page_number": 1,
      "text": "Invoice\nDate: 2024-01-15\nAmount: $1,234.56",
      "confidence": 0.96,
      "words": [
        {
          "text": "Invoice",
          "confidence": 0.98,
          "bounding_box": {
            "x": 100,
            "y": 50,
            "width": 80,
            "height": 20
          }
        }
      ],
      "lines": [
        {
          "text": "Invoice",
          "confidence": 0.98,
          "bounding_box": {
            "x": 100,
            "y": 50,
            "width": 80,
            "height": 20
          }
        }
      ]
    }
  ],
  "extracted_entities": [
    {
      "type": "date",
      "value": "2024-01-15",
      "confidence": 0.95,
      "page": 1
    },
    {
      "type": "amount",
      "value": "$1,234.56",
      "confidence": 0.97,
      "page": 1
    }
  ]
}
```

#### **Form Processing**
```bash
# Process structured forms
curl -X POST "https://api.computer-genie.com/v1/ocr/forms" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "form=@tax_form.pdf" \
  -F "form_type=tax_1040" \
  -F "options={\"validate_fields\":true}"

# Response
{
  "form_id": "form_1234567890",
  "form_type": "tax_1040",
  "status": "processed",
  "fields": [
    {
      "field_name": "taxpayer_name",
      "value": "John Doe",
      "confidence": 0.98,
      "validation": {
        "is_valid": true,
        "format": "name"
      }
    },
    {
      "field_name": "ssn",
      "value": "XXX-XX-1234",
      "confidence": 0.95,
      "validation": {
        "is_valid": true,
        "format": "ssn"
      }
    }
  ],
  "validation_summary": {
    "total_fields": 25,
    "valid_fields": 24,
    "invalid_fields": 1,
    "overall_confidence": 0.94
  }
}
```

### **Workflow Automation API**

#### **Create Workflow**
```bash
# Create automation workflow
curl -X POST "https://api.computer-genie.com/v1/workflows" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Invoice Processing Workflow",
    "description": "Automated invoice processing and approval",
    "trigger": {
      "type": "document_upload",
      "conditions": {
        "category": "invoice",
        "file_type": ["pdf", "jpg", "png"]
      }
    },
    "steps": [
      {
        "id": "ocr_extraction",
        "type": "ocr",
        "parameters": {
          "language": "en",
          "extract_entities": true
        }
      },
      {
        "id": "data_validation",
        "type": "validation",
        "parameters": {
          "required_fields": ["amount", "date", "vendor"]
        }
      },
      {
        "id": "approval_routing",
        "type": "human_review",
        "parameters": {
          "approver_role": "finance_manager",
          "timeout": "24h"
        }
      }
    ]
  }'

# Response
{
  "workflow_id": "workflow_1234567890",
  "name": "Invoice Processing Workflow",
  "status": "active",
  "created_at": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "steps": [
    {
      "id": "ocr_extraction",
      "type": "ocr",
      "status": "configured"
    },
    {
      "id": "data_validation",
      "type": "validation",
      "status": "configured"
    },
    {
      "id": "approval_routing",
      "type": "human_review",
      "status": "configured"
    }
  ]
}
```

#### **Execute Workflow**
```bash
# Execute workflow manually
curl -X POST "https://api.computer-genie.com/v1/workflows/workflow_1234567890/execute" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "document_id": "doc_1234567890"
    },
    "parameters": {
      "priority": "high",
      "notify_completion": true
    }
  }'

# Response
{
  "execution_id": "exec_1234567890",
  "workflow_id": "workflow_1234567890",
  "status": "running",
  "started_at": "2024-01-15T10:35:00Z",
  "current_step": "ocr_extraction",
  "progress": {
    "completed_steps": 0,
    "total_steps": 3,
    "percentage": 0
  }
}
```

### **Analytics API**

#### **Get Processing Statistics**
```bash
# Get processing statistics
curl -X GET "https://api.computer-genie.com/v1/analytics/processing?start_date=2024-01-01&end_date=2024-01-31&granularity=daily" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response
{
  "period": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "granularity": "daily"
  },
  "metrics": {
    "total_documents": 1500,
    "processed_documents": 1450,
    "failed_documents": 50,
    "average_processing_time": 2.5,
    "accuracy_rate": 0.96
  },
  "daily_stats": [
    {
      "date": "2024-01-01",
      "documents_processed": 45,
      "average_processing_time": 2.3,
      "accuracy_rate": 0.97
    }
  ]
}
```

#### **Custom Analytics Query**
```bash
# Custom analytics query
curl -X POST "https://api.computer-genie.com/v1/analytics/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "metrics": ["document_count", "processing_time", "accuracy"],
      "dimensions": ["category", "department"],
      "filters": {
        "date_range": {
          "start": "2024-01-01",
          "end": "2024-01-31"
        },
        "status": "processed"
      },
      "aggregation": "sum"
    }
  }'

# Response
{
  "query_id": "query_1234567890",
  "results": [
    {
      "category": "invoice",
      "department": "finance",
      "document_count": 500,
      "avg_processing_time": 2.1,
      "accuracy": 0.98
    },
    {
      "category": "contract",
      "department": "legal",
      "document_count": 200,
      "avg_processing_time": 3.5,
      "accuracy": 0.94
    }
  ],
  "total_results": 2,
  "execution_time": 0.25
}
```

---

## 🔄 **Integration Patterns**

### **Synchronous Integration**
```python
# Python example - Synchronous processing
import requests
import time

def process_document_sync(file_path, api_key):
    # Upload document
    with open(file_path, 'rb') as file:
        response = requests.post(
            'https://api.computer-genie.com/v1/documents',
            headers={'Authorization': f'Bearer {api_key}'},
            files={'file': file},
            data={'metadata': '{"priority": "high"}'}
        )
    
    document_id = response.json()['id']
    
    # Poll for completion
    while True:
        status_response = requests.get(
            f'https://api.computer-genie.com/v1/documents/{document_id}',
            headers={'Authorization': f'Bearer {api_key}'}
        )
        
        status = status_response.json()['status']
        if status == 'processed':
            return status_response.json()
        elif status == 'failed':
            raise Exception('Processing failed')
        
        time.sleep(2)  # Wait 2 seconds before next check
```

### **Asynchronous Integration with Webhooks**
```python
# Python example - Asynchronous processing with webhooks
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

def process_document_async(file_path, api_key, webhook_url):
    # Upload document with webhook
    with open(file_path, 'rb') as file:
        response = requests.post(
            'https://api.computer-genie.com/v1/documents',
            headers={'Authorization': f'Bearer {api_key}'},
            files={'file': file},
            data={
                'metadata': '{"priority": "high"}',
                'webhook_url': webhook_url,
                'webhook_events': '["processing_completed", "processing_failed"]'
            }
        )
    
    return response.json()['id']

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    event = request.json
    
    if event['event_type'] == 'processing_completed':
        document_id = event['data']['document_id']
        # Handle successful processing
        print(f"Document {document_id} processed successfully")
    
    elif event['event_type'] == 'processing_failed':
        document_id = event['data']['document_id']
        error = event['data']['error']
        # Handle processing failure
        print(f"Document {document_id} failed: {error}")
    
    return jsonify({'status': 'received'})
```

### **Batch Processing Integration**
```python
# Python example - Batch processing
import requests
import json

def process_batch(file_paths, api_key):
    # Prepare files for batch upload
    files = []
    for i, file_path in enumerate(file_paths):
        files.append(('files[]', open(file_path, 'rb')))
    
    # Upload batch
    response = requests.post(
        'https://api.computer-genie.com/v1/documents/batch',
        headers={'Authorization': f'Bearer {api_key}'},
        files=files,
        data={'batch_metadata': json.dumps({'project': 'Q1_2024'})}
    )
    
    batch_id = response.json()['batch_id']
    
    # Monitor batch progress
    while True:
        status_response = requests.get(
            f'https://api.computer-genie.com/v1/batches/{batch_id}',
            headers={'Authorization': f'Bearer {api_key}'}
        )
        
        batch_status = status_response.json()
        if batch_status['status'] == 'completed':
            return batch_status
        elif batch_status['status'] == 'failed':
            raise Exception('Batch processing failed')
        
        time.sleep(5)  # Check every 5 seconds
    
    # Close files
    for _, file in files:
        file.close()
```

---

## 📚 **SDKs & Libraries**

### **Python SDK**
```bash
# Installation
pip install computer-genie-sdk

# Usage
from computer_genie import ComputerGenieClient

# Initialize client
client = ComputerGenieClient(
    api_key='your_api_key',
    base_url='https://api.computer-genie.com/v1'
)

# Upload and process document
document = client.documents.upload(
    file_path='document.pdf',
    metadata={'category': 'invoice'}
)

# Wait for processing
result = client.documents.wait_for_processing(document.id)

# Get extracted data
extracted_data = result.extracted_data
```

### **JavaScript/Node.js SDK**
```bash
# Installation
npm install computer-genie-sdk

# Usage
const ComputerGenie = require('computer-genie-sdk');

// Initialize client
const client = new ComputerGenie({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.computer-genie.com/v1'
});

// Upload document
const document = await client.documents.upload({
  filePath: 'document.pdf',
  metadata: { category: 'invoice' }
});

// Process with callback
client.documents.process(document.id, {
  onComplete: (result) => {
    console.log('Processing completed:', result);
  },
  onError: (error) => {
    console.error('Processing failed:', error);
  }
});
```

### **Java SDK**
```java
// Maven dependency
<dependency>
    <groupId>com.abhishektech</groupId>
    <artifactId>computer-genie-sdk</artifactId>
    <version>1.0.0</version>
</dependency>

// Usage
import com.abhishektech.computergenie.ComputerGenieClient;
import com.abhishektech.computergenie.models.Document;

// Initialize client
ComputerGenieClient client = new ComputerGenieClient.Builder()
    .apiKey("your_api_key")
    .baseUrl("https://api.computer-genie.com/v1")
    .build();

// Upload document
Document document = client.documents()
    .upload(new File("document.pdf"))
    .metadata(Map.of("category", "invoice"))
    .execute();

// Wait for processing
Document result = client.documents()
    .waitForProcessing(document.getId())
    .timeout(Duration.ofMinutes(5))
    .execute();
```

### **.NET SDK**
```csharp
// NuGet package
Install-Package ComputerGenie.SDK

// Usage
using ComputerGenie.SDK;

// Initialize client
var client = new ComputerGenieClient(new ComputerGenieOptions
{
    ApiKey = "your_api_key",
    BaseUrl = "https://api.computer-genie.com/v1"
});

// Upload document
var document = await client.Documents.UploadAsync(
    filePath: "document.pdf",
    metadata: new { category = "invoice" }
);

// Process document
var result = await client.Documents.ProcessAsync(document.Id);
```

---

## 🔔 **Webhooks**

### **Webhook Configuration**
```bash
# Create webhook endpoint
curl -X POST "https://api.computer-genie.com/v1/webhooks" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/webhooks/computer-genie",
    "events": [
      "document.uploaded",
      "document.processed",
      "document.failed",
      "workflow.completed",
      "batch.completed"
    ],
    "secret": "your_webhook_secret",
    "active": true
  }'

# Response
{
  "webhook_id": "webhook_1234567890",
  "url": "https://your-app.com/webhooks/computer-genie",
  "events": [
    "document.uploaded",
    "document.processed",
    "document.failed",
    "workflow.completed",
    "batch.completed"
  ],
  "secret": "your_webhook_secret",
  "active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### **Webhook Event Examples**

#### **Document Processed Event**
```json
{
  "event_id": "event_1234567890",
  "event_type": "document.processed",
  "timestamp": "2024-01-15T10:35:00Z",
  "data": {
    "document_id": "doc_1234567890",
    "filename": "invoice.pdf",
    "status": "processed",
    "processing_time": 2.5,
    "confidence_score": 0.96,
    "extracted_data": {
      "text": "Invoice #12345...",
      "entities": [
        {
          "type": "amount",
          "value": "$1,234.56",
          "confidence": 0.98
        }
      ]
    }
  }
}
```

#### **Workflow Completed Event**
```json
{
  "event_id": "event_1234567891",
  "event_type": "workflow.completed",
  "timestamp": "2024-01-15T10:40:00Z",
  "data": {
    "workflow_id": "workflow_1234567890",
    "execution_id": "exec_1234567890",
    "status": "completed",
    "duration": 300,
    "steps_completed": 3,
    "output_data": {
      "approval_status": "approved",
      "approver": "finance_manager@company.com"
    }
  }
}
```

### **Webhook Security**
```python
# Python example - Webhook signature verification
import hmac
import hashlib

def verify_webhook_signature(payload, signature, secret):
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(
        f"sha256={expected_signature}",
        signature
    )

# Flask webhook handler with verification
@app.route('/webhook', methods=['POST'])
def handle_webhook():
    signature = request.headers.get('X-Computer-Genie-Signature')
    payload = request.get_data(as_text=True)
    
    if not verify_webhook_signature(payload, signature, WEBHOOK_SECRET):
        return jsonify({'error': 'Invalid signature'}), 401
    
    event = request.json
    # Process event...
    
    return jsonify({'status': 'received'})
```

---

## 🏢 **Enterprise Integrations**

### **Salesforce Integration**
```apex
// Apex class for Salesforce integration
public class ComputerGenieIntegration {
    private static final String API_BASE_URL = 'https://api.computer-genie.com/v1';
    private static final String API_KEY = 'your_api_key';
    
    @future(callout=true)
    public static void processAttachment(Id attachmentId) {
        Attachment attachment = [SELECT Name, Body FROM Attachment WHERE Id = :attachmentId];
        
        HttpRequest req = new HttpRequest();
        req.setEndpoint(API_BASE_URL + '/documents');
        req.setMethod('POST');
        req.setHeader('Authorization', 'Bearer ' + API_KEY);
        
        // Create multipart form data
        String boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW';
        req.setHeader('Content-Type', 'multipart/form-data; boundary=' + boundary);
        
        String body = '--' + boundary + '\r\n';
        body += 'Content-Disposition: form-data; name="file"; filename="' + attachment.Name + '"\r\n';
        body += 'Content-Type: application/octet-stream\r\n\r\n';
        body += EncodingUtil.base64Encode(attachment.Body) + '\r\n';
        body += '--' + boundary + '--';
        
        req.setBody(body);
        
        Http http = new Http();
        HttpResponse res = http.send(req);
        
        if (res.getStatusCode() == 200) {
            // Handle successful upload
            Map<String, Object> response = (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
            String documentId = (String) response.get('id');
            
            // Store document ID for future reference
            // Create custom object record or update existing record
        }
    }
}
```

### **SAP Integration**
```abap
* ABAP code for SAP integration
CLASS zcl_computer_genie_integration DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    METHODS: process_document
      IMPORTING
        iv_document_path TYPE string
        iv_category TYPE string
      RETURNING
        VALUE(rv_document_id) TYPE string.

  PRIVATE SECTION.
    CONSTANTS: c_api_url TYPE string VALUE 'https://api.computer-genie.com/v1',
               c_api_key TYPE string VALUE 'your_api_key'.

ENDCLASS.

CLASS zcl_computer_genie_integration IMPLEMENTATION.
  METHOD process_document.
    DATA: lo_http_client TYPE REF TO if_http_client,
          lv_response TYPE string,
          lv_status_code TYPE i.

    " Create HTTP client
    CALL METHOD cl_http_client=>create_by_url
      EXPORTING
        url    = |{ c_api_url }/documents|
      IMPORTING
        client = lo_http_client.

    " Set headers
    lo_http_client->request->set_header_field(
      name  = 'Authorization'
      value = |Bearer { c_api_key }|
    ).

    " Set method
    lo_http_client->request->set_method( 'POST' ).

    " Add file data (simplified)
    " In real implementation, you would read the file and create multipart form data

    " Send request
    CALL METHOD lo_http_client->send
      EXCEPTIONS
        http_communication_failure = 1
        http_invalid_state         = 2
        http_processing_failed     = 3.

    " Get response
    CALL METHOD lo_http_client->receive
      EXCEPTIONS
        http_communication_failure = 1
        http_invalid_state         = 2
        http_processing_failed     = 3.

    lv_response = lo_http_client->response->get_cdata( ).
    lv_status_code = lo_http_client->response->get_status( )-code.

    " Parse response and extract document ID
    " Implementation depends on your JSON parsing library

    lo_http_client->close( ).
  ENDMETHOD.
ENDCLASS.
```

### **Microsoft Dynamics Integration**
```csharp
// C# code for Microsoft Dynamics integration
using System;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Xrm.Sdk;
using Microsoft.Xrm.Sdk.Query;

public class ComputerGenieIntegration
{
    private readonly IOrganizationService _service;
    private readonly HttpClient _httpClient;
    private const string ApiBaseUrl = "https://api.computer-genie.com/v1";
    private const string ApiKey = "your_api_key";

    public ComputerGenieIntegration(IOrganizationService service)
    {
        _service = service;
        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {ApiKey}");
    }

    public async Task<string> ProcessDocumentAsync(Guid annotationId)
    {
        // Retrieve attachment from Dynamics
        var annotation = _service.Retrieve("annotation", annotationId, 
            new ColumnSet("filename", "documentbody", "mimetype"));

        var fileName = annotation.GetAttributeValue<string>("filename");
        var documentBody = annotation.GetAttributeValue<string>("documentbody");
        var mimeType = annotation.GetAttributeValue<string>("mimetype");

        // Convert base64 to byte array
        var fileBytes = Convert.FromBase64String(documentBody);

        // Create multipart form content
        using var content = new MultipartFormDataContent();
        using var fileContent = new ByteArrayContent(fileBytes);
        fileContent.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(mimeType);
        content.Add(fileContent, "file", fileName);

        // Send to Computer Genie API
        var response = await _httpClient.PostAsync($"{ApiBaseUrl}/documents", content);
        var responseContent = await response.Content.ReadAsStringAsync();

        if (response.IsSuccessStatusCode)
        {
            // Parse response and return document ID
            dynamic result = Newtonsoft.Json.JsonConvert.DeserializeObject(responseContent);
            return result.id;
        }

        throw new Exception($"API call failed: {responseContent}");
    }
}
```

---

## ⚡ **Rate Limiting**

### **Rate Limit Headers**
```http
# Response headers for rate limiting
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642262400
X-RateLimit-Window: 3600
Retry-After: 3600
```

### **Rate Limit Tiers**
```yaml
# Rate limiting by plan
rate_limits:
  free_tier:
    requests_per_hour: 100
    requests_per_day: 1000
    concurrent_requests: 5
    
  professional:
    requests_per_hour: 1000
    requests_per_day: 10000
    concurrent_requests: 20
    
  enterprise:
    requests_per_hour: 10000
    requests_per_day: 100000
    concurrent_requests: 100
    
  custom:
    requests_per_hour: "unlimited"
    requests_per_day: "unlimited"
    concurrent_requests: 500
```

### **Rate Limit Handling**
```python
# Python example - Rate limit handling
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class RateLimitAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        
        if response.status_code == 429:  # Too Many Requests
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limit exceeded. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return self.send(request, **kwargs)
        
        return response

# Configure session with rate limit handling
session = requests.Session()
session.mount('https://', RateLimitAdapter())

# Use session for API calls
response = session.get(
    'https://api.computer-genie.com/v1/documents',
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)
```

---

## ❌ **Error Handling**

### **Error Response Format**
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is invalid",
    "details": "Missing required field: file",
    "request_id": "req_1234567890",
    "timestamp": "2024-01-15T10:30:00Z",
    "documentation_url": "https://docs.computer-genie.com/errors/invalid-request"
  }
}
```

### **Error Codes**
```yaml
# Common error codes
error_codes:
  authentication:
    INVALID_API_KEY: "The provided API key is invalid"
    EXPIRED_TOKEN: "The access token has expired"
    INSUFFICIENT_PERMISSIONS: "Insufficient permissions for this operation"
    
  validation:
    INVALID_REQUEST: "The request is malformed or invalid"
    MISSING_REQUIRED_FIELD: "A required field is missing"
    INVALID_FILE_TYPE: "The file type is not supported"
    FILE_TOO_LARGE: "The file size exceeds the maximum limit"
    
  processing:
    PROCESSING_FAILED: "Document processing failed"
    UNSUPPORTED_FORMAT: "The document format is not supported"
    CORRUPTED_FILE: "The file appears to be corrupted"
    
  rate_limiting:
    RATE_LIMIT_EXCEEDED: "Rate limit exceeded"
    QUOTA_EXCEEDED: "Monthly quota exceeded"
    
  server:
    INTERNAL_ERROR: "An internal server error occurred"
    SERVICE_UNAVAILABLE: "The service is temporarily unavailable"
    TIMEOUT: "The request timed out"
```

### **Error Handling Best Practices**
```python
# Python example - Comprehensive error handling
import requests
from requests.exceptions import RequestException

class ComputerGenieAPIError(Exception):
    def __init__(self, error_code, message, details=None):
        self.error_code = error_code
        self.message = message
        self.details = details
        super().__init__(f"{error_code}: {message}")

def make_api_request(url, headers, **kwargs):
    try:
        response = requests.request(**kwargs, url=url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        
        elif response.status_code == 401:
            raise ComputerGenieAPIError("AUTHENTICATION_FAILED", "Invalid credentials")
        
        elif response.status_code == 403:
            raise ComputerGenieAPIError("INSUFFICIENT_PERMISSIONS", "Access denied")
        
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 60)
            raise ComputerGenieAPIError("RATE_LIMIT_EXCEEDED", f"Retry after {retry_after} seconds")
        
        elif response.status_code >= 400:
            error_data = response.json().get('error', {})
            raise ComputerGenieAPIError(
                error_data.get('code', 'UNKNOWN_ERROR'),
                error_data.get('message', 'An error occurred'),
                error_data.get('details')
            )
        
        else:
            response.raise_for_status()
            
    except RequestException as e:
        raise ComputerGenieAPIError("NETWORK_ERROR", f"Network error: {str(e)}")

# Usage with error handling
try:
    result = make_api_request(
        url='https://api.computer-genie.com/v1/documents',
        headers={'Authorization': 'Bearer YOUR_TOKEN'},
        method='GET'
    )
    print("Success:", result)
    
except ComputerGenieAPIError as e:
    print(f"API Error: {e.error_code} - {e.message}")
    if e.details:
        print(f"Details: {e.details}")
```

---

## 🎯 **Best Practices**

### **API Usage Best Practices**

#### **1. Authentication Security**
```python
# ✅ Good: Store API keys securely
import os
api_key = os.environ.get('COMPUTER_GENIE_API_KEY')

# ❌ Bad: Hardcode API keys
api_key = 'sk-1234567890abcdef'  # Never do this!
```

#### **2. Request Optimization**
```python
# ✅ Good: Batch requests when possible
documents = client.documents.batch_upload([
    'file1.pdf', 'file2.pdf', 'file3.pdf'
])

# ❌ Bad: Individual requests for each file
for file in files:
    client.documents.upload(file)  # Inefficient
```

#### **3. Error Handling**
```python
# ✅ Good: Comprehensive error handling
try:
    result = client.documents.process(document_id)
except ComputerGenieAPIError as e:
    if e.error_code == 'RATE_LIMIT_EXCEEDED':
        time.sleep(60)  # Wait and retry
        result = client.documents.process(document_id)
    else:
        logger.error(f"Processing failed: {e}")
        raise

# ❌ Bad: No error handling
result = client.documents.process(document_id)  # May fail silently
```

#### **4. Asynchronous Processing**
```python
# ✅ Good: Use webhooks for long-running operations
client.documents.upload(
    file_path='large_document.pdf',
    webhook_url='https://your-app.com/webhook',
    webhook_events=['processing_completed']
)

# ❌ Bad: Polling for completion
while True:
    status = client.documents.get_status(document_id)
    if status == 'completed':
        break
    time.sleep(1)  # Inefficient polling
```

### **Performance Optimization**

#### **1. Connection Pooling**
```python
# ✅ Good: Use connection pooling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
adapter = HTTPAdapter(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=Retry(total=3, backoff_factor=0.3)
)
session.mount('https://', adapter)
```

#### **2. Caching**
```python
# ✅ Good: Cache frequently accessed data
from functools import lru_cache

@lru_cache(maxsize=100)
def get_document_metadata(document_id):
    return client.documents.get(document_id)
```

#### **3. Pagination**
```python
# ✅ Good: Use pagination for large datasets
def get_all_documents():
    documents = []
    offset = 0
    limit = 100
    
    while True:
        batch = client.documents.list(offset=offset, limit=limit)
        documents.extend(batch['documents'])
        
        if not batch['has_more']:
            break
            
        offset += limit
    
    return documents
```

### **Security Best Practices**

#### **1. Input Validation**
```python
# ✅ Good: Validate file types and sizes
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.png', '.tiff'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def validate_file(file_path):
    if not os.path.exists(file_path):
        raise ValueError("File does not exist")
    
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        raise ValueError("File too large")
```

#### **2. Data Sanitization**
```python
# ✅ Good: Sanitize metadata
import re

def sanitize_metadata(metadata):
    sanitized = {}
    for key, value in metadata.items():
        # Remove potentially dangerous characters
        clean_key = re.sub(r'[^\w\-_]', '', key)
        clean_value = str(value)[:1000]  # Limit length
        sanitized[clean_key] = clean_value
    return sanitized
```

---

## 📞 **API Support**

### **Developer Resources**
- 📚 **API Documentation**: [docs.computer-genie.com/api](https://docs.computer-genie.com/api)
- 🔧 **Interactive API Explorer**: [api-explorer.computer-genie.com](https://api-explorer.computer-genie.com)
- 💬 **Developer Forum**: [forum.computer-genie.com](https://forum.computer-genie.com)
- 📖 **Code Examples**: [github.com/abhishektech/computer-genie-examples](https://github.com/abhishektech/computer-genie-examples)

### **Support Channels**
- 📧 **API Support**: api-support@abhishektech.com
- 💬 **Live Chat**: Available in developer portal
- 📞 **Phone Support**: +91-XXX-XXX-XXXX (Enterprise customers)
- 🎫 **Support Tickets**: [support.computer-genie.com](https://support.computer-genie.com)

### **SLA & Response Times**
| Support Tier | Response Time | Availability |
|--------------|---------------|--------------|
| Community | 48 hours | Business hours |
| Professional | 24 hours | Business hours |
| Enterprise | 4 hours | 24/7 |
| Critical | 1 hour | 24/7 |

---

**© 2024 Abhishek Technologies Pvt Ltd. All rights reserved.**

*This documentation is subject to change. Please refer to the latest version at [docs.computer-genie.com](https://docs.computer-genie.com)*