# 🔗 Computer Genie Integration Examples

<div align="center">

![Integrations](https://img.shields.io/badge/Integrations-Examples-purple?style=for-the-badge&logo=api)

**Platform & Service Integration Examples**

</div>

---

## 📋 **Table of Contents**

1. [Web Frameworks](#web-frameworks)
2. [Cloud Platforms](#cloud-platforms)
3. [Enterprise Systems](#enterprise-systems)
4. [Development Tools](#development-tools)
5. [Database Integrations](#database-integrations)
6. [API Integrations](#api-integrations)
7. [Automation Platforms](#automation-platforms)
8. [Monitoring & Analytics](#monitoring--analytics)

---

## 🌐 **Web Frameworks**

### **Flask Integration**
```python
from flask import Flask, request, jsonify, render_template
from computer_genie import ComputerGenie
import base64
import io
from PIL import Image

app = Flask(__name__)
genie = ComputerGenie()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        prompt = request.form.get('prompt', 'Describe this image')
        
        # Save temporary file
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        # Analyze with Computer Genie
        result = genie.vision.analyze_image(temp_path, prompt)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'description': result.description,
            'confidence': result.confidence,
            'metadata': result.metadata
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/screenshot', methods=['POST'])
def take_screenshot():
    """Take and analyze screenshot"""
    try:
        prompt = request.json.get('prompt', 'Describe the screen')
        
        # Take screenshot
        screenshot = genie.vision.screenshot()
        
        # Analyze screenshot
        analysis = genie.vision.analyze_image(screenshot.path, prompt)
        
        # Convert image to base64 for web display
        with open(screenshot.path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        return jsonify({
            'success': True,
            'image_data': f"data:image/png;base64,{img_data}",
            'analysis': analysis.description,
            'confidence': analysis.confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/automate', methods=['POST'])
def automate_task():
    """Automate UI task"""
    try:
        task = request.json.get('task')
        
        if not task:
            return jsonify({'error': 'No task specified'}), 400
        
        # Execute automation task
        result = genie.automation.execute(task)
        
        return jsonify({
            'success': True,
            'result': result.description,
            'status': result.status
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### **FastAPI Integration**
```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from computer_genie import ComputerGenie
import tempfile
import os

app = FastAPI(title="Computer Genie API", version="1.0.0")
genie = ComputerGenie()

class AnalysisRequest(BaseModel):
    prompt: str = "Describe this image"

class AutomationRequest(BaseModel):
    task: str
    timeout: int = 30

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...), request: AnalysisRequest = None):
    """Analyze uploaded image with Computer Genie"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Analyze with Computer Genie
        prompt = request.prompt if request else "Describe this image"
        result = genie.vision.analyze_image(tmp_path, prompt)
        
        # Clean up
        os.unlink(tmp_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "description": result.description,
            "confidence": result.confidence,
            "metadata": result.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/screenshot/")
async def take_screenshot(request: AnalysisRequest = None):
    """Take and analyze screenshot"""
    
    try:
        # Take screenshot
        screenshot = genie.vision.screenshot()
        
        # Analyze if prompt provided
        if request and request.prompt:
            analysis = genie.vision.analyze_image(screenshot.path, request.prompt)
            return {
                "success": True,
                "screenshot_path": screenshot.path,
                "analysis": analysis.description,
                "confidence": analysis.confidence
            }
        else:
            return {
                "success": True,
                "screenshot_path": screenshot.path,
                "message": "Screenshot taken successfully"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/automate/")
async def automate_task(request: AutomationRequest):
    """Execute automation task"""
    
    try:
        result = genie.automation.execute(request.task, timeout=request.timeout)
        
        return {
            "success": True,
            "task": request.task,
            "result": result.description,
            "status": result.status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    try:
        status = genie.get_api_status()
        return {
            "status": "healthy",
            "computer_genie_status": status,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### **Django Integration**
```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from computer_genie import ComputerGenie
import json
import tempfile
import os

genie = ComputerGenie()

@csrf_exempt
@require_http_methods(["POST"])
def analyze_image_view(request):
    """Django view for image analysis"""
    
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        prompt = request.POST.get('prompt', 'Describe this image')
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            for chunk in image_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Analyze with Computer Genie
        result = genie.vision.analyze_image(tmp_path, prompt)
        
        # Clean up
        os.unlink(tmp_path)
        
        return JsonResponse({
            'success': True,
            'description': result.description,
            'confidence': result.confidence,
            'metadata': result.metadata
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def automation_view(request):
    """Django view for automation tasks"""
    
    try:
        data = json.loads(request.body)
        task = data.get('task')
        
        if not task:
            return JsonResponse({'error': 'No task specified'}, status=400)
        
        result = genie.automation.execute(task)
        
        return JsonResponse({
            'success': True,
            'result': result.description,
            'status': result.status
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/analyze-image/', views.analyze_image_view, name='analyze_image'),
    path('api/automate/', views.automation_view, name='automate'),
]

# models.py
from django.db import models

class AnalysisResult(models.Model):
    """Store analysis results"""
    image_name = models.CharField(max_length=255)
    prompt = models.TextField()
    description = models.TextField()
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class AutomationTask(models.Model):
    """Store automation task results"""
    task_description = models.TextField()
    result = models.TextField()
    status = models.CharField(max_length=50)
    executed_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-executed_at']
```

---

## ☁️ **Cloud Platforms**

### **AWS Lambda Integration**
```python
import json
import boto3
import base64
from computer_genie import ComputerGenie
import tempfile
import os

def lambda_handler(event, context):
    """AWS Lambda function for Computer Genie processing"""
    
    try:
        # Initialize Computer Genie
        genie = ComputerGenie()
        
        # Parse request
        body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        operation = body.get('operation')
        
        if operation == 'analyze_image':
            return handle_image_analysis(genie, body)
        elif operation == 'screenshot':
            return handle_screenshot(genie, body)
        elif operation == 'automate':
            return handle_automation(genie, body)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid operation'})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def handle_image_analysis(genie, body):
    """Handle image analysis request"""
    
    # Decode base64 image
    image_data = base64.b64decode(body['image_data'])
    prompt = body.get('prompt', 'Describe this image')
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(image_data)
        tmp_path = tmp_file.name
    
    try:
        # Analyze image
        result = genie.vision.analyze_image(tmp_path, prompt)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'description': result.description,
                'confidence': result.confidence
            })
        }
    finally:
        # Clean up
        os.unlink(tmp_path)

def handle_screenshot(genie, body):
    """Handle screenshot request"""
    
    prompt = body.get('prompt', 'Describe the screen')
    
    # Take screenshot
    screenshot = genie.vision.screenshot()
    
    # Analyze if requested
    if prompt:
        analysis = genie.vision.analyze_image(screenshot.path, prompt)
        
        # Upload to S3 (optional)
        s3_url = upload_to_s3(screenshot.path, 'screenshots-bucket')
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'screenshot_url': s3_url,
                'analysis': analysis.description,
                'confidence': analysis.confidence
            })
        }

def upload_to_s3(file_path, bucket_name):
    """Upload file to S3 and return URL"""
    s3_client = boto3.client('s3')
    
    file_name = os.path.basename(file_path)
    s3_client.upload_file(file_path, bucket_name, file_name)
    
    return f"https://{bucket_name}.s3.amazonaws.com/{file_name}"

# requirements.txt for Lambda
"""
computer-genie
boto3
Pillow
"""

# serverless.yml for Serverless Framework
"""
service: computer-genie-api

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  environment:
    COMPUTER_GENIE_API_KEY: ${env:COMPUTER_GENIE_API_KEY}

functions:
  processImage:
    handler: handler.lambda_handler
    timeout: 300
    memorySize: 1024
    events:
      - http:
          path: process
          method: post
          cors: true

plugins:
  - serverless-python-requirements
"""
```

### **Google Cloud Functions Integration**
```python
import functions_framework
from google.cloud import storage
from computer_genie import ComputerGenie
import tempfile
import json
import base64
import os

@functions_framework.http
def computer_genie_processor(request):
    """Google Cloud Function for Computer Genie processing"""
    
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type'
    }
    
    if request.method == 'OPTIONS':
        return ('', 204, headers)
    
    try:
        # Initialize Computer Genie
        genie = ComputerGenie()
        
        # Parse request
        request_json = request.get_json()
        operation = request_json.get('operation')
        
        if operation == 'analyze_image':
            result = process_image_analysis(genie, request_json)
        elif operation == 'screenshot':
            result = process_screenshot(genie, request_json)
        else:
            result = {'error': 'Invalid operation'}, 400
        
        return (json.dumps(result), 200, headers)
        
    except Exception as e:
        return (json.dumps({'error': str(e)}), 500, headers)

def process_image_analysis(genie, request_data):
    """Process image analysis request"""
    
    # Decode image
    image_data = base64.b64decode(request_data['image_data'])
    prompt = request_data.get('prompt', 'Describe this image')
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(image_data)
        tmp_path = tmp_file.name
    
    try:
        # Analyze image
        result = genie.vision.analyze_image(tmp_path, prompt)
        
        # Upload to Google Cloud Storage (optional)
        gcs_url = upload_to_gcs(tmp_path, 'computer-genie-results')
        
        return {
            'success': True,
            'description': result.description,
            'confidence': result.confidence,
            'image_url': gcs_url
        }
    finally:
        os.unlink(tmp_path)

def upload_to_gcs(file_path, bucket_name):
    """Upload file to Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blob_name = f"images/{os.path.basename(file_path)}"
    blob = bucket.blob(blob_name)
    
    blob.upload_from_filename(file_path)
    
    return f"gs://{bucket_name}/{blob_name}"

# requirements.txt
"""
functions-framework==3.*
computer-genie
google-cloud-storage
Pillow
"""
```

### **Azure Functions Integration**
```python
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from computer_genie import ComputerGenie
import json
import base64
import tempfile
import os

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function for Computer Genie processing"""
    
    try:
        # Initialize Computer Genie
        genie = ComputerGenie()
        
        # Parse request
        req_body = req.get_json()
        operation = req_body.get('operation')
        
        if operation == 'analyze_image':
            result = handle_image_analysis(genie, req_body)
        elif operation == 'screenshot':
            result = handle_screenshot(genie, req_body)
        elif operation == 'automate':
            result = handle_automation(genie, req_body)
        else:
            return func.HttpResponse(
                json.dumps({'error': 'Invalid operation'}),
                status_code=400,
                mimetype="application/json"
            )
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype="application/json"
        )

def handle_image_analysis(genie, req_body):
    """Handle image analysis in Azure"""
    
    # Decode image
    image_data = base64.b64decode(req_body['image_data'])
    prompt = req_body.get('prompt', 'Describe this image')
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        tmp_file.write(image_data)
        tmp_path = tmp_file.name
    
    try:
        # Analyze image
        result = genie.vision.analyze_image(tmp_path, prompt)
        
        # Upload to Azure Blob Storage (optional)
        blob_url = upload_to_blob_storage(tmp_path, 'computer-genie-results')
        
        return {
            'success': True,
            'description': result.description,
            'confidence': result.confidence,
            'blob_url': blob_url
        }
    finally:
        os.unlink(tmp_path)

def upload_to_blob_storage(file_path, container_name):
    """Upload file to Azure Blob Storage"""
    
    # Get connection string from environment
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    
    blob_name = f"images/{os.path.basename(file_path)}"
    
    with open(file_path, 'rb') as data:
        blob_service_client.upload_blob(
            name=blob_name,
            data=data,
            container=container_name,
            overwrite=True
        )
    
    return f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"

# requirements.txt
"""
azure-functions
computer-genie
azure-storage-blob
Pillow
"""

# function.json
"""
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "function",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["post"]
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
"""
```

---

## 🏢 **Enterprise Systems**

### **Salesforce Integration**
```python
from simple_salesforce import Salesforce
from computer_genie import ComputerGenie
import base64
import os

class SalesforceComputerGenieIntegration:
    """Integration between Salesforce and Computer Genie"""
    
    def __init__(self, sf_username, sf_password, sf_security_token):
        self.sf = Salesforce(
            username=sf_username,
            password=sf_password,
            security_token=sf_security_token
        )
        self.genie = ComputerGenie()
    
    def process_lead_documents(self, lead_id):
        """Process documents attached to a Salesforce lead"""
        
        # Get lead information
        lead = self.sf.Lead.get(lead_id)
        
        # Get attachments
        attachments = self.sf.query(
            f"SELECT Id, Name, Body FROM Attachment WHERE ParentId = '{lead_id}'"
        )
        
        results = []
        
        for attachment in attachments['records']:
            try:
                # Download attachment
                attachment_data = self.sf.Attachment.get(attachment['Id'])
                
                # Decode base64 content
                file_content = base64.b64decode(attachment_data['Body'])
                
                # Save temporarily
                temp_path = f"temp_{attachment['Name']}"
                with open(temp_path, 'wb') as f:
                    f.write(file_content)
                
                # Process with Computer Genie
                if attachment['Name'].lower().endswith('.pdf'):
                    analysis = self.genie.document.process_pdf(temp_path)
                else:
                    analysis = self.genie.vision.analyze_image(
                        temp_path, 
                        "Extract key information from this document"
                    )
                
                # Create note in Salesforce
                note_data = {
                    'ParentId': lead_id,
                    'Title': f"Computer Genie Analysis: {attachment['Name']}",
                    'Body': f"Analysis: {analysis.description}\nConfidence: {analysis.confidence}"
                }
                
                note_result = self.sf.Note.create(note_data)
                
                results.append({
                    'attachment_name': attachment['Name'],
                    'analysis': analysis.description,
                    'note_id': note_result['id'],
                    'status': 'success'
                })
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                results.append({
                    'attachment_name': attachment['Name'],
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results
    
    def create_case_from_screenshot(self, account_id, description):
        """Create Salesforce case with screenshot analysis"""
        
        try:
            # Take screenshot
            screenshot = self.genie.vision.screenshot()
            
            # Analyze screenshot
            analysis = self.genie.vision.analyze_image(
                screenshot.path,
                f"Analyze this screenshot for: {description}"
            )
            
            # Create case in Salesforce
            case_data = {
                'AccountId': account_id,
                'Subject': f"Computer Genie Analysis: {description}",
                'Description': f"Analysis: {analysis.description}\nConfidence: {analysis.confidence}",
                'Status': 'New',
                'Priority': 'Medium'
            }
            
            case_result = self.sf.Case.create(case_data)
            
            # Attach screenshot to case
            with open(screenshot.path, 'rb') as f:
                screenshot_data = base64.b64encode(f.read()).decode()
            
            attachment_data = {
                'ParentId': case_result['id'],
                'Name': 'screenshot.png',
                'Body': screenshot_data,
                'ContentType': 'image/png'
            }
            
            attachment_result = self.sf.Attachment.create(attachment_data)
            
            return {
                'case_id': case_result['id'],
                'attachment_id': attachment_result['id'],
                'analysis': analysis.description,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

# Usage example
sf_integration = SalesforceComputerGenieIntegration(
    sf_username='your_username',
    sf_password='your_password',
    sf_security_token='your_token'
)

# Process lead documents
lead_results = sf_integration.process_lead_documents('00Q1234567890ABC')
print(f"Processed {len(lead_results)} documents")

# Create case with screenshot
case_result = sf_integration.create_case_from_screenshot(
    '0011234567890ABC',
    'System error on dashboard'
)
print(f"Created case: {case_result['case_id']}")
```

### **SAP Integration**
```python
import pyrfc
from computer_genie import ComputerGenie
import json

class SAPComputerGenieIntegration:
    """Integration between SAP and Computer Genie"""
    
    def __init__(self, sap_config):
        self.sap_conn = pyrfc.Connection(**sap_config)
        self.genie = ComputerGenie()
    
    def process_invoice_documents(self, company_code):
        """Process invoice documents from SAP"""
        
        try:
            # Call SAP RFC to get pending invoices
            result = self.sap_conn.call(
                'BAPI_INCOMINGINVOICE_GETLIST',
                COMPANYCODE=company_code,
                INVOICESTATUS='1'  # Pending status
            )
            
            invoices = result['INVOICELIST']
            processed_invoices = []
            
            for invoice in invoices:
                try:
                    # Get invoice document
                    doc_result = self.sap_conn.call(
                        'BAPI_DOCUMENT_GETDETAIL2',
                        DOCUMENTNUMBER=invoice['INVOICEDOCNUMBER']
                    )
                    
                    # Process document with Computer Genie
                    if doc_result['DOCUMENT_PATH']:
                        analysis = self.genie.document.extract_invoice_data(
                            doc_result['DOCUMENT_PATH']
                        )
                        
                        # Validate against SAP data
                        validation_result = self.validate_invoice_data(
                            invoice, analysis
                        )
                        
                        # Update SAP with validation results
                        if validation_result['is_valid']:
                            self.approve_invoice_in_sap(invoice['INVOICEDOCNUMBER'])
                        else:
                            self.flag_invoice_for_review(
                                invoice['INVOICEDOCNUMBER'],
                                validation_result['issues']
                            )
                        
                        processed_invoices.append({
                            'invoice_number': invoice['INVOICEDOCNUMBER'],
                            'analysis': analysis,
                            'validation': validation_result,
                            'status': 'processed'
                        })
                
                except Exception as e:
                    processed_invoices.append({
                        'invoice_number': invoice['INVOICEDOCNUMBER'],
                        'error': str(e),
                        'status': 'failed'
                    })
            
            return processed_invoices
            
        except Exception as e:
            raise Exception(f"SAP integration error: {e}")
    
    def validate_invoice_data(self, sap_invoice, genie_analysis):
        """Validate Computer Genie analysis against SAP data"""
        
        validation_result = {
            'is_valid': True,
            'issues': [],
            'confidence_score': genie_analysis.confidence
        }
        
        # Check invoice number
        if sap_invoice['INVOICENUMBER'] != genie_analysis.invoice_number:
            validation_result['issues'].append(
                f"Invoice number mismatch: SAP={sap_invoice['INVOICENUMBER']}, "
                f"Genie={genie_analysis.invoice_number}"
            )
            validation_result['is_valid'] = False
        
        # Check amount (with tolerance)
        sap_amount = float(sap_invoice['GROSSAMOUNT'])
        genie_amount = float(genie_analysis.total_amount)
        tolerance = 0.01
        
        if abs(sap_amount - genie_amount) > tolerance:
            validation_result['issues'].append(
                f"Amount mismatch: SAP={sap_amount}, Genie={genie_amount}"
            )
            validation_result['is_valid'] = False
        
        # Check vendor
        if sap_invoice['VENDOR'] != genie_analysis.vendor_id:
            validation_result['issues'].append(
                f"Vendor mismatch: SAP={sap_invoice['VENDOR']}, "
                f"Genie={genie_analysis.vendor_id}"
            )
            validation_result['is_valid'] = False
        
        return validation_result
    
    def approve_invoice_in_sap(self, invoice_number):
        """Approve invoice in SAP system"""
        
        try:
            result = self.sap_conn.call(
                'BAPI_INCOMINGINVOICE_APPROVE',
                INVOICEDOCNUMBER=invoice_number
            )
            
            if result['RETURN']['TYPE'] == 'S':
                return {'status': 'approved', 'message': 'Invoice approved successfully'}
            else:
                return {'status': 'failed', 'message': result['RETURN']['MESSAGE']}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def flag_invoice_for_review(self, invoice_number, issues):
        """Flag invoice for manual review in SAP"""
        
        try:
            # Create workflow item for review
            result = self.sap_conn.call(
                'SAP_WAPI_CREATE_WORKITEM',
                TASK='TS12345678',  # Review task template
                OBJECT_KEY=invoice_number,
                TEXT=f"Computer Genie validation issues: {'; '.join(issues)}"
            )
            
            return {
                'status': 'flagged',
                'workitem_id': result['WORKITEM_ID'],
                'issues': issues
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# Usage example
sap_config = {
    'ashost': 'sap-server.company.com',
    'sysnr': '00',
    'client': '100',
    'user': 'sap_user',
    'passwd': 'sap_password'
}

sap_integration = SAPComputerGenieIntegration(sap_config)

# Process invoices
results = sap_integration.process_invoice_documents('1000')
print(f"Processed {len(results)} invoices")

for result in results:
    if result['status'] == 'processed':
        print(f"Invoice {result['invoice_number']}: {result['validation']['is_valid']}")
    else:
        print(f"Failed to process {result['invoice_number']}: {result['error']}")
```

### **Microsoft Dynamics Integration**
```python
import requests
from computer_genie import ComputerGenie
import json
import base64

class DynamicsComputerGenieIntegration:
    """Integration between Microsoft Dynamics 365 and Computer Genie"""
    
    def __init__(self, dynamics_url, access_token):
        self.dynamics_url = dynamics_url
        self.headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'OData-MaxVersion': '4.0',
            'OData-Version': '4.0'
        }
        self.genie = ComputerGenie()
    
    def process_customer_documents(self, account_id):
        """Process documents attached to a Dynamics customer account"""
        
        # Get account information
        account_url = f"{self.dynamics_url}/api/data/v9.2/accounts({account_id})"
        account_response = requests.get(account_url, headers=self.headers)
        account_data = account_response.json()
        
        # Get related documents
        documents_url = f"{self.dynamics_url}/api/data/v9.2/accounts({account_id})/account_Annotations"
        documents_response = requests.get(documents_url, headers=self.headers)
        documents = documents_response.json()['value']
        
        results = []
        
        for document in documents:
            if document.get('documentbody'):  # Has file attachment
                try:
                    # Decode document
                    file_content = base64.b64decode(document['documentbody'])
                    
                    # Save temporarily
                    temp_path = f"temp_{document['filename']}"
                    with open(temp_path, 'wb') as f:
                        f.write(file_content)
                    
                    # Process with Computer Genie
                    if document['filename'].lower().endswith('.pdf'):
                        analysis = self.genie.document.process_pdf(temp_path)
                    else:
                        analysis = self.genie.vision.analyze_image(
                            temp_path,
                            "Extract key customer information from this document"
                        )
                    
                    # Create note in Dynamics
                    note_data = {
                        'subject': f"Computer Genie Analysis: {document['filename']}",
                        'notetext': f"Analysis: {analysis.description}\nConfidence: {analysis.confidence}",
                        'objectid_account@odata.bind': f"/accounts({account_id})"
                    }
                    
                    note_url = f"{self.dynamics_url}/api/data/v9.2/annotations"
                    note_response = requests.post(note_url, headers=self.headers, json=note_data)
                    
                    results.append({
                        'document_name': document['filename'],
                        'analysis': analysis.description,
                        'note_id': note_response.json()['annotationid'],
                        'status': 'success'
                    })
                    
                    # Clean up
                    os.remove(temp_path)
                    
                except Exception as e:
                    results.append({
                        'document_name': document['filename'],
                        'error': str(e),
                        'status': 'failed'
                    })
        
        return results
    
    def create_opportunity_from_analysis(self, account_id, screenshot_analysis):
        """Create opportunity in Dynamics based on screenshot analysis"""
        
        try:
            # Extract opportunity details from analysis
            opportunity_data = {
                'name': f"Computer Genie Identified Opportunity",
                'description': screenshot_analysis,
                'parentaccountid@odata.bind': f"/accounts({account_id})",
                'estimatedvalue': 10000,  # Default value
                'stepname': 'Qualify',
                'salesstage': 0  # Qualify stage
            }
            
            # Create opportunity
            opportunity_url = f"{self.dynamics_url}/api/data/v9.2/opportunities"
            response = requests.post(opportunity_url, headers=self.headers, json=opportunity_data)
            
            if response.status_code == 204:
                opportunity_id = response.headers['OData-EntityId'].split('(')[1].split(')')[0]
                return {
                    'opportunity_id': opportunity_id,
                    'status': 'created',
                    'analysis': screenshot_analysis
                }
            else:
                return {
                    'error': response.text,
                    'status': 'failed'
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def automate_lead_qualification(self, lead_id):
        """Automate lead qualification using Computer Genie"""
        
        try:
            # Get lead information
            lead_url = f"{self.dynamics_url}/api/data/v9.2/leads({lead_id})"
            lead_response = requests.get(lead_url, headers=self.headers)
            lead_data = lead_response.json()
            
            # Take screenshot of lead form
            screenshot = self.genie.vision.screenshot()
            
            # Analyze lead qualification criteria
            analysis = self.genie.vision.analyze_image(
                screenshot.path,
                "Analyze this lead form and determine qualification score based on: "
                "budget, authority, need, timeline (BANT criteria)"
            )
            
            # Extract qualification score (simplified)
            qualification_score = self.extract_qualification_score(analysis.description)
            
            # Update lead with qualification results
            update_data = {
                'description': f"Computer Genie Analysis: {analysis.description}",
                'leadqualitycode': qualification_score  # 1=Hot, 2=Warm, 3=Cold
            }
            
            update_response = requests.patch(lead_url, headers=self.headers, json=update_data)
            
            if update_response.status_code == 204:
                return {
                    'lead_id': lead_id,
                    'qualification_score': qualification_score,
                    'analysis': analysis.description,
                    'status': 'updated'
                }
            else:
                return {
                    'error': update_response.text,
                    'status': 'failed'
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def extract_qualification_score(self, analysis_text):
        """Extract qualification score from analysis text"""
        
        # Simplified scoring logic
        analysis_lower = analysis_text.lower()
        
        if any(word in analysis_lower for word in ['high budget', 'decision maker', 'urgent', 'immediate']):
            return 1  # Hot lead
        elif any(word in analysis_lower for word in ['medium budget', 'influencer', 'soon', 'interested']):
            return 2  # Warm lead
        else:
            return 3  # Cold lead

# Usage example
dynamics_integration = DynamicsComputerGenieIntegration(
    dynamics_url='https://yourorg.crm.dynamics.com',
    access_token='your_access_token'
)

# Process customer documents
account_id = '12345678-1234-1234-1234-123456789012'
doc_results = dynamics_integration.process_customer_documents(account_id)
print(f"Processed {len(doc_results)} documents")

# Automate lead qualification
lead_id = '87654321-4321-4321-4321-210987654321'
qualification_result = dynamics_integration.automate_lead_qualification(lead_id)
print(f"Lead qualification: {qualification_result['qualification_score']}")
```

---

## 🔧 **Development Tools**

### **GitHub Actions Integration**
```yaml
# .github/workflows/computer-genie-analysis.yml
name: Computer Genie Analysis

on:
  pull_request:
    branches: [ main ]
  push:
    branches: [ main ]

jobs:
  analyze-screenshots:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install computer-genie
        pip install -r requirements.txt
    
    - name: Run Computer Genie Analysis
      env:
        COMPUTER_GENIE_API_KEY: ${{ secrets.COMPUTER_GENIE_API_KEY }}
      run: |
        python scripts/analyze_ui_changes.py
    
    - name: Upload Analysis Results
      uses: actions/upload-artifact@v3
      with:
        name: computer-genie-analysis
        path: analysis-results/

# scripts/analyze_ui_changes.py
import os
from computer_genie import ComputerGenie
import json

def analyze_ui_changes():
    """Analyze UI changes in pull request"""
    
    genie = ComputerGenie()
    
    # Get list of changed files
    changed_files = os.popen('git diff --name-only HEAD~1').read().strip().split('\n')
    
    # Filter for UI-related files
    ui_files = [f for f in changed_files if f.endswith(('.html', '.css', '.js', '.jsx', '.vue', '.tsx'))]
    
    if not ui_files:
        print("No UI files changed")
        return
    
    # Take screenshot of application
    screenshot = genie.vision.screenshot()
    
    # Analyze UI changes
    analysis = genie.vision.analyze_image(
        screenshot.path,
        f"Analyze this UI for potential issues related to changes in: {', '.join(ui_files)}"
    )
    
    # Generate report
    report = {
        'changed_files': ui_files,
        'analysis': analysis.description,
        'confidence': analysis.confidence,
        'recommendations': extract_recommendations(analysis.description)
    }
    
    # Save report
    os.makedirs('analysis-results', exist_ok=True)
    with open('analysis-results/ui-analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis complete. Confidence: {analysis.confidence}")
    print(f"Report saved to analysis-results/ui-analysis.json")

def extract_recommendations(analysis_text):
    """Extract actionable recommendations from analysis"""
    # Simplified recommendation extraction
    recommendations = []
    
    if 'accessibility' in analysis_text.lower():
        recommendations.append("Review accessibility compliance")
    
    if 'responsive' in analysis_text.lower():
        recommendations.append("Test responsive design on different screen sizes")
    
    if 'error' in analysis_text.lower():
        recommendations.append("Check for UI errors or broken elements")
    
    return recommendations

if __name__ == "__main__":
    analyze_ui_changes()
```

### **Jenkins Integration**
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        COMPUTER_GENIE_API_KEY = credentials('computer-genie-api-key')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup') {
            steps {
                sh 'pip install computer-genie'
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Computer Genie Analysis') {
            steps {
                script {
                    sh 'python jenkins/computer_genie_analysis.py'
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'computer-genie-results/**/*', fingerprint: true
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'computer-genie-results',
                        reportFiles: 'report.html',
                        reportName: 'Computer Genie Analysis Report'
                    ])
                }
            }
        }
        
        stage('Quality Gate') {
            steps {
                script {
                    def analysisResult = readJSON file: 'computer-genie-results/analysis.json'
                    
                    if (analysisResult.confidence < 0.8) {
                        error("Computer Genie analysis confidence too low: ${analysisResult.confidence}")
                    }
                    
                    if (analysisResult.issues && analysisResult.issues.size() > 0) {
                        echo "Issues found: ${analysisResult.issues}"
                        currentBuild.result = 'UNSTABLE'
                    }
                }
            }
        }
    }
}
```

```python
# jenkins/computer_genie_analysis.py
import os
from computer_genie import ComputerGenie
import json
from datetime import datetime

def run_jenkins_analysis():
    """Run Computer Genie analysis in Jenkins pipeline"""
    
    genie = ComputerGenie()
    
    # Create results directory
    os.makedirs('computer-genie-results', exist_ok=True)
    
    # Take screenshot of application
    screenshot = genie.vision.screenshot()
    
    # Analyze application state
    analysis = genie.vision.analyze_image(
        screenshot.path,
        "Analyze this application for UI issues, broken elements, or quality problems"
    )
    
    # Extract issues
    issues = extract_issues(analysis.description)
    
    # Generate detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'build_number': os.environ.get('BUILD_NUMBER', 'unknown'),
        'analysis': {
            'description': analysis.description,
            'confidence': analysis.confidence
        },
        'issues': issues,
        'screenshot_path': screenshot.path,
        'quality_score': calculate_quality_score(analysis.description, issues)
    }
    
    # Save JSON report
    with open('computer-genie-results/analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate HTML report
    generate_html_report(report)
    
    print(f"Analysis complete. Quality score: {report['quality_score']}")
    print(f"Issues found: {len(issues)}")

def extract_issues(analysis_text):
    """Extract issues from analysis text"""
    issues = []
    
    # Define issue patterns
    issue_patterns = {
        'broken_link': ['broken link', 'dead link', 'link error'],
        'missing_image': ['missing image', 'broken image', 'image not found'],
        'layout_issue': ['layout problem', 'misaligned', 'overlapping'],
        'accessibility': ['accessibility issue', 'contrast problem', 'missing alt'],
        'performance': ['slow loading', 'performance issue', 'lag']
    }
    
    analysis_lower = analysis_text.lower()
    
    for issue_type, patterns in issue_patterns.items():
        for pattern in patterns:
            if pattern in analysis_lower:
                issues.append({
                    'type': issue_type,
                    'description': pattern,
                    'severity': 'medium'  # Default severity
                })
    
    return issues

def calculate_quality_score(analysis_text, issues):
    """Calculate quality score based on analysis"""
    base_score = 100
    
    # Deduct points for issues
    for issue in issues:
        if issue['severity'] == 'high':
            base_score -= 20
        elif issue['severity'] == 'medium':
            base_score -= 10
        else:
            base_score -= 5
    
    # Ensure score doesn't go below 0
    return max(0, base_score)

def generate_html_report(report):
    """Generate HTML report"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Computer Genie Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .issue {{ background-color: #ffe6e6; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background-color: #e6ffe6; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .score {{ font-size: 24px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Computer Genie Analysis Report</h1>
            <p>Build: {report['build_number']}</p>
            <p>Timestamp: {report['timestamp']}</p>
            <p class="score">Quality Score: {report['quality_score']}/100</p>
        </div>
        
        <h2>Analysis Results</h2>
        <p><strong>Confidence:</strong> {report['analysis']['confidence']}</p>
        <p><strong>Description:</strong> {report['analysis']['description']}</p>
        
        <h2>Issues Found ({len(report['issues'])})</h2>
    """
    
    if report['issues']:
        for issue in report['issues']:
            html_content += f"""
            <div class="issue">
                <strong>{issue['type'].replace('_', ' ').title()}:</strong> {issue['description']}
                <br><small>Severity: {issue['severity']}</small>
            </div>
            """
    else:
        html_content += '<div class="success">No issues found!</div>'
    
    html_content += """
        </body>
    </html>
    """
    
    with open('computer-genie-results/report.html', 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    run_jenkins_analysis()
```

---

## 📊 **Database Integrations**

### **PostgreSQL Integration**
```python
import psycopg2
from psycopg2.extras import RealDictCursor
from computer_genie import ComputerGenie
import json
from datetime import datetime

class PostgreSQLComputerGenieIntegration:
    """Integration between PostgreSQL and Computer Genie"""
    
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.genie = ComputerGenie()
        self.setup_tables()
    
    def setup_tables(self):
        """Create necessary tables for Computer Genie integration"""
        
        with self.conn.cursor() as cursor:
            # Create analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS computer_genie_analyses (
                    id SERIAL PRIMARY KEY,
                    file_path VARCHAR(500),
                    prompt TEXT,
                    description TEXT,
                    confidence FLOAT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create automation logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS computer_genie_automation_logs (
                    id SERIAL PRIMARY KEY,
                    task_description TEXT,
                    result TEXT,
                    status VARCHAR(50),
                    execution_time FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create document processing queue
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_processing_queue (
                    id SERIAL PRIMARY KEY,
                    file_path VARCHAR(500),
                    document_type VARCHAR(100),
                    status VARCHAR(50) DEFAULT 'pending',
                    priority INTEGER DEFAULT 5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    result JSONB
                )
            """)
            
            self.conn.commit()
    
    def queue_document_for_processing(self, file_path, document_type, priority=5):
        """Add document to processing queue"""
        
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO document_processing_queue (file_path, document_type, priority)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (file_path, document_type, priority))
            
            queue_id = cursor.fetchone()[0]
            self.conn.commit()
            
            return queue_id
    
    def process_document_queue(self, batch_size=10):
        """Process documents from queue"""
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get pending documents
            cursor.execute("""
                SELECT * FROM document_processing_queue 
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC
                LIMIT %s
            """, (batch_size,))
            
            documents = cursor.fetchall()
            
            results = []
            
            for doc in documents:
                try:
                    # Update status to processing
                    cursor.execute("""
                        UPDATE document_processing_queue 
                        SET status = 'processing' 
                        WHERE id = %s
                    """, (doc['id'],))
                    self.conn.commit()
                    
                    # Process with Computer Genie
                    if doc['document_type'] == 'pdf':
                        analysis = self.genie.document.process_pdf(doc['file_path'])
                    elif doc['document_type'] == 'image':
                        analysis = self.genie.vision.analyze_image(
                            doc['file_path'], 
                            "Extract key information from this document"
                        )
                    else:
                        analysis = self.genie.document.extract_general_content(doc['file_path'])
                    
                    # Store analysis result
                    analysis_id = self.store_analysis_result(
                        doc['file_path'],
                        "Document processing",
                        analysis.description,
                        analysis.confidence,
                        analysis.metadata
                    )
                    
                    # Update queue item
                    result_data = {
                        'analysis_id': analysis_id,
                        'description': analysis.description,
                        'confidence': analysis.confidence,
                        'metadata': analysis.metadata
                    }
                    
                    cursor.execute("""
                        UPDATE document_processing_queue 
                        SET status = 'completed', processed_at = %s, result = %s
                        WHERE id = %s
                    """, (datetime.now(), json.dumps(result_data), doc['id']))
                    
                    results.append({
                        'queue_id': doc['id'],
                        'file_path': doc['file_path'],
                        'analysis_id': analysis_id,
                        'status': 'completed'
                    })
                    
                except Exception as e:
                    # Update status to failed
                    cursor.execute("""
                        UPDATE document_processing_queue 
                        SET status = 'failed', processed_at = %s, result = %s
                        WHERE id = %s
                    """, (datetime.now(), json.dumps({'error': str(e)}), doc['id']))
                    
                    results.append({
                        'queue_id': doc['id'],
                        'file_path': doc['file_path'],
                        'error': str(e),
                        'status': 'failed'
                    })
                
                self.conn.commit()
            
            return results
    
    def store_analysis_result(self, file_path, prompt, description, confidence, metadata):
        """Store Computer Genie analysis result in database"""
        
        with self.conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO computer_genie_analyses 
                (file_path, prompt, description, confidence, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (file_path, prompt, description, confidence, json.dumps(metadata)))
            
            analysis_id = cursor.fetchone()[0]
            self.conn.commit()
            
            return analysis_id
    
    def get_analysis_history(self, file_path=None, limit=100):
        """Get analysis history from database"""
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            if file_path:
                cursor.execute("""
                    SELECT * FROM computer_genie_analyses 
                    WHERE file_path = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (file_path, limit))
            else:
                cursor.execute("""
                    SELECT * FROM computer_genie_analyses 
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
            
            return cursor.fetchall()
    
    def get_processing_statistics(self):
        """Get processing statistics"""
        
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_analyses,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) FILTER (WHERE confidence > 0.8) as high_confidence_count,
                    DATE_TRUNC('day', created_at) as date,
                    COUNT(*) as daily_count
                FROM computer_genie_analyses 
                WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE_TRUNC('day', created_at)
                ORDER BY date DESC
            """)
            
            daily_stats = cursor.fetchall()
            
            cursor.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM document_processing_queue
                GROUP BY status
            """)
            
            queue_stats = cursor.fetchall()
            
            return {
                'daily_statistics': daily_stats,
                'queue_statistics': queue_stats
            }

# Usage example
db_config = {
    'host': 'localhost',
    'database': 'computer_genie_db',
    'user': 'postgres',
    'password': 'password'
}

db_integration = PostgreSQLComputerGenieIntegration(db_config)

# Queue documents for processing
queue_id = db_integration.queue_document_for_processing(
    '/path/to/document.pdf', 
    'pdf', 
    priority=1
)
print(f"Document queued with ID: {queue_id}")

# Process queue
results = db_integration.process_document_queue(batch_size=5)
print(f"Processed {len(results)} documents")

# Get statistics
stats = db_integration.get_processing_statistics()
print(f"Processing statistics: {stats}")
```

### **MongoDB Integration**
```python
from pymongo import MongoClient
from computer_genie import ComputerGenie
from datetime import datetime
import gridfs
import base64

class MongoDBComputerGenieIntegration:
    """Integration between MongoDB and Computer Genie"""
    
    def __init__(self, mongo_uri, database_name):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.genie = ComputerGenie()
        self.fs = gridfs.GridFS(self.db)
        
        # Collections
        self.analyses = self.db.computer_genie_analyses
        self.automation_logs = self.db.automation_logs
        self.document_queue = self.db.document_processing_queue
        
        self.setup_indexes()
    
    def setup_indexes(self):
        """Create indexes for better performance"""
        
        # Analysis collection indexes
        self.analyses.create_index([("file_path", 1)])
        self.analyses.create_index([("created_at", -1)])
        self.analyses.create_index([("confidence", -1)])
        
        # Queue collection indexes
        self.document_queue.create_index([("status", 1), ("priority", 1), ("created_at", 1)])
        
        # Automation logs indexes
        self.automation_logs.create_index([("created_at", -1)])
        self.automation_logs.create_index([("status", 1)])
    
    def store_file_and_analyze(self, file_path, prompt, metadata=None):
        """Store file in GridFS and analyze with Computer Genie"""
        
        try:
            # Store file in GridFS
            with open(file_path, 'rb') as f:
                file_id = self.fs.put(
                    f, 
                    filename=file_path,
                    metadata=metadata or {}
                )
            
            # Analyze with Computer Genie
            if file_path.lower().endswith('.pdf'):
                analysis = self.genie.document.process_pdf(file_path)
            else:
                analysis = self.genie.vision.analyze_image(file_path, prompt)
            
            # Store analysis result
            analysis_doc = {
                'file_id': file_id,
                'file_path': file_path,
                'prompt': prompt,
                'description': analysis.description,
                'confidence': analysis.confidence,
                'metadata': analysis.metadata,
                'created_at': datetime.utcnow()
            }
            
            analysis_id = self.analyses.insert_one(analysis_doc).inserted_id
            
            return {
                'analysis_id': str(analysis_id),
                'file_id': str(file_id),
                'description': analysis.description,
                'confidence': analysis.confidence,
                'status': 'completed'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }
    
    def batch_analyze_documents(self, file_paths, prompt):
        """Batch analyze multiple documents"""
        
        results = []
        
        for file_path in file_paths:
            result = self.store_file_and_analyze(file_path, prompt)
            results.append(result)
        
        return results
    
    def search_analyses(self, query_filter, limit=50):
        """Search analysis results"""
        
        return list(self.analyses.find(query_filter).limit(limit).sort("created_at", -1))
    
    def get_analytics_dashboard_data(self):
        """Get data for analytics dashboard"""
        
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$created_at"},
                        "month": {"$month": "$created_at"},
                        "day": {"$dayOfMonth": "$created_at"}
                    },
                    "count": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"},
                    "high_confidence_count": {
                        "$sum": {"$cond": [{"$gte": ["$confidence", 0.8]}, 1, 0]}
                    }
                }
            },
            {"$sort": {"_id": -1}},
            {"$limit": 30}
        ]
        
        daily_stats = list(self.analyses.aggregate(pipeline))
        
        # Get confidence distribution
        confidence_pipeline = [
            {
                "$bucket": {
                    "groupBy": "$confidence",
                    "boundaries": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "default": "Other",
                    "output": {"count": {"$sum": 1}}
                }
            }
        ]
        
        confidence_distribution = list(self.analyses.aggregate(confidence_pipeline))
        
        return {
            'daily_statistics': daily_stats,
            'confidence_distribution': confidence_distribution,
            'total_analyses': self.analyses.count_documents({}),
            'recent_analyses': list(self.analyses.find().limit(10).sort("created_at", -1))
        }

# Usage example
mongo_integration = MongoDBComputerGenieIntegration(
    'mongodb://localhost:27017/',
    'computer_genie_db'
)

# Analyze and store file
result = mongo_integration.store_file_and_analyze(
    '/path/to/document.pdf',
    'Extract key information from this document'
)
print(f"Analysis completed: {result['analysis_id']}")

# Get dashboard data
dashboard_data = mongo_integration.get_analytics_dashboard_data()
print(f"Total analyses: {dashboard_data['total_analyses']}")
```

---

## 🔗 **API Integrations**

### **REST API Integration**
```python
import requests
from computer_genie import ComputerGenie
import json
import base64

class RESTAPIComputerGenieIntegration:
    """Generic REST API integration for Computer Genie"""
    
    def __init__(self, api_base_url, api_key=None):
        self.api_base_url = api_base_url.rstrip('/')
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Computer-Genie-Integration/1.0'
        }
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
        self.genie = ComputerGenie()
    
    def analyze_and_post(self, image_path, prompt, endpoint='/analysis'):
        """Analyze image and post results to API"""
        
        try:
            # Analyze with Computer Genie
            analysis = self.genie.vision.analyze_image(image_path, prompt)
            
            # Prepare payload
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            payload = {
                'image_data': image_data,
                'prompt': prompt,
                'analysis': {
                    'description': analysis.description,
                    'confidence': analysis.confidence,
                    'metadata': analysis.metadata
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Post to API
            response = requests.post(
                f"{self.api_base_url}{endpoint}",
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            
            return {
                'status': 'success',
                'api_response': response.json(),
                'analysis': analysis.description,
                'confidence': analysis.confidence
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def webhook_handler(self, webhook_data):
        """Handle incoming webhook with Computer Genie processing"""
        
        try:
            # Extract image from webhook
            if 'image_url' in webhook_data:
                # Download image
                img_response = requests.get(webhook_data['image_url'])
                img_response.raise_for_status()
                
                # Save temporarily
                temp_path = 'temp_webhook_image.jpg'
                with open(temp_path, 'wb') as f:
                    f.write(img_response.content)
                
                # Analyze
                prompt = webhook_data.get('prompt', 'Analyze this image')
                analysis = self.genie.vision.analyze_image(temp_path, prompt)
                
                # Send results back
                callback_url = webhook_data.get('callback_url')
                if callback_url:
                    callback_payload = {
                        'webhook_id': webhook_data.get('id'),
                        'analysis': analysis.description,
                        'confidence': analysis.confidence,
                        'status': 'completed'
                    }
                    
                    requests.post(callback_url, json=callback_payload)
                
                # Clean up
                os.remove(temp_path)
                
                return {
                    'status': 'processed',
                    'analysis': analysis.description,
                    'confidence': analysis.confidence
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

# Usage example
api_integration = RESTAPIComputerGenieIntegration(
    'https://api.example.com',
    'your-api-key'
)

# Analyze and post to API
result = api_integration.analyze_and_post(
    '/path/to/image.jpg',
    'Describe this image',
    '/v1/analysis'
)
print(f"API integration result: {result['status']}")
```

---

## 🤖 **Automation Platforms**

### **Zapier Integration**
```python
from computer_genie import ComputerGenie
import json
import requests

class ZapierComputerGenieIntegration:
    """Zapier integration for Computer Genie"""
    
    def __init__(self):
        self.genie = ComputerGenie()
    
    def zapier_webhook_handler(self, zapier_data):
        """Handle Zapier webhook trigger"""
        
        try:
            action_type = zapier_data.get('action_type')
            
            if action_type == 'analyze_image':
                return self.handle_image_analysis(zapier_data)
            elif action_type == 'take_screenshot':
                return self.handle_screenshot(zapier_data)
            elif action_type == 'automate_task':
                return self.handle_automation(zapier_data)
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown action type: {action_type}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def handle_image_analysis(self, data):
        """Handle image analysis from Zapier"""
        
        image_url = data.get('image_url')
        prompt = data.get('prompt', 'Describe this image')
        
        if not image_url:
            return {
                'status': 'error',
                'message': 'No image URL provided'
            }
        
        # Download image
        response = requests.get(image_url)
        temp_path = 'zapier_temp_image.jpg'
        
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        # Analyze
        analysis = self.genie.vision.analyze_image(temp_path, prompt)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            'status': 'success',
            'description': analysis.description,
            'confidence': analysis.confidence,
            'metadata': analysis.metadata
        }
    
    def handle_screenshot(self, data):
        """Handle screenshot request from Zapier"""
        
        prompt = data.get('prompt', 'Describe the screen')
        
        # Take screenshot
        screenshot = self.genie.vision.screenshot()
        
        # Analyze if prompt provided
        if prompt:
            analysis = self.genie.vision.analyze_image(screenshot.path, prompt)
            
            return {
                'status': 'success',
                'screenshot_path': screenshot.path,
                'analysis': analysis.description,
                'confidence': analysis.confidence
            }
        else:
            return {
                'status': 'success',
                'screenshot_path': screenshot.path,
                'message': 'Screenshot taken successfully'
            }

# Zapier webhook endpoint (Flask example)
from flask import Flask, request, jsonify

app = Flask(__name__)
zapier_integration = ZapierComputerGenieIntegration()

@app.route('/zapier/webhook', methods=['POST'])
def zapier_webhook():
    """Zapier webhook endpoint"""
    
    data = request.get_json()
    result = zapier_integration.zapier_webhook_handler(data)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 📈 **Monitoring & Analytics**

### **Grafana Integration**
```python
import requests
from computer_genie import ComputerGenie
import json
from datetime import datetime, timedelta

class GrafanaComputerGenieIntegration:
    """Grafana integration for Computer Genie monitoring"""
    
    def __init__(self, grafana_url, api_key):
        self.grafana_url = grafana_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.genie = ComputerGenie()
    
    def create_computer_genie_dashboard(self):
        """Create Grafana dashboard for Computer Genie metrics"""
        
        dashboard_config = {
            "dashboard": {
                "title": "Computer Genie Analytics",
                "tags": ["computer-genie", "ai", "automation"],
                "timezone": "browser",
                "panels": [
                    {
                        "title": "Analysis Confidence Over Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "computer_genie_analysis_confidence",
                                "legendFormat": "Confidence Score"
                            }
                        ]
                    },
                    {
                        "title": "Daily Analysis Count",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "sum(computer_genie_analyses_total)",
                                "legendFormat": "Total Analyses"
                            }
                        ]
                    },
                    {
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(computer_genie_errors_total[5m])",
                                "legendFormat": "Error Rate"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-24h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        response = requests.post(
            f"{self.grafana_url}/api/dashboards/db",
            headers=self.headers,
            json=dashboard_config
        )
        
        return response.json()
    
    def send_metrics_to_prometheus(self, metrics_data):
        """Send Computer Genie metrics to Prometheus"""
        
        # This would typically use a Prometheus client library
        # Here's a simplified example
        
        prometheus_metrics = []
        
        for metric in metrics_data:
            prometheus_metrics.append(
                f"computer_genie_{metric['name']} {metric['value']} {int(datetime.now().timestamp())}"
            )
        
        # Send to Prometheus pushgateway
        metrics_payload = '\n'.join(prometheus_metrics)
        
        response = requests.post(
            'http://prometheus-pushgateway:9091/metrics/job/computer_genie',
            data=metrics_payload,
            headers={'Content-Type': 'text/plain'}
        )
        
        return response.status_code == 200

# Usage example
grafana_integration = GrafanaComputerGenieIntegration(
    'http://grafana.example.com',
    'your-grafana-api-key'
)

# Create dashboard
dashboard_result = grafana_integration.create_computer_genie_dashboard()
print(f"Dashboard created: {dashboard_result}")

# Send metrics
metrics = [
    {'name': 'analysis_confidence', 'value': 0.95},
    {'name': 'analyses_total', 'value': 150},
    {'name': 'errors_total', 'value': 2}
]

metrics_sent = grafana_integration.send_metrics_to_prometheus(metrics)
print(f"Metrics sent: {metrics_sent}")
```

---

## 🎯 **Best Practices**

### **Error Handling**
```python
import logging
from computer_genie import ComputerGenie
from functools import wraps
import time

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
                    
                    logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            
        return wrapper
    return decorator

class RobustComputerGenieIntegration:
    """Robust Computer Genie integration with error handling"""
    
    def __init__(self):
        self.genie = ComputerGenie()
        self.logger = logging.getLogger(__name__)
    
    @retry_on_failure(max_retries=3, delay=2)
    def safe_analyze_image(self, image_path, prompt):
        """Safely analyze image with retry logic"""
        
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not prompt or len(prompt.strip()) == 0:
                raise ValueError("Prompt cannot be empty")
            
            # Perform analysis
            result = self.genie.vision.analyze_image(image_path, prompt)
            
            # Validate result
            if not result or not result.description:
                raise ValueError("Analysis returned empty result")
            
            if result.confidence < 0.1:
                self.logger.warning(f"Low confidence analysis: {result.confidence}")
            
            return {
                'status': 'success',
                'description': result.description,
                'confidence': result.confidence,
                'metadata': result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'image_path': image_path,
                'prompt': prompt
            }
    
    def batch_process_with_error_handling(self, file_paths, prompt):
        """Process multiple files with comprehensive error handling"""
        
        results = {
            'successful': [],
            'failed': [],
            'summary': {
                'total': len(file_paths),
                'success_count': 0,
                'failure_count': 0
            }
        }
        
        for file_path in file_paths:
            try:
                result = self.safe_analyze_image(file_path, prompt)
                
                if result['status'] == 'success':
                    results['successful'].append(result)
                    results['summary']['success_count'] += 1
                else:
                    results['failed'].append(result)
                    results['summary']['failure_count'] += 1
                    
            except Exception as e:
                error_result = {
                    'status': 'failed',
                    'error': str(e),
                    'file_path': file_path
                }
                results['failed'].append(error_result)
                results['summary']['failure_count'] += 1
        
        return results

# Usage example
robust_integration = RobustComputerGenieIntegration()

# Safe single analysis
result = robust_integration.safe_analyze_image(
    '/path/to/image.jpg',
    'Describe this image'
)

# Batch processing with error handling
file_paths = ['/path/to/img1.jpg', '/path/to/img2.jpg', '/path/to/img3.jpg']
batch_results = robust_integration.batch_process_with_error_handling(
    file_paths,
    'Extract text from this image'
)

print(f"Batch processing: {batch_results['summary']['success_count']} successful, "
      f"{batch_results['summary']['failure_count']} failed")
```

---

## 📞 **Support & Resources**

- **Documentation**: [Computer Genie Docs](https://docs.computer-genie.ai)
- **API Reference**: [API Documentation](https://api.computer-genie.ai/docs)
- **Community**: [Discord Server](https://discord.gg/computer-genie)
- **Support**: [support@abhishektechnologies.com](mailto:support@abhishektechnologies.com)
- **Enterprise**: [enterprise@abhishektechnologies.com](mailto:enterprise@abhishektechnologies.com)

---

<div align="center">

**🚀 Ready to integrate Computer Genie with your platform?**

[Get Started](../tutorials/quick_start.md) | [API Reference](../api/README.md) | [Enterprise Solutions](../enterprise/README.md)

---

*Powered by **Abhishek Technologies Pvt Ltd** - Transforming businesses through intelligent automation*

</div>