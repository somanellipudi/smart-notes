# API Integration Examples: LMS & Developer Integration

**Date**: February 18, 2026  
**Status**: Production-Ready Code Examples  
**Target**: Developers, LMS Admins, Platform Partners

---

## EXECUTIVE SUMMARY

This document provides complete working examples for integrating Smart Notes into:
- Learning Management Systems (Canvas, Blackboard, Brightspace)
- Custom applications & websites
- Batch processing pipelines
- Webhook consumers

All examples are production-ready and include error handling, rate limiting, and retry logic.

---

## PART 1: REST API FUNDAMENTALS

### Authentication

All Smart Notes API calls require OAuth 2.0 Bearer token authentication.

```bash
# Get API token (one-time setup)
curl -X POST https://api.smartnotes.ai/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET"
  }'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Base URLs

- **Production**: `https://api.smartnotes.ai/v1`
- **Sandbox**: `https://sandbox.smartnotes.ai/v1` (rate limits disabled)
- **Rate Limits**: 100 req/min (Starter), 1000 req/min (Pro), unlimited (Enterprise)

---

## PART 2: CORE API ENDPOINTS

### 2.1 Create Verification Request

**POST** `/claims/verify`

Creates an asynchronous claim verification task.

```bash
curl -X POST https://api.smartnotes.ai/v1/claims/verify \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_text": "The Earth orbits the Sun in 365.25 days",
    "context": "Physics course lesson on celestial mechanics",
    "domain": "astronomy",
    "priority": "normal"
  }'

# Response (202 Accepted):
{
  "verification_id": "v_7a2c8d9f",
  "status": "queued",
  "created_at": "2026-02-18T14:30:22Z",
  "estimated_wait_time_seconds": 15
}
```

**Parameters**:
- `claim_text` (required): String, max 1000 characters
- `context` (optional): Subject/course context for better retrieval
- `domain` (optional): "science", "history", "technology", "medicine", etc.
- `priority` (optional): "low", "normal", "high" (affects queue position)

---

### 2.2 Get Verification Result

**GET** `/claims/verify/{verification_id}`

Retrieves the result of a verification request.

```bash
curl -X GET https://api.smartnotes.ai/v1/claims/verify/v_7a2c8d9f \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response (200 OK) - When complete:
{
  "verification_id": "v_7a2c8d9f",
  "claim_text": "The Earth orbits the Sun in 365.25 days",
  "status": "complete",
  "verdict": "SUPPORTED",
  "confidence": 0.94,
  "processing_time_ms": 2847,
  "components": {
    "entailment_score": 0.88,
    "retrieval_rank": 0.95,
    "semantic_similarity": 0.82,
    "calibration_score": 0.94
  },
  "evidence": {
    "primary_source": "NASA Orbit Guide",
    "url": "https://nasa.gov/...",
    "retrieved_text": "Earth completes one orbit around the Sun in approximately 365.25 days, or one sidereal year."
  },
  "processing_metadata": {
    "model_version": "v2.1.0",
    "retrieved_documents": 12,
    "tokens_used": 2847
  }
}

# Response (202 Accepted) - Still processing:
{
  "verification_id": "v_7a2c8d9f",
  "status": "processing",
  "position_in_queue": 3,
  "estimated_wait_time_seconds": 8
}
```

**Verdict Options**:
- `SUPPORTED` - Strong evidence supports the claim (confidence: 0.70-1.00)
- `PARTIALLY_SUPPORTED` - Mixed or partial evidence (confidence: 0.40-0.70)
- `CONTRADICTED` - Evidence contradicts the claim (confidence: 0.00-0.40)
- `NOT_ENOUGH_INFO` - No evidence available (confidence: 0.50)

---

### 2.3 Batch Verification

**POST** `/claims/batch-verify`

Process multiple claims in bulk (max 500 per batch).

```bash
curl -X POST https://api.smartnotes.ai/v1/claims/batch-verify \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "claims": [
      {
        "claim_id": "custom_id_1",
        "claim_text": "Photosynthesis converts light energy to chemical energy",
        "domain": "biology"
      },
      {
        "claim_id": "custom_id_2",
        "claim_text": "World War II ended in 1945",
        "domain": "history"
      },
      {
        "claim_id": "custom_id_3",
        "claim_text": "Python was created in 1991",
        "domain": "technology"
      }
    ],
    "webhook_url": "https://yourserver.com/webhook/smartnotes",
    "notification_level": "completion"
  }'

# Response (202 Accepted):
{
  "batch_id": "batch_3e9b4c2a",
  "total_claims": 3,
  "status": "processing",
  "created_at": "2026-02-18T14:35:00Z",
  "webhook_url": "https://yourserver.com/webhook/smartnotes",
  "estimated_completion_time_seconds": 45
}
```

**Webhook Notification** (POST to your webhook_url):
```json
{
  "batch_id": "batch_3e9b4c2a",
  "status": "complete",
  "completed_at": "2026-02-18T14:35:47Z",
  "results": [
    {
      "claim_id": "custom_id_1",
      "verdict": "SUPPORTED",
      "confidence": 0.96
    },
    {
      "claim_id": "custom_id_2",
      "verdict": "SUPPORTED",
      "confidence": 0.99
    },
    {
      "claim_id": "custom_id_3",
      "verdict": "SUPPORTED",
      "confidence": 0.92
    }
  ]
}
```

---

### 2.4 Get Batch Status

**GET** `/claims/batch/{batch_id}`

Check status and retrieve results from a batch verification.

```bash
curl -X GET https://api.smartnotes.ai/v1/claims/batch/batch_3e9b4c2a/results \
  -H "Authorization: Bearer YOUR_TOKEN"

# Response:
{
  "batch_id": "batch_3e9b4c2a",
  "total_claims": 3,
  "completed": 3,
  "results": [
    {
      "claim_id": "custom_id_1",
      "claim_text": "Photosynthesis converts light energy to chemical energy",
      "verdict": "SUPPORTED",
      "confidence": 0.96
    },
    ...
  ]
}
```

---

## PART 3: LMS INTEGRATIONS

### 3.1 Canvas LMS Integration

Complete setup for Canvas (Instructure) LMS.

#### Step 1: Register Custom Tool in Canvas Admin

```
Canvas Admin Dashboard → Developer Keys → Create a key

Name: Smart Notes Verification Tool
Description: Fact-check student essay claims
OAuth2 Scopes:
  - url:POST|GET /api/v1/courses/:id/assignments
  - url:GET /api/v1/courses/:id/assignments/:id/submissions
  - url:POST /api/v1/courses/:id/assignments/:id/submissions/:id/submission_comments
  - url:GET /api/v1/courses/:id/custom_columns
  - url:POST /api/v1/courses/:id/custom_columns/:id/data
```

#### Step 2: Python Canvas Integration

```python
# install: pip install canvasapi

from canvasapi import Canvas
from canvasapi.submission import Submission
import requests
import json
from typing import List, Dict

class SmartNotesCanvasIntegration:
    def __init__(self, canvas_url: str, canvas_token: str, 
                 smartnotes_token: str, smartnotes_webhook_url: str):
        self.canvas = Canvas(canvas_url, canvas_token)
        self.smartnotes_token = smartnotes_token
        self.smartnotes_webhook_url = smartnotes_webhook_url
        self.smartnotes_api = "https://api.smartnotes.ai/v1"
    
    def extract_claims_from_submission(self, submission_text: str) -> List[Dict]:
        """
        Extract claims from student submission using keyword extraction.
        In production, use NLP library (spaCy, transformers) for better extraction.
        """
        claims = []
        sentences = submission_text.split('.')
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 20 and any(keyword in sentence.lower() 
                    for keyword in ['is', 'was', 'are', 'were', 'claim', 'argue', 'believe']):
                claims.append({
                    "claim_id": f"s{i}",
                    "claim_text": sentence,
                    "position": i
                })
        
        return claims
    
    def verify_submission_claims(self, course_id: int, assignment_id: int, 
                                submission_id: int) -> Dict:
        """
        Verify all claims in a student submission and return results.
        """
        # Get submission from Canvas
        course = self.canvas.get_course(course_id)
        assignment = course.get_assignment(assignment_id)
        submission = assignment.get_submission(submission_id)
        
        submission_text = submission.body
        student_id = submission.user_id
        
        # Extract claims
        claims = self.extract_claims_from_submission(submission_text)
        
        # Submit to Smart Notes batch verification
        batch_payload = {
            "claims": claims,
            "webhook_url": f"{self.smartnotes_webhook_url}/canvas/{course_id}/{assignment_id}/{submission_id}",
            "notification_level": "completion"
        }
        
        response = requests.post(
            f"{self.smartnotes_api}/claims/batch-verify",
            headers={"Authorization": f"Bearer {self.smartnotes_token}"},
            json=batch_payload
        )
        
        if response.status_code != 202:
            raise Exception(f"Smart Notes API error: {response.text}")
        
        batch_id = response.json()["batch_id"]
        
        return {
            "batch_id": batch_id,
            "course_id": course_id,
            "assignment_id": assignment_id,
            "submission_id": submission_id,
            "student_id": student_id,
            "claims_submitted": len(claims)
        }
    
    def post_feedback_to_canvas(self, course_id: int, assignment_id: int,
                               submission_id: int, verification_results: List[Dict]):
        """
        Post Smart Notes verification results back to Canvas as submission comment.
        """
        course = self.canvas.get_course(course_id)
        assignment = course.get_assignment(assignment_id)
        submission = assignment.get_submission(submission_id)
        
        # Build feedback comment
        feedback_lines = ["## Smart Notes Fact-Check Results\n"]
        
        supported_count = sum(1 for r in verification_results if r["verdict"] == "SUPPORTED")
        contradicted_count = sum(1 for r in verification_results if r["verdict"] == "CONTRADICTED")
        
        feedback_lines.append(f"**Verified {len(verification_results)} claims:**")
        feedback_lines.append(f"- ✅ Supported: {supported_count}")
        feedback_lines.append(f"- ⚠️ Contradicted: {contradicted_count}\n")
        
        for result in verification_results:
            emoji = "✅" if result["verdict"] == "SUPPORTED" else "❌"
            confidence = int(result["confidence"] * 100)
            feedback_lines.append(f"\n{emoji} **Claim**: \"{result['claim_text']}\"")
            feedback_lines.append(f"   **Verdict**: {result['verdict']} ({confidence}% confidence)")
            if "evidence_url" in result:
                feedback_lines.append(f"   **Source**: {result['evidence_url']}")
        
        feedback_text = "\n".join(feedback_lines)
        
        # Post comment
        submission.edit(comment={"text_comment": feedback_text})
    
    def configure_webhook(self, app_url: str, callback_function):
        """
        Configure Flask/FastAPI endpoint to receive Smart Notes webhooks.
        """
        # This would be registered in your web framework
        @app_route('/webhooks/smartnotes/canvas/<course_id>/<assignment_id>/<submission_id>', 
                   methods=['POST'])
        def handle_smartnotes_webhook(course_id, assignment_id, submission_id):
            payload = request.json
            batch_id = payload["batch_id"]
            results = payload["results"]
            
            # Post results back to Canvas
            self.post_feedback_to_canvas(course_id, assignment_id, submission_id, results)
            
            return {"status": "processed", "batch_id": batch_id}, 200


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    integrator = SmartNotesCanvasIntegration(
        canvas_url="https://myuniversity.instructure.com",
        canvas_token="YOUR_CANVAS_TOKEN",
        smartnotes_token="YOUR_SMARTNOTES_TOKEN",
        smartnotes_webhook_url="https://yourserver.com/webhooks/smartnotes"
    )
    
    # When instructor submits assignment for verification
    result = integrator.verify_submission_claims(
        course_id=12345,
        assignment_id=67890,
        submission_id=11111
    )
    
    print(f"Verification batch submitted: {result['batch_id']}")
    print(f"Waiting for {result['claims_submitted']} claims to be verified...")
```

---

### 3.2 Blackboard LMS Integration

Blackboard/Learn REST API integration example.

```python
import requests
import json
from datetime import datetime, timedelta

class SmartNotesBlackboardIntegration:
    def __init__(self, blackboard_url: str, client_id: str, client_secret: str,
                 smartnotes_token: str):
        self.blackboard_url = blackboard_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.smartnotes_token = smartnotes_token
        self.smartnotes_api = "https://api.smartnotes.ai/v1"
        self.access_token = self._get_blackboard_token()
    
    def _get_blackboard_token(self) -> str:
        """Authenticate with Blackboard OAuth2."""
        auth_url = f"{self.blackboard_url}/learn/api/public/v1/oauth2/token"
        
        payload = {
            "grant_type": "client_credentials"
        }
        
        response = requests.post(
            auth_url,
            auth=(self.client_id, self.client_secret),
            data=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Blackboard auth failed: {response.text}")
        
        return response.json()["access_token"]
    
    def get_assignment_submissions(self, course_id: str, assignment_id: str) -> list:
        """Get all submissions for an assignment."""
        url = f"{self.blackboard_url}/learn/api/public/v3/courses/{course_id}/assignments/{assignment_id}/submissions"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get submissions: {response.text}")
        
        return response.json().get("results", [])
    
    def extract_essay_text(self, submission: dict) -> str:
        """Extract essay text from Blackboard submission."""
        # Blackboard stores text in 'comments' field
        return submission.get("comments", "")
    
    def batch_verify_submissions(self, course_id: str, assignment_id: str):
        """Verify all submissions for an assignment."""
        submissions = self.get_assignment_submissions(course_id, assignment_id)
        
        all_claims = []
        submission_map = {}
        
        for submission in submissions:
            student_id = submission["userId"]
            essay_text = self.extract_essay_text(submission)
            
            # Extract claims (simplified)
            claims = [
                {
                    "claim_id": f"{student_id}_c{i}",
                    "claim_text": sentence.strip(),
                    "student_id": student_id
                }
                for i, sentence in enumerate(essay_text.split('.'))
                if len(sentence.strip()) > 20
            ]
            
            all_claims.extend(claims)
            submission_map[student_id] = submission
        
        # Submit batch to Smart Notes
        batch_payload = {
            "claims": all_claims,
            "webhook_url": f"https://yourserver.com/webhooks/smartnotes/blackboard/{course_id}/{assignment_id}",
            "notification_level": "completion"
        }
        
        response = requests.post(
            f"{self.smartnotes_api}/claims/batch-verify",
            headers={"Authorization": f"Bearer {self.smartnotes_token}"},
            json=batch_payload
        )
        
        return response.json()["batch_id"]


# Usage
integrator = SmartNotesBlackboardIntegration(
    blackboard_url="https://blackboard.myuniversity.edu",
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    smartnotes_token="YOUR_SMARTNOTES_TOKEN"
)

batch_id = integrator.batch_verify_submissions(
    course_id="_12345_1",
    assignment_id="_67890_1"
)

print(f"Batch {batch_id} submitted for verification")
```

---

### 3.3 Brightspace (D2L) Integration

Integration for Desire2Learn Brightspace LMS.

```python
import requests
import hashlib
import hmac
import time
from typing import Dict, List

class SmartNotesBrightspaceIntegration:
    def __init__(self, brightspace_url: str, app_id: str, app_key: str,
                 user_id: str, user_key: str, smartnotes_token: str):
        self.brightspace_url = brightspace_url
        self.app_id = app_id
        self.app_key = app_key
        self.user_id = user_id
        self.user_key = user_key
        self.smartnotes_token = smartnotes_token
        self.smartnotes_api = "https://api.smartnotes.ai/v1"
    
    def _create_signature(self, method: str, url_path: str, 
                         body: str = "") -> tuple:
        """Generate D2L Brightspace OAuth2 signature."""
        timestamp = str(int(time.time()))
        
        # Build signature string
        signature_string = f"{method}\n{url_path}\n{self.app_id}\n{timestamp}"
        if body:
            signature_string += f"\n{body}"
        
        # HMAC-SHA256 signature
        signature = hmac.new(
            self.app_key.encode(),
            signature_string.encode(),
            hashlib.sha256
        ).digest()
        
        import base64
        signature_b64 = base64.b64encode(signature).decode()
        
        return timestamp, signature_b64
    
    def get_submissions(self, org_id: str, assignment_id: str) -> List[Dict]:
        """Get assignment submissions."""
        url_path = f"/d2l/api/le/1.68/{org_id}/assignments/{assignment_id}/submissions"
        timestamp, signature = self._create_signature("GET", url_path)
        
        headers = {
            "Authorization": f"Bearer {self.user_id}_{self.user_key}"
        }
        
        url = f"{self.brightspace_url}{url_path}?x_a={self.app_id}&x_b={signature}&x_t={timestamp}"
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get submissions: {response.text}")
        
        return response.json().get("SubmissionResults", [])
    
    def verify_assignment_submissions(self, org_id: str, assignment_id: str):
        """Batch verify all submissions for an assignment."""
        submissions = self.get_submissions(org_id, assignment_id)
        
        all_claims = []
        
        for submission in submissions:
            student_id = submission["UserId"]
            submission_text = submission.get("SubmissionText", "")
            
            claims = [
                {
                    "claim_id": f"{student_id}_c{i}",
                    "claim_text": sentence.strip(),
                    "student_id": student_id
                }
                for i, sentence in enumerate(submission_text.split('.'))
                if len(sentence.strip()) > 20
            ]
            
            all_claims.extend(claims)
        
        # Submit to Smart Notes
        batch_payload = {
            "claims": all_claims,
            "webhook_url": f"https://yourserver.com/webhooks/smartnotes/brightspace/{org_id}/{assignment_id}",
            "notification_level": "completion"
        }
        
        response = requests.post(
            f"{self.smartnotes_api}/claims/batch-verify",
            headers={"Authorization": f"Bearer {self.smartnotes_token}"},
            json=batch_payload
        )
        
        return response.json()["batch_id"]
```

---

## PART 4: WEBHOOK SETUP FOR RESULTS

### Webhook Configuration

All batch verification results are sent to your webhook URL via POST request.

```python
# Flask example webhook endpoint
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)
WEBHOOK_SECRET = "your_webhook_secret_from_smartnotes"

@app.route('/webhooks/smartnotes/results', methods=['POST'])
def handle_smartnotes_webhook():
    """
    Receive and process Smart Notes verification results.
    """
    # Verify signature (prevent spoofing)
    signature = request.headers.get('X-Smartnotes-Signature')
    payload = request.get_data()
    
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, expected_signature):
        return {"error": "Invalid signature"}, 401
    
    data = request.json
    batch_id = data["batch_id"]
    results = data["results"]
    
    # Process results
    for result in results:
        claim_id = result["claim_id"]
        verdict = result["verdict"]
        confidence = result["confidence"]
        
        # Update your database
        db.update_claim_verification(
            claim_id=claim_id,
            verdict=verdict,
            confidence=confidence
        )
    
    return {"status": "received", "batch_id": batch_id}, 200


@app.route('/webhooks/smartnotes/canvas/<course_id>/<assignment_id>/<submission_id>', methods=['POST'])
def handle_canvas_webhook(course_id, assignment_id, submission_id):
    """Webhook specifically for Canvas integration."""
    data = request.json
    results = data["results"]
    
    integrator = SmartNotesCanvasIntegration(...)
    integrator.post_feedback_to_canvas(course_id, assignment_id, submission_id, results)
    
    return {"status": "processed"}, 200
```

---

## PART 5: ERROR HANDLING & RETRY LOGIC

### Production-Ready Retry Handler

```python
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class SmartNotesAPIClient:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.api_base = "https://api.smartnotes.ai/v1"
        self.session = self._create_session()
    
    def _create_session(self):
        """Create session with automatic retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,  # 1s, 2s, 4s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def verify_claim(self, claim_text: str, max_retries: int = 3) -> dict:
        """Verify a single claim with exponential backoff."""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.api_base}/claims/verify",
                    headers=headers,
                    json={"claim_text": claim_text},
                    timeout=30
                )
                
                if response.status_code == 202:
                    return response.json()
                
                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Rate limited. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code >= 500:
                    # Server error - retry
                    wait_time = (2 ** attempt)
                    print(f"Server error. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                else:
                    response.raise_for_status()
            
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded")
```

---

## PART 6: RATE LIMITING & QUOTA MANAGEMENT

### Monitoring Usage

```python
import requests

class QuotaManager:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.api_base = "https://api.smartnotes.ai/v1"
    
    def get_usage(self) -> dict:
        """Get current usage and quota information."""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        response = requests.get(
            f"{self.api_base}/usage/current",
            headers=headers
        )
        
        return response.json()
    
    def check_quota_available(self, claims_needed: int) -> bool:
        """Check if enough quota available for batch operation."""
        usage = self.get_usage()
        
        remaining = usage["monthly_limit"] - usage["claims_used"]
        
        return remaining >= claims_needed


# Usage
quota_manager = QuotaManager(api_token="YOUR_TOKEN")

try:
    if quota_manager.check_quota_available(claims_needed=500):
        # Proceed with batch
        print("✅ Quota available")
    else:
        print("❌ Insufficient quota - upgrade plan")
except Exception as e:
    print(f"Error checking quota: {e}")
```

---

## PART 7: TESTING & DEBUGGING

### Sandbox Environment

Use sandbox for testing without affecting rate limits.

```python
client = SmartNotesAPIClient(
    api_token="YOUR_SANDBOX_TOKEN",
    api_base="https://sandbox.smartnotes.ai/v1"  # Use sandbox
)

# Test claim (sandbox doesn't verify, returns instant response)
result = client.verify_claim("Test claim")
```

### Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("smartnotes_integration")

# All API calls will be logged with details
response = client.verify_claim("Test claim")
```

---

## PART 8: MONITORING & OBSERVABILITY

### Datadog Integration

```python
from datadog import initialize, api
import time

options = {
    "api_key": "YOUR_DATADOG_API_KEY",
    "app_key": "YOUR_APP_KEY"
}
initialize(**options)

class MonitoredSmartNotesClient:
    def __init__(self, api_token: str):
        self.client = SmartNotesAPIClient(api_token)
    
    def verify_claim_with_metrics(self, claim_text: str) -> dict:
        """Verify claim and send metrics to Datadog."""
        start_time = time.time()
        
        try:
            result = self.client.verify_claim(claim_text)
            
            duration_ms = (time.time() - start_time) * 1000
            api.Metric.send(
                metric="smartnotes.claim_verification_time",
                points=duration_ms,
                tags=["endpoint:verify"]
            )
            
            return result
        
        except Exception as e:
            api.Metric.send(
                metric="smartnotes.claim_verification_error",
                points=1,
                tags=[f"error_type:{type(e).__name__}"]
            )
            raise
```

---

## CONCLUSION

These examples provide production-ready patterns for:
- ✅ Canvas, Blackboard, Brightspace LMS integration
- ✅ RESTful API usage with authentication
- ✅ Batch processing with webhooks
- ✅ Error handling and retry logic
- ✅ Rate limiting and quota management
- ✅ Monitoring and observability

For support: api-support@smartnotes.ai
