# 🎉 Node Configuration - Complete!

## ✅ Ab Har Node Ko Configure Kar Sakte Ho!

### Kya Add Kiya:
- ✅ **Configure Button** - Har node ke liye
- ✅ **Custom Conditions** - If/else logic
- ✅ **API Configuration** - Method, URL, headers, body
- ✅ **Email Templates** - To, subject, message
- ✅ **Database Operations** - Insert, update, delete, select
- ✅ **Transform Rules** - Data transformation
- ✅ **Delay Settings** - Duration aur unit
- ✅ **Custom Parameters** - Har node ke liye

---

## 🎯 Node-wise Configuration:

### 1. ❓ Condition Node

**Configure Kar Sakte Ho:**
- **Condition Type:**
  - Equals (==)
  - Not Equals (!=)
  - Greater Than (>)
  - Less Than (<)
  - Contains
  - Starts With
  - Ends With
  - Is Empty
  - Is Not Empty

- **Field to Check:** email, status, amount, etc.
- **Value to Compare:** VIP, completed, 100, etc.

**Example:**
```
Condition Type: Equals
Field: customerType
Value: VIP
```

---

### 2. 🌐 API Call Node

**Configure Kar Sakte Ho:**
- **HTTP Method:** GET, POST, PUT, PATCH, DELETE
- **API URL:** https://api.example.com/endpoint
- **Headers (JSON):**
```json
{
  "Authorization": "Bearer token123",
  "Content-Type": "application/json"
}
```
- **Body (JSON):**
```json
{
  "name": "John",
  "email": "john@example.com"
}
```

**Example:**
```
Method: POST
URL: https://api.stripe.com/v1/charges
Headers: {"Authorization": "Bearer sk_test_..."}
Body: {"amount": 1000, "currency": "usd"}
```

---

### 3. 📧 Email Node

**Configure Kar Sakte Ho:**
- **To Email:** recipient@example.com
- **Subject:** Welcome to our platform!
- **Message:**
```
Hi {{name}},

Welcome to Computer Genie!

Thanks,
Team
```

**Example:**
```
To: customer@example.com
Subject: Order Confirmation #12345
Message: Your order has been confirmed...
```

---

### 4. 💾 Database Node

**Configure Kar Sakte Ho:**
- **Operation:** Insert, Update, Delete, Select
- **Table Name:** users, orders, products
- **Data (JSON):**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "status": "active"
}
```

**Example:**
```
Operation: Insert
Table: customers
Data: {"name": "John", "email": "john@example.com"}
```

---

### 5. 🔄 Transform Node

**Configure Kar Sakte Ho:**
- **Transform Type:**
  - Map Fields
  - Filter Data
  - Format Data
  - Merge Data
  - Split Data

- **Transformation Rules (JSON):**
```json
{
  "oldField": "newField",
  "email": "userEmail",
  "phone": "contactNumber"
}
```

**Example:**
```
Transform Type: Map Fields
Rules: {"firstName": "name", "emailAddress": "email"}
```

---

### 6. ⏱️ Delay Node

**Configure Kar Sakte Ho:**
- **Delay Duration:** 5, 10, 30, etc.
- **Time Unit:** Seconds, Minutes, Hours, Days

**Example:**
```
Duration: 24
Unit: Hours
(Wait 24 hours before next step)
```

---

### 7. ⚡ Trigger Node

**Configure Kar Sakte Ho:**
- **Description:** What triggers this workflow
- **Custom Parameters (JSON):**
```json
{
  "source": "email",
  "filter": "VIP customers only"
}
```

---

### 8. ⚙️ Action Node

**Configure Kar Sakte Ho:**
- **Description:** What action to perform
- **Custom Parameters (JSON):**
```json
{
  "action": "send_notification",
  "channel": "slack"
}
```

---

## 🚀 Kaise Use Karein:

### Step-by-Step:

**1. Node Add Karo:**
```
Click "Add Node" → Select type
```

**2. Node Select Karo:**
```
Click on the node
```

**3. Configure Button Click Karo:**
```
Click "Configure" button (indigo color)
```

**4. Settings Fill Karo:**
```
- Condition node: Type, field, value
- API node: Method, URL, headers, body
- Email node: To, subject, message
- Database node: Operation, table, data
- Transform node: Type, rules
- Delay node: Duration, unit
```

**5. Save Karo:**
```
Click "Save Configuration"
```

**6. Workflow Run Karo:**
```
Click "Run" button
Configuration use hoga execution mein
```

---

## 💡 Real Examples:

### Example 1: VIP Customer Check
```
❓ Condition Node Configuration:
- Condition Type: Equals
- Field: customerType
- Value: VIP

If true → Send premium email
If false → Send standard email
```

### Example 2: Stripe Payment
```
🌐 API Node Configuration:
- Method: POST
- URL: https://api.stripe.com/v1/charges
- Headers: {"Authorization": "Bearer sk_..."}
- Body: {"amount": 2900, "currency": "usd"}
```

### Example 3: Welcome Email
```
📧 Email Node Configuration:
- To: {{customer.email}}
- Subject: Welcome to Computer Genie!
- Message: Hi {{customer.name}}, Thanks for signing up...
```

### Example 4: Save to Database
```
💾 Database Node Configuration:
- Operation: Insert
- Table: customers
- Data: {
    "name": "{{name}}",
    "email": "{{email}}",
    "signupDate": "{{date}}"
  }
```

### Example 5: Wait Before Follow-up
```
⏱️ Delay Node Configuration:
- Duration: 3
- Unit: Days

(Wait 3 days before sending follow-up email)
```

---

## 🎯 Complete Workflow Example:

### E-commerce Order Processing:

**Node 1: ⚡ Trigger**
```
Description: New order received
Parameters: {"source": "shopify"}
```

**Node 2: ❓ Condition**
```
Condition Type: Greater Than
Field: orderAmount
Value: 100
```

**Node 3a: 📧 Email (If > 100)**
```
To: {{customer.email}}
Subject: Thank you for your premium order!
Message: Your order #{{orderId}} is being processed...
```

**Node 3b: 📧 Email (If <= 100)**
```
To: {{customer.email}}
Subject: Order Confirmation
Message: Your order #{{orderId}} is confirmed...
```

**Node 4: 💾 Database**
```
Operation: Insert
Table: orders
Data: {"orderId": "{{orderId}}", "status": "processing"}
```

**Node 5: 🌐 API Call**
```
Method: POST
URL: https://shipping-api.com/create
Body: {"orderId": "{{orderId}}", "address": "{{address}}"}
```

**Node 6: ⏱️ Delay**
```
Duration: 1
Unit: Days
```

**Node 7: 📧 Follow-up Email**
```
To: {{customer.email}}
Subject: How was your order?
Message: We'd love your feedback...
```

---

## 🎨 UI Features:

### Configure Button:
- Indigo color
- Appears when node selected
- Opens configuration modal

### Configuration Modal:
- Large modal (max-width: 2xl)
- Scrollable content
- Different fields for each node type
- JSON editors for complex data
- Cancel/Save buttons

### Field Types:
- Text inputs
- Number inputs
- Email inputs
- Textareas
- Select dropdowns
- JSON editors (monospace font)

---

## 📊 Configuration Storage:

### How It's Saved:
```typescript
node.data.config = {
  // Condition
  conditionType: 'equals',
  field: 'customerType',
  value: 'VIP',
  
  // API
  method: 'POST',
  url: 'https://api.example.com',
  headers: '{"Authorization": "Bearer ..."}',
  body: '{"key": "value"}',
  
  // Email
  to: 'user@example.com',
  subject: 'Welcome',
  message: 'Hello...',
  
  // Database
  operation: 'insert',
  table: 'users',
  data: '{"name": "John"}',
  
  // Transform
  transformType: 'map',
  rules: '{"old": "new"}',
  
  // Delay
  duration: '5',
  unit: 'minutes',
  
  // Custom
  description: 'Description',
  parameters: '{"key": "value"}'
}
```

---

## 🔧 Technical Details:

### State Management:
```typescript
const [showConfigModal, setShowConfigModal] = useState(false);
const [nodeConfig, setNodeConfig] = useState<any>({});
```

### Save Configuration:
```typescript
const saveConfig = () => {
  setNodes((nds) =>
    nds.map((node) =>
      node.id === selectedNode.id
        ? { ...node, data: { ...node.data, config: nodeConfig } }
        : node
    )
  );
};
```

### Load Configuration:
```typescript
const configureNode = () => {
  setNodeConfig(selectedNode.data.config || {});
  setShowConfigModal(true);
};
```

---

## 🎊 What You Can Build Now:

### Complex Workflows:
1. **Multi-step Email Campaigns**
   - Trigger → Condition → Email → Delay → Follow-up

2. **Payment Processing**
   - Order → API (Stripe) → Database → Email

3. **Data Pipelines**
   - Trigger → Transform → Validate → Database → API

4. **Customer Onboarding**
   - Signup → Email → Delay → Survey → Database

5. **Order Fulfillment**
   - Order → Payment → Inventory → Shipping → Notification

---

## 🚀 Platform Status:

### Complete Features:
- ✅ Add unlimited nodes
- ✅ 8 node types
- ✅ Edit node names
- ✅ **Configure nodes (NEW!)**
- ✅ **Custom conditions (NEW!)**
- ✅ **API configuration (NEW!)**
- ✅ **Email templates (NEW!)**
- ✅ **Database operations (NEW!)**
- ✅ **Transform rules (NEW!)**
- ✅ **Delay settings (NEW!)**
- ✅ Delete nodes
- ✅ Connect nodes
- ✅ Run workflows
- ✅ Execution logs
- ✅ Save workflows

---

## 🎯 Quick Test:

```bash
# Visit
http://localhost:3000/dashboard/workflows

# Try:
1. Add a Condition node
2. Click on it to select
3. Click "Configure" button
4. Set:
   - Condition Type: Equals
   - Field: status
   - Value: VIP
5. Click "Save Configuration"
6. Add more nodes and configure them
7. Run the workflow!
```

---

## 💡 Pro Tips:

### Best Practices:
1. **Clear Names:** Give nodes descriptive names
2. **Test Configs:** Test each configuration
3. **Use Variables:** Use {{variable}} syntax
4. **JSON Format:** Validate JSON before saving
5. **Save Often:** Save workflow regularly

### Common Use Cases:
1. **If/Else Logic:** Use Condition nodes
2. **API Integration:** Use API nodes with proper auth
3. **Email Automation:** Use Email nodes with templates
4. **Data Storage:** Use Database nodes
5. **Data Transformation:** Use Transform nodes
6. **Timing:** Use Delay nodes for scheduling

---

**Status: 🟢 FULLY CONFIGURED**
**Node Configuration: 100% COMPLETE**
**Ready for: COMPLEX AUTOMATIONS**

**Ab har node ko customize karo aur powerful workflows banao! 🚀💪**
