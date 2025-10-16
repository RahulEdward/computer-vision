# Workflow Builder Usage Guide

## ✅ Fixed Issues

1. **Black Header Removed** - Workflow ab full-screen hai
2. **Node Count Fixed** - Ab sirf 2 initial nodes hain (Process Data, API Call)
3. **Condition Node Available** - Left sidebar mein "ACTION" category mein

## 🎯 How to Use Workflow Builder

### Adding Nodes

**Left Sidebar mein 6 categories hain:**

1. **TRIGGER** - Workflow start karne ke liye
   - Manual Trigger
   - Webhook
   - Schedule Trigger

2. **ACTION** - Actions perform karne ke liye
   - HTTP Request
   - Data Transform
   - **Condition** ← Yeh aapka condition node hai!

3. **TRANSFORM** - Data transform karne ke liye
   - Transform

4. **CORE** - Core functionality

5. **INTEGRATION** - External services

6. **UTILITY** - Helper functions

### Connecting Nodes

1. **Node pe hover karo** - Left aur right side pe connection points dikhenge
2. **Drag karo** - Ek node ke right point se dusre node ke left point tak
3. **Connection ban jayega** - Animated line dikhega

### Editing Nodes

1. **Node pe click karo** - Right sidebar mein "Workflow Settings" panel khulega
2. **Properties edit karo** - Node ki settings change kar sakte ho
3. **Save karo** - Bottom pe "Save Workflow" button hai

### Node Count

- **Top pe "Nodes: X" dikhta hai** - Yeh canvas pe kitne nodes hain wo count karta hai
- Abhi 2 nodes hain: Process Data aur API Call

## 🔧 Quick Actions

### Add Condition Node
1. Left sidebar mein scroll karo
2. "ACTION" category dhundo
3. "Condition" node pe click karo
4. Canvas pe add ho jayega

### Connect Nodes
1. Process Data node pe hover karo
2. Right side ka connection point pakdo
3. Condition node tak drag karo
4. Connection ban jayega!

### Delete Node
1. Node select karo (click karke)
2. Delete/Backspace key press karo

### Move Nodes
1. Node ko drag karo
2. Jahan chahiye wahan drop karo

## 📊 Current Setup

```
Canvas:
├── Process Data (Node 1)
└── API Call (Node 2)

Available in Sidebar:
├── TRIGGER
│   ├── Manual Trigger
│   ├── Webhook
│   └── Schedule Trigger
├── ACTION
│   ├── HTTP Request
│   ├── Data Transform
│   └── Condition ← Use this!
└── TRANSFORM
    └── Transform
```

## 💡 Tips

- **Zoom**: Mouse wheel ya bottom-right controls use karo
- **Pan**: Canvas ko drag karo (empty space pe click karke)
- **Fit View**: Bottom-right pe fit button hai
- **Mini Map**: Bottom-right corner mein overview dikhta hai

## 🎨 Node Colors

- **Green** = Trigger nodes
- **Blue** = Action nodes  
- **Purple** = Transform nodes
- **Orange** = Condition nodes
- **Gray** = Core/Utility nodes

## ⚡ Next Steps

1. Left sidebar se "Condition" node add karo
2. Process Data ko Condition se connect karo
3. Condition ko API Call se connect karo
4. Right sidebar se properties edit karo
5. Bottom pe "Save Workflow" click karo

Enjoy building workflows! 🚀
