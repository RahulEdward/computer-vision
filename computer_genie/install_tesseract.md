# Tesseract OCR Installation Guide (Windows)

## Text Reading के लिए Tesseract OCR Install करें

### Step 1: Tesseract Download करें
1. यहां जाएं: https://github.com/UB-Mannheim/tesseract/wiki
2. Windows के लिए latest installer download करें
3. File name कुछ इस तरह होगी: `tesseract-ocr-w64-setup-v5.x.x.exe`

### Step 2: Install करें
1. Downloaded file को run करें
2. Installation wizard follow करें
3. Default settings रखें
4. Install location note करें (usually: `C:\Program Files\Tesseract-OCR`)

### Step 3: Environment Variable Set करें
1. Windows key + R दबाएं
2. `sysdm.cpl` type करें और Enter दबाएं
3. "Advanced" tab पर जाएं
4. "Environment Variables" button click करें
5. "System Variables" में "Path" select करें
6. "Edit" button click करें
7. "New" button click करें
8. Add करें: `C:\Program Files\Tesseract-OCR`
9. OK click करें सभी dialogs में

### Step 4: Verify Installation
Command prompt या PowerShell में run करें:
```bash
tesseract --version
```

### Step 5: Test with Computer Genie
```bash
python take_screenshot.py
```

## Alternative: Chocolatey से Install करें
अगर आपके पास Chocolatey है:
```bash
choco install tesseract
```

## Alternative: Conda से Install करें
अगर आपके पास Conda है:
```bash
conda install -c conda-forge tesseract
```

## Troubleshooting
- अगर command not found error आए तो computer restart करें
- Path में spaces हों तो quotes use करें
- Admin privileges की जरूरत हो सकती है

## After Installation
Tesseract install होने के बाद आप:
1. Screenshot में text पढ़ सकेंगे
2. Software names detect कर सकेंगे  
3. Image में लिखा text extract कर सकेंगे