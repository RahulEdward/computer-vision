#!/usr/bin/env python3
"""Quick test script for Computer Genie"""

import asyncio
from computer_genie.vision.screenshort import Screenshot
from computer_genie.vision import OCR

async def quick_test():
    print("🤖 Computer Genie Quick Test")
    print("=" * 40)
    
    # Test screenshot
    screenshot = Screenshot()
    image = await screenshot.capture()
    
    if image:
        print("✅ Screenshot: OK")
        print(f"📏 Size: {image.size[0]} x {image.size[1]}")
        
        # Test OCR
        import numpy as np
        image_array = np.array(image)
        ocr = OCR()
        text = await ocr.extract_text(image_array)
        
        if "not available" not in text.lower():
            print("✅ OCR: OK")
            print(f"📖 Text found: {len(text.split())} words")
        else:
            print("⚠️  OCR: Tesseract not available")
    else:
        print("❌ Screenshot: Failed")
    
    screenshot.close()
    print("=" * 40)

if __name__ == "__main__":
    asyncio.run(quick_test())