#!/usr/bin/env python3
"""
Screenshot Viewer और Text Reader
यह script screenshot लेती है, save करती है, और उसे open भी करती है
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path so we can import computer_genie
sys.path.insert(0, str(Path(__file__).parent))

from computer_genie.vision.screenshort import Screenshot
from computer_genie.vision import OCR
from computer_genie.utils import setup_logger

def open_image(image_path):
    """Image को default viewer में open करता है"""
    try:
        if os.name == 'nt':  # Windows
            os.startfile(image_path)
        elif os.name == 'posix':  # macOS/Linux
            subprocess.run(['open', image_path])  # macOS
            # subprocess.run(['xdg-open', image_path])  # Linux
        print(f"🖼️  Image opened: {image_path}")
        return True
    except Exception as e:
        print(f"❌ Image open करने में error: {e}")
        return False

async def take_and_view_screenshot():
    """Screenshot लेकर view करता है और text भी पढ़ता है"""
    
    # Setup logging
    logger = setup_logger("screenshot_viewer")
    
    print("🖥️  Computer Genie Screenshot Viewer & Text Reader")
    print("=" * 60)
    
    try:
        # Screenshot object बनाएं
        screenshot = Screenshot()
        logger.info("Screenshot utility initialized")
        
        # Screenshot लें
        print("📸 Screenshot ली जा रही है...")
        image = await screenshot.capture()
        
        if image:
            # Screenshot को file में save करें
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshot_{timestamp}.png"
            image.save(screenshot_path)
            
            print(f"✅ Screenshot saved: {screenshot_path}")
            print(f"📏 Image size: {image.size[0]} x {image.size[1]} pixels")
            print(f"💾 File size: {os.path.getsize(screenshot_path)} bytes")
            
            # Image को open करें
            print("\n🖼️  Opening screenshot...")
            if open_image(screenshot_path):
                print("✅ Screenshot opened in default viewer")
            
            # OCR के लिए image को numpy array में convert करें
            import numpy as np
            image_array = np.array(image)
            
            # OCR object बनाएं और text पढ़ें
            print("\n📖 Screen पर text reading...")
            ocr = OCR()
            text = await ocr.extract_text(image_array)
            
            print("\n" + "=" * 60)
            print("📖 SCREEN पर मिला TEXT:")
            print("=" * 60)
            
            if text and "not available" not in text.lower() and "error" not in text.lower():
                # Text को clean करें
                lines = text.strip().split('\n')
                cleaned_lines = [line.strip() for line in lines if line.strip()]
                
                if cleaned_lines:
                    for i, line in enumerate(cleaned_lines, 1):
                        print(f"{i:2d}. {line}")
                else:
                    print("🔍 कोई readable text नहीं मिला")
            else:
                print("⚠️  TEXT READING के लिए TESSERACT OCR INSTALL करना होगा")
                print("📥 Download: https://github.com/UB-Mannheim/tesseract/wiki")
                print(f"🔍 Status: {text}")
                print("\n💡 Installation guide: install_tesseract.md file देखें")
            
            print("=" * 60)
            
        else:
            print("❌ Screenshot लेने में problem हुई")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        screenshot.close()

async def main():
    """Main function"""
    await take_and_view_screenshot()
    
    print("\n✨ Script completed!")
    print("💡 Screenshot file आपके current directory में save हुई है")
    print("🔧 Text reading के लिए Tesseract install करें")
    
    # User को options दें
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("1. Screenshot file को manually भी open कर सकते हैं")
    print("2. Tesseract install करके text reading enable करें")
    print("3. Script को फिर से run करके text reading test करें")
    print("=" * 60)

if __name__ == "__main__":
    # Run the script
    asyncio.run(main())