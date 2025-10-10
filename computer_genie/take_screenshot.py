#!/usr/bin/env python3
"""
Screenshot और Text Reading Script
यह script screenshot लेती है और उसे file में save करती है
साथ ही screen पर मौजूद text भी पढ़ती है
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path so we can import computer_genie
sys.path.insert(0, str(Path(__file__).parent))

from computer_genie.vision.screenshort import Screenshot
from computer_genie.vision import OCR
from computer_genie.utils import setup_logger

async def take_screenshot_and_read_text():
    """Screenshot लेकर text भी पढ़ता है"""
    
    # Setup logging
    logger = setup_logger("screenshot_script")
    logger.info("Screenshot और Text Reading Script शुरू हो रही है...")
    
    try:
        # Screenshot object बनाएं
        screenshot = Screenshot()
        logger.info("Screenshot utility initialized")
        
        # Screenshot लें
        logger.info("Screenshot ली जा रही है...")
        image = await screenshot.capture()
        
        if image:
            # Screenshot को file में save करें
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshot_{timestamp}.png"
            image.save(screenshot_path)
            logger.info(f"Screenshot saved: {screenshot_path}")
            print(f"✅ Screenshot saved: {screenshot_path}")
            print(f"📏 Image size: {image.size}")
            
            # OCR के लिए image को numpy array में convert करें
            import numpy as np
            image_array = np.array(image)
            
            # OCR object बनाएं और text पढ़ें
            ocr = OCR()
            logger.info("Text reading शुरू हो रही है...")
            text = await ocr.extract_text(image_array)
            
            print(f"\n📖 Screen पर मिला Text:")
            print("=" * 50)
            if text and "not available" not in text.lower() and "error" not in text.lower():
                print(text)
            else:
                print("⚠️ Text reading के लिए Tesseract OCR install करना होगा")
                print("📥 Download link: https://github.com/UB-Mannheim/tesseract/wiki")
                print(f"🔍 OCR Status: {text}")
            print("=" * 50)
            
        else:
            logger.error("Screenshot लेने में error आई")
            print("❌ Screenshot लेने में problem हुई")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        screenshot.close()

async def main():
    """Main function"""
    print("🖥️  Computer Genie Screenshot & Text Reader")
    print("=" * 50)
    
    await take_screenshot_and_read_text()
    
    print("\n✨ Script completed!")
    print("💡 अगर text reading नहीं हो रही तो Tesseract install करें")

if __name__ == "__main__":
    # Run the script
    asyncio.run(main())