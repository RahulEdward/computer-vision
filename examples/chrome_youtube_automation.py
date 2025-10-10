#!/usr/bin/env python3
"""
Chrome Browser + YouTube Automation Example
==========================================

यह example दिखाता है कि Computer Genie का उपयोग करके:
1. Chrome browser कैसे खोलें
2. YouTube पर कैसे navigate करें
3. Videos कैसे search करें और play करें
4. Browser automation के advanced features

Author: Computer Genie Team
"""

import asyncio
import time
import sys
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, List, Any

# Ensure we can import the local package
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Import Computer Genie components
    from computer_genie import VisionAgent
    from computer_genie.exceptions import ElementNotFoundError, ActionFailedError
    IMPORTS_AVAILABLE = True
    print("✅ Computer Genie components imported successfully")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("📝 Running in demo mode")
    IMPORTS_AVAILABLE = False


class ChromeYouTubeAutomation:
    """Chrome Browser और YouTube automation के लिए class."""
    
    def __init__(self):
        """Initialize the automation system."""
        self.agent = None
        self.browser_opened = False
        
    async def setup(self):
        """Setup the automation system."""
        print("🚀 Setting up Chrome YouTube Automation...")
        
        if IMPORTS_AVAILABLE:
            try:
                self.agent = VisionAgent()
                print("✅ VisionAgent initialized")
            except Exception as e:
                print(f"⚠️  VisionAgent setup failed: {e}")
                print("📝 Continuing in demo mode")
        else:
            print("🎭 Running in demo mode")
            
        print("✅ Setup completed")
    
    async def open_chrome_browser(self):
        """Chrome browser खोलें।"""
        print("\n🌐 Opening Chrome Browser...")
        
        try:
            if IMPORTS_AVAILABLE and self.agent:
                # Real mode: Use VisionAgent to interact with Chrome
                print("  🔍 Using VisionAgent to open Chrome...")
                await asyncio.sleep(1)
                
                # Try to find and click Chrome icon
                try:
                    await self.agent.click("Chrome")
                    print("  ✅ Chrome opened via VisionAgent")
                except ElementNotFoundError:
                    print("  📝 Chrome icon not found, using system command")
                    self._open_chrome_system()
            else:
                # Demo mode: Use system command
                print("  🎭 Demo mode: Opening Chrome via system command")
                self._open_chrome_system()
                
            self.browser_opened = True
            await asyncio.sleep(2)  # Wait for browser to load
            print("  ✅ Chrome browser opened successfully")
            
        except Exception as e:
            print(f"  ❌ Failed to open Chrome: {e}")
            return False
            
        return True
    
    def _open_chrome_system(self):
        """System command के द्वारा Chrome खोलें।"""
        try:
            # Windows में Chrome खोलने के different ways
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                "chrome"  # If Chrome is in PATH
            ]
            
            for chrome_path in chrome_paths:
                try:
                    subprocess.Popen([chrome_path])
                    print(f"  ✅ Chrome opened using: {chrome_path}")
                    return True
                except FileNotFoundError:
                    continue
                    
            # Fallback: Use webbrowser module
            webbrowser.open("about:blank")
            print("  ✅ Browser opened using webbrowser module")
            return True
            
        except Exception as e:
            print(f"  ⚠️  System Chrome open failed: {e}")
            return False
    
    async def navigate_to_youtube(self):
        """YouTube पर navigate करें।"""
        print("\n📺 Navigating to YouTube...")
        
        try:
            if IMPORTS_AVAILABLE and self.agent:
                # Real mode: Use VisionAgent
                print("  🔍 Using VisionAgent to navigate...")
                
                # Click on address bar
                try:
                    await self.agent.click("address bar")
                    await asyncio.sleep(0.5)
                except ElementNotFoundError:
                    print("  📝 Address bar not found, trying alternative method")
                    # Press Ctrl+L to focus address bar
                    await self.agent.key_combination(["ctrl", "l"])
                    await asyncio.sleep(0.5)
                
                # Type YouTube URL
                await self.agent.type("https://www.youtube.com")
                await asyncio.sleep(0.5)
                
                # Press Enter
                await self.agent.key("enter")
                await asyncio.sleep(3)  # Wait for page to load
                
                print("  ✅ Navigated to YouTube using VisionAgent")
                
            else:
                # Demo mode: Use webbrowser
                print("  🎭 Demo mode: Opening YouTube in browser")
                webbrowser.open("https://www.youtube.com")
                await asyncio.sleep(2)
                print("  ✅ YouTube opened in browser")
                
        except Exception as e:
            print(f"  ❌ Failed to navigate to YouTube: {e}")
            return False
            
        return True
    
    async def search_video(self, search_query: str):
        """YouTube पर video search करें।"""
        print(f"\n🔍 Searching for: '{search_query}'...")
        
        try:
            if IMPORTS_AVAILABLE and self.agent:
                # Real mode: Use VisionAgent
                print("  🔍 Using VisionAgent to search...")
                
                # Find and click search box
                try:
                    await self.agent.click("search")
                    await asyncio.sleep(0.5)
                except ElementNotFoundError:
                    print("  📝 Search box not found, trying alternative")
                    # Press / to focus search (YouTube shortcut)
                    await self.agent.key("/")
                    await asyncio.sleep(0.5)
                
                # Type search query
                await self.agent.type(search_query)
                await asyncio.sleep(0.5)
                
                # Press Enter to search
                await self.agent.key("enter")
                await asyncio.sleep(3)  # Wait for search results
                
                print(f"  ✅ Searched for '{search_query}' using VisionAgent")
                
            else:
                # Demo mode: Simulate search
                print(f"  🎭 Demo mode: Simulating search for '{search_query}'")
                await asyncio.sleep(2)
                print(f"  ✅ Search simulation completed")
                
        except Exception as e:
            print(f"  ❌ Failed to search: {e}")
            return False
            
        return True
    
    async def play_first_video(self):
        """पहली video को play करें।"""
        print("\n▶️  Playing first video...")
        
        try:
            if IMPORTS_AVAILABLE and self.agent:
                # Real mode: Use VisionAgent
                print("  🔍 Using VisionAgent to play video...")
                
                # Find and click first video thumbnail
                try:
                    await self.agent.click("video thumbnail")
                    await asyncio.sleep(2)
                    print("  ✅ First video clicked using VisionAgent")
                except ElementNotFoundError:
                    print("  📝 Video thumbnail not found, trying alternative")
                    # Try clicking on video title
                    await self.agent.click("video title")
                    await asyncio.sleep(2)
                    print("  ✅ Video title clicked")
                
            else:
                # Demo mode: Simulate video play
                print("  🎭 Demo mode: Simulating video play")
                await asyncio.sleep(2)
                print("  ✅ Video play simulation completed")
                
        except Exception as e:
            print(f"  ❌ Failed to play video: {e}")
            return False
            
        return True
    
    async def control_video_playback(self):
        """Video playback को control करें।"""
        print("\n🎮 Controlling video playback...")
        
        try:
            if IMPORTS_AVAILABLE and self.agent:
                # Real mode: Use VisionAgent
                print("  🔍 Using VisionAgent for playback control...")
                
                # Pause video (spacebar)
                print("    ⏸️  Pausing video...")
                await self.agent.key("space")
                await asyncio.sleep(2)
                
                # Resume video (spacebar again)
                print("    ▶️  Resuming video...")
                await self.agent.key("space")
                await asyncio.sleep(2)
                
                # Adjust volume (up arrow)
                print("    🔊 Increasing volume...")
                await self.agent.key("up")
                await asyncio.sleep(1)
                
                # Fullscreen (f key)
                print("    🖥️  Entering fullscreen...")
                await self.agent.key("f")
                await asyncio.sleep(2)
                
                # Exit fullscreen (escape)
                print("    🪟 Exiting fullscreen...")
                await self.agent.key("escape")
                await asyncio.sleep(1)
                
                print("  ✅ Video playback controlled using VisionAgent")
                
            else:
                # Demo mode: Simulate controls
                print("  🎭 Demo mode: Simulating playback controls")
                controls = ["Pause", "Resume", "Volume Up", "Fullscreen", "Exit Fullscreen"]
                for control in controls:
                    print(f"    🎮 {control}...")
                    await asyncio.sleep(0.5)
                print("  ✅ Playback control simulation completed")
                
        except Exception as e:
            print(f"  ❌ Failed to control playback: {e}")
            return False
            
        return True
    
    async def run_complete_demo(self, search_query: str = "Python programming tutorial"):
        """Complete YouTube automation demo चलाएं।"""
        print("🎯 Starting Complete Chrome YouTube Automation Demo")
        print("=" * 60)
        
        results = {
            'browser_opened': False,
            'youtube_loaded': False,
            'search_completed': False,
            'video_played': False,
            'controls_tested': False
        }
        
        try:
            # Step 1: Open Chrome
            results['browser_opened'] = await self.open_chrome_browser()
            
            # Step 2: Navigate to YouTube
            if results['browser_opened']:
                results['youtube_loaded'] = await self.navigate_to_youtube()
            
            # Step 3: Search for video
            if results['youtube_loaded']:
                results['search_completed'] = await self.search_video(search_query)
            
            # Step 4: Play first video
            if results['search_completed']:
                results['video_played'] = await self.play_first_video()
            
            # Step 5: Control playback
            if results['video_played']:
                results['controls_tested'] = await self.control_video_playback()
            
            # Generate report
            self.generate_automation_report(results, search_query)
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            
        return results
    
    def generate_automation_report(self, results: Dict, search_query: str):
        """Automation results की report generate करें।"""
        print("\n📊 CHROME YOUTUBE AUTOMATION REPORT")
        print("=" * 50)
        
        success_count = sum(results.values())
        total_steps = len(results)
        success_rate = (success_count / total_steps) * 100
        
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
        print(f"📝 Search Query: '{search_query}'")
        
        print("\n📋 Step-by-Step Results:")
        step_names = {
            'browser_opened': '🌐 Chrome Browser Opening',
            'youtube_loaded': '📺 YouTube Navigation',
            'search_completed': '🔍 Video Search',
            'video_played': '▶️  Video Playback',
            'controls_tested': '🎮 Playback Controls'
        }
        
        for step, result in results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {step_names[step]}")
        
        print(f"\n🚀 AUTOMATION CAPABILITIES:")
        print(f"  ✅ Browser Launch & Control")
        print(f"  ✅ Web Navigation")
        print(f"  ✅ Element Detection & Interaction")
        print(f"  ✅ Keyboard & Mouse Automation")
        print(f"  ✅ Video Platform Integration")
        
        if success_rate >= 80:
            print(f"\n🎉 EXCELLENT: Automation system working perfectly!")
        elif success_rate >= 60:
            print(f"\n👍 GOOD: Most features working correctly")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT: Some features need attention")
        
        print(f"\n💡 Use Cases:")
        print(f"   • Automated video content testing")
        print(f"   • Educational content navigation")
        print(f"   • Social media automation")
        print(f"   • Browser-based task automation")
        print(f"   • Quality assurance testing")


async def main():
    """Main execution function."""
    print("🚀 Chrome YouTube Automation Example")
    print("=" * 40)
    
    automation = ChromeYouTubeAutomation()
    
    try:
        # Setup
        await automation.setup()
        
        # Run complete demo
        search_query = input("\n🔍 Enter search query (or press Enter for default): ").strip()
        if not search_query:
            search_query = "Python programming tutorial"
        
        results = await automation.run_complete_demo(search_query)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Automation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Automation failed: {e}")
        return 1


if __name__ == "__main__":
    print("🎬 Starting Chrome YouTube Automation...")
    exit_code = asyncio.run(main())
    print(f"\n🏁 Automation completed with exit code: {exit_code}")
    sys.exit(exit_code)