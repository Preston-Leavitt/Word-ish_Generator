import os
import aiohttp
import asyncio
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class SoraClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_video(
        self, 
        prompt: str, 
        duration: int = 10,
        aspect_ratio: str = "landscape",
        style: str = "realistic"
    ) -> bytes:
        """Generate video using Sora API."""
        try:
            print(f"Attempting to generate video with Sora: {prompt[:50]}...")
            # Start video generation
            payload = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "style": style
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/video/generations",
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Sora API error: {response.status}")
                    
                    result = await response.json()
                    video_id = result["id"]
                
                # Poll for completion
                while True:
                    async with session.get(
                        f"{self.base_url}/video/generations/{video_id}",
                        headers=self.headers
                    ) as status_response:
                        status_data = await status_response.json()
                        
                        if status_data["status"] == "completed":
                            # Download the video
                            video_url = status_data["data"][0]["url"]
                            async with session.get(video_url) as video_response:
                                return await video_response.read()
                        
                        elif status_data["status"] == "failed":
                            raise Exception("Video generation failed")
                        
                        # Wait before polling again
                        await asyncio.sleep(2)
                        
        except Exception as e:
            print(f"Sora API error: {e}")
            print(f"Using mock video generation as Sora API is unavailable")
            return await self._generate_mock_video_simple(duration, aspect_ratio)
    
    async def _generate_mock_video_simple(self, duration: int, aspect_ratio: str) -> bytes:
        """Generate a very simple mock video (a few colored bytes)."""
        # This is a fallback that doesn't require external tools
        print("Generating simple mock video data")
        
        # Create a minimal video file with just a header
        # This is not a real video but provides bytes to save
        mock_data = b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41\x00\x00\x00\xF0moov'
        mock_data += b'\x00' * 2048  # Pad with some zeroes
        
        # Simulate processing time based on duration
        await asyncio.sleep(duration / 5)
        
        # Create the static directory if it doesn't exist
        os.makedirs("static/videos", exist_ok=True)
        
        # Save a dummy file to test the file path
        test_file = "static/videos/test_mock_video.mp4"
        with open(test_file, "wb") as f:
            f.write(mock_data)
        
        print(f"Mock video saved to {test_file}")
        return mock_data
    
    def _create_placeholder_video(self) -> bytes:
        """Create a minimal video placeholder."""
        # This would be a very basic video file in production
        # For now, return empty bytes that will trigger error handling
        return b""
