"""
Unit tests for cookie format transformation in ApifyBackend
"""
import os
import sys
import unittest
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import under test
from tests.scrape_and_train import ApifyBackend

class TestCookieTransformation(unittest.TestCase):
    def test_transform_cookies_with_good_format(self):
        """Test with already well-formatted cookies"""
        input_cookies = ["li_at=ABCD1234"]
        result = ApifyBackend.transform_cookies(input_cookies)
        self.assertEqual(result, ["li_at=ABCD1234"])
    
    def test_transform_cookies_with_attributes(self):
        """Test with cookie containing extra attributes"""
        input_cookies = ["li_at=ABCD1234; Path=/; Domain=.linkedin.com; Expires=Thu, 01 Jan 2023 00:00:00 GMT"]
        result = ApifyBackend.transform_cookies(input_cookies)
        self.assertEqual(result, ["li_at=ABCD1234"])
    
    def test_transform_cookies_with_jsession(self):
        """Test with both li_at and JSESSIONID"""
        input_cookies = ["li_at=ABCD1234; JSESSIONID=ajax:1234567890"]
        result = ApifyBackend.transform_cookies(input_cookies)
        self.assertEqual(result, ["li_at=ABCD1234; JSESSIONID=ajax:1234567890"])
    
    def test_transform_cookies_with_encoded_values(self):
        """Test with URL-encoded values"""
        input_cookies = ["li_at=ABCD%201234%3D"]
        result = ApifyBackend.transform_cookies(input_cookies)
        self.assertEqual(result, ["li_at=ABCD 1234="])
    
    def test_transform_cookies_with_empty_input(self):
        """Test with empty input"""
        self.assertEqual(ApifyBackend.transform_cookies([]), [])
        self.assertEqual(ApifyBackend.transform_cookies([""]), [])
        self.assertEqual(ApifyBackend.transform_cookies(None), [])
    
    def test_transform_cookies_with_malformed_input(self):
        """Test with malformed input"""
        input_cookies = ["not_a_valid_cookie", "also_invalid=123"]
        result = ApifyBackend.transform_cookies(input_cookies)
        self.assertEqual(result, [])
    
    def test_transform_cookies_extracts_jsession_only_if_format_valid(self):
        """Test that only properly formatted JSESSIONID cookies are kept"""
        # Valid format 
        input_cookies = ["JSESSIONID=ajax:1234567890"]
        result = ApifyBackend.transform_cookies(input_cookies)
        self.assertEqual(result, ["JSESSIONID=ajax:1234567890"])
        
        # Invalid format (should be skipped)
        input_cookies = ["JSESSIONID=invalid_format_123"]
        result = ApifyBackend.transform_cookies(input_cookies)
        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()
