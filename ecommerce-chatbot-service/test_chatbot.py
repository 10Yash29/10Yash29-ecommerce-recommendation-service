#!/usr/bin/env python3
"""
Test script for FootyBot Chatbot Service
Tests various endpoints and functionality
"""

import requests
import json
import time
from colorama import init, Fore, Style

init(autoreset=True)

BASE_URL = "http://localhost:8000"

def print_header(text):
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{text:^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

def print_success(text):
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_error(text):
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_info(text):
    print(f"{Fore.YELLOW}ℹ️  {text}{Style.RESET_ALL}")

def test_health_check():
    """Test health check endpoint"""
    print_header("Testing Health Check Endpoint")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Status: {data.get('status')}")
            
            components = data.get('components', {})
            print(f"   Components:")
            print(f"     - Index loaded: {components.get('index_loaded')}")
            print(f"     - Chunks loaded: {components.get('chunks_loaded')}")
            print(f"     - Embedder loaded: {components.get('embedder_loaded')}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to server. Is it running?")
        return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_chatbot_endpoint(message):
    """Test chatbot endpoint with a message"""
    print_header(f"Testing Chatbot: '{message}'")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/chatbot",
            json={"message": message},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Response received in {elapsed_time:.2f}s")
            print(f"\n{Fore.BLUE}User:{Style.RESET_ALL} {message}")
            print(f"{Fore.GREEN}Bot:{Style.RESET_ALL} {data.get('response')}")
            print(f"{Fore.YELLOW}Confidence:{Style.RESET_ALL} {data.get('confidence', 'unknown')}")
            return True
        else:
            print_error(f"Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print_error("Request timed out")
        return False
    except Exception as e:
        print_error(f"Chatbot error: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print_header("Testing Error Handling")
    
    # Test 1: Empty message
    print_info("Test 1: Empty message")
    try:
        response = requests.post(
            f"{BASE_URL}/chatbot",
            json={"message": ""},
            timeout=10
        )
        if response.status_code == 400:
            print_success("Empty message correctly rejected")
        else:
            print_error(f"Unexpected status: {response.status_code}")
    except Exception as e:
        print_error(f"Error: {e}")
    
    # Test 2: No message field
    print_info("Test 2: Missing message field")
    try:
        response = requests.post(
            f"{BASE_URL}/chatbot",
            json={},
            timeout=10
        )
        if response.status_code == 400:
            print_success("Missing message correctly rejected")
        else:
            print_error(f"Unexpected status: {response.status_code}")
    except Exception as e:
        print_error(f"Error: {e}")
    
    # Test 3: Very long message
    print_info("Test 3: Message exceeding limit")
    try:
        long_message = "a" * 501
        response = requests.post(
            f"{BASE_URL}/chatbot",
            json={"message": long_message},
            timeout=10
        )
        if response.status_code == 400:
            print_success("Long message correctly rejected")
        else:
            print_error(f"Unexpected status: {response.status_code}")
    except Exception as e:
        print_error(f"Error: {e}")

def main():
    print(f"\n{Fore.MAGENTA}{'*'*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'FootyBot Chatbot Test Suite':^60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'*'*60}{Style.RESET_ALL}\n")
    
    # Test health check
    if not test_health_check():
        print_error("\nServer is not running or not healthy. Exiting.")
        return
    
    # Test chatbot with various queries
    test_queries = [
        "What products do you sell?",
        "What is your return policy?",
        "Do you ship internationally?",
        "Tell me about your size guide",
        "How can I track my order?"
    ]
    
    for query in test_queries:
        test_chatbot_endpoint(query)
        time.sleep(1)  # Avoid rate limiting
    
    # Test error handling
    test_error_handling()
    
    print_header("Test Suite Complete!")
    print_success("All tests finished")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("\n\nTests interrupted by user")
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
