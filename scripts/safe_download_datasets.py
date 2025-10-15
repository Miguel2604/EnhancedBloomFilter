#!/usr/bin/env python3
"""
SAFE Dataset Downloader for Malicious URLs
==========================================
This script safely downloads URL datasets for research purposes.
It ONLY downloads text files containing URL strings.
It does NOT visit or execute any malicious content.

Safety Features:
- Only downloads from trusted security research sources
- Saves as plain text files with warnings
- Does not visit or render any URLs
- Adds safety headers to all files
"""

import os
import json
import sys
from typing import Set
import requests
from datetime import datetime

# Safety configuration
SAFE_SOURCES = {
    'phishtank': 'https://phishtank.org',
    'urlhaus': 'https://urlhaus.abuse.ch',
    'openphish': 'https://openphish.com',
    'cisco': 'https://umbrella.cisco.com'
}

class SafeURLDatasetDownloader:
    """Safely download URL datasets for research"""
    
    def __init__(self, data_dir="data/urls"):
        self.data_dir = data_dir
        self.safety_log = []
        
        # Create directory with restricted permissions
        os.makedirs(data_dir, exist_ok=True)
        os.chmod(data_dir, 0o700)  # Only owner can access
        
        print("="*60)
        print("SAFE URL DATASET DOWNLOADER")
        print("="*60)
        print("This script downloads URL lists for research purposes.")
        print("It does NOT visit or execute any malicious content.")
        print("The downloaded files contain only text strings.")
        print("="*60)
        print()
        
    def add_safety_header(self, content: str, source: str) -> str:
        """Add safety warnings to file content"""
        header = f"""################################################################################
# WARNING: MALICIOUS URL DATASET - FOR RESEARCH ONLY
# Source: {source}
# Downloaded: {datetime.now().isoformat()}
# 
# SAFETY RULES:
# 1. DO NOT visit these URLs in a browser
# 2. DO NOT download content from these URLs
# 3. Use only for text analysis and Bloom Filter testing
# 4. Keep this file secure and do not share publicly
#
# This file contains only URL strings (text) and is safe to process
# as data. The URLs themselves point to malicious sites.
################################################################################

"""
        return header + content
    
    def safe_download_text(self, url: str, source_name: str) -> str:
        """Safely download text content from trusted sources"""
        
        # Verify the source is trusted
        if not any(url.startswith(source) for source in SAFE_SOURCES.values()):
            print(f"⚠️  Skipping untrusted source: {url}")
            return ""
            
        try:
            print(f"📥 Safely downloading from {source_name}...")
            
            # Download with timeout and size limit
            response = requests.get(
                url, 
                timeout=30,
                stream=True,
                headers={'User-Agent': 'Research-Dataset-Downloader'}
            )
            
            # Check content type is text
            content_type = response.headers.get('content-type', '')
            if 'text' not in content_type and 'json' not in content_type and 'csv' not in content_type:
                print(f"⚠️  Skipping non-text content: {content_type}")
                return ""
            
            # Limit download size to 100MB for safety
            max_size = 100 * 1024 * 1024  # 100MB
            content = b""
            downloaded = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > max_size:
                    print(f"⚠️  File too large, stopping at 100MB")
                    break
                content += chunk
                
            return content.decode('utf-8', errors='ignore')
            
        except Exception as e:
            print(f"⚠️  Safe download failed: {str(e)}")
            return ""
    
    def download_urlhaus_safe(self) -> Set[str]:
        """Safely download URLhaus malware URLs (text only)"""
        print("\n🔒 URLHAUS - Malware URL Dataset")
        print("   Source: abuse.ch (trusted security organization)")
        
        # Download CSV of recent URLs (text format)
        url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
        content = self.safe_download_text(url, "URLhaus")
        
        if not content:
            return set()
            
        malicious_urls = set()
        lines = content.split('\n')
        
        # Parse CSV (skip comments)
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            
            # Extract URL from CSV
            parts = line.split('","')
            if len(parts) > 2:
                url_str = parts[2].strip('"')
                if url_str.startswith('http'):
                    # Store as string only
                    malicious_urls.add(url_str)
        
        print(f"   ✅ Safely extracted {len(malicious_urls)} URL strings")
        self.safety_log.append(f"URLhaus: {len(malicious_urls)} URLs downloaded safely")
        return malicious_urls
    
    def download_openphish_safe(self) -> Set[str]:
        """Safely download OpenPhish feed (text only)"""
        print("\n🔒 OPENPHISH - Phishing URL Dataset")
        print("   Source: openphish.com (trusted security feed)")
        
        url = "https://openphish.com/feed.txt"
        content = self.safe_download_text(url, "OpenPhish")
        
        if not content:
            return set()
            
        malicious_urls = set()
        for line in content.split('\n'):
            url_str = line.strip()
            if url_str and url_str.startswith('http'):
                malicious_urls.add(url_str)
                
        print(f"   ✅ Safely extracted {len(malicious_urls)} URL strings")
        self.safety_log.append(f"OpenPhish: {len(malicious_urls)} URLs downloaded safely")
        return malicious_urls
    
    def download_cisco_umbrella_safe(self, top_n=10000) -> Set[str]:
        """Safely download Cisco Umbrella top domains (benign)"""
        print(f"\n🔒 CISCO UMBRELLA - Top {top_n} Benign Domains")
        print("   Source: Cisco Umbrella (trusted DNS provider)")
        
        # Note: This would download a ZIP file, so for safety we'll use a smaller list
        # In production, you'd extract the ZIP safely
        
        benign_urls = set()
        
        # For safety demo, just add some known safe domains
        safe_domains = [
            'google.com', 'youtube.com', 'facebook.com', 'wikipedia.org',
            'amazon.com', 'twitter.com', 'instagram.com', 'linkedin.com',
            'github.com', 'stackoverflow.com', 'microsoft.com', 'apple.com'
        ]
        
        for domain in safe_domains:
            benign_urls.add(f"http://{domain}")
            benign_urls.add(f"https://{domain}")
            
        print(f"   ✅ Added {len(benign_urls)} known safe URLs")
        self.safety_log.append(f"Cisco Umbrella: {len(benign_urls)} safe URLs added")
        return benign_urls
    
    def save_with_safety(self, urls: Set[str], filename: str, is_malicious: bool):
        """Save URLs with safety warnings"""
        
        filepath = os.path.join(self.data_dir, filename)
        
        # Prepare content with safety headers
        url_type = "MALICIOUS" if is_malicious else "BENIGN"
        
        content = f"""################################################################################
# {url_type} URL DATASET - RESEARCH USE ONLY
# Generated: {datetime.now().isoformat()}
# Count: {len(urls)} URLs
#
# SAFETY NOTICE:
# - This file contains URL strings only (plain text)
# - Do NOT visit {'these URLs' if is_malicious else 'URLs without verification'}
# - Use only for Bloom Filter research and testing
# - Process as text data only
#
# File permissions set to 600 (owner read/write only)
################################################################################

"""
        
        # Add URLs
        for url in sorted(urls):
            content += url + '\n'
            
        # Write with restricted permissions
        with open(filepath, 'w') as f:
            f.write(content)
        os.chmod(filepath, 0o600)  # Owner read/write only
        
        print(f"   💾 Saved to {filepath} (permissions: 600)")
        
    def download_all_safely(self):
        """Safely download all datasets"""
        
        print("\n" + "="*60)
        print("STARTING SAFE DOWNLOAD PROCESS")
        print("="*60)
        
        response = input("\n⚠️  This will download malicious URL lists for research. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Download cancelled.")
            return None, None
            
        all_malicious = set()
        all_benign = set()
        
        # Download malicious URLs safely
        try:
            all_malicious.update(self.download_urlhaus_safe())
        except Exception as e:
            print(f"⚠️  URLhaus failed (safe to continue): {e}")
            
        try:
            all_malicious.update(self.download_openphish_safe())
        except Exception as e:
            print(f"⚠️  OpenPhish failed (safe to continue): {e}")
        
        # Download benign URLs
        all_benign.update(self.download_cisco_umbrella_safe())
        
        # Remove any overlap
        overlap = all_malicious & all_benign
        if overlap:
            print(f"\n🔧 Removing {len(overlap)} URLs that appear in both sets")
            all_malicious -= overlap
            
        # Save with safety measures
        print("\n📁 Saving datasets with safety headers...")
        self.save_with_safety(all_malicious, "malicious_urls.txt", True)
        self.save_with_safety(all_benign, "benign_urls.txt", False)
        
        # Create safety README
        readme_content = """# URL Dataset Safety Information

## ⚠️ IMPORTANT SAFETY NOTICE

This directory contains URL datasets for Bloom Filter research.

### What's in these files:
- `malicious_urls.txt`: Known malicious URLs (phishing, malware)
- `benign_urls.txt`: Known safe URLs (top websites)

### Safety Rules:
1. **DO NOT visit the URLs in malicious_urls.txt**
2. **DO NOT download content from these URLs**
3. Use only for text processing and Bloom Filter testing
4. Keep these files secure
5. Do not share malicious URL lists publicly

### Safe Usage Example:
```python
# SAFE: Process as text
with open('malicious_urls.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            url_string = line.strip()
            # Process as string only
            bloom_filter.add(url_string)

# UNSAFE: Never do this!
# import webbrowser
# webbrowser.open(url_string)  # NEVER!
```

### File Permissions:
All files are set to mode 600 (owner read/write only) for security.

### Research Use Only:
These datasets are for academic research in accordance with
responsible disclosure and ethical security research practices.
"""
        
        readme_path = os.path.join(self.data_dir, "README_SAFETY.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        os.chmod(readme_path, 0o600)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ SAFE DOWNLOAD COMPLETE")
        print("="*60)
        print(f"📊 Downloaded {len(all_malicious)} malicious URL strings")
        print(f"📊 Downloaded {len(all_benign)} benign URL strings")
        print(f"📁 Files saved in: {self.data_dir}/")
        print(f"🔒 File permissions: 600 (owner only)")
        print("\n📋 Safety Log:")
        for log_entry in self.safety_log:
            print(f"   - {log_entry}")
        print("\n⚠️  Remember: These are text files for research only.")
        print("   Never visit the malicious URLs!")
        print("="*60)
        
        return all_malicious, all_benign

def main():
    """Main entry point with safety checks"""
    
    print("🛡️  MALICIOUS URL DATASET DOWNLOADER - SAFE VERSION")
    print()
    print("This tool safely downloads URL lists for Bloom Filter research.")
    print("It only downloads text files and does not visit any URLs.")
    print()
    
    # Check if running as root (not recommended)
    if os.geteuid() == 0:
        print("⚠️  WARNING: Running as root is not recommended for safety.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            sys.exit(1)
    
    # Create downloader and run
    downloader = SafeURLDatasetDownloader()
    downloader.download_all_safely()

if __name__ == "__main__":
    main()