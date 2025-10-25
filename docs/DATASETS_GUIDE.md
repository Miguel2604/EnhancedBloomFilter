# Malicious URL Datasets Guide

## Overview
URL filtering is one of the most practical applications for Bloom Filters. You need both malicious (positive set) and benign (negative set) URLs to train and test the Learned Bloom Filter.

---

## 🔴 Malicious URL Datasets

### 1. **PhishTank** (Most Popular)
- **Description**: Community-driven database of verified phishing URLs
- **Size**: ~2M+ verified phishing URLs (updated hourly)
- **Format**: JSON, CSV, XML
- **Access**: Free with API key
- **Link**: https://phishtank.org/developer_info.php

**How to download**:
```bash
# Get free API key from website first
wget http://data.phishtank.com/data/YOUR_API_KEY/online-valid.json
wget http://data.phishtank.com/data/YOUR_API_KEY/online-valid.csv
```

**Sample data structure**:
```json
{
  "phish_id": "7930714",
  "url": "http://malicious-example.com/phish",
  "phish_detail_url": "http://phishtank.org/phish_detail.php?phish_id=7930714",
  "submission_time": "2023-01-15T20:31:42+00:00",
  "verified": "yes",
  "verification_time": "2023-01-15T20:35:12+00:00",
  "online": "yes",
  "target": "PayPal"
}
```

---

### 2. **URLhaus** (Malware URLs)
- **Description**: Malware URL exchange by abuse.ch
- **Size**: 500K+ malware distribution sites
- **Format**: CSV, TXT
- **Access**: Free, no registration
- **Link**: https://urlhaus.abuse.ch/downloads/

**Direct downloads**:
```bash
# Full database (30-day window)
wget https://urlhaus.abuse.ch/downloads/csv_recent/

# Currently online threats only
wget https://urlhaus.abuse.ch/downloads/csv_online/

# Full historical data
wget https://urlhaus.abuse.ch/downloads/csv/
```

**Fields**: url, threat_type, date_added, reporter, tags

---

### 3. **ISCX-URL-2016** (Academic Dataset)
- **Description**: University of New Brunswick dataset
- **Size**: 36,707 malicious + 35,300 benign URLs
- **Format**: CSV with extracted features
- **Access**: Free for research
- **Link**: https://www.unb.ca/cic/datasets/url-2016.html

**Features included**:
- URL length, number of dots, special characters
- Domain age, WHOIS data
- Already labeled for ML training

---

### 4. **Malicious URLs Dataset** (Kaggle)
- **Description**: Curated dataset for ML
- **Size**: 450K+ URLs (balanced classes)
- **Format**: CSV
- **Access**: Free with Kaggle account
- **Link**: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

**Download**:
```bash
# Install Kaggle CLI first
pip install kaggle

# Download dataset
kaggle datasets download -d sid321axn/malicious-urls-dataset
unzip malicious-urls-dataset.zip
```

**Structure**:
```csv
url,type
http://example.com,benign
http://phishing.site,malicious
http://defacement.org,defacement
```

---

### 5. **OpenPhish** (Active Phishing)
- **Description**: Real-time phishing feed
- **Size**: 10K+ active phishing URLs
- **Format**: TXT (one URL per line)
- **Access**: Free tier available
- **Link**: https://openphish.com/

**Free feed**:
```bash
wget https://openphish.com/feed.txt
```

---

### 6. **VirusTotal API** (Comprehensive)
- **Description**: Aggregated threat intelligence
- **Size**: Unlimited (API-based)
- **Format**: JSON API
- **Access**: Free tier: 500 requests/day
- **Link**: https://developers.virustotal.com/

**Python example**:
```python
import requests

api_key = "YOUR_API_KEY"
url_to_check = "http://suspicious-site.com"

headers = {"x-apikey": api_key}
url_id = base64.urlsafe_b64encode(url_to_check.encode()).decode().strip("=")
response = requests.get(
    f"https://www.virustotal.com/api/v3/urls/{url_id}",
    headers=headers
)
```

---

## 🟢 Benign URL Datasets

### 1. **Alexa Top 1 Million** (Discontinued but archived)
- **Description**: Top visited websites globally
- **Size**: 1M domains
- **Format**: CSV
- **Access**: Free (archived versions)
- **Link**: https://www.domcop.com/files/top/

**Alternative - Cisco Umbrella**:
```bash
wget http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip
unzip top-1m.csv.zip
```

### 2. **Majestic Million**
- **Description**: Top million websites by referring subnets
- **Size**: 1M domains
- **Format**: CSV
- **Access**: Free
- **Link**: https://majestic.com/reports/majestic-million

```bash
wget http://downloads.majestic.com/majestic_million.csv
```

### 3. **DMOZ/Open Directory** (Historical)
- **Description**: Human-curated directory
- **Size**: 5M+ URLs
- **Format**: RDF/XML
- **Access**: Free (archived)
- **Link**: https://dmoz-odp.org/

---

## 🔄 Mixed Datasets (Both Malicious & Benign)

### 1. **UCI Machine Learning Repository - URL Dataset**
- **Description**: Anonymized URL dataset
- **Size**: 2.4M URLs
- **Format**: SVM format
- **Link**: https://archive.ics.uci.edu/ml/datasets/URL+Reputation

**Features**: 3M+ features including lexical, host-based, and popularity metrics

### 2. **WEBSPAM-UK2007**
- **Description**: Web spam detection dataset
- **Size**: 105M pages from 114K hosts
- **Format**: Various
- **Link**: https://chato.cl/webspam/datasets/uk2007/

---

## 📊 Dataset Statistics & Recommendations

### For Your Thesis, I Recommend:

**Primary Dataset Combination**:
1. **Malicious**: PhishTank (500K recent) + URLhaus (200K malware)
2. **Benign**: Cisco Umbrella Top 100K
3. **Validation**: ISCX-URL-2016 (pre-labeled academic dataset)

**Why this combination**:
- Diverse threat types (phishing, malware, defacement)
- Recent, actively maintained data
- Good class balance (700K malicious, 100K benign)
- Academic validation set for comparison

---

## 🔧 Dataset Preparation Script

Create `scripts/download_datasets.py`:

```python
#!/usr/bin/env python3
"""
Download and prepare malicious URL datasets for Bloom Filter experiments
"""

import os
import json
import pandas as pd
import requests
from typing import Set, Tuple
import zipfile
import time

class URLDatasetDownloader:
    def __init__(self, data_dir="data/urls"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_phishtank(self, api_key=None) -> Set[str]:
        """Download PhishTank verified phishing URLs"""
        print("Downloading PhishTank dataset...")
        
        if api_key:
            url = f"http://data.phishtank.com/data/{api_key}/online-valid.json"
        else:
            # Use public feed (smaller, no API key needed)
            url = "http://data.phishtank.com/data/online-valid.json"
            
        response = requests.get(url)
        data = response.json()
        
        malicious_urls = set()
        for entry in data:
            if entry.get('verified') == 'yes' and entry.get('online') == 'yes':
                malicious_urls.add(entry['url'])
                
        print(f"  Downloaded {len(malicious_urls)} verified phishing URLs")
        return malicious_urls
    
    def download_urlhaus(self) -> Set[str]:
        """Download URLhaus malware URLs"""
        print("Downloading URLhaus dataset...")
        
        url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
        response = requests.get(url)
        
        malicious_urls = set()
        lines = response.text.split('\n')
        
        for line in lines[9:]:  # Skip header
            if line.strip():
                parts = line.split('","')
                if len(parts) > 2:
                    url = parts[2].strip('"')
                    if url.startswith('http'):
                        malicious_urls.add(url)
                        
        print(f"  Downloaded {len(malicious_urls)} malware URLs")
        return malicious_urls
    
    def download_openphish(self) -> Set[str]:
        """Download OpenPhish feed"""
        print("Downloading OpenPhish dataset...")
        
        url = "https://openphish.com/feed.txt"
        response = requests.get(url)
        
        malicious_urls = set()
        for line in response.text.split('\n'):
            url = line.strip()
            if url:
                malicious_urls.add(url)
                
        print(f"  Downloaded {len(malicious_urls)} phishing URLs")
        return malicious_urls
    
    def download_cisco_umbrella(self, top_n=100000) -> Set[str]:
        """Download Cisco Umbrella top domains"""
        print(f"Downloading Cisco Umbrella top {top_n} domains...")
        
        url = "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"
        zip_path = os.path.join(self.data_dir, "umbrella.zip")
        
        # Download zip file
        response = requests.get(url)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
            
        # Extract and read
        benign_urls = set()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
            
        csv_path = os.path.join(self.data_dir, "top-1m.csv")
        df = pd.read_csv(csv_path, names=['rank', 'domain'], nrows=top_n)
        
        for domain in df['domain']:
            # Add both HTTP and HTTPS versions
            benign_urls.add(f"http://{domain}")
            benign_urls.add(f"https://{domain}")
            
        # Cleanup
        os.remove(zip_path)
        os.remove(csv_path)
        
        print(f"  Downloaded {len(benign_urls)} benign URLs")
        return benign_urls
    
    def download_kaggle_dataset(self) -> Tuple[Set[str], Set[str]]:
        """Download Kaggle malicious URLs dataset (requires kaggle API)"""
        print("Downloading Kaggle dataset...")
        
        try:
            os.system("kaggle datasets download -d sid321axn/malicious-urls-dataset -p " + self.data_dir)
            
            zip_path = os.path.join(self.data_dir, "malicious-urls-dataset.zip")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                
            csv_path = os.path.join(self.data_dir, "malicious_phish.csv")
            df = pd.read_csv(csv_path)
            
            malicious = set(df[df['type'].isin(['malware', 'phishing', 'defacement'])]['url'])
            benign = set(df[df['type'] == 'benign']['url'])
            
            print(f"  Downloaded {len(malicious)} malicious and {len(benign)} benign URLs")
            return malicious, benign
            
        except Exception as e:
            print(f"  Failed to download Kaggle dataset: {e}")
            return set(), set()
    
    def save_datasets(self, malicious_urls: Set[str], benign_urls: Set[str]):
        """Save datasets to files"""
        print("\nSaving datasets...")
        
        # Save as text files
        with open(os.path.join(self.data_dir, "malicious_urls.txt"), 'w') as f:
            for url in malicious_urls:
                f.write(url + '\n')
                
        with open(os.path.join(self.data_dir, "benign_urls.txt"), 'w') as f:
            for url in benign_urls:
                f.write(url + '\n')
                
        # Save as CSV with labels
        all_urls = []
        for url in malicious_urls:
            all_urls.append({'url': url, 'label': 1, 'type': 'malicious'})
        for url in benign_urls:
            all_urls.append({'url': url, 'label': 0, 'type': 'benign'})
            
        df = pd.DataFrame(all_urls)
        df.to_csv(os.path.join(self.data_dir, "all_urls_labeled.csv"), index=False)
        
        print(f"  Saved {len(malicious_urls)} malicious URLs")
        print(f"  Saved {len(benign_urls)} benign URLs")
        print(f"  Total: {len(all_urls)} URLs")
        
        # Save statistics
        stats = {
            'malicious_count': len(malicious_urls),
            'benign_count': len(benign_urls),
            'total_count': len(all_urls),
            'malicious_ratio': len(malicious_urls) / len(all_urls),
            'download_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(self.data_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
            
    def download_all(self):
        """Download all available datasets"""
        print("Starting dataset download...\n")
        
        all_malicious = set()
        all_benign = set()
        
        # Download malicious URLs
        try:
            all_malicious.update(self.download_phishtank())
        except Exception as e:
            print(f"  PhishTank failed: {e}")
            
        try:
            all_malicious.update(self.download_urlhaus())
        except Exception as e:
            print(f"  URLhaus failed: {e}")
            
        try:
            all_malicious.update(self.download_openphish())
        except Exception as e:
            print(f"  OpenPhish failed: {e}")
            
        # Download benign URLs
        try:
            all_benign.update(self.download_cisco_umbrella(top_n=100000))
        except Exception as e:
            print(f"  Cisco Umbrella failed: {e}")
            
        # Try Kaggle dataset
        kaggle_mal, kaggle_ben = self.download_kaggle_dataset()
        all_malicious.update(kaggle_mal)
        all_benign.update(kaggle_ben)
        
        # Remove any overlap (URLs that appear in both sets)
        overlap = all_malicious & all_benign
        if overlap:
            print(f"\nRemoving {len(overlap)} URLs that appear in both sets")
            all_malicious -= overlap
            all_benign -= overlap
            
        # Save all datasets
        self.save_datasets(all_malicious, all_benign)
        
        return all_malicious, all_benign

if __name__ == "__main__":
    downloader = URLDatasetDownloader()
    malicious, benign = downloader.download_all()
    
    print("\n" + "="*50)
    print("Dataset Download Complete!")
    print("="*50)
    print(f"Malicious URLs: {len(malicious):,}")
    print(f"Benign URLs: {len(benign):,}")
    print(f"Total URLs: {len(malicious) + len(benign):,}")
    print(f"\nFiles saved in: data/urls/")
```

---

## 🧪 Dataset Quality Checks

```python
def validate_dataset(malicious_file, benign_file):
    """Validate dataset quality"""
    
    # Check for duplicates
    mal_urls = set(open(malicious_file).read().splitlines())
    ben_urls = set(open(benign_file).read().splitlines())
    
    print(f"Unique malicious: {len(mal_urls)}")
    print(f"Unique benign: {len(ben_urls)}")
    
    # Check for overlap
    overlap = mal_urls & ben_urls
    print(f"Overlapping URLs: {len(overlap)}")
    
    # Check URL validity
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    invalid_mal = [u for u in mal_urls if not url_pattern.match(u)]
    invalid_ben = [u for u in ben_urls if not url_pattern.match(u)]
    
    print(f"Invalid malicious URLs: {len(invalid_mal)}")
    print(f"Invalid benign URLs: {len(invalid_ben)}")
    
    # Distribution analysis
    from urllib.parse import urlparse
    
    mal_domains = [urlparse(u).netloc for u in mal_urls]
    ben_domains = [urlparse(u).netloc for u in ben_urls]
    
    print(f"Unique malicious domains: {len(set(mal_domains))}")
    print(f"Unique benign domains: {len(set(ben_domains))}")
```

---

## 🚀 Quick Start

1. **Install dependencies**:
```bash
pip install requests pandas kaggle
```

2. **Run the download script**:
```bash
python scripts/download_datasets.py
```

3. **Verify datasets**:
```bash
wc -l data/urls/*.txt
head -5 data/urls/malicious_urls.txt
head -5 data/urls/benign_urls.txt
```

---

## 📈 Expected Dataset Sizes

After running the download script, you should have approximately:
- **Malicious URLs**: 500,000 - 1,000,000
- **Benign URLs**: 200,000 - 400,000
- **Total dataset**: 700,000 - 1,400,000 URLs

This is perfect for:
- Training the Learned Bloom Filter
- Testing performance at scale
- Demonstrating real-world applicability

---

## ⚠️ Legal & Ethical Considerations

1. **Use for research only**: These URLs are for academic research
2. **Don't visit malicious URLs**: Use isolated environments if needed
3. **Respect rate limits**: Don't overload free APIs
4. **Attribution**: Cite dataset sources in your thesis
5. **No redistribution**: Don't share raw malicious URL lists publicly

---

## 📚 Citation Examples

For your thesis:

```bibtex
@misc{phishtank2023,
  title = {PhishTank: An Anti-Phishing Service},
  author = {PhishTank},
  year = {2023},
  url = {https://phishtank.org}
}

@misc{urlhaus2023,
  title = {URLhaus: Malware URL Exchange},
  author = {abuse.ch},
  year = {2023},
  url = {https://urlhaus.abuse.ch}
}

@article{mamun2016detecting,
  title = {Detecting Malicious URLs Using Lexical Analysis},
  author = {Mamun, Mohammad Saiful Islam and others},
  journal = {Network and System Security},
  year = {2016},
  publisher = {Springer}
}
```