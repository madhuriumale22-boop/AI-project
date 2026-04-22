import re
import urllib.parse
import tldextract

def is_ip_address(domain):
    # Regex for checking if a domain is an IPv4 or IPv6 address
    ipv4_pattern = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
    ipv6_pattern = re.compile(r"^(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}$", re.IGNORECASE)
    if ipv4_pattern.match(domain) or ipv6_pattern.match(domain):
        return 1
    return 0

def has_suspicious_keywords(url):
    suspicious_keywords = ['login', 'verify', 'bank', 'secure', 'update', 'account', 'auth', 'signin']
    url_lower = url.lower()
    for keyword in suspicious_keywords:
        if keyword in url_lower:
            return 1
    return 0

def uses_shortener(domain):
    # Common URL shortening services
    shorteners = [
        "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", 
        "buff.ly", "adf.ly", "bit.do", "mcaf.ee", "su.pr", "cutt.ly"
    ]
    if any(shortener in domain for shortener in shorteners):
        return 1
    return 0

def extract_features(url):
    """
    Extracts features from a given URL to be used by the ML model.
    Returns a list of numerical features.
    """
    # Parse URL
    parsed_url = urllib.parse.urlparse(url)
    
    # If scheme is not present, parsing might not correctly identify netloc, so let's handle that
    if not parsed_url.scheme:
        url = "http://" + url
        parsed_url = urllib.parse.urlparse(url)
    
    domain = parsed_url.netloc
    path = parsed_url.path
    
    # Extract domain parts
    extracted = tldextract.extract(url)
    
    # 1. URL length
    url_length = len(url)
    
    # 2. Count of .
    dot_count = url.count('.')
    
    # 3. Count of -
    dash_count = url.count('-')
    
    # 4. Count of /
    slash_count = url.count('/')
    
    # 5. Count of digits
    digit_count = sum(c.isdigit() for c in url)
    
    # 6. Count of subdomains
    # tldextract groups all subdomains into one string separated by dots.
    # E.g., 'a.b.c' -> 'a.b.c', so split by dot. If empty, it's 0.
    subdomain_count = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
    
    # 7. Contains @ or not
    contains_at = 1 if '@' in url else 0
    
    # 8. Contains https or not
    contains_https = 1 if parsed_url.scheme == 'https' else 0
    
    # 9. Contains IP address or not
    contains_ip = is_ip_address(domain)
    
    # 10. Contains suspicious keywords or not
    contains_suspicious = has_suspicious_keywords(url)
    
    # 11. Uses URL shortener or not
    uses_short = uses_shortener(domain)
    
    return [
        url_length,
        dot_count,
        dash_count,
        slash_count,
        digit_count,
        subdomain_count,
        contains_at,
        contains_https,
        contains_ip,
        contains_suspicious,
        uses_short
    ]
