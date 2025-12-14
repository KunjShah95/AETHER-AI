"""
API Client - HTTP Request Testing Tool
Provides comprehensive API testing capabilities similar to Postman/Insomnia
"""

import json
import time
import requests
from typing import Dict, Optional, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
import os

console = Console()


class APIClient:
    """Professional API testing and HTTP client"""
    
    def __init__(self):
        self.session = requests.Session()
        self.headers = {}
        self.auth = None
        self.base_url = ""
        self.collections = {}
        self.history = []
        self.max_history = 100
        self._load_collections()
        
    def _load_collections(self):
        """Load saved request collections"""
        try:
            collections_path = os.path.join(os.path.expanduser("~"), ".aetherai", "api_collections.json")
            if os.path.exists(collections_path):
                with open(collections_path, 'r') as f:
                    self.collections = json.load(f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load collections: {e}[/yellow]")
    
    def _save_collections(self):
        """Save request collections to disk"""
        try:
            collections_path = os.path.join(os.path.expanduser("~"), ".aetherai", "api_collections.json")
            os.makedirs(os.path.dirname(collections_path), exist_ok=True)
            with open(collections_path, 'w') as f:
                json.dump(self.collections, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving collections: {e}[/red]")
    
    def set_headers(self, headers_str: str) -> str:
        """Set custom headers from key:value pairs"""
        try:
            headers = {}
            for pair in headers_str.split(','):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    headers[key.strip()] = value.strip()
            self.headers.update(headers)
            return f"✅ Headers set: {', '.join(headers.keys())}"
        except Exception as e:
            return f"❌ Error setting headers: {str(e)}"
    
    def set_auth(self, auth_type: str, credentials: str) -> str:
        """Set authentication"""
        try:
            if auth_type.lower() == "bearer":
                self.headers["Authorization"] = f"Bearer {credentials}"
                return "✅ Bearer token authentication set"
            elif auth_type.lower() == "basic":
                from requests.auth import HTTPBasicAuth
                username, password = credentials.split(':', 1)
                self.auth = HTTPBasicAuth(username, password)
                return "✅ Basic authentication set"
            elif auth_type.lower() == "apikey":
                key, value = credentials.split(':', 1)
                self.headers[key] = value
                return f"✅ API key authentication set ({key})"
            else:
                return f"❌ Unknown auth type. Use: bearer, basic, or apikey"
        except Exception as e:
            return f"❌ Error setting auth: {str(e)}"
    
    def _make_request(self, method: str, url: str, data: Optional[str] = None, 
                      params: Optional[Dict] = None) -> Dict[str, Any]:
        """Internal method to make HTTP requests"""
        try:
            # Add base URL if set
            full_url = url if url.startswith('http') else f"{self.base_url}{url}"
            
            # Parse data if provided
            json_data = None
            if data:
                try:
                    json_data = json.loads(data)
                except json.JSONDecodeError:
                    # If not JSON, treat as form data
                    json_data = None
            
            start_time = time.time()
            
            # Make request
            response = self.session.request(
                method=method.upper(),
                url=full_url,
                json=json_data,
                headers=self.headers,
                auth=self.auth,
                params=params,
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            # Parse response
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            result = {
                "status_code": response.status_code,
                "status_text": response.reason,
                "elapsed": elapsed,
                "headers": dict(response.headers),
                "data": response_data,
                "url": full_url,
                "method": method.upper()
            }
            
            # Add to history
            self.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": method.upper(),
                "url": full_url,
                "status": response.status_code,
                "time": f"{elapsed:.3f}s"
            })
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            return result
            
        except requests.exceptions.Timeout:
            return {"error": "Request timeout (30s)"}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection error - check URL and internet"}
        except Exception as e:
            return {"error": str(e)}
    
    def _format_response(self, result: Dict[str, Any]) -> str:
        """Format response for display"""
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        # Status indicator
        status = result['status_code']
        if status < 300:
            status_color = "green"
            status_icon = "✅"
        elif status < 400:
            status_color = "yellow"
            status_icon = "⚠️"
        else:
            status_color = "red"
            status_icon = "❌"
        
        # Build response display
        output = []
        output.append(f"\n{status_icon} [{status_color}]{result['method']} {result['url']}[/{status_color}]")
        output.append(f"Status: [{status_color}]{status} {result['status_text']}[/{status_color}]")
        output.append(f"Time: {result['elapsed']:.3f}s")
        
        # Response headers (key ones)
        important_headers = ['content-type', 'content-length', 'server']
        header_info = []
        for key in important_headers:
            if key in result['headers']:
                header_info.append(f"{key}: {result['headers'][key]}")
        if header_info:
            output.append(f"Headers: {', '.join(header_info)}")
        
        # Response body
        data = result['data']
        if isinstance(data, dict) or isinstance(data, list):
            json_str = json.dumps(data, indent=2)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            console.print("\n" + "\n".join(output))
            console.print("\n📄 Response Body:")
            console.print(syntax)
            return ""
        else:
            output.append(f"\n📄 Response:\n{str(data)[:1000]}")
        
        return "\n".join(output)
    
    def get(self, url: str, params: Optional[str] = None) -> str:
        """Make GET request"""
        param_dict = {}
        if params:
            for pair in params.split('&'):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    param_dict[key] = value
        
        result = self._make_request("GET", url, params=param_dict)
        return self._format_response(result)
    
    def post(self, url: str, data: str) -> str:
        """Make POST request"""
        result = self._make_request("POST", url, data=data)
        return self._format_response(result)
    
    def put(self, url: str, data: str) -> str:
        """Make PUT request"""
        result = self._make_request("PUT", url, data=data)
        return self._format_response(result)
    
    def delete(self, url: str) -> str:
        """Make DELETE request"""
        result = self._make_request("DELETE", url)
        return self._format_response(result)
    
    def patch(self, url: str, data: str) -> str:
        """Make PATCH request"""
        result = self._make_request("PATCH", url, data=data)
        return self._format_response(result)
    
    def save_collection(self, name: str, method: str, url: str, 
                       data: Optional[str] = None, headers: Optional[str] = None) -> str:
        """Save request to collection"""
        try:
            self.collections[name] = {
                "method": method.upper(),
                "url": url,
                "data": data,
                "headers": headers,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self._save_collections()
            return f"✅ Request '{name}' saved to collection"
        except Exception as e:
            return f"❌ Error saving collection: {str(e)}"
    
    def list_collections(self) -> str:
        """List all saved collections"""
        if not self.collections:
            return "📋 No saved collections yet"
        
        table = Table(title="🗂️ Saved API Collections", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="white")
        table.add_column("Method", style="yellow")
        table.add_column("URL", style="green")
        table.add_column("Created", style="dim")
        
        for name, req in self.collections.items():
            table.add_row(
                name,
                req['method'],
                req['url'][:50] + "..." if len(req['url']) > 50 else req['url'],
                req.get('created', 'Unknown')
            )
        
        console.print(table)
        return ""
    
    def run_collection(self, name: str) -> str:
        """Run a saved request from collection"""
        if name not in self.collections:
            return f"❌ Collection '{name}' not found"
        
        req = self.collections[name]
        method = req['method'].lower()
        
        # Restore headers if saved
        if req.get('headers'):
            self.set_headers(req['headers'])
        
        # Execute request
        if method == "get":
            return self.get(req['url'])
        elif method == "post":
            return self.post(req['url'], req.get('data', '{}'))
        elif method == "put":
            return self.put(req['url'], req.get('data', '{}'))
        elif method == "delete":
            return self.delete(req['url'])
        elif method == "patch":
            return self.patch(req['url'], req.get('data', '{}'))
        else:
            return f"❌ Unknown method: {method}"
    
    def show_history(self) -> str:
        """Show request history"""
        if not self.history:
            return "📜 No request history yet"
        
        table = Table(title="📜 Request History", show_header=True, header_style="bold cyan")
        table.add_column("Time", style="dim")
        table.add_column("Method", style="yellow")
        table.add_column("URL", style="green")
        table.add_column("Status", style="white")
        table.add_column("Duration", style="cyan")
        
        for req in reversed(self.history[-20:]):  # Last 20 requests
            status_style = "green" if req['status'] < 300 else ("yellow" if req['status'] < 400 else "red")
            table.add_row(
                req['timestamp'],
                req['method'],
                req['url'][:40] + "..." if len(req['url']) > 40 else req['url'],
                f"[{status_style}]{req['status']}[/{status_style}]",
                req['time']
            )
        
        console.print(table)
        return ""
    
    def test_endpoint(self, url: str, expected_status: int = 200) -> str:
        """Test API endpoint"""
        result = self._make_request("GET", url)
        
        if "error" in result:
            return f"❌ Test Failed: {result['error']}"
        
        actual_status = result['status_code']
        if actual_status == expected_status:
            return f"✅ Test Passed: {url}\n   Status: {actual_status} (Expected: {expected_status})\n   Time: {result['elapsed']:.3f}s"
        else:
            return f"❌ Test Failed: {url}\n   Status: {actual_status} (Expected: {expected_status})\n   Time: {result['elapsed']:.3f}s"
    
    def benchmark(self, url: str, num_requests: int = 10) -> str:
        """Benchmark API endpoint"""
        try:
            num_requests = min(max(1, num_requests), 100)  # Limit to 100
            
            console.print(f"\n🏃 Benchmarking {url} with {num_requests} requests...")
            
            times = []
            statuses = {}
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task(f"Running {num_requests} requests...", total=num_requests)
                
                for i in range(num_requests):
                    result = self._make_request("GET", url)
                    if "error" not in result:
                        times.append(result['elapsed'])
                        status = result['status_code']
                        statuses[status] = statuses.get(status, 0) + 1
                    progress.update(task, advance=1)
            
            if not times:
                return "❌ All requests failed"
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            output = [
                f"\n📊 Benchmark Results for {url}",
                f"Total Requests: {num_requests}",
                f"Successful: {len(times)}",
                f"",
                f"⏱️ Response Times:",
                f"  Average: {avg_time:.3f}s",
                f"  Min: {min_time:.3f}s",
                f"  Max: {max_time:.3f}s",
                f"",
                f"📈 Status Codes:"
            ]
            
            for status, count in sorted(statuses.items()):
                output.append(f"  {status}: {count} requests")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Benchmark error: {str(e)}"
    
    def clear_history(self) -> str:
        """Clear request history"""
        self.history = []
        return "✅ Request history cleared"
    
    def reset(self) -> str:
        """Reset all headers and auth"""
        self.headers = {}
        self.auth = None
        return "✅ Headers and authentication reset"
