#!/usr/bin/env python3
# /// script
# dependencies = [
#   "huggingface-hub>=0.23",
#   "PyYAML>=6.0"
# ]
# ///
"""
CLI script to download and manage LLM models from HuggingFace.
Usage: python3 download_model.py [search|download] [query]
Compatible with uv: `uv run download_model.py`.
"""

import os
import sys
import json
import argparse
import subprocess
import urllib.request
import urllib.error
import yaml
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# Optional: huggingface_hub for snapshot_download (installed lazily)
try:
    from huggingface_hub import snapshot_download
except ImportError:  # install on demand
    snapshot_download = None


def _load_env_file() -> None:
    """Load variables from a simple .env file if they exist and are not defined."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


MODELS_DIR = Path("/models")
MODELS_YAML = Path(__file__).parent / "models.yaml"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_hf_hub():
    """Ensure huggingface_hub is available."""
    global snapshot_download
    if snapshot_download is not None:
        return
    try:
        import importlib
        # try ensurepip first if pip missing
        try:
            import pip  # noqa: F401
        except ImportError:
            try:
                import ensurepip
                ensurepip.bootstrap()
            except Exception:
                # fallback: get-pip.py
                import urllib.request, tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix="get-pip.py") as tmp:
                    tmp.write(urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py").read())
                    tmp.flush()
                    import subprocess, sys as _sys
                    subprocess.check_call([_sys.executable, tmp.name])
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub>=0.23"], stdout=subprocess.DEVNULL)
        snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    except Exception as e:
        print(f"{Colors.RED}❌ Failed to install huggingface_hub: {e}{Colors.RESET}")
        snapshot_download = None

# Terminal colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def search_huggingface(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for models on HuggingFace Hub"""
    try:
        print(f"{Colors.CYAN}🔍 Searching for '{query}' on HuggingFace...{Colors.RESET}")
        url = "https://huggingface.co/api/models"
        params = {
            "search": query,
            "limit": limit,
            "sort": "downloads",
            "direction": -1,
            "filter": "text-generation"
        }
        
        # Build URL with parameters
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        full_url = f"{url}?{query_string}"
        
        req = urllib.request.Request(full_url)
        req.add_header('User-Agent', 'llm-gateway-downloader/1.0')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            models = json.loads(response.read().decode('utf-8'))
        
        if not models:
            print(f"{Colors.YELLOW}⚠️  No models found.{Colors.RESET}")
            return []
        
        return models
    except Exception as e:
        print(f"{Colors.RED}❌ Search error: {e}{Colors.RESET}")
        return []


def format_size(size_in_bytes: int) -> str:
    """Converts bytes to readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f}{unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f}PB"


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a model"""
    try:
        urls = [
            f"https://huggingface.co/api/models/{model_id}?expand=files",
            f"https://huggingface.co/api/models/{model_id}"
        ]
        for url in urls:
            try:
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'llm-gateway-downloader/1.0')
                with urllib.request.urlopen(req, timeout=10) as response:
                    return json.loads(response.read().decode('utf-8'))
            except urllib.error.HTTPError:
                continue
    except Exception as e:
        print(f"{Colors.RED}❌ Error getting info: {e}{Colors.RESET}")
        return None


def _total_size_bytes(info: Dict[str, Any]) -> Optional[int]:
    """Sum the size of all known files in the repo."""
    try:
        # Direct field if available
        used = info.get('usedStorage')
        if isinstance(used, (int, float)) and used > 0:
            return int(used)
        siblings = info.get('siblings', [])
        sizes = [s.get('size') for s in siblings if isinstance(s, dict) and s.get('size')]
        total = sum(sizes) if sizes else None
        return total
    except Exception:
        return None


def _params_label(info: Dict[str, Any]) -> str:
    """Try to extract the number of parameters (in B) from tags or metadata."""
    try:
        tags = info.get('tags', []) or []
        # format params:70b
        for t in tags:
            if isinstance(t, str) and t.lower().startswith('params:'):
                return t.split(':', 1)[1].upper()
        # tags with '7b', '32b', '1.5b'
        for t in tags:
            if isinstance(t, str) and re.match(r"^\d+(\.\d+)?b$", t.lower()):
                return t.upper()
        # metadata with numeric parameter
        card = info.get('model_card_data') or {}
        n_params = card.get('parameters') or card.get('n_parameters')
        if isinstance(n_params, (int, float)) and n_params > 0:
            return f"{n_params/1e9:.1f}B"
        # infer from model name
        for key in [info.get('id', ''), info.get('modelId', '')]:
            if not key:
                continue
            m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", key)
            if m:
                return f"{float(m.group(1)):.1f}B" if '.' in m.group(1) else f"{m.group(1)}B"
    except Exception:
        pass
    return "N/A"


def display_models(models: List[Dict[str, Any]]) -> None:
    """Display found models in table format"""
    if not models:
        return
    
    print(f"\n{Colors.BOLD}┌─ Search results:{Colors.RESET}")
    
    for i, model in enumerate(models, 1):
        model_id = model.get('id', 'N/A')
        likes = model.get('likes', 0)
        downloads = model.get('downloads', 0)
        
        # Get detailed info (size and parameters)
        size_str = 'N/A'
        params_str = 'N/A'
        last_modified = 'N/A'
        try:
            info = get_model_info(model_id)
            if info:
                total_bytes = _total_size_bytes(info)
                if total_bytes:
                    size_str = f"{total_bytes/1024**3:.1f} GB"
                params_str = _params_label(info)
                last_modified = info.get('lastModified') or info.get('last_modified', 'N/A')
        except Exception:
            pass
        
        print(f"\n{Colors.CYAN}[{i}]{Colors.RESET} {Colors.BOLD}{model_id}{Colors.RESET}")
        print(f"    ❤️  Likes: {likes:,} | 📥 Downloads: {downloads:,}")
        print(f"    📦 Size: {size_str} | 🧠 Parameters: {params_str}")
        print(f"    🕐 Updated: {last_modified}")
        
        # Show description if exists
        if model.get('description'):
            desc = model['description'][:80] + "..." if len(model['description']) > 80 else model['description']
            print(f"    📝 {desc}")


def download_model(model_id: str, allow_patterns: Optional[str] = None) -> bool:
    """Download a model from HuggingFace. Tries to use huggingface_hub cache (/models/hub)."""
    try:
        print(f"\n{Colors.CYAN}📥 Downloading {model_id}...{Colors.RESET}")
        
        # Get model info
        info = get_model_info(model_id)
        if not info:
            print(f"{Colors.RED}❌ Cannot get model information{Colors.RESET}")
            return False
        
        print(f"{Colors.GREEN}✓ Model found{Colors.RESET}")
        
        # Try snapshot_download to HF cache
        cache_dir = MODELS_DIR / "hub"
        cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = None
        _ensure_hf_hub()
        if snapshot_download:
            try:
                print(f"{Colors.YELLOW}Downloading with huggingface_hub (cache at {cache_dir})...{Colors.RESET}")
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(cache_dir),
                    resume_download=True,
                    local_files_only=False,
                    allow_patterns=allow_patterns,
                )
                print(f"{Colors.GREEN}✓ Download complete{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}⚠️ Fallback to git (snapshot_download failed: {e}){Colors.RESET}")
                downloaded_path = None
        
        if downloaded_path is None:
            # Fallback: git clone to /models/<model_id>
            model_dir = MODELS_DIR / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"{Colors.YELLOW}Cloning repository with git to {model_dir}...{Colors.RESET}")
            git_url = f"https://huggingface.co/{model_id}"
            result = subprocess.run(
                ["git", "clone", git_url, str(model_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                if "already exists" in result.stderr:
                    print(f"{Colors.YELLOW}⚠️  Model already exists at {model_dir}{Colors.RESET}")
                else:
                    print(f"{Colors.RED}❌ Clone error: {result.stderr}{Colors.RESET}")
                    return False
            else:
                print(f"{Colors.GREEN}✓ Repository cloned successfully{Colors.RESET}")
            downloaded_path = str(model_dir)
        
        # Show path
        print(f"\n{Colors.GREEN}✅ Model available at:{Colors.RESET}")
        print(f"   {Colors.BOLD}{downloaded_path}{Colors.RESET}")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Download error: {e}{Colors.RESET}")
        return False


def add_model_to_config(model_id: str, gpu: str = "0", priority: int = 50) -> bool:
    """Add the model to models.yaml"""
    try:
        if not MODELS_YAML.exists():
            print(f"{Colors.RED}❌ {MODELS_YAML} not found{Colors.RESET}")
            return False
        
        with open(MODELS_YAML, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        if 'models' not in config:
            config['models'] = {}
        
        # Create container name
        container_name = f"vllm-{model_id.split('/')[-1].lower()}"
        
        # Default configuration
        new_model = {
            "container_name": container_name,
            "image": "vllm/vllm-openai:latest",
            "gpu": gpu,
            "gpus": f"device={gpu}" if gpu != "all" else "all",
            "min_free_gib": 30,
            "priority": priority,
            "warm": False,
            "host_port": 8001 + len(config['models']),
            "env": {
                "HF_HOME": "/models",
                "TZ": "Europe/Madrid"
            },
            "volumes": ["/models:/models"],
            "args": [
                "--host",
                "0.0.0.0",
                "--model",
                model_id,
                "--max-model-len",
                "2048",
                "--gpu-memory-utilization",
                "0.75"
            ]
        }
        
        # Use short model name
        model_key = model_id.split('/')[-1].lower().replace('_', '-')
        config['models'][model_key] = new_model
        
        # Save
        with open(MODELS_YAML, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"{Colors.GREEN}✅ Model added to {MODELS_YAML}{Colors.RESET}")
        print(f"   Key: {Colors.BOLD}{model_key}{Colors.RESET}")
        print(f"   Port: {new_model['host_port']}")
        print(f"   GPU: {gpu}")
        
        return True
        
    except Exception as e:
        print(f"{Colors.RED}❌ Error saving config: {e}{Colors.RESET}")
        return False


def interactive_search() -> None:
    """Interactive search"""
    print(f"\n{Colors.BOLD}=== HuggingFace Model Finder ==={Colors.RESET}\n")
    
    while True:
        query = input(f"{Colors.CYAN}🔍 Search for a model (or 'exit'): {Colors.RESET}").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query:
            continue
        
        models = search_huggingface(query, limit=5)
        if not models:
            continue
        
        display_models(models)
        
        try:
            choice = input(f"\n{Colors.CYAN}Choose model [1-{len(models)}] or press Enter to search again: {Colors.RESET}").strip()
            
            if not choice:
                continue
            
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_id = models[idx]['id']
                
                print(f"\n{Colors.BOLD}Configure {model_id}:{Colors.RESET}")
                gpu = input(f"  GPU [0]: ").strip() or "0"
                priority = input(f"  Priority [50]: ").strip() or "50"
                
                if download_model(model_id):
                    if input(f"\n{Colors.CYAN}Add to models.yaml? [y/n]: {Colors.RESET}").lower() in ['y', 'yes']:
                        add_model_to_config(model_id, gpu, int(priority))
                
            else:
                print(f"{Colors.RED}Invalid option{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Invalid input{Colors.RESET}")
            continue
        except KeyboardInterrupt:
            break


def main():
    # Load .env if exists (HF_TOKEN, etc.)
    _load_env_file()

    parser = argparse.ArgumentParser(
        description="LLM Model Downloader for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 download_model.py search "mistral"
  python3 download_model.py search "llama" --limit 10
  python3 download_model.py download "mistralai/Mistral-7B-Instruct-v0.1"
  python3 download_model.py download "meta-llama/Llama-2-7b" --gpu 0 --priority 60
  python3 download_model.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # search command
    search_parser = subparsers.add_parser('search', help='Search for models')
    search_parser.add_argument('query', help='Search term')
    search_parser.add_argument('--limit', type=int, default=5, help='Results limit')
    
    # download command
    download_parser = subparsers.add_parser('download', help='Download model')
    download_parser.add_argument('model_id', help='Model ID (e.g.: mistralai/Mistral-7B)')
    download_parser.add_argument('--gpu', default='0', help='GPU to use [default: 0]')
    download_parser.add_argument('--priority', type=int, default=50, help='Priority [default: 50]')
    download_parser.add_argument('--add-config', action='store_true', help='Add to models.yaml automatically')
    
    # interactive command
    subparsers.add_parser('interactive', help='Interactive mode with search')
    
    args = parser.parse_args()
    
    # No arguments, interactive mode
    if not args.command:
        interactive_search()
        return
    
    if args.command == 'search':
        models = search_huggingface(args.query, limit=args.limit)
        display_models(models)
    
    elif args.command == 'download':
        if download_model(args.model_id):
            if args.add_config:
                add_model_to_config(args.model_id, args.gpu, args.priority)
            else:
                if input(f"\n{Colors.CYAN}Add to models.yaml? [y/n]: {Colors.RESET}").lower() in ['y', 'yes']:
                    add_model_to_config(args.model_id, args.gpu, args.priority)
    
    elif args.command == 'interactive':
        interactive_search()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Cancelled by user{Colors.RESET}")
        sys.exit(0)
