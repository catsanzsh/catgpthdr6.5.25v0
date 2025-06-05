from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import textwrap
import tkinter as tk
import importlib.util
import requests
import traceback
from datetime import datetime
from pathlib import Path
from threading import Thread
from types import ModuleType
from typing import Any, Dict, List, Tuple, Optional, Union

try:
    import aiohttp
    ASYNC_MODE = True
except ImportError:
    ASYNC_MODE = False

from tkinter import messagebox, scrolledtext, simpledialog

# ────────────────────────────── Logging Setup ───────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# ────────────────────────────── Runtime Globals ─────────────────────────────
RUNTIME_API_KEY: Optional[str] = None # Will store API key in memory

# ────────────────────────────── Runtime Paths ───────────────────────────────
HOME = Path.home()
ARCHIVE_DIR = HOME / "Documents" / "CatGPT_Agent_Archive"
PLUGIN_DIR = ARCHIVE_DIR / "plugins"
MEMORY_FILE = ARCHIVE_DIR / "memory.json"
MODEL_FILE = ARCHIVE_DIR / "models.json"
ARCHIVE_INDEX = ARCHIVE_DIR / "archive_index.json"

# Ensure directories exist
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
PLUGIN_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────── Constants ──────────────────────────────────
# DEFAULT_API_KEY has been removed. API key is now handled by get_api_key() and prompt.
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY" # Environment variable name
DEFAULT_MODELS = ["meta-llama/llama-4-maverick", "gpt-3.5-turbo", "claude-3-opus", "gpt-4"]
LLM_TIMEOUT = 120  # seconds

# UI Theme elements
UI_THEME = {
    "bg_primary": "#f5f5f5",
    "bg_secondary": "#ffffff",
    "bg_tertiary": "#f0e6ff", # Evolution archive background
    "bg_chat_display": "#fafafa",
    "bg_chat_input": "#f9f9f9",
    "bg_editor": "#1e2838",
    "bg_editor_header": "#34495e",
    "bg_button_primary": "#10a37f",
    "bg_button_secondary": "#3498db",
    "bg_button_danger": "#e74c3c",
    "bg_button_warning": "#f39c12",
    "bg_button_evolution": "#9b59b6",
    "bg_button_evo_compile": "#27ae60",
    "bg_listbox_select": "#6c5ce7", # Evolution archive listbox selection
    "fg_primary": "#2c3e50",
    "fg_secondary": "#ecf0f1", # For dark backgrounds like editor
    "fg_button_light": "#ffffff",
    "fg_evolution_header": "#6c5ce7",
    "font_default": ("Consolas", 11),
    "font_chat": ("Consolas", 11),
    "font_button_main": ("Arial", 11, "bold"),
    "font_button_small": ("Arial", 10),
    "font_title": ("Arial", 14, "bold"),
    "font_editor": ("Consolas", 11),
    "font_listbox": ("Consolas", 9),
}

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def now_ts() -> str:
    """Generates a timestamp string for the current time."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_api_key() -> str:
    """
    Returns the API key stored in memory (RUNTIME_API_KEY).
    This key is expected to be set at startup via environment variable or user prompt.
    """
    global RUNTIME_API_KEY
    if RUNTIME_API_KEY:
        return RUNTIME_API_KEY
    # This fallback primarily handles cases where the key wasn't set as expected.
    # The UI or initial setup should ensure RUNTIME_API_KEY is populated.
    logger.warning("API Key not found in memory. Ensure it was set at startup via prompt or environment variable.")
    return "" # Fallback to empty string

async def call_llm_async(session: aiohttp.ClientSession, payload: Dict[str, Any]) -> str:
    """
    Fires a single completion request to OpenRouter (asynchronously).
    Args:
        session: The aiohttp client session.
        payload: The request payload for the LLM.
    Returns:
        The LLM's response content.
    Raises:
        RuntimeError: If the API call fails or API key is missing.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = get_api_key()
    if not api_key:
        logger.error("API Key is missing. Cannot call LLM.")
        raise RuntimeError("API Key is missing. Please configure it by restarting or setting the OPENROUTER_API_KEY environment variable.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with session.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"LLM API call failed with status {resp.status}: {error_text}")
                raise RuntimeError(f"API Error (Status {resp.status}): {error_text}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    except aiohttp.ClientError as e:
        logger.error(f"AIOHTTP client error during LLM call: {e}")
        raise RuntimeError(f"Network Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during async LLM call: {e}")
        raise RuntimeError(f"Unexpected API Error: {e}")


def call_llm_sync(payload: Dict[str, Any]) -> str:
    """
    Fires a single completion request to OpenRouter (synchronously).
    Args:
        payload: The request payload for the LLM.
    Returns:
        The LLM's response content.
    Raises:
        RuntimeError: If the API call fails or API key is missing.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = get_api_key()
    if not api_key:
        logger.error("API Key is missing. Cannot call LLM.")
        raise RuntimeError("API Key is missing. Please configure it by restarting or setting the OPENROUTER_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        if response.status_code != 200:
            logger.error(f"LLM API call failed with status {response.status_code}: {response.text}")
            raise RuntimeError(f"API Error (Status {response.status_code}): {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.error(f"Requests error during LLM call: {e}")
        raise RuntimeError(f"Network Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during sync LLM call: {e}")
        raise RuntimeError(f"Unexpected API Error: {e}")

# ----------------------------------------------------------------------------
# DarwinAgent – Fusion of both agent systems
# ----------------------------------------------------------------------------

class DarwinAgent:
    """LLM agent that can fork, load, and evaluate child agents with fitness tracking."""

    def __init__(self):
        self.models: List[str] = self._load_models()
        self.cfg: Dict[str, Any] = {
            "model": self.models[0] if self.models else DEFAULT_MODELS[1], # Ensure a fallback
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        self.history: List[Dict[str, str]] = self._load_memory()
        # Archive stores (timestamp, filename, status, error_message_or_none)
        self.agent_archive: List[Tuple[str, str, str, Optional[str]]] = self._load_agent_archive()
        self.plugins: Dict[str, ModuleType] = {}
        self._discover_plugins()
        logger.info("DarwinAgent initialized.")

    # ────────── Persistence ──────────
    def _load_json_file(self, file_path: Path, default_value: Union[List, Dict]) -> Union[List, Dict]:
        """Helper to load JSON data from a file."""
        if file_path.exists():
            try:
                return json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {file_path}. Returning default.")
                return default_value
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}. Returning default.")
                return default_value
        return default_value

    def _save_json_file(self, file_path: Path, data: Union[List, Dict]):
        """Helper to save JSON data to a file."""
        try:
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {e}")

    def _load_memory(self) -> List[Dict[str, str]]:
        return self._load_json_file(MEMORY_FILE, [])

    def _save_memory(self):
        self._save_json_file(MEMORY_FILE, self.history[-2000:]) # Save last 2000 history items

    def _load_models(self) -> List[str]:
        models = self._load_json_file(MODEL_FILE, [])
        if not models: # If file didn't exist or was empty/corrupt
            logger.info(f"{MODEL_FILE} not found or empty. Writing default models.")
            self._save_json_file(MODEL_FILE, DEFAULT_MODELS)
            return DEFAULT_MODELS
        return models

    def _load_agent_archive(self) -> List[Tuple[str, str, str, Optional[str]]]:
        return self._load_json_file(ARCHIVE_INDEX, [])

    def _save_agent_archive(self):
        self._save_json_file(ARCHIVE_INDEX, self.agent_archive)

    # ────────── Plugin system ──────────
    def _discover_plugins(self):
        """Discovers and loads plugins from the PLUGIN_DIR."""
        if str(PLUGIN_DIR) not in sys.path:
            sys.path.insert(0, str(PLUGIN_DIR))

        loaded_plugins = {}
        for py_file in PLUGIN_DIR.glob("*.py"):
            name = py_file.stem
            if name == "__init__": # Skip __init__.py files
                continue
            try:
                spec = importlib.util.spec_from_file_location(name, py_file)
                if not spec or not spec.loader:
                    logger.warning(f"Could not create spec for plugin {name} at {py_file}")
                    continue
                
                # If plugin already loaded, try to reload it
                if name in self.plugins:
                    mod = importlib.reload(self.plugins[name])
                else:
                    mod = importlib.util.module_from_spec(spec)
                
                spec.loader.exec_module(mod)
                if hasattr(mod, "run"):
                    loaded_plugins[name] = mod
                    logger.info(f"Plugin '{name}' loaded successfully.")
                else:
                    logger.warning(f"Plugin {name} loaded but has no 'run' method.")
            except Exception as e:
                logger.error(f"Plugin '{name}' failed to load: {e}\n{traceback.format_exc()}")
        self.plugins = loaded_plugins


    async def _run_plugin_async(self, name: str, args: str) -> str:
        """Runs a plugin asynchronously."""
        plugin = self.plugins.get(name)
        if not plugin:
            return f"[plugin‑error] no such plugin: {name}"
        try:
            # Assuming plugin.run might be sync or async
            if inspect.iscoroutinefunction(plugin.run):
                ret = await plugin.run(args)
            else:
                ret = await asyncio.to_thread(plugin.run, args)
            return str(ret)
        except Exception as e:
            logger.error(f"Error running plugin {name} with args '{args}': {e}")
            return f"[plugin‑error] {name}: {e}"

    def _run_plugin_sync(self, name: str, args: str) -> str:
        """Runs a plugin synchronously."""
        plugin = self.plugins.get(name)
        if not plugin:
            return f"[plugin‑error] no such plugin: {name}"
        try:
            # Assuming plugin.run is synchronous
            if inspect.iscoroutinefunction(plugin.run):
                return f"[plugin‑error] {name}: cannot run async plugin synchronously"
            ret = plugin.run(args)
            return str(ret)
        except Exception as e:
            logger.error(f"Error running plugin {name} with args '{args}': {e}")
            return f"[plugin‑error] {name}: {e}"

    # ────────── Inference ──────────
    def _prepare_payload(self) -> Dict[str, Any]:
        """Prepares the payload for the LLM API call."""
        return {
            "model": self.cfg["model"],
            "messages": self.history[-20:],  # Truncate history for API context
            "temperature": float(self.cfg["temperature"]),
            "max_tokens": int(self.cfg["max_tokens"]),
            "top_p": float(self.cfg["top_p"]),
            "frequency_penalty": float(self.cfg["frequency_penalty"]),
            "presence_penalty": float(self.cfg["presence_penalty"]),
        }

    async def ask_async(self, user_msg: str) -> str:
        """Handles user input, either as a command or a message to the LLM (async)."""
        if user_msg.startswith("/model "):
            mdl = user_msg.split(maxsplit=1)[1]
            if mdl in self.models:
                self.cfg["model"] = mdl
                logger.info(f"Model switched to {mdl}")
                return f"Model switched to {mdl}"
            return f"Unknown model id {mdl}. Known: {', '.join(self.models)}"

        if user_msg.startswith("/tool "):
            _, rest = user_msg.split(maxsplit=1)
            name, *arg_tokens = rest.split(maxsplit=1)
            args = arg_tokens[0] if arg_tokens else ""
            return await self._run_plugin_async(name, args)

        self.history.append({"role": "user", "content": user_msg})
        payload = self._prepare_payload()
        
        try:
            if ASYNC_MODE: # Prioritize aiohttp if available
                async with aiohttp.ClientSession() as sess:
                    assistant_msg = await call_llm_async(sess, payload)
            else: # Fallback for when aiohttp is not installed
                assistant_msg = await asyncio.to_thread(call_llm_sync, payload)
        except RuntimeError as e: # Catch errors from call_llm_* (e.g. API key missing)
            return f"[LLM-Error] {e}"

        self.history.append({"role": "assistant", "content": assistant_msg})
        self._save_memory()
        return assistant_msg

    def ask_sync(self, user_msg: str) -> str:
        """Handles user input, either as a command or a message to the LLM (sync)."""
        if user_msg.startswith("/model "):
            mdl = user_msg.split(maxsplit=1)[1]
            if mdl in self.models:
                self.cfg["model"] = mdl
                logger.info(f"Model switched to {mdl}")
                return f"Model switched to {mdl}"
            return f"Unknown model id {mdl}. Known: {', '.join(self.models)}"

        if user_msg.startswith("/tool "):
            _, rest = user_msg.split(maxsplit=1)
            name, *arg_tokens = rest.split(maxsplit=1)
            args = arg_tokens[0] if arg_tokens else ""
            return self._run_plugin_sync(name, args)

        self.history.append({"role": "user", "content": user_msg})
        payload = self._prepare_payload()
        
        try:
            assistant_msg = call_llm_sync(payload)
        except RuntimeError as e: # Catch errors from call_llm_sync (e.g. API key missing)
            return f"[LLM-Error] {e}"
            
        self.history.append({"role": "assistant", "content": assistant_msg})
        self._save_memory()
        return assistant_msg

    # ────────── Darwin recompile with FIT/QUARANTINE ──────────
    def try_agent_compile(self, path: Path, code: str) -> Tuple[str, Optional[str]]:
        """
        Attempts to import the agent file and create a class instance. Survival of the fittest.
        Returns:
            A tuple (status: str, error_message: Optional[str]).
            Status can be "FIT" or "QUARANTINED".
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            
            spec = importlib.util.spec_from_file_location("TestAgent", path)
            if not spec or not spec.loader:
                return "QUARANTINED", "Could not load module spec"
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            agent_class = getattr(module, "DarwinAgent", None)
            if agent_class is None:
                agent_class = getattr(module, "CatGPT", None) # Fallback for legacy
                if agent_class is None:
                    return "QUARANTINED", "No DarwinAgent or CatGPT class found."

            # Test instantiation and basic functionality
            if not get_api_key(): # Check if API key is available for test
                logger.warning("Skipping agent health check during compile test as API key is not set.")
            elif agent_class.__name__ == "DarwinAgent": # Only test if API key is present
                test_instance = agent_class() # Assumes DarwinAgent has a no-arg constructor
                if ASYNC_MODE:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop) # Required for the new thread
                    try:
                        answer = loop.run_until_complete(test_instance.ask_async("health-check"))
                    finally:
                        loop.close()
                else:
                    answer = test_instance.ask_sync("health-check")
                
                if not answer: # Basic check, could be more sophisticated
                    return "QUARANTINED", "Agent health check (ask method) failed to return a response."
            elif agent_class.__name__ == "CatGPT": # Legacy CatGPT class check
                test_instance = agent_class.__new__(agent_class) # Avoid calling __init__ if it takes arguments
                if not hasattr(test_instance, "create_widgets"): # Example check for legacy
                    return "QUARANTINED", "Legacy CatGPT class missing 'create_widgets' method."
            
            return "FIT", None
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Agent compilation/test failed: {e}\n{tb}")
            return "QUARANTINED", tb

    def recompile(self, new_code: str) -> Tuple[str, str, Optional[str]]:
        """
        Recompiles the agent with new code, saves it, and archives it.
        Returns:
            A tuple (filename: str, status: str, error_message: Optional[str]).
        """
        ts = now_ts()
        fname_stem = f"CatGPT_Agent_v{ts}"
        agent_file_path = ARCHIVE_DIR / f"{fname_stem}.py"
        
        status, error = self.try_agent_compile(agent_file_path, new_code)
        
        # Save with metadata header
        error_display = str(error) if error else 'None'
        final_code_header = f'''"""
Darwin CatGPT Fusion Agent - Generated by self-recompile
Timestamp: {datetime.now()}
Status: {status}
Error: {textwrap.shorten(error_display, width=100, placeholder="...")}
"""

'''
        final_code = final_code_header + new_code
        agent_file_path.write_text(final_code, encoding="utf-8")
        
        # Create README
        readme_path = ARCHIVE_DIR / f"README_{fname_stem}.txt"
        readme_content = f"""Darwin CatGPT Fusion - Agent Version
Saved at: {datetime.now()}
File: {agent_file_path.name}
Status: {status}
Error: {error_display}

This agent was generated through evolutionary self-modification.
Edit, share, or branch this file as your own agent!
Meowvolution continues!
"""
        readme_path.write_text(readme_content, encoding="utf-8")
        
        # Update archive
        self.agent_archive.append((ts, agent_file_path.name, status, error))
        self._save_agent_archive()
        
        logger.info(f"Agent recompiled: {agent_file_path.name}, Status: {status}")
        return agent_file_path.name, status, error

# ----------------------------------------------------------------------------
# Tkinter UI layer – Fusion UI combining both versions
# ----------------------------------------------------------------------------

class CatGPTFusion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Darwin CatGPT Fusion Edition - Self-Evolving AI")
        self.geometry("1100x750")
        self.config(bg=UI_THEME["bg_primary"])
        
        # Prompt for API key BEFORE initializing the agent or UI that might depend on it
        self._prompt_for_api_key_if_missing() 

        self.agent = DarwinAgent()
        self.intro_message = "Welcome to Darwin CatGPT Fusion! (Multiple models available)"
        
        self._build_ui()
        logger.info("CatGPTFusion UI initialized.")

    def _prompt_for_api_key_if_missing(self):
        """
        Checks for API key in environment variables. If not found, prompts the user.
        Sets the RUNTIME_API_KEY global variable.
        """
        global RUNTIME_API_KEY
        
        # 1. Check environment variable first
        env_key = os.environ.get(OPENROUTER_API_KEY_ENV_VAR)
        if env_key and env_key.strip():
            RUNTIME_API_KEY = env_key.strip()
            logger.info(f"API Key loaded from {OPENROUTER_API_KEY_ENV_VAR} environment variable.")
            return

        # 2. If not in env, and RUNTIME_API_KEY is not already set (e.g. by tests or other means)
        if RUNTIME_API_KEY: # Check if it was already set (e.g. if this method were called multiple times)
            return

        # 3. Prompt the user
        # This needs the main window (self) to be at least minimally initialized for the dialog parent.
        # `super().__init__()` should be sufficient.
        api_key_input = simpledialog.askstring(
            "API Key Required",
            f"{OPENROUTER_API_KEY_ENV_VAR} not found in environment variables.\n"
            "Please enter your OpenRouter API Key for this session:",
            parent=self 
        )
        if api_key_input and api_key_input.strip():
            RUNTIME_API_KEY = api_key_input.strip()
            logger.info("API Key set from user prompt for this session.")
        else:
            RUNTIME_API_KEY = "" # Explicitly set to empty if user provides nothing or cancels
            logger.warning("API Key not provided by user. LLM functionality will be impaired.")
            messagebox.showwarning(
                "API Key Missing",
                "No API Key was entered. AI features may not work.\n"
                f"You can set the {OPENROUTER_API_KEY_ENV_VAR} environment variable and restart the application.",
                parent=self
            )

    def _build_ui(self):
        """Builds the main UI components."""
        # Main container
        main_container = tk.Frame(self, bg=UI_THEME["bg_primary"])
        main_container.pack(fill="both", expand=True)

        self._build_left_panel(main_container)
        self._build_right_panel(main_container)
        
        self._display_initial_messages()

    def _build_left_panel(self, parent_container):
        """Builds the left panel (chat interface)."""
        left_frame = tk.Frame(parent_container, bg=UI_THEME["bg_secondary"], relief=tk.RAISED, bd=1)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self._build_chat_display(left_frame)
        self._build_input_area(left_frame)
        self._build_control_buttons(left_frame)

    def _build_chat_display(self, parent_frame):
        """Builds the chat display area."""
        self.chat_window = scrolledtext.ScrolledText(
            parent_frame, width=75, height=32, bg=UI_THEME["bg_chat_display"], fg=UI_THEME["fg_primary"],
            font=UI_THEME["font_chat"], wrap=tk.WORD, relief=tk.SOLID, bd=1
        )
        self.chat_window.pack(fill="both", expand=True, padx=10, pady=10)
        self.chat_window.configure(state=tk.DISABLED) # Read-only initially

    def _build_input_area(self, parent_frame):
        """Builds the user input field and send/clear buttons."""
        input_frame = tk.Frame(parent_frame, bg=UI_THEME["bg_secondary"])
        input_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.input_field = tk.Text(
            input_frame, width=60, height=3, bg=UI_THEME["bg_chat_input"], fg=UI_THEME["fg_primary"],
            font=UI_THEME["font_chat"], wrap=tk.WORD, relief=tk.SOLID, bd=1,
            insertbackground=UI_THEME["fg_primary"] # Cursor color
        )
        self.input_field.pack(side="left", fill="x", expand=True)
        self.input_field.bind("<Return>", lambda e: self._on_send() if not (e.state & 0x1) else None) # Send on Enter, allow Shift+Enter for newline

        # Buttons frame (Send, Clear)
        btn_frame = tk.Frame(input_frame, bg=UI_THEME["bg_secondary"])
        btn_frame.pack(side="left", padx=(5, 0))

        tk.Button(
            btn_frame, text="Send", command=self._on_send,
            bg=UI_THEME["bg_button_primary"], fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_main"],
            padx=15, pady=5, cursor="hand2", relief=tk.RAISED, bd=1
        ).pack(pady=2, fill="x")

        tk.Button(
            btn_frame, text="Clear", command=self._clear_history,
            bg=UI_THEME["bg_button_danger"], fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_small"],
            padx=10, pady=3, relief=tk.RAISED, bd=1
        ).pack(pady=2, fill="x")
        
    def _build_control_buttons(self, parent_frame):
        """Builds control buttons (Model Config, Recompile, Plugin Manager)."""
        control_frame = tk.Frame(parent_frame, bg=UI_THEME["bg_secondary"])
        control_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        buttons_config = [
            ("Model Config", self._model_config_window, UI_THEME["bg_button_secondary"]),
            ("Agent Recompile", self._agent_recompile_window, UI_THEME["bg_button_evolution"]),
            ("Plugin Manager", self._plugin_manager, UI_THEME["bg_button_warning"]),
        ]

        for text, command, bg_color in buttons_config:
            tk.Button(
                control_frame, text=text, command=command,
                bg=bg_color, fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_small"], 
                padx=10, relief=tk.RAISED, bd=1, cursor="hand2"
            ).pack(side="left", padx=2, pady=2)

    def _build_right_panel(self, parent_container):
        """Builds the right panel (Evolution Archive)."""
        right_frame = tk.Frame(parent_container, bg=UI_THEME["bg_tertiary"], relief=tk.RIDGE, bd=2)
        right_frame.pack(side="left", fill="y", padx=(0, 10), pady=10)

        tk.Label(
            right_frame, text="🧬 Evolution Archive",
            font=UI_THEME["font_title"], bg=UI_THEME["bg_tertiary"], fg=UI_THEME["fg_evolution_header"]
        ).pack(pady=8)

        self._build_archive_listbox(right_frame)
        self._build_archive_buttons(right_frame)

        self.stats_label = tk.Label(
            right_frame, text="", font=("Consolas", 10), bg=UI_THEME["bg_tertiary"], fg=UI_THEME["fg_primary"]
        )
        self.stats_label.pack(pady=5)
        self._refresh_archive_listbox()

    def _build_archive_listbox(self, parent_frame):
        """Builds the listbox for displaying agent archive."""
        list_frame = tk.Frame(parent_frame, bg=UI_THEME["bg_tertiary"])
        list_frame.pack(fill="both", expand=True, padx=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.archive_listbox = tk.Listbox(
            list_frame, width=50, height=25, font=UI_THEME["font_listbox"],
            bg=UI_THEME["bg_secondary"], fg=UI_THEME["fg_primary"], 
            selectbackground=UI_THEME["bg_listbox_select"], selectforeground=UI_THEME["fg_button_light"],
            yscrollcommand=scrollbar.set, relief=tk.SOLID, bd=1
        )
        self.archive_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.archive_listbox.yview)
    
    def _build_archive_buttons(self, parent_frame):
        """Builds buttons for interacting with the archive."""
        archive_btn_frame = tk.Frame(parent_frame, bg=UI_THEME["bg_tertiary"])
        archive_btn_frame.pack(fill="x", padx=5, pady=5)

        tk.Button(
            archive_btn_frame, text="Open/Edit Selected", command=self._open_selected_agent,
            bg=UI_THEME["bg_listbox_select"], fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_small"], 
            padx=10, relief=tk.RAISED, bd=1, cursor="hand2"
        ).pack(side="left", padx=2, pady=2)

        tk.Button(
            archive_btn_frame, text="Delete Selected", command=self._delete_selected,
            bg=UI_THEME["bg_button_danger"], fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_small"],
            padx=10, relief=tk.RAISED, bd=1, cursor="hand2"
        ).pack(side="left", padx=2, pady=2)

    def _display_initial_messages(self):
        """Displays the welcome message and initial info in the chat window."""
        self.chat_window.configure(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"CatGPT: {self.intro_message}\n")
        self.chat_window.insert(tk.END, f"Models: {', '.join(self.agent.models)}\n")
        self.chat_window.insert(tk.END, "Commands: /model <name>, /tool <plugin> <args>\n")
        if not get_api_key():
            self.chat_window.insert(tk.END, "\nWARNING: API Key not set. LLM features will not work.\n"
                                            f"Set the {OPENROUTER_API_KEY_ENV_VAR} environment variable or restart to be prompted.\n\n")
        else:
            self.chat_window.insert(tk.END, "\nAPI Key is configured for this session.\n\n")
        self.chat_window.configure(state=tk.DISABLED)

    def _append_chat(self, who: str, txt: str):
        """Appends a message to the chat window."""
        self.chat_window.configure(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"\n{who}: {txt}\n")
        self.chat_window.see(tk.END) # Scroll to the end
        self.chat_window.configure(state=tk.DISABLED)

    def _on_send(self):
        """Handles sending a message from the input field."""
        user_msg = self.input_field.get("1.0", "end-1c").strip()
        if not user_msg:
            return
        
        if not get_api_key() and not user_msg.startswith("/"): # If no API key and not a command
            messagebox.showerror("API Key Missing", "Cannot send message to LLM. API Key is not configured.", parent=self)
            return

        self.input_field.delete("1.0", tk.END)
        self._append_chat("You", user_msg)
        
        # Start agent processing in a separate thread to keep UI responsive
        Thread(target=self._worker, args=(user_msg,), daemon=True).start()

    def _worker(self, msg: str):
        """Worker thread to interact with the DarwinAgent."""
        try:
            if ASYNC_MODE: # If aiohttp is available
                # Create a new event loop for this thread if running async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    answer = loop.run_until_complete(self.agent.ask_async(msg))
                finally:
                    loop.close()
            else: # Fallback to synchronous method
                answer = self.agent.ask_sync(msg)
        except Exception as e:
            logger.error(f"Error in worker thread: {e}\n{traceback.format_exc()}")
            answer = f"[error] An unexpected error occurred: {e}"
        
        # Schedule UI update on the main thread
        self.after(0, lambda: self._append_chat("CatGPT", answer))

    def _clear_history(self):
        """Clears the chat history and agent memory."""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the chat history and agent memory?"):
            self.agent.history = []
            self.agent._save_memory() # Save empty memory
            
            self.chat_window.configure(state=tk.NORMAL)
            self.chat_window.delete('1.0', tk.END)
            self._display_initial_messages() # Re-add initial messages
            logger.info("Chat history and agent memory cleared.")

    def _model_config_window(self):
        """Opens a window to configure LLM parameters."""
        win = tk.Toplevel(self)
        win.title("Model Configuration")
        win.geometry("400x350") # Adjusted height
        win.config(bg=UI_THEME["bg_secondary"])
        win.transient(self) # Keep window on top of main
        win.grab_set() # Modal behavior

        frame = tk.Frame(win, bg=UI_THEME["bg_secondary"], padx=20, pady=20)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Model:", font=UI_THEME["font_default"], bg=UI_THEME["bg_secondary"]).grid(row=0, column=0, sticky="w", pady=5)
        model_var = tk.StringVar(value=self.agent.cfg["model"])
        model_menu = tk.OptionMenu(frame, model_var, *self.agent.models if self.agent.models else [DEFAULT_MODELS[1]])
        model_menu.config(width=25, relief=tk.SOLID, bd=1)
        model_menu.grid(row=0, column=1, pady=5, sticky="ew")

        params_config = [
            ("Temperature:", "temperature", (0.0, 2.0), float),
            ("Max Tokens:", "max_tokens", (1, 8192), int), # Increased max_tokens upper limit
            ("Top P:", "top_p", (0.0, 1.0), float),
            ("Frequency Penalty:", "frequency_penalty", (-2.0, 2.0), float),
            ("Presence Penalty:", "presence_penalty", (-2.0, 2.0), float)
        ]
        entries: Dict[str, tk.Entry] = {}

        for i, (label_text, key, (min_val, max_val), param_type) in enumerate(params_config, 1):
            tk.Label(frame, text=label_text, font=UI_THEME["font_default"], bg=UI_THEME["bg_secondary"]).grid(row=i, column=0, sticky="w", pady=5)
            entry = tk.Entry(frame, width=20, relief=tk.SOLID, bd=1)
            entry.insert(0, str(self.agent.cfg.get(key, "")))
            entry.grid(row=i, column=1, pady=5, sticky="ew")
            entries[key] = entry

        def save_config():
            try:
                new_cfg = {"model": model_var.get()}
                for key, entry_widget in entries.items():
                    param_type = next(p[3] for p in params_config if p[1] == key)
                    min_val, max_val = next(p[2] for p in params_config if p[1] == key)
                    
                    val_str = entry_widget.get()
                    if not val_str: # Handle empty string for numeric fields if necessary, or validate
                        raise ValueError(f"{key.replace('_', ' ').title()} cannot be empty.")

                    value = param_type(val_str)
                    
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"{key.replace('_', ' ').title()} must be between {min_val} and {max_val}.")
                    new_cfg[key] = value
                
                self.agent.cfg.update(new_cfg)
                logger.info(f"Model configuration updated: {self.agent.cfg}")
                messagebox.showinfo("Success", "Configuration saved!", parent=win)
                win.destroy()
            except ValueError as ve:
                messagebox.showerror("Validation Error", str(ve), parent=win)
            except Exception as e:
                logger.error(f"Error saving model configuration: {e}")
                messagebox.showerror("Error", f"Could not save configuration: {e}", parent=win)

        tk.Button(
            frame, text="Save Configuration", command=save_config,
            bg=UI_THEME["bg_button_primary"], fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_main"],
            padx=20, pady=10, relief=tk.RAISED, bd=1
        ).grid(row=len(params_config)+1, column=0, columnspan=2, pady=20)
        
        frame.grid_columnconfigure(1, weight=1) # Make entry column resizable

    def _agent_recompile_window(self):
        """Opens a window for editing and recompiling the agent's source code."""
        win = tk.Toplevel(self)
        win.title("Agent Recompile & Evolution")
        win.geometry("900x700")
        win.config(bg=UI_THEME["bg_editor"]) 
        win.transient(self)
        win.grab_set()

        header = tk.Frame(win, bg=UI_THEME["bg_editor_header"], height=50)
        header.pack(fill="x")
        tk.Label(
            header, text="🧬 Darwin Agent Evolution Chamber",
            font=("Arial", 16, "bold"), bg=UI_THEME["bg_editor_header"], fg=UI_THEME["fg_secondary"]
        ).pack(pady=10)

        editor_frame = tk.Frame(win, bg=UI_THEME["bg_editor"])
        editor_frame.pack(fill="both", expand=True, padx=10, pady=10)

        editor = tk.Text(
            editor_frame, width=100, height=35, bg=UI_THEME["bg_editor"], fg=UI_THEME["fg_secondary"],
            insertbackground=UI_THEME["bg_button_danger"], font=UI_THEME["font_editor"], wrap=tk.NONE,
            undo=True # Enable undo/redo
        )
        # Add scrollbars to editor
        v_scroll = tk.Scrollbar(editor_frame, orient="vertical", command=editor.yview)
        h_scroll = tk.Scrollbar(editor_frame, orient="horizontal", command=editor.xview)
        editor.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        editor.pack(side="left", fill="both", expand=True)
        
        try:
            current_source = inspect.getsource(sys.modules[__name__])
            editor.insert(tk.END, current_source)
        except Exception as e:
            logger.error(f"Could not load current source for recompile: {e}")
            editor.insert(tk.END, f"# Error loading current source: {e}")

        btn_frame = tk.Frame(win, bg=UI_THEME["bg_editor"])
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        def compile_and_evolve():
            new_code = editor.get("1.0", "end-1c")
            filename, status, error = self.agent.recompile(new_code)
            self._refresh_archive_listbox()
            
            error_msg_display = str(error) if error else "None"
            if status == "FIT":
                messagebox.showinfo(
                    "Evolution Success! 🎉",
                    f"Agent evolved successfully!\n\nFile: {filename}\nStatus: {status}\n\nThe agent has passed all fitness tests.",
                    parent=win
                )
            else:
                messagebox.showwarning(
                    "Evolution Failed ⚠️",
                    f"Agent quarantined!\n\nFile: {filename}\nStatus: {status}\n\nError:\n{textwrap.shorten(error_msg_display, 500, placeholder='...')}",
                    parent=win
                )
            win.destroy()

        tk.Button(
            btn_frame, text="🚀 Compile & Evolve", command=compile_and_evolve,
            bg=UI_THEME["bg_button_evo_compile"], fg=UI_THEME["fg_button_light"], font=("Arial", 14, "bold"),
            padx=30, pady=15, cursor="hand2", relief=tk.RAISED, bd=1
        ).pack(side="left", padx=5, pady=5)

        tk.Button(
            btn_frame, text="Cancel", command=win.destroy,
            bg=UI_THEME["bg_button_danger"], fg=UI_THEME["fg_button_light"], font=("Arial", 12),
            padx=20, pady=10, relief=tk.RAISED, bd=1
        ).pack(side="left", padx=5, pady=5)

    def _plugin_manager(self):
        """Opens a window to manage and view plugins."""
        win = tk.Toplevel(self)
        win.title("Plugin Manager")
        win.geometry("600x400")
        win.config(bg=UI_THEME["bg_secondary"])
        win.transient(self)
        win.grab_set()

        tk.Label(
            win, text="Installed Plugins", font=UI_THEME["font_title"], bg=UI_THEME["bg_secondary"]
        ).pack(pady=10)

        plugin_list_frame = tk.Frame(win, bg=UI_THEME["bg_secondary"])
        plugin_list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        plugin_scrollbar = tk.Scrollbar(plugin_list_frame)
        plugin_scrollbar.pack(side="right", fill="y")

        plugin_list = tk.Listbox(
            plugin_list_frame, width=70, height=15, font=UI_THEME["font_listbox"],
            relief=tk.SOLID, bd=1, yscrollcommand=plugin_scrollbar.set
        )
        plugin_list.pack(side="left", fill="both", expand=True)
        plugin_scrollbar.config(command=plugin_list.yview)

        self.agent._discover_plugins() # Ensure list is fresh before displaying
        if self.agent.plugins:
            for name in self.agent.plugins:
                plugin_list.insert(tk.END, f"✓ {name}")
        else:
            plugin_list.insert(tk.END, "No plugins found or loaded.")

        tk.Label(
            win, text=f"Plugin Directory: {PLUGIN_DIR}", font=("Arial", 9), 
            bg=UI_THEME["bg_secondary"], fg="#666"
        ).pack(pady=5)

        def refresh_plugins():
            self.agent._discover_plugins()
            plugin_list.delete(0, tk.END) # Clear existing list
            if self.agent.plugins:
                for name in self.agent.plugins:
                    plugin_list.insert(tk.END, f"✓ {name}")
            else:
                plugin_list.insert(tk.END, "No plugins found or loaded.")
            logger.info("Plugin list refreshed.")

        tk.Button(
            win, text="Refresh Plugins", command=refresh_plugins,
            bg=UI_THEME["bg_button_secondary"], fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_small"],
            relief=tk.RAISED, bd=1
        ).pack(pady=10)

    def _refresh_archive_listbox(self):
        """Refreshes the listbox displaying agent archive entries."""
        self.archive_listbox.delete(0, tk.END)
        fit_count = 0
        quarantine_count = 0
        
        # Display most recent first
        for ts, filename, status, error in reversed(self.agent.agent_archive):
            icon = "✅" if status == "FIT" else "🔒" # Fit / Quarantined
            if status == "FIT":
                fit_count += 1
            else:
                quarantine_count += 1
            
            display_text = f"{icon} {ts} | {filename}"
            self.archive_listbox.insert(tk.END, display_text)
        
        total = len(self.agent.agent_archive)
        self.stats_label.config(
            text=f"Total: {total} | Fit: {fit_count} | Quarantined: {quarantine_count}"
        )
        logger.debug("Archive listbox refreshed.")


    def _open_selected_agent(self):
        """Opens the selected agent's code in a new editor window."""
        selection = self.archive_listbox.curselection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select an agent version from the archive to open.", parent=self)
            return
        
        # Map listbox index to reversed archive index
        idx = selection[0]
        try:
            ts, filename, status, error = self.agent.agent_archive[::-1][idx]
        except IndexError:
            messagebox.showerror("Error", "Could not retrieve selected agent. Please refresh.", parent=self)
            return

        agent_path = ARCHIVE_DIR / filename
        if not agent_path.exists():
            messagebox.showerror("File Not Found", f"Agent file not found: {filename}", parent=self)
            # Optionally remove from archive if file is missing
            # self.agent.agent_archive = [item for item in self.agent.agent_archive if item[1] != filename]
            # self.agent._save_agent_archive()
            # self._refresh_archive_listbox()
            return

        win = tk.Toplevel(self)
        win.title(f"View/Edit Agent: {filename} ({status})")
        win.geometry("850x650")
        win.config(bg=UI_THEME["bg_editor"])
        win.transient(self)
        win.grab_set()

        status_bg = UI_THEME["bg_button_evo_compile"] if status == "FIT" else UI_THEME["bg_button_danger"]
        status_frame = tk.Frame(win, bg=status_bg, height=30)
        status_frame.pack(fill="x")
        tk.Label(
            status_frame, text=f"Status: {status}", font=("Arial", 12, "bold"),
            bg=status_bg, fg=UI_THEME["fg_button_light"]
        ).pack(pady=5)

        editor_frame = tk.Frame(win, bg=UI_THEME["bg_editor"]) # Frame to hold editor and scrollbars
        editor_frame.pack(fill="both", expand=True, padx=5, pady=5)


        editor = tk.Text(
            editor_frame, width=95, height=35, bg=UI_THEME["bg_editor"], fg=UI_THEME["fg_secondary"],
            insertbackground=UI_THEME["bg_button_secondary"], font=UI_THEME["font_editor"], wrap=tk.NONE, undo=True
        )
        v_scroll = tk.Scrollbar(editor_frame, orient="vertical", command=editor.yview)
        h_scroll = tk.Scrollbar(editor_frame, orient="horizontal", command=editor.xview)
        editor.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        editor.pack(side="left", fill="both", expand=True)


        try:
            editor.insert(tk.END, agent_path.read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Error Reading File", f"Could not read agent file: {e}", parent=win)
            editor.insert(tk.END, f"# Error reading file: {e}")
            
        def save_as_new_branch():
            new_code = editor.get("1.0", "end-1c")
            branch_filename, branch_status, branch_error = self.agent.recompile(new_code)
            self._refresh_archive_listbox()
            
            branch_error_msg = str(branch_error) if branch_error else "None"
            if branch_status == "FIT":
                messagebox.showinfo("Branch Created!", f"New agent branch created: {branch_filename}\nStatus: FIT", parent=win)
            else:
                messagebox.showwarning("Branch Quarantined", f"Agent branch created but quarantined: {branch_filename}\nError: {textwrap.shorten(branch_error_msg,100,placeholder='...')}", parent=win)
            win.destroy()
        
        button_container = tk.Frame(win, bg=UI_THEME["bg_editor"]) # Container for the button below editor
        button_container.pack(fill="x", pady=(5,10))

        tk.Button(
            button_container, text="Save as New Evolved Agent", command=save_as_new_branch,
            bg=UI_THEME["bg_button_secondary"], fg=UI_THEME["fg_button_light"], font=("Arial", 12, "bold"),
            padx=20, pady=10, relief=tk.RAISED, bd=1
        ).pack() # Centered by default in its own frame

    def _delete_selected(self):
        """Deletes the selected agent version from the archive and filesystem."""
        selection = self.archive_listbox.curselection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select an agent to delete.", parent=self)
            return
        
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to permanently delete this agent version and its files?", parent=self):
            idx = selection[0]
            archive_entry_to_delete = None
            try:
                # The listbox is reversed, so we need to find the correct item in the original list
                reversed_archive = self.agent.agent_archive[::-1]
                ts_to_delete, filename_to_delete, _, _ = reversed_archive[idx]

                # Find and remove the item from the original self.agent.agent_archive
                original_index = -1
                for i, item in enumerate(self.agent.agent_archive):
                    if item[0] == ts_to_delete and item[1] == filename_to_delete:
                        original_index = i
                        break
                
                if original_index != -1:
                    archive_entry_to_delete = self.agent.agent_archive.pop(original_index)
                    self.agent._save_agent_archive()
                else:
                    logger.error(f"Could not find agent {filename_to_delete} (ts: {ts_to_delete}) in archive for deletion.")
                    messagebox.showerror("Error", "Could not find agent in archive index.", parent=self)
                    self._refresh_archive_listbox() # Refresh to reflect any inconsistency
                    return

            except IndexError:
                messagebox.showerror("Error", "Could not retrieve selected agent for deletion. Please refresh.", parent=self)
                return

            # Delete files if entry was successfully removed from archive list
            if archive_entry_to_delete:
                ts, filename, _, _ = archive_entry_to_delete
                agent_file = ARCHIVE_DIR / filename
                # Construct readme name carefully based on how it's created in recompile()
                # Original: readme_path = ARCHIVE_DIR / f"README_{fname_stem}.txt"
                # fname_stem = f"CatGPT_Agent_v{ts}"
                readme_file = ARCHIVE_DIR / f"README_CatGPT_Agent_v{ts}.txt" 
                
                files_deleted_count = 0
                try:
                    if agent_file.exists():
                        agent_file.unlink()
                        files_deleted_count +=1
                        logger.info(f"Deleted agent file: {agent_file}")
                    else:
                        logger.warning(f"Agent file not found for deletion: {agent_file}")

                    if readme_file.exists():
                        readme_file.unlink()
                        files_deleted_count +=1
                        logger.info(f"Deleted readme file: {readme_file}")
                    else:
                        logger.warning(f"Readme file not found for deletion: {readme_file}")
                        
                    if files_deleted_count > 0:
                        messagebox.showinfo("Deletion Successful", f"Agent '{filename}' and associated files deleted.", parent=self)
                    elif not agent_file.exists() and not readme_file.exists():
                        messagebox.showwarning("Deletion Note", f"Agent '{filename}' files were not found on disk. Removed from archive index.", parent=self)
                    else: # Some files might still exist if logic above fails for one but not other
                        messagebox.showwarning("Deletion Incomplete", f"Agent '{filename}' removed from index. Some files might not have been found/deleted.", parent=self)


                except Exception as e:
                    logger.error(f"Error deleting agent files for {filename}: {e}")
                    messagebox.showerror("Deletion Error", f"Could not delete all files for agent '{filename}': {e}", parent=self)
            
            self._refresh_archive_listbox()


# ----------------------------------------------------------------------------
# App entrypoint
# ----------------------------------------------------------------------------

def main():
    # Check for async support
    if not ASYNC_MODE:
        logger.warning("aiohttp not installed. Running in sync mode for LLM calls.")
        logger.warning("Install with: pip install aiohttp")
    
    # On Windows: fix asyncio event loop policy for Tkinter threads
    # This is crucial if using asyncio in threads with Tkinter
    if sys.platform.startswith("win") and ASYNC_MODE:
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info("Set WindowsSelectorEventLoopPolicy for asyncio.")
        except Exception as e:
            logger.error(f"Could not set WindowsSelectorEventLoopPolicy: {e}")
            logger.warning("Async operations in threads might not work correctly on Windows.")
    
    app = CatGPTFusion()
    app.mainloop()

if __name__ == "__main__":
    main()
