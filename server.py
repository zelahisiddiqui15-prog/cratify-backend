import time
import threading
import shutil
import subprocess
import requests
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import load_config, BACKEND_URL

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".aiff", ".aif", ".flac", ".ogg",
    ".fxp", ".nmsv", ".vital", ".xpf", ".aupreset",
    ".mid", ".midi",
}


def osascript_confirm_sort(filename, category, key, dest):
    """Ask user to confirm before sorting a file."""
    label = f"{category}/{key}" if key else category
    escaped_filename = filename.replace('"', '\\"')
    escaped_label = label.replace('"', '\\"')
    escaped_dest = str(dest).replace('"', '\\"')
    script = (
        f'tell application "System Events" to display dialog '
        f'"SortDrop wants to move:\\n\\n{escaped_filename}\\n\\n→ {escaped_label}\\n\\nDest: {escaped_dest}" '
        f'with title "SortDrop — Confirm Sort" '
        f'buttons {{"Skip", "Sort It"}} default button "Sort It"'
    )
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return result.returncode == 0 and "Sort It" in result.stdout


class MusicFileHandler(FileSystemEventHandler):
    def __init__(self, callbacks: dict):
        super().__init__()
        self._processing = set()
        self._lock = threading.Lock()
        self._callbacks = callbacks

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        with self._lock:
            if str(path) in self._processing:
                return
            self._processing.add(str(path))
        threading.Thread(target=self._process, args=(path,), daemon=True).start()

    def _process(self, path: Path):
        time.sleep(2.5)
        if not path.exists():
            self._processing.discard(str(path))
            return

        cfg = load_config()
        user_id = cfg.get("user_id")
        output_folder = cfg.get("output_folder")
        confirm_mode = cfg.get("confirm_before_sort", False)

        if not user_id or not output_folder:
            self._processing.discard(str(path))
            return

        on_sorting = self._callbacks.get("on_sorting")
        if on_sorting:
            on_sorting(path.name)

        try:
            resp = requests.post(
                f"{BACKEND_URL}/classify",
                json={"filename": path.name, "user_id": user_id},
                timeout=15,
            )

            if resp.status_code == 200:
                result = resp.json()
                dest = self._build_dest(output_folder, result, path.name, cfg)
                dest.parent.mkdir(parents=True, exist_ok=True)

                # Handle filename collisions
                if dest.exists():
                    stem, suffix = dest.stem, dest.suffix
                    counter = 1
                    while dest.exists():
                        dest = dest.parent / f"{stem}_{counter}{suffix}"
                        counter += 1

                # Confirm before sort if enabled
                if confirm_mode:
                    category = result.get("category", "Other")
                    key = result.get("key", "")
                    confirmed = osascript_confirm_sort(path.name, category, key, dest)
                    if not confirmed:
                        on_skip = self._callbacks.get("on_skip")
                        if on_skip:
                            on_skip(path.name)
                        self._processing.discard(str(path))
                        return

                shutil.move(str(path), str(dest))

                on_success = self._callbacks.get("on_success")
                if on_success:
                    on_success(path.name, result, str(dest))

            elif resp.status_code == 402:
                on_trial = self._callbacks.get("on_trial_exhausted")
                if on_trial:
                    on_trial()
            else:
                on_error = self._callbacks.get("on_error")
                if on_error:
                    on_error(f"Server error: {resp.status_code}")

        except Exception as e:
            on_error = self._callbacks.get("on_error")
            if on_error:
                on_error(str(e))

        self._processing.discard(str(path))

    def _build_dest(self, output: str, result: dict, filename: str, cfg: dict) -> Path:
        base = Path(output)
        category = result.get("category", "Other")
        drum_type = result.get("drum_type", None)
        key = result.get("key", "")
        file_type = result.get("file_type", "stem")
        mode = cfg.get("subfolder_mode", "category_key")

        if file_type == "preset":
            folder = base / "Presets" / category
        elif file_type == "midi":
            folder = base / "MIDI" / category
        elif category == "Drum":
            # Drums get sorted by type, not key
            drum_folder = drum_type if drum_type else "Full Loop"
            folder = base / "Drums" / drum_folder
        elif mode == "category_key" and key:
            folder = base / category / key
        elif mode == "category_only":
            folder = base / category
        else:
            folder = base / category

        return folder / filename


class FolderWatcher:
    def __init__(self, callbacks: dict):
        self._callbacks = callbacks
        self._observers = []

    def start(self) -> tuple:
        cfg = load_config()
        watch_folders = cfg.get("watch_folders", [])

        # Backwards compat — support old single watch_folder key
        single = cfg.get("watch_folder")
        if single and single not in watch_folders:
            watch_folders.append(single)

        if not watch_folders:
            return False, "Watch folder not configured or doesn't exist."

        valid_folders = [f for f in watch_folders if Path(f).exists()]
        if not valid_folders:
            return False, "Watch folder not configured or doesn't exist."

        handler = MusicFileHandler(self._callbacks)
        observer = Observer()
        for folder in valid_folders:
            observer.schedule(handler, folder, recursive=False)

        observer.start()
        self._observers = [observer]
        return True, ""

    def stop(self):
        for obs in self._observers:
            if obs.is_alive():
                obs.stop()
                obs.join()
        self._observers = []

    @property
    def is_running(self):
        return any(obs.is_alive() for obs in self._observers)