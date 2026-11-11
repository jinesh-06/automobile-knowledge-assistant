import os
import shutil
import logging
import pytesseract
from PIL import Image
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, tesseract_cmd: Optional[str] = None):
        self.is_available = False
        self.tesseract_version = None
        self.executable_path = None
        self.unavailable_reason = ""
        self.last_error = None
        
        self._configure_tesseract(tesseract_cmd)

    def _configure_tesseract(self, explicit_cmd: Optional[str] = None):
        """
        Detect and validate Tesseract executable from:
        1. Explicit parameter
        2. TESSERACT_CMD / TESSERACT_PATH environment variables
        3. Existing pytesseract.pytesseract.tesseract_cmd if already valid
        4. System PATH via shutil.which('tesseract')
        5. Common Windows installation paths
        """
        candidates = []
        if explicit_cmd:
            candidates.append(explicit_cmd)
        
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd:
            candidates.append(env_cmd)
            
        env_path = os.environ.get("TESSERACT_PATH")
        if env_path:
            # If env_path is a directory, append tesseract.exe
            if os.path.isdir(env_path):
                candidates.append(os.path.join(env_path, "tesseract.exe" if os.name == "nt" else "tesseract"))
            else:
                candidates.append(env_path)

        # Current pytesseract setting if already configured and not default
        current_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        if current_cmd and current_cmd != "tesseract":
            candidates.append(current_cmd)

        # System PATH
        which_tesseract = shutil.which("tesseract")
        if which_tesseract:
            candidates.append(which_tesseract)
        candidates.append("tesseract")

        # Standard Windows installation paths if on Windows
        if os.name == "nt":
            win_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            ]
            candidates.extend(win_paths)

        # Deduplicate candidates while preserving order
        unique_candidates = []
        for c in candidates:
            if c and c not in unique_candidates:
                unique_candidates.append(c)

        original_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
        tested_errors = []

        for candidate in unique_candidates:
            # If candidate contains directory separators, verify file exists before attempting
            if ("/" in candidate or "\\" in candidate) and not os.path.isfile(candidate):
                tested_errors.append(f"'{candidate}': file not found")
                continue

            try:
                pytesseract.pytesseract.tesseract_cmd = candidate
                ver = pytesseract.get_tesseract_version()
                self.is_available = True
                self.tesseract_version = str(ver)
                self.executable_path = candidate
                self.unavailable_reason = ""
                logger.info(f"Tesseract OCR initialized successfully: {candidate} (v{ver})")
                return
            except Exception as e:
                tested_errors.append(f"'{candidate}': {e}")

        # If none worked, restore original and mark unavailable
        pytesseract.pytesseract.tesseract_cmd = original_cmd
        self.is_available = False
        self.unavailable_reason = (
            "Tesseract is not installed or executable could not be discovered. "
            "Please install Tesseract OCR and ensure it is in your system PATH, or set the TESSERACT_PATH environment variable."
        )
        logger.warning(f"Tesseract OCR not available: {self.unavailable_reason}")

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current availability and diagnostic status of Tesseract OCR.
        """
        return {
            "is_available": self.is_available,
            "version": self.tesseract_version,
            "executable": self.executable_path,
            "unavailable_reason": self.unavailable_reason
        }

    def perform_ocr(self, image: Image.Image) -> str:
        """
        Perform OCR on a PIL Image and return extracted text.
        """
        if not self.is_available:
            logger.warning("OCR requested but Tesseract is not available.")
            return ""

        try:
            text = pytesseract.image_to_string(image)
            self.last_error = None
            return text or ""
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"OCR execution failed: {e}")
            return ""
