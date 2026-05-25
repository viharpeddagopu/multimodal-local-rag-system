import re
import fitz

from pathlib import Path
from dataclasses import dataclass

from src.config import PAGE_IMAGES_DIR


# =========================================================
# PAGE DATA STRUCTURE
# =========================================================

@dataclass
class PageRecord:
    document: str
    page: int
    text: str
    image_path: Path


# =========================================================
# TEXT CLEANING
# =========================================================

HYPHEN_PATTERN = re.compile(r"(\w)-\s*\n\s*(\w)")
MULTIPLE_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text):

    text = HYPHEN_PATTERN.sub(r"\1\2", text)

    text = MULTIPLE_NEWLINES.sub("\n\n", text)

    return text.strip()


# =========================================================
# PDF PROCESSING
# =========================================================

def extract_pdf_text(
    pdf_paths,
    render_dpi=144
):

    pages = []

    for pdf_path in pdf_paths:

        doc = fitz.open(pdf_path)

        document_name = pdf_path.name

        # ---------------------------------------------
        # Create image folder for this document
        # ---------------------------------------------

        document_image_dir = (
            PAGE_IMAGES_DIR / document_name
        )

        document_image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for page_num, page in enumerate(doc):

            # -----------------------------------------
            # Better reading-order extraction
            # -----------------------------------------

            blocks = page.get_text("blocks")

            blocks_sorted = sorted(
                blocks,
                key=lambda b: (
                    round(b[1], 1),
                    round(b[0], 1)
                )
            )

            raw_text = "\n".join(
                block[4]
                for block in blocks_sorted
                if isinstance(block[4], str)
            )

            cleaned_text = clean_text(raw_text)

            # -----------------------------------------
            # Render page image
            # -----------------------------------------

            pix = page.get_pixmap(
                dpi=render_dpi,
                alpha=False
            )

            image_path = (
                document_image_dir
                / f"page_{page_num + 1}.png"
            )

            pix.save(str(image_path))

            # -----------------------------------------
            # Store page record
            # -----------------------------------------

            pages.append(
                PageRecord(
                    document=document_name,
                    page=page_num + 1,
                    text=cleaned_text,
                    image_path=image_path
                )
            )

    return pages