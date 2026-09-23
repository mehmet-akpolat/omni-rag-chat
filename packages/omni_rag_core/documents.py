import re
from pathlib import Path

from pypdf import PdfReader

from .domain import DocumentPreview, PagePreview


class InvalidDocument(ValueError):
    pass


class DocumentStore:
    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, content: bytes) -> Path:
        path = self.upload_dir / f"{document_id}.pdf"
        path.write_bytes(content)
        return path

    def path_for(self, document_id: str) -> Path:
        path = self.upload_dir / f"{document_id}.pdf"
        if not path.exists():
            raise FileNotFoundError(document_id)
        return path

    def extract(self, path: Path) -> list[str]:
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise InvalidDocument("Encrypted PDFs are not supported")
            pages = [page.extract_text() or "" for page in reader.pages]
        except InvalidDocument:
            raise
        except Exception as exc:
            raise InvalidDocument("The file is not a readable PDF") from exc
        if not pages:
            raise InvalidDocument("The PDF has no pages")
        return pages

    def preview(self, document_id: str, filename: str, pages: list[str]) -> DocumentPreview:
        safe_name = re.sub(r"[^\w.() -]", "_", Path(filename).name)[:180]
        return DocumentPreview(
            document_id=document_id,
            source=safe_name,
            mime_type="application/pdf",
            total_pages=len(pages),
            pages=[
                PagePreview(page_number=index, excerpt=" ".join(text.split())[:420])
                for index, text in enumerate(pages, 1)
            ],
        )
