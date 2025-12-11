
import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Simulation of StreamedFile
@dataclass
class StreamedFile:
    path: str
    filename: Optional[str] = None
    _size: Optional[int] = None
    _cleanup: bool = False

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        pass

    @classmethod
    def from_upload(cls, upload_file):
        # Simulate what from_upload does
        fd, path = tempfile.mkstemp()
        os.close(fd)
        return cls(path=path, filename="test.xlsx", _cleanup=True)

# Simulation of ExcelProcessor
class ExcelProcessor:
    async def stream_xlsx_processing(self, file_path: str, filename: str, user_id: str):
        print(f"Processing {file_path}")
        # Simulate polars read
        if not isinstance(file_path, str):
            raise TypeError(f"Expected str, got {type(file_path)}")
        return {"sheets": {}}

async def main():
    try:
        # Simulate the endpoint logic
        streamed_file = StreamedFile.from_upload("fake_file")
        
        # Simulate implicit context?
        # with streamed_file:
        #    pass
        
        # Verify file path usage
        processor = ExcelProcessor()
        await processor.stream_xlsx_processing(
            file_path=streamed_file.path,
            filename="test.xlsx",
            user_id="user1"
        )
        print("Success!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
