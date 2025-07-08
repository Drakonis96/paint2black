# Paint2Black & OCR PDF Service

A Flask web application offering a two-step process for PDF manipulation:
1.  **Color Conversion:** Convert PDF pages to monochrome (black & white) or grayscale.
2.  **OCR Processing:** Apply Optical Character Recognition (OCR) to a PDF to make its text searchable and selectable.

The application provides a clean web interface with separate tabs for each step, featuring real-time progress updates using Server-Sent Events (SSE).

## DISCLAIMER
⚠️ Use at your own risk. This app is in testing, completely experimental, and may contain bugs.

## Features

*   **Two-Step Workflow:** Clearly separated functionalities for color conversion and OCR accessible via UI tabs.
*   **Step 1: Color Conversion:**
    *   Upload PDF files.
    *   Choose conversion mode:
        *   **Monochrome:** Uses Otsu's thresholding via OpenCV for black & white output.
        *   **Grayscale:** Converts pages to grayscale using OpenCV.
    *   Real-time, page-by-page progress bar powered by Server-Sent Events (SSE).
    *   Download the resulting color-converted PDF.
*   **Step 2: OCR Processing:**
    *   Upload PDF files (often the output from Step 1, but any PDF is acceptable).
    *   Select the language for OCR (default support for English, Spanish, French, German - extensible).
    *   Utilizes Tesseract OCR engine via `pytesseract` wrapper.
    *   Generates a PDF with an embedded, searchable text layer.
    *   Real-time, page-by-page progress bar using SSE.
    *   Parallel processing of pages for OCR using `concurrent.futures.ProcessPoolExecutor` for improved speed on multi-core systems.
    *   Uses PyMuPDF (Fitz) for robust merging of OCR'd pages, preserving the text layer.
    *   Download the final OCR'd PDF.
*   **Background Processing:** Long-running tasks (conversion and OCR) are executed in background threads to keep the UI responsive.
*   **Robust PDF Handling:** Uses PyMuPDF (Fitz) for reliable PDF reading, rendering, and manipulation.
*   **Web Interface:** Clean UI with Drag & Drop support for file uploads.

## Technology Stack

*   **Backend:** Python 3.10+, Flask
*   **WSGI Server:** Gunicorn
*   **PDF Processing:** PyMuPDF (Fitz), OpenCV-Python
*   **OCR:** Tesseract OCR, pytesseract
*   **Concurrency:** `threading`, `concurrent.futures.ProcessPoolExecutor`
*   **Frontend:** HTML, CSS, JavaScript (Fetch API, Server-Sent Events)
*   **Containerization:** Docker

## Setup & Installation (Using Docker)

1.  **Prerequisites:** Ensure Docker is installed and running on your system.
2.  **Get the Code:** Clone the repository or download the source files (`app.py`, `Dockerfile`, `requirements.txt`, `templates/` folder).
3.  **Create Required Directories:** In the project's root directory, create the temporary folder:
    ```bash
    mkdir temp
    ```
4.  **Build the Docker Image:** Open a terminal in the project's root directory and run:
    ```bash
    docker build -t paint2black-ocr .
    ```
5.  **Run the Docker Container:**
    ```bash
    # The -v flag maps the local 'temp' folder to the container's '/app/temp'
    # --rm automatically removes the container when it stops
    docker run -p 5018:5018 -v "$(pwd)/temp:/app/temp" --rm --name paint2black-app paint2black-ocr
    ```
    *(Note: On Windows PowerShell, use `${pwd}` instead of `$(pwd)` for the volume path)*

    The container's startup script automatically sets correct permissions on
    `/app/temp`, so the application can write uploaded files without any manual
    permission changes on the host.
6.  **Access the Application:** Open your web browser and navigate to `http://localhost:5018`.

## Usage

1.  Navigate to the application URL (`http://localhost:5018`).
2.  **Color Conversion:**
    *   Select the "Step 1: Convert Color" tab.
    *   Drag & Drop or click to select the PDF file you want to convert.
    *   Choose either "Monochrome" or "Grayscale".
    *   Click "Start Conversion".
    *   Monitor the progress bar.
    *   Once complete, click the "Download Converted PDF" button.
3.  **OCR Processing:**
    *   Select the "Step 2: Perform OCR" tab.
    *   Drag & Drop or click to select the PDF file you want to apply OCR to (this can be the file downloaded from Step 1 or any other PDF).
    *   Select the primary language of the document text from the dropdown.
    *   Click "Start OCR".
    *   Monitor the progress bar.
    *   Once complete, click the "Download OCR'd PDF" button. The resulting PDF will have selectable text.

## Configuration

Several parameters can be adjusted near the top of the `app.py` file:

*   `app.config['MAX_CONTENT_LENGTH']`: Maximum allowed upload size (in bytes).
*   `app.config['ALLOWED_EXTENSIONS']`: Allowed file extensions (currently just 'pdf').
*   `app.config['SUPPORTED_OCR_LANGUAGES']`: List of Tesseract language codes available. Ensure the corresponding `tesseract-ocr-[lang]` packages are installed in the `Dockerfile`.
*   `app.config['CONVERSION_DPI']`: Resolution (Dots Per Inch) used when rendering PDF pages to images for processing. Higher values improve quality but increase processing time and memory usage.

The number of parallel workers for OCR (`max_ocr_workers`) and Gunicorn settings (`--workers`, `--timeout`) are configured within `app.py` and the `Dockerfile` respectively.

## Known Limitations

*   **In-Memory Progress Storage:** Task status is tracked in the application's memory. If the Flask/Gunicorn server process restarts, progress information for any running tasks will be lost. For production use, an external store like Redis or a database is recommended.
*   **Single Gunicorn Worker Requirement:** Due to the in-memory progress storage, the application is configured to run with a single Gunicorn worker (`--workers 1` in the Dockerfile `CMD`). This limits its ability to handle many simultaneous user requests concurrently. Background tasks *within* a request are parallelized where appropriate (OCR).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. (Add details if applicable)

## License

MIT License